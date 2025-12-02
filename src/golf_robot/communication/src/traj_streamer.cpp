// stream_dq_like_test_with_logging.cpp
// Reads dq0..dq5 from a trajectory CSV and streams them like test.cpp,
// while logging measured robot state to a CSV file.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <atomic>
#include <thread>
#include <chrono>
#include <csignal>
#include <mutex>
#include <condition_variable>
#include <filesystem>
#include <cmath>

#include "../include/communication/ur_driver.h"
#include "../include/communication/ur5.h"

static const std::string CSV_IN   = "log/trajectory_sim.csv";      // input
// static const std::string CSV_OUT  = "log/streamed_measurements.csv"; // output
static const std::string CSV_OUT =
    "/home/thomas/Documents/masters_project/golf_robot/log/streamed_measurements.csv";

// static const std::string ROBOT_IP = "192.168.56.101";              // robot/sim IP
static const std::string ROBOT_IP = "192.38.66.227"; 
static const int         REVERSE_PORT = 5007;
static const double      ACCEL  = 5.0;
static const double      DT     = 0.008;
static const double      VCLAMP = 3.0;
static const double      kp = 0.1;
static const double      ki = 1.0;

static const double MAX_ERROR_DEG = 1.0;
static const double MAX_ERROR_RAD = MAX_ERROR_DEG * 3.14 / 180.0;

// --- Minimal 2x2 per-joint KF (q, dq) ---
struct KF2 {
    double q=0.0, dq=0.0;
    double P00=1e-4, P01=0.0, P10=0.0, P11=1.0;
    void predict(double dt, double sigma_a){
        if (dt <= 0) return;
        const double g0=0.5*dt*dt, g1=dt, s2=sigma_a*sigma_a;
        // x = F x
        q  += dt*dq;
        // P = FPF^T + Q
        const double nP00 = P00 + dt*(P01+P10) + dt*dt*P11 + s2*g0*g0;
        const double nP01 = P01 + dt*P11 + s2*g0*g1;
        const double nP10 = P10 + dt*P11 + s2*g0*g1;
        const double nP11 = P11 + s2*g1*g1;
        P00=nP00; P01=nP01; P10=nP10; P11=nP11;
    }
    void update_pos(double z, double R){
        const double S = P00 + R;
        const double K0 = P00 / S;
        const double K1 = P10 / S;
        const double r  = z - q;
        q  += K0 * r;
        dq += K1 * r;
        const double nP00 = (1.0 - K0)*P00;
        const double nP01 = (1.0 - K0)*P01;
        const double nP10 = P10 - K1*P00;
        const double nP11 = P11 - K1*P01;
        P00=nP00; P01=nP01; P10=nP10; P11=nP11;
    }
};



std::atomic<bool> keep_running(true);
void on_sigint(int){ keep_running = false; }

static inline double clamp(double x, double lo, double hi){
    return x < lo ? lo : (x > hi ? hi : x);
}

static std::vector<std::string> split_csv(const std::string& line){
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string tok;
    while(std::getline(ss, tok, ',')) out.push_back(tok);
    return out;
}

int main(){
    std::signal(SIGINT, on_sigint);

    std::condition_variable rt_msg_cond_;
    std::condition_variable msg_cond_;
    UrDriver mydriver(rt_msg_cond_, msg_cond_, ROBOT_IP, REVERSE_PORT);
    mydriver.start();

    // Wait for at least one RT packet so state is valid before streaming
    {
        std::mutex msg_lock;
        std::unique_lock<std::mutex> locker(msg_lock);
        while(!mydriver.rt_interface_->robot_state_->getDataPublished()){
            rt_msg_cond_.wait(locker);
        }
        // clear the flag so subsequent waits will block until the next packet
        mydriver.rt_interface_->robot_state_->setDataPublished();
    }

    // Ensure the parent directory for the CSV_OUT exists (use absolute path's parent)
    try {
        std::filesystem::path outp(CSV_OUT);
        if(!outp.has_parent_path()){
            std::filesystem::create_directories("./");
        } else {
            std::filesystem::create_directories(outp.parent_path());
        }
    } catch(const std::exception &e){
        std::cerr << "[WARN] Failed to create parent directories for " << CSV_OUT << ": " << e.what() << "\n";
    }

    // --- Load dq trajectories ---
    std::ifstream ifs(CSV_IN);
    if(!ifs){
        std::cerr << "[ERROR] Cannot open CSV at " << CSV_IN << "\n";
        return 1;
    }

    std::vector<std::array<double,6>> dq_rows;
    std::vector<std::array<double,6>> q_rows;
    dq_rows.reserve(100000);
    q_rows.reserve(100000);
    std::string line;
    bool header_checked = false;

    while(std::getline(ifs, line)){
        if(line.empty()) continue;
        auto toks = split_csv(line);
        if(toks.size() < 13) continue;

        if(!header_checked){
            header_checked = true;
            try { (void)std::stod(toks[7]); } 
            catch(...) { continue; } // skip header
        }

        std::array<double,6> dq{};
        std::array<double,6> q{};
        try {
            for(int j=0;j<6;++j){
                q[j] = std::stod(toks[1+j]);
                dq[j] = std::stod(toks[7+j]);
            }
            q_rows.push_back(q);
            dq_rows.push_back(dq);
        } catch(...){
            continue;
        }
    }
    if(dq_rows.empty()){
        std::cerr << "[ERROR] No dq rows parsed.\n";
        return 1;
    }

    std::cout << "[INFO] Loaded " << dq_rows.size() << " dq rows\n";
    
    // We'll collect measurements in-memory and write the CSV at the end of the run.
    std::cout << "[INFO] Will write measurements to: " << CSV_OUT << " when run completes\n";

    // --- MOVE TO START (optional but recommended) ---
    // Use the first CSV joint positions as the start pose.
    std::vector<double> q_start(6);
    for(int i=0;i<6;i++) q_start[i] = q_rows.front()[i];

    // Wait for a valid current measurement (we did an initial wait earlier), then move
    std::vector<double> q_current = mydriver.rt_interface_->robot_state_->getQActual();

    // Use URScript movej to go to start position (then poll until reached)
    {
        const double a_move = 1.2; // accel
        const double v_move = 0.25; // vel
        char cmdbuf[512];
        snprintf(cmdbuf, sizeof(cmdbuf), "movej([%.6f,%.6f,%.6f,%.6f,%.6f,%.6f], a=%.3f, v=%.3f)\n",
                 q_start[0], q_start[1], q_start[2], q_start[3], q_start[4], q_start[5], a_move, v_move);
        mydriver.rt_interface_->addCommandToQueue(std::string(cmdbuf));

        // Wait until robot actual joints are close to q_start (or timeout)
        const double tol = 1e-3; // radians
        const double timeout_s = 10.0; // safety timeout
        auto wait_t0 = std::chrono::steady_clock::now();
        bool reached = false;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_t0).count() < timeout_s) {
            std::vector<double> cur = mydriver.rt_interface_->robot_state_->getQActual();
            if (cur.size() == 6) {
                double err = 0.0;
                for (int k = 0; k < 6; ++k) err += fabs(cur[k] - q_start[k]);
                if (err < tol) { reached = true; break; }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (!reached) {
            std::cerr << "[WARN] move-to-start did not reach target within timeout\n";
            return 1;
        } else {
            std::cout << "[INFO] Move-to-start reached target\n";
            
        }
    }



    // --- Control loop with KF (event-driven updates) and fixed 8 ms pacing ---
    using clock = std::chrono::steady_clock;

    const bool   USE_KF   = true;      // toggle here
    const double SIGMA_A  = 3.0;       // rad/s^2   (process accel noise)
    const double SIGMA_Q  = 1e-3;      // rad       (meas noise std for q)

    std::array<KF2,6> kf;
    for (int j=0;j<6;++j) { kf[j].q = q_rows.front()[j]; kf[j].dq = 0.0; }

    // controller params (you can tune)
    std::array<double,6> Kp_pos  = {kp, kp, kp, kp, kp, kp};
    std::array<double,6> VCLAMPs = {VCLAMP, VCLAMP, VCLAMP, VCLAMP, VCLAMP, VCLAMP};
    // --- I-term params and state ---
    std::array<double,6> Ki_pos  = {ki, ki, ki, ki, ki, ki};   // rad/s per rad  (tune >0 to enable)
    std::array<double,6> I_CLAMP = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};   // clamp on integral state (rad)
    double I_LEAK = 0.0;                                         // 0 = no leak; e.g. 0.01 for slow bleed

    std::array<double,6> I = {0,0,0,0,0,0};  // integral state per joint (integrates position error)


    auto* state = mydriver.rt_interface_->robot_state_;

    auto   t0       = clock::now();
    auto   kf_time  = t0;                // time the KF state corresponds to
    auto   next_tick= t0;                // fixed send times
    size_t N        = dq_rows.size();

    // Logs:
    //  - meas_events: event-driven (measurement arrival times)
    //  - kf_pred: fixed-rate (prediction at each send time)
    std::vector<std::array<double,13>> meas_events;
    std::vector<std::array<double,13>> kf_pred(N);
    std::mutex meas_log_mtx;

    // keep last measurement values for logging (not for control when KF is on)
    std::vector<double> last_q_meas(6, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> last_dq_meas(6, std::numeric_limits<double>::quiet_NaN());

    // KF shared (protected) if you later extend to multi-threaded predicts;
    // here, updates run on a thread, predicts in main thread
    std::mutex kf_mtx;

    // --------- Event-driven measurement thread ---------
    std::atomic<bool> run_meas(true);
    std::thread meas_thread([&]{
        std::mutex wait_mtx;
        uint64_t seen = state->packet_count_.load(std::memory_order_relaxed);

        while (run_meas.load(std::memory_order_relaxed)) {
            std::unique_lock<std::mutex> lk(wait_mtx);
            rt_msg_cond_.wait(lk, [&]{
                return !run_meas.load(std::memory_order_relaxed) ||
                    state->packet_count_.load(std::memory_order_relaxed) > seen;
            });
            if (!run_meas.load(std::memory_order_relaxed)) break;
            lk.unlock();

            seen = state->packet_count_.load(std::memory_order_relaxed);
            auto t_meas = clock::now();

            // fetch measurement
            const std::vector<double> q_meas  = state->getQActual();
            const std::vector<double> dq_meas = state->getQdActual();

            // KF: advance to measurement time, then update
            if (USE_KF) {
                std::lock_guard<std::mutex> g(kf_mtx);
                const double dt_meas = std::chrono::duration<double>(t_meas - kf_time).count();
                for (int j=0;j<6;++j) kf[j].predict(dt_meas, SIGMA_A);
                kf_time = t_meas;
                for (int j=0;j<6;++j)
                    if (j < (int)q_meas.size()) kf[j].update_pos(q_meas[j], SIGMA_Q*SIGMA_Q);
            }

            // store last measurement (for logging)
            for (int j=0;j<6;++j){
                if (j < (int)q_meas.size())  last_q_meas[j]  = q_meas[j];
                if (j < (int)dq_meas.size()) last_dq_meas[j] = dq_meas[j];
            }

            // log the measurement event row (t, q[0..5], dq[0..5]) at t_meas
            std::array<double,13> row{};
            row[0] = std::chrono::duration<double>(t_meas - t0).count();
            for (int j=0;j<6;++j) row[j+1] = (j<(int)q_meas.size()?  q_meas[j]  : std::numeric_limits<double>::quiet_NaN());
            for (int j=0;j<6;++j) row[j+7] = (j<(int)dq_meas.size()? dq_meas[j] : std::numeric_limits<double>::quiet_NaN());
            {
                std::lock_guard<std::mutex> lg(meas_log_mtx);
                meas_events.push_back(row);
            }
        }
    });

    bool aborted_due_to_error = false;
    // --------- Fixed 8 ms command loop ---------
    for (size_t i = 0; i < N; ++i) {
        next_tick += std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(DT));

        // Predict KF to the exact send time, then read the predicted state
        std::array<double,6> q_hat{}, dq_hat{};
        {
            std::lock_guard<std::mutex> g(kf_mtx);
            if (USE_KF) {
                const double dt_to_cmd = std::chrono::duration<double>(next_tick - kf_time).count();
                for (int j=0;j<6;++j) kf[j].predict(dt_to_cmd, SIGMA_A);
                kf_time = next_tick;
                for (int j=0;j<6;++j){ q_hat[j] = kf[j].q; dq_hat[j] = kf[j].dq; }
            } else {
                // fallback: use last measurement (or desired) as estimate
                for (int j=0;j<6;++j){
                    q_hat[j]  = std::isnan(last_q_meas[j]) ? q_rows[i][j] : last_q_meas[j];
                    dq_hat[j] = std::isnan(last_dq_meas[j])? dq_rows[i][j] : last_dq_meas[j];
                }
            }
        }

    // Build command: FF + P on position error + I on position error
    const auto& v_ff = dq_rows[i];
    std::array<double,6> v_cmd = v_ff;
    
    bool large_error_this_step = false;


    for (int j=0;j<6;++j){
        const double e = q_rows[i][j] - q_hat[j];

        if (std::fabs(e) > MAX_ERROR_RAD) {
            large_error_this_step = true;
        }

        // --- I-term update (integrate error over time between sends) ---
        // simple rectangle integration + optional leak
        I[j] = (1.0 - I_LEAK) * I[j] + e * DT;

        // anti-windup clamp on the integrator state itself
        if (I[j] >  I_CLAMP[j]) I[j] =  I_CLAMP[j];
        if (I[j] < -I_CLAMP[j]) I[j] = -I_CLAMP[j];

        // control law: v = v_ff + P*(e)/DT + Ki*I
        double u = v_ff[j] + (Kp_pos[j] * e) / DT + Ki_pos[j] * I[j];

        // output clamp (velocity clamp)
        if (u >  VCLAMPs[j]) u =  VCLAMPs[j];
        if (u < -VCLAMPs[j]) u = -VCLAMPs[j];

        // optional back-calculation anti-windup (uncomment to use)
        // const double u_unsat = v_ff[j] + (Kp_pos[j] * e) / DT + Ki_pos[j] * I[j];
        // const double aw = u - u_unsat;            // saturation error (0 when unsaturated)
        // const double Kaw = 0.2;                   // back-calc gain (tune small)
        // I[j] += (-Kaw * aw) / std::max(1e-6, Ki_pos[j]);

        v_cmd[j] = u;
    }

    if (large_error_this_step) {
        std::cerr << "[ERROR] joint error exceeded "
                 << MAX_ERROR_DEG << " deg at step " << i 
                 << ". Stopping and returning to start.\n";
        
        mydriver.setSpeed(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ACCEL, 1);

        mydriver.rt_interface_->addCommandToQueue("stopj(1.0)\n");

        std::this_thread::sleep_for(std::chrono::milliseconds(1500));

        aborted_due_to_error = true;
        break;
    }

        // Log KF prediction at send time
        std::array<double,13> prow{};
        prow[0] = std::chrono::duration<double>(next_tick - t0).count();
        for (int j=0;j<6;++j){ prow[j+1] = q_hat[j]; prow[j+7] = dq_hat[j]; }
        kf_pred[i] = prow;

        // Send on schedule
        std::this_thread::sleep_until(next_tick);
        mydriver.setSpeed(v_cmd[0], v_cmd[1], v_cmd[2], v_cmd[3], v_cmd[4], v_cmd[5], ACCEL, 0.03);
    }

    // Stop the meas thread before return-to-start
    run_meas.store(false);
    rt_msg_cond_.notify_all();
    if (meas_thread.joinable()) meas_thread.join();




    mydriver.rt_interface_->addCommandToQueue("stopj(1.0)\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    // --- RETURN TO START (optional) ---
    // std::cout << "[INFO] Performing return-to-start...\n";
    // {
    //     std::vector<double> q_end(6);
    //     for(int i=0;i<6;i++) q_end[i] = q_rows.back()[i];
    //     const double a_move = 1.2;
    //     const double v_move = 0.25;
    //     char cmdbuf[512];
    //     // snprintf(cmdbuf, sizeof(cmdbuf), "movej([%.6f,%.6f,%.6f,%.6f,%.6f,%.6f], a=%.3f, v=%.3f)\n",
    //     //          q_start[0], q_start[1], q_start[2], q_start[3], q_start[4], q_start[5], a_move, v_move);
    //     snprintf(cmdbuf, sizeof(cmdbuf), "movej([-2.520, -2.434, -0.898,  0.802,  0.913,  0.501], a=%.3f, v=%.3f)\n",
    //              a_move, v_move);
    //     mydriver.rt_interface_->addCommandToQueue(std::string(cmdbuf));

    //     const double tol = 1e-3;
    //     const double timeout_s = 10.0;
    //     auto wait_t0 = std::chrono::steady_clock::now();
    //     bool reached = false;
    //     while (std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_t0).count() < timeout_s) {
    //         std::vector<double> cur = mydriver.rt_interface_->robot_state_->getQActual();
    //         if (cur.size() == 6) {
    //             double err = 0.0;
    //             for (int k = 0; k < 6; ++k) err += fabs(cur[k] - q_start[k]);
    //             if (err < tol) { reached = true; break; }
    //         }
    //         std::this_thread::sleep_for(std::chrono::milliseconds(50));
    //     }
    //     if (!reached) {
    //         std::cerr << "[WARN] return-to-start did not reach target within timeout\n";
    //     } else {
    //         std::cout << "[INFO] Return-to-start reached target\n";
    //     }
    // }
    
    std::cout << "[INFO] Performing return-to-start...\n";
    {
        std::vector<double> q_end(6);
        for(int i=0;i<6;i++) q_end[i] = q_rows.back()[i];
        const double a_move = 1.2;
        const double v_move = 0.25;
        char cmdbuf[512];
        snprintf(cmdbuf, sizeof(cmdbuf), "movej([%.6f,%.6f,%.6f,%.6f,%.6f,%.6f], a=%.3f, v=%.3f)\n",
                 q_start[0], q_start[1], q_start[2], q_start[3], q_start[4], q_start[5], a_move, v_move);
        // snprintf(cmdbuf, sizeof(cmdbuf), "movej([-2.520, -2.434, -0.898,  0.802,  0.913,  0.501], a=%.3f, v=%.3f)\n",
        //          a_move, v_move);
        mydriver.rt_interface_->addCommandToQueue(std::string(cmdbuf));

        const double tol = 1e-3;
        const double timeout_s = 10.0;
        auto wait_t0 = std::chrono::steady_clock::now();
        bool reached = false;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_t0).count() < timeout_s) {
            std::vector<double> cur = mydriver.rt_interface_->robot_state_->getQActual();
            if (cur.size() == 6) {
                double err = 0.0;
                for (int k = 0; k < 6; ++k) err += fabs(cur[k] - q_start[k]);
                if (err < tol) { reached = true; break; }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (!reached) {
            std::cerr << "[WARN] return-to-start did not reach target within timeout\n";
        } else {
            std::cout << "[INFO] Return-to-start reached target\n";
        }
    }
    mydriver.halt();
    std::cout << "[DONE] Writing logs...\n";

    // Derive KF csv path next to CSV_OUT
    std::filesystem::path out_meas(CSV_OUT);
    std::filesystem::path out_kf   = out_meas.parent_path() / "kf_predictions.csv";

    // ---- Write measurements to CSV_OUT (event-driven) ----
    {
        std::ofstream file(out_meas);
        if (!file.is_open()) {
            std::cerr << "[ERROR] could not open " << out_meas << " for writing.\n";
            return 1;
        }
        file << "t";
        for (int j=0;j<6;++j) file << ",q"  << j;
        for (int j=0;j<6;++j) file << ",dq" << j;
        file << "\n";

        // meas_events was filled by the measurement thread
        {
            std::lock_guard<std::mutex> lg(meas_log_mtx);
            for (const auto& row : meas_events) {
                for (int c=0;c<13;++c) { file << row[c]; if (c<12) file << ","; }
                file << "\n";
            }
        }
    }

    // ---- Write KF predictions at send times ----
    {
        std::ofstream file(out_kf);
        if (!file.is_open()) {
            std::cerr << "[ERROR] could not open " << out_kf << " for writing.\n";
            return 1;
        }
        file << "t";
        for (int j=0;j<6;++j) file << ",qhat"  << j;
        for (int j=0;j<6;++j) file << ",dqhat" << j;
        file << "\n";

        for (const auto& row : kf_pred) {
            for (int c=0;c<13;++c) { file << row[c]; if (c<12) file << ","; }
            file << "\n";
        }
    }

    std::cout << "[DONE] Logged measurements to " << out_meas << "\n";
    std::cout << "[DONE] Logged KF predictions to " << out_kf   << "\n";
    return 0;

}
