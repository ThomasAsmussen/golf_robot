
// --- Minimal 2x2 per-joint KF (q, dq) ---
// struct KF2 {
//     double q=0.0, dq=0.0;
//     double P00=1e-4, P01=0.0, P10=0.0, P11=1.0;
//     void predict(double dt, double sigma_a){
//         if (dt <= 0) return;
//         const double g0=0.5*dt*dt, g1=dt, s2=sigma_a*sigma_a;
//         // x = F x
//         q  += dt*dq;
//         // P = FPF^T + Q
//         const double nP00 = P00 + dt*(P01+P10) + dt*dt*P11 + s2*g0*g0;
//         const double nP01 = P01 + dt*P11 + s2*g0*g1;
//         const double nP10 = P10 + dt*P11 + s2*g0*g1;
//         const double nP11 = P11 + s2*g1*g1;
//         P00=nP00; P01=nP01; P10=nP10; P11=nP11;
//     }
//     void update_pos(double z, double R){
//         const double S = P00 + R;
//         const double K0 = P00 / S;
//         const double K1 = P10 / S;
//         const double r  = z - q;
//         q  += K0 * r;
//         dq += K1 * r;
//         const double nP00 = (1.0 - K0)*P00;
//         const double nP01 = (1.0 - K0)*P01;
//         const double nP10 = P10 - K1*P00;
//         const double nP11 = P11 - K1*P01;
//         P00=nP00; P01=nP01; P10=nP10; P11=nP11;
//     }
// };

// struct KF2 {
//     double q = 0.0, dq = 0.0;
//     double P00 = 1e-4, P01 = 0.0, P10 = 0.0, P11 = 1.0;

//     void predict(double dt, double sigma_a) {
//         if (dt <= 0) return;
//         const double g0 = 0.5 * dt * dt;
//         const double g1 = dt;
//         const double s2 = sigma_a * sigma_a;

//         // x = F x
//         q  += dt * dq;

//         // P = F P F^T + Q (white acceleration model)
//         const double nP00 = P00 + dt*(P01+P10) + dt*dt*P11 + s2*g0*g0;
//         const double nP01 = P01 + dt*P11 + s2*g0*g1;
//         const double nP10 = P10 + dt*P11 + s2*g0*g1;
//         const double nP11 = P11 + s2*g1*g1;
//         P00 = nP00; P01 = nP01; P10 = nP10; P11 = nP11;
//     }

//     // measurement: z = [q_meas, dq_meas]; R = diag(Rq, Rdq)
//     void update_q_dq(double zq, double zdq, double Rq, double Rdq) {
//         // S = P + R
//         double S00 = P00 + Rq;
//         double S01 = P01;
//         double S10 = P10;
//         double S11 = P11 + Rdq;

//         // inv(S) for 2x2
//         double det = S00 * S11 - S01 * S10;
//         if (std::fabs(det) < 1e-12) {
//             // fallback: only position update if S is near singular
//             return;
//         }
//         double invS00 =  S11 / det;
//         double invS01 = -S01 / det;
//         double invS10 = -S10 / det;
//         double invS11 =  S00 / det;

//         // K = P * S^{-1}
//         double K00 = P00 * invS00 + P01 * invS10;
//         double K01 = P00 * invS01 + P01 * invS11;
//         double K10 = P10 * invS00 + P11 * invS10;
//         double K11 = P10 * invS01 + P11 * invS11;

//         // innovation r = z - x
//         double rq  = zq  - q;
//         double rdq = zdq - dq;

//         // update state
//         q  += K00 * rq + K01 * rdq;
//         dq += K10 * rq + K11 * rdq;

//         // update covariance: P = (I - K) P
//         double I00 = 1.0 - K00;
//         double I01 =    - K01;
//         double I10 =    - K10;
//         double I11 = 1.0 - K11;

//         double nP00 = I00*P00 + I01*P10;
//         double nP01 = I00*P01 + I01*P11;
//         double nP10 = I10*P00 + I11*P10;
//         double nP11 = I10*P01 + I11*P11;

//         P00 = nP00; P01 = nP01; P10 = nP10; P11 = nP11;
//     }
// };


// stream_dq_like_test_with_logging.cpp
// Reads q0..q5 and dq0..dq5 from a trajectory CSV,
// streams joint velocities with speedj (via mydriver.setSpeed),
// and logs:
//  1) event-driven measurements (q_actual, dq_actual) -> streamed_measurements.csv
//  2) KF predictions at each 8 ms send tick            -> kf_predictions.csv
//
// KF uses a simple actuator model to "feed commanded velocity" into predict():
//   dq_next = a*dq + (1-a)*dq_cmd,   a = exp(-dt/tau)
//
// Build:
//   g++ -O2 -std=c++17 -pthread stream_dq_like_test_with_logging.cpp -o stream_test
// (adjust include/lib paths to your project)

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
#include <limits>

#include "../include/communication/ur_driver.h"
#include "../include/communication/ur5.h"

//static const std::string CSV_IN   = "C:/Users/marti/'OneDrive - Danmarks Tekniske Universitet'/DTU/GitHub/golf_robot/log/trajectory_sim.csv";      // input
static const std::string CSV_IN   = "log/trajectory_sim.csv";      // input
static const std::string CSV_OUT  = "log/streamed_measurements.csv"; // output
//static const std::string CSV_OUT =
//    "/home/thomas/Documents/masters_project/golf_robot/log/streamed_measurements.csv";
//static const std::string CSV_OUT =
//    "C:/Users/marti/'OneDrive - Danmarks Tekniske Universitet'/DTU/GitHub/golf_robot/log/streamed_measurements.csv";
// static const std::string CSV_IN =
//     "/mnt/c/Users/marti/OneDrive - Danmarks Tekniske Universitet/DTU/GitHub/golf_robot/log/trajectory_sim.csv";

// static const std::string CSV_OUT =
//     "/mnt/c/Users/marti/OneDrive - Danmarks Tekniske Universitet/DTU/GitHub/golf_robot/log/streamed_measurements.csv";


static const std::string ROBOT_IP = "192.38.66.227";
static const int         REVERSE_PORT = 5007;
static const std::array<double,6> END_POS = {-2.47, -2.38, -1.55, 1.66, 0.49, -0.26};

static const double ACCEL  = 6.0;
static const double DT     = 0.008;     // 8 ms
static const double VCLAMP = 3.5;

static const double kp = 0.0;
static const double kd = 0.0;
static const double ki = 0.0;
// Anti-windup / safety
static const double ITERM_CLAMP = 1.0;  // max magnitude of integral contribution in rad/s
static const double IQ_CLAMP    = 10.0; // max magnitude of integral state in "rad*s" (backup clamp)


static const double MAX_ERROR_DEG = 2.0;
static const double MAX_ERROR_RAD = MAX_ERROR_DEG * 3.14159265358979323846 / 180.0;

// KF tuning
static const bool   USE_KF   = true;
static const double SIGMA_A  = 8.0e-1;     // rad/s^2 process accel noise
static const double SIGMA_Q  = 1e-4;    // rad meas std
static const double SIGMA_DQ = 10;    // rad/s meas std (NOTE: 1e-4 is often too optimistic on real robot)
// static const std::array<double,6> TAU_CMD = {
//     0.0180,  // joint 0
//     0.0182,  // joint 1
//     0.0165,  // joint 2
//     0.0137,  // joint 3
//     0.0137,  // joint 4
//     0.0129   // joint 5
// };
static const std::array<double,6> TAU_CMD = {
    1,  // joint 0
    1,  // joint 1
    1,  // joint 2
    1,  // joint 3
    1,  // joint 4
    1   // joint 5
};
// static const double TAU_CMD  = 0.024;    // seconds: actuator time constant for dq tracking

// -------------------- Utilities --------------------
std::atomic<bool> keep_running(true);
void on_sigint(int){ keep_running = false; }

static std::vector<std::string> split_csv(const std::string& line){
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string tok;
    while(std::getline(ss, tok, ',')) out.push_back(tok);
    return out;
}

// -------------------- Minimal 2x2 KF per joint: x=[q,dq] --------------------
struct KF2 {
    double q = 0.0, dq = 0.0;
    double P00 = 1e-4, P01 = 0.0, P10 = 0.0, P11 = 1.0;

    // Predict using commanded dq via a first-order lag model.
    // dq_next = a*dq + (1-a)*dq_cmd,  a=exp(-dt/tau)
    void predict_cmd(double dt, double dq_cmd, double tau, double sigma_a) {
        if (dt <= 0) return;

        const double tau_eff = std::max(1e-4, tau);
        const double a = std::exp(-dt / tau_eff);

        const double dq_next = a * dq + (1.0 - a) * dq_cmd;

        // State propagation (integrate with dq_next)
        q  += dt * dq_next;
        dq  = dq_next;

        // Linearized dynamics: x_next = F x + B u + w
        // dq_next = a*dq + (1-a)*u
        // q_next  = q + dt*dq_next = q + dt*a*dq + dt*(1-a)*u
        const double F00 = 1.0;
        const double F01 = dt * a;
        const double F10 = 0.0;
        const double F11 = a;

        // Simple Q from white-acceleration noise (good enough here)
        const double g0 = 0.5 * dt * dt;
        const double g1 = dt;
        const double s2 = sigma_a * sigma_a;
        const double Q00 = s2 * g0 * g0;
        const double Q01 = s2 * g0 * g1;
        const double Q10 = s2 * g0 * g1;
        const double Q11 = s2 * g1 * g1;

        // P = F P F^T + Q
        const double nP00 = F00*P00*F00 + F00*P01*F01 + F01*P10*F00 + F01*P11*F01 + Q00;
        const double nP01 = F00*P00*F10 + F00*P01*F11 + F01*P10*F10 + F01*P11*F11 + Q01;
        const double nP10 = F10*P00*F00 + F10*P01*F01 + F11*P10*F00 + F11*P11*F01 + Q10;
        const double nP11 = F10*P00*F10 + F10*P01*F11 + F11*P10*F10 + F11*P11*F11 + Q11;

        P00 = nP00; P01 = nP01; P10 = nP10; P11 = nP11;
    }

    // Measurement update with z=[q_meas, dq_meas], R = diag(Rq, Rdq)
    void update_q_dq(double zq, double zdq, double Rq, double Rdq) {
        // S = P + R
        const double S00 = P00 + Rq;
        const double S01 = P01;
        const double S10 = P10;
        const double S11 = P11 + Rdq;

        const double det = S00 * S11 - S01 * S10;
        if (std::fabs(det) < 1e-12) return;

        const double invS00 =  S11 / det;
        const double invS01 = -S01 / det;
        const double invS10 = -S10 / det;
        const double invS11 =  S00 / det;

        // K = P S^{-1}
        const double K00 = P00 * invS00 + P01 * invS10;
        const double K01 = P00 * invS01 + P01 * invS11;
        const double K10 = P10 * invS00 + P11 * invS10;
        const double K11 = P10 * invS01 + P11 * invS11;

        // innovation
        const double rq  = zq  - q;
        const double rdq = zdq - dq;

        // state update
        q  += K00 * rq + K01 * rdq;
        dq += K10 * rq + K11 * rdq;

        // covariance update: P = (I-K)P
        const double I00 = 1.0 - K00;
        const double I01 =    - K01;
        const double I10 =    - K10;
        const double I11 = 1.0 - K11;

        const double nP00 = I00*P00 + I01*P10;
        const double nP01 = I00*P01 + I01*P11;
        const double nP10 = I10*P00 + I11*P10;
        const double nP11 = I10*P01 + I11*P11;

        P00 = nP00; P01 = nP01; P10 = nP10; P11 = nP11;
    }
};

int main(){
    std::signal(SIGINT, on_sigint);

    // ---- Driver start ----
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
        mydriver.rt_interface_->robot_state_->setDataPublished();
    }

    // Ensure output directory exists
    try {
        std::filesystem::path outp(CSV_OUT);
        if(outp.has_parent_path())
            std::filesystem::create_directories(outp.parent_path());
    } catch(const std::exception &e){
        std::cerr << "[WARN] Failed to create parent dirs for " << CSV_OUT << ": " << e.what() << "\n";
    }

    // ---- Load trajectory CSV ----
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
            catch(...) { continue; } // header
        }

        std::array<double,6> dq{};
        std::array<double,6> q{};
        try {
            for(int j=0;j<6;++j){
                q[j]  = std::stod(toks[1+j]);
                dq[j] = std::stod(toks[7+j]);
            }
            q_rows.push_back(q);
            dq_rows.push_back(dq);
        } catch(...) {
            continue;
        }
    }

    if(dq_rows.empty()){
        std::cerr << "[ERROR] No dq rows parsed.\n";
        return 1;
    }

    std::cout << "[INFO] Loaded " << dq_rows.size() << " rows\n";
    std::cout << "[INFO] Will write measurements to: " << CSV_OUT << "\n";

    // ---- Move to start pose ----
    std::vector<double> q_start(6);
    for(int i=0;i<6;i++) q_start[i] = q_rows.front()[i];

    {
        const double a_move = 1.2;
        const double v_move = 0.25;
        char cmdbuf[512];
        snprintf(cmdbuf, sizeof(cmdbuf),
                 "movej([%.6f,%.6f,%.6f,%.6f,%.6f,%.6f], a=%.3f, v=%.3f)\n",
                 q_start[0], q_start[1], q_start[2], q_start[3], q_start[4], q_start[5],
                 a_move, v_move);
        mydriver.rt_interface_->addCommandToQueue(std::string(cmdbuf));

        const double tol = 1e-3;
        const double timeout_s = 10.0;
        auto t_wait0 = std::chrono::steady_clock::now();
        bool reached = false;

        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t_wait0).count() < timeout_s) {
            std::vector<double> cur = mydriver.rt_interface_->robot_state_->getQActual();
            if (cur.size() == 6) {
                double err = 0.0;
                for (int k = 0; k < 6; ++k) err += std::fabs(cur[k] - q_start[k]);
                if (err < tol) { reached = true; break; }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (!reached) {
            std::cerr << "[WARN] move-to-start did not reach target within timeout\n";
            return 1;
        }
        std::cout << "[INFO] Move-to-start reached target\n";
    }

    using clock = std::chrono::steady_clock;
    auto* state = mydriver.rt_interface_->robot_state_;
    std::cout << "[GAINS] kp=" << kp << " kd=" << kd << " ki=" << ki
          << " ITERM_CLAMP=" << ITERM_CLAMP << " IQ_CLAMP=" << IQ_CLAMP << "\n";

    // ---- KF init ----
    std::array<KF2,6> kf;
    for (int j=0;j<6;++j) { kf[j].q = q_rows.front()[j]; kf[j].dq = 0.0; }

    // Command shared with KF thread
    std::array<std::atomic<double>,6> last_cmd;
    for (int j=0;j<6;++j) last_cmd[j].store(0.0, std::memory_order_relaxed);

    // Controller params
    // std::array<double,6> Kp_pos = {kp,kp,kp,kp,kp,kp};
    std::array<double,6> Kd_vel = {kd,kd,kd,kd,kd,kd};
    std::array<double,6> Ki_pos = {ki, ki, ki, ki, ki, ki};
    std::array<double,6> Kp_pos = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    // std::array<double,6> Kd_vel = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    // std::array<double,6> Ki_pos = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    std::array<double,6> VCLAMPs = {VCLAMP,VCLAMP,VCLAMP,VCLAMP,VCLAMP,VCLAMP};

    // Logging buffers
    const size_t N = dq_rows.size();
    std::vector<std::array<double,13>> meas_events;
    std::vector<std::array<double,13>> kf_pred(N);
    std::mutex meas_log_mtx;

    std::vector<double> last_q_meas(6, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> last_dq_meas(6, std::numeric_limits<double>::quiet_NaN());

    std::mutex kf_mtx;

    auto t0 = clock::now();
    auto kf_time = t0;
    auto next_tick = t0;

    // ---- Measurement thread: event-driven KF update ----
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

            const std::vector<double> q_meas  = state->getQActual();
            const std::vector<double> dq_meas = state->getQdActual();

            if (USE_KF) {
                std::lock_guard<std::mutex> g(kf_mtx);

                const double dt_meas = std::chrono::duration<double>(t_meas - kf_time).count();
                if (dt_meas > 0) {
                    std::array<double,6> u{};
                    for (int j=0;j<6;++j) u[j] = last_cmd[j].load(std::memory_order_relaxed);

                    for (int j=0;j<6;++j) {
                        kf[j].predict_cmd(dt_meas, u[j], TAU_CMD[j], SIGMA_A);
                    }
                    kf_time = t_meas;
                }

                // Update with measured q,dq
                const double Rq  = SIGMA_Q  * SIGMA_Q;
                const double Rdq = SIGMA_DQ * SIGMA_DQ;
                for (int j=0;j<6;++j) {
                    if (j < (int)q_meas.size() && j < (int)dq_meas.size()) {
                        kf[j].update_q_dq(q_meas[j], dq_meas[j], Rq, Rdq);
                    }
                }
            }

            // store last measurement (for logging/fallback)
            for (int j=0;j<6;++j){
                if (j < (int)q_meas.size())  last_q_meas[j]  = q_meas[j];
                if (j < (int)dq_meas.size()) last_dq_meas[j] = dq_meas[j];
            }

            // log measurement event
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

// -------------------- Position integrator state --------------------
std::array<double,6> Iq{};
Iq.fill(0.0);

// Integrator gating + anti-windup tuning
static const double EINT_MAX_DEG = 1.0;  // only integrate when |e_q| < this
static const double EINT_MAX_RAD = EINT_MAX_DEG * 3.14159265358979323846 / 180.0;

// Soft back-calculation (0..1). Start small; 0 disables back-calc entirely.
static const double KAW = 0.2;

// ---- Fixed-rate 8ms command loop ----
for (size_t i = 0; i < N && keep_running.load(); ++i) {
    next_tick += std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(DT));

    // Predict KF to the send time (read-only copy prediction)
    std::array<double,6> q_hat{}, dq_hat{};
    {
        std::lock_guard<std::mutex> g(kf_mtx);

        if (USE_KF) {
            double dt_pred = std::chrono::duration<double>(next_tick - kf_time).count();
            if (dt_pred < 0) dt_pred = 0.0;

            std::array<double,6> u{};
            for (int j = 0; j < 6; ++j) u[j] = last_cmd[j].load(std::memory_order_relaxed);

            for (int j = 0; j < 6; ++j) {
                KF2 kf_copy = kf[j];
                if (dt_pred > 0) kf_copy.predict_cmd(dt_pred, u[j], TAU_CMD[j], SIGMA_A);
                q_hat[j]  = kf_copy.q;
                dq_hat[j] = kf_copy.dq;
            }
        } else {
            for (int j = 0; j < 6; ++j) {
                q_hat[j]  = std::isnan(last_q_meas[j])  ? q_rows[i][j]  : last_q_meas[j];
                dq_hat[j] = std::isnan(last_dq_meas[j]) ? dq_rows[i][j] : last_dq_meas[j];
            }
        }
    }

    // Build commanded velocity: feedforward + feedback
    const auto& v_ff = dq_rows[i];
    std::array<double,6> v_cmd = v_ff;

    bool large_error_this_step = false;

    for (int j = 0; j < 6; ++j) {
        const double e_q  = q_rows[i][j] - q_hat[j];
        const double e_dq = v_ff[j]      - dq_hat[j];

        if (std::fabs(e_q) > MAX_ERROR_RAD) {
            large_error_this_step = true;
            continue; // still compute others, but we will abort below
        }

        // ---- Controller (FF + PD + I on position) ----
        // Use current integral state in the pre-sat command
        const double u_unsat =
            v_ff[j]
            + Kp_pos[j] * e_q
            + Kd_vel[j] * e_dq
            + Ki_pos[j] * Iq[j];

        // Saturate to speed limits
        double u_sat = u_unsat;
        if (u_sat >  VCLAMPs[j]) u_sat =  VCLAMPs[j];
        if (u_sat < -VCLAMPs[j]) u_sat = -VCLAMPs[j];
        const bool sat = (u_sat != u_unsat);

        // ---- Integrator update (gated) + soft back-calc anti-windup ----
        double Iq_new = Iq[j];

        if (Ki_pos[j] > 0.0) {
            // Integrate only when close AND not saturated
            if (!sat && (std::fabs(e_q) < EINT_MAX_RAD)) {
                Iq_new += e_q * DT;
            }

            // Soft back-calculation only when saturated
            if (sat && (KAW > 0.0)) {
                Iq_new += KAW * (u_sat - u_unsat) / Ki_pos[j];
            }

            // Backup clamp on raw integrator state
            if (Iq_new >  IQ_CLAMP) Iq_new =  IQ_CLAMP;
            if (Iq_new < -IQ_CLAMP) Iq_new = -IQ_CLAMP;

            // Clamp integral contribution directly (most meaningful)
            double iterm = Ki_pos[j] * Iq_new;
            if (iterm >  ITERM_CLAMP) iterm =  ITERM_CLAMP;
            if (iterm < -ITERM_CLAMP) iterm = -ITERM_CLAMP;
            Iq_new = iterm / Ki_pos[j];
        }

        Iq[j]   = Iq_new;
        // v_cmd[j] = u_sat;
        v_cmd[j] = u_unsat;
    }

    // Abort on large error: stop robot and reset integrators
    if (large_error_this_step) {
        std::cerr << "[ERROR] joint error exceeded " << MAX_ERROR_DEG
                  << " deg at step " << i << ". Stopping.\n";

        Iq.fill(0.0);

        mydriver.setSpeed(0, 0, 0, 0, 0, 0, ACCEL, 1);
        mydriver.rt_interface_->addCommandToQueue("stopj(1.0)\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        aborted_due_to_error = true;
        break;
    }

    // Log KF prediction at send time (q_hat,dq_hat)
    {
        std::array<double,13> prow{};
        prow[0] = std::chrono::duration<double>(next_tick - t0).count();
        for (int j = 0; j < 6; ++j) { prow[j+1] = q_hat[j]; prow[j+7] = dq_hat[j]; }
        kf_pred[i] = prow;
    }

    // Send on schedule
    std::this_thread::sleep_until(next_tick);
    mydriver.setSpeed(v_cmd[0], v_cmd[1], v_cmd[2], v_cmd[3], v_cmd[4], v_cmd[5], ACCEL, 0.03);

    // Publish last_cmd ONCE per tick, after we know what we actually sent
    for (int j = 0; j < 6; ++j) last_cmd[j].store(v_cmd[j], std::memory_order_relaxed);
}


    // Stop measurement thread
    run_meas.store(false);
    rt_msg_cond_.notify_all();
    if (meas_thread.joinable()) meas_thread.join();

    // Stop robot motion
    mydriver.rt_interface_->addCommandToQueue("stopj(1.0)\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    // Return to start pose (even if aborted)
    std::cout << "[INFO] Performing return-to-start...\n";
    {
        const double a_move = 1.2;
        const double v_move = 0.25;
        char cmdbuf[512];
        // snprintf(cmdbuf, sizeof(cmdbuf),
        //          "movej([%.6f,%.6f,%.6f,%.6f,%.6f,%.6f], a=%.3f, v=%.3f)\n",
        //          q_start[0], q_start[1], q_start[2], q_start[3], q_start[4], q_start[5],
        //          a_move, v_move);
        snprintf(cmdbuf, sizeof(cmdbuf),
                 "movej([%.6f,%.6f,%.6f,%.6f,%.6f,%.6f], a=%.3f, v=%.3f)\n",
                 END_POS[0], END_POS[1], END_POS[2], END_POS[3], END_POS[4], END_POS[5],
                 a_move, v_move);
        mydriver.rt_interface_->addCommandToQueue(std::string(cmdbuf));

        const double tol = 1e-3;
        const double timeout_s = 10.0;
        auto t_wait0 = std::chrono::steady_clock::now();
        bool reached = false;

        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t_wait0).count() < timeout_s) {
            std::vector<double> cur = mydriver.rt_interface_->robot_state_->getQActual();
            if (cur.size() == 6) {
                double err = 0.0;
                for (int k = 0; k < 6; ++k) err += std::fabs(cur[k] - q_start[k]);
                if (err < tol) { reached = true; break; }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (!reached) std::cerr << "[WARN] return-to-start did not reach target within timeout\n";
        else          std::cout << "[INFO] Return-to-start reached target\n";
    }

    mydriver.halt();

    // ---- Write logs ----
    std::cout << "[DONE] Writing logs...\n";

    std::filesystem::path out_meas(CSV_OUT);
    std::filesystem::path out_kf   = out_meas.parent_path() / "kf_predictions.csv";

    // Measurements (event-driven)
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

        std::lock_guard<std::mutex> lg(meas_log_mtx);
        for (const auto& row : meas_events) {
            for (int c=0;c<13;++c) { file << row[c]; if (c<12) file << ","; }
            file << "\n";
        }
    }

    // KF predictions at send ticks
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
    if (aborted_due_to_error) return 2;
    return 0;
}
