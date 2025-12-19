#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <limits>

struct KF2 {
    double q = 0.0, dq = 0.0;
    double P00 = 1e-4, P01 = 0.0, P10 = 0.0, P11 = 1.0;

    void predict(double dt, double sigma_a) {
        if (dt <= 0) return;
        const double g0 = 0.5 * dt * dt;
        const double g1 = dt;
        const double s2 = sigma_a * sigma_a;

        // x = F x
        q  += dt * dq;

        // P = F P F^T + Q (white acceleration model)
        const double nP00 = P00 + dt*(P01+P10) + dt*dt*P11 + s2*g0*g0;
        const double nP01 = P01 + dt*P11 + s2*g0*g1;
        const double nP10 = P10 + dt*P11 + s2*g0*g1;
        const double nP11 = P11 + s2*g1*g1;
        P00 = nP00; P01 = nP01; P10 = nP10; P11 = nP11;
    }

    // measurement: z = [q_meas, dq_meas]; R = diag(Rq, Rdq)
    void update_q_dq(double zq, double zdq, double Rq, double Rdq) {
        // S = P + R
        double S00 = P00 + Rq;
        double S01 = P01;
        double S10 = P10;
        double S11 = P11 + Rdq;

        // inv(S) for 2x2
        double det = S00 * S11 - S01 * S10;
        if (std::fabs(det) < 1e-12) {
            // fallback: only position update if S is near singular
            return;
        }
        double invS00 =  S11 / det;
        double invS01 = -S01 / det;
        double invS10 = -S10 / det;
        double invS11 =  S00 / det;

        // K = P * S^{-1}
        double K00 = P00 * invS00 + P01 * invS10;
        double K01 = P00 * invS01 + P01 * invS11;
        double K10 = P10 * invS00 + P11 * invS10;
        double K11 = P10 * invS01 + P11 * invS11;

        // innovation r = z - x
        double rq  = zq  - q;
        double rdq = zdq - dq;

        // update state
        q  += K00 * rq + K01 * rdq;
        dq += K10 * rq + K11 * rdq;

        // update covariance: P = (I - K) P
        double I00 = 1.0 - K00;
        double I01 =    - K01;
        double I10 =    - K10;
        double I11 = 1.0 - K11;

        double nP00 = I00*P00 + I01*P10;
        double nP01 = I00*P01 + I01*P11;
        double nP10 = I10*P00 + I11*P10;
        double nP11 = I10*P01 + I11*P11;

        P00 = nP00; P01 = nP01; P10 = nP10; P11 = nP11;
    }
};


static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(tok);
    return out;
}

int main(int argc, char** argv) {
    const std::string CSV_IN  = "log/streamed_measurements.csv";
    const std::string CSV_OUT = "log/offline_kf_predictions.csv";

    std::ifstream ifs(CSV_IN);
    if (!ifs) {
        std::cerr << "Cannot open " << CSV_IN << "\n";
        return 1;
    }

    std::string line;
    // skip header
    if (!std::getline(ifs, line)) {
        std::cerr << "Empty file\n";
        return 1;
    }

    std::vector<double> t;
    std::vector<std::array<double,6>> q_meas, dq_meas;

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        auto toks = split_csv(line);
        if (toks.size() < 13) continue;

        try {
            double ti = std::stod(toks[0]);
            std::array<double,6> q{}, dq{};
            for (int j=0;j<6;++j) {
                q[j]  = std::stod(toks[1 + j]);
                dq[j] = std::stod(toks[7 + j]);
            }
            t.push_back(ti);
            q_meas.push_back(q);
            dq_meas.push_back(dq);
        } catch (...) {
            continue;
        }
    }

    size_t N = t.size();
    if (N < 2) {
        std::cerr << "Not enough data\n";
        return 1;
    }

    // KF params to tune
    double SIGMA_A  = 3.0;
    double SIGMA_Q  = 1e-3;
    double SIGMA_DQ = 5e-3;

    if (argc >= 2) SIGMA_A = std::stod(argv[1]);
    if (argc >= 3) SIGMA_Q = std::stod(argv[2]);
    if (argc >= 4) SIGMA_DQ = std::stod(argv[3]);

    std::cout << "Using SIGMA_A=" << SIGMA_A
              << ", SIGMA_Q=" << SIGMA_Q
              << ", SIGMA_DQ=" << SIGMA_DQ << "\n";
    std::array<KF2,6> kf;
    // init from first sample
    for (int j=0;j<6;++j) {
        kf[j].q  = q_meas[0][j];
        kf[j].dq = dq_meas[0][j];
    }

    // --- NEW: fixed 8 ms prediction grid -------------------------
    const double DT = 0.008;  // 8 ms
    const double t0 = t[0];
    const double t_last = t.back();

    // total number of fixed steps from t0 to t_last
    size_t Nsteps = static_cast<size_t>(std::lround((t_last - t0) / DT));
    // we have step indices s = 0..Nsteps
    // s=0 corresponds to time t0

    // map each measurement k to the nearest step index
    std::vector<size_t> meas_step_idx(N);
    meas_step_idx[0] = 0;
    for (size_t k = 1; k < N; ++k) {
        double rel = (t[k] - t0) / DT;
        long idx = std::lround(rel);
        if (idx < 0) idx = 0;
        if (static_cast<size_t>(idx) > Nsteps) idx = static_cast<long>(Nsteps);
        meas_step_idx[k] = static_cast<size_t>(idx);
    }

    // output buffer: one line per fixed step
    std::vector<std::array<double,13>> kf_out(Nsteps + 1);

    // init output at step 0
    kf_out[0][0] = t0;
    for (int j=0;j<6;++j) {
        kf_out[0][1 + j] = kf[j].q;
        kf_out[0][7 + j] = kf[j].dq;
    }

    double mse_q  = 0.0;
    double mse_dq = 0.0;
    size_t mse_count = 0;

    size_t m = 1; // index into measurement vectors (we already used k=0)

    // main fixed-step loop
    for (size_t s = 1; s <= Nsteps; ++s) {
        // 1) predict exactly 8 ms
        for (int j=0; j<6; ++j) {
            kf[j].predict(DT, SIGMA_A);
        }

        // 2) if a measurement is scheduled for this step, apply update
        // (in case of multiple mapping to same step, update in order)
        while (m < N && meas_step_idx[m] == s) {
            for (int j=0; j<6; ++j) {
                kf[j].update_q_dq(
                    q_meas[m][j], dq_meas[m][j],
                    SIGMA_Q*SIGMA_Q, SIGMA_DQ*SIGMA_DQ
                );

                double eq  = kf[j].q  - q_meas[m][j];
                double edq = kf[j].dq - dq_meas[m][j];
                mse_q  += eq*eq;
                mse_dq += edq*edq;
            }
            ++m;
            ++mse_count;
        }

        // 3) store state at this fixed time
        double ts = t0 + s * DT;
        kf_out[s][0] = ts;
        for (int j=0; j<6; ++j) {
            kf_out[s][1 + j] = kf[j].q;
            kf_out[s][7 + j] = kf[j].dq;
        }
    }

    if (mse_count > 0) {
        mse_q  /= (mse_count * 6);
        mse_dq /= (mse_count * 6);
    } else {
        mse_q  = std::numeric_limits<double>::quiet_NaN();
        mse_dq = std::numeric_limits<double>::quiet_NaN();
    }

    std::cout << "MSE q: "  << mse_q  << "\n";
    std::cout << "MSE dq: " << mse_dq << "\n";

    // write CSV: one line per fixed step
    std::ofstream ofs(CSV_OUT);
    ofs << "t";
    for (int j=0;j<6;++j) ofs << ",qhat"  << j;
    for (int j=0;j<6;++j) ofs << ",dqhat" << j;
    ofs << "\n";

    for (size_t s=0; s<=Nsteps; ++s) {
        for (int c=0;c<13;++c) {
            ofs << kf_out[s][c];
            if (c < 12) ofs << ",";
        }
        ofs << "\n";
    }

    std::cout << "Wrote " << CSV_OUT << "\n";
    return 0;
}
