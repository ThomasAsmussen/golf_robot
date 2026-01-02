// return_to_start.cpp
// Minimal Exe B: connect to UR, then movej to END_POS (return pose), then exit.
//
// Build:
//   g++ -O2 -std=c++17 -pthread return_to_start.cpp -o return_to_start
//
// Run:
//   ./return_to_start
//
// Notes:
// - This assumes your UrDriver / rt_interface_ / robot_state_ are the same as Exe A.
// - If you want this to return to the trajectory's q_start instead, you can read q_start from a file
//   or pass it via argv. For now it's hardcoded END_POS.

#include <iostream>
#include <array>
#include <atomic>
#include <thread>
#include <chrono>
#include <csignal>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <vector>

#include "../include/communication/ur_driver.h"
#include "../include/communication/ur5.h"

static const std::string ROBOT_IP = "192.38.66.227";
static const int         REVERSE_PORT = 5007;

// This is your "return pose" (same as you had in Exe A)
static const std::array<double,6> END_POS = {-2.47, -2.38, -1.55, 1.66, 0.49, -0.26};

// Move parameters
static const double A_MOVE = 1.2;
static const double V_MOVE = 0.25;

// Safety / stopping
static const double ACCEL  = 6.0;
static const double SPEEDJ_T = 0.03;

// Wait tolerance / timeout
static const double TOL_SUMABS = 1e-3;   // sum |q - target|
static const double TIMEOUT_S  = 15.0;

std::atomic<bool> keep_running(true);
void on_sigint(int){ keep_running = false; }

static double sum_abs_err(const std::vector<double>& q, const std::array<double,6>& target){
    if (q.size() < 6) return 1e9;
    double e = 0.0;
    for(int i=0;i<6;i++) e += std::fabs(q[i] - target[i]);
    return e;
}

int main(int argc, char** argv){
    std::signal(SIGINT, on_sigint);

    std::condition_variable rt_msg_cond_;
    std::condition_variable msg_cond_;
    UrDriver mydriver(rt_msg_cond_, msg_cond_, ROBOT_IP, REVERSE_PORT);
    mydriver.start();

    // Wait for at least one RT packet so state is valid
    {
        std::mutex msg_lock;
        std::unique_lock<std::mutex> locker(msg_lock);

        while(!mydriver.rt_interface_->robot_state_->getDataPublished() && keep_running.load()){
            rt_msg_cond_.wait_for(locker, std::chrono::milliseconds(200));
        }
        mydriver.rt_interface_->robot_state_->setDataPublished();
    }

    if(!keep_running.load()){
        std::cerr << "[INFO] Aborted before motion.\n";
        mydriver.halt();
        return 2;
    }

    auto* state = mydriver.rt_interface_->robot_state_;

    // --- Ensure we are not "stuck" in speed mode from previous exe ---
    // This is cheap and helps a lot in practice.
    try {
        mydriver.setSpeed(0,0,0,0,0,0, ACCEL, SPEEDJ_T);
    } catch(...) {
        // ignore if your wrapper throws
    }
    mydriver.rt_interface_->addCommandToQueue("stopj(2.0)\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // --- Send movej to END_POS ---
    {
        char cmdbuf[512];
        std::snprintf(cmdbuf, sizeof(cmdbuf),
            "movej([%.6f,%.6f,%.6f,%.6f,%.6f,%.6f], a=%.3f, v=%.3f)\n",
            END_POS[0], END_POS[1], END_POS[2], END_POS[3], END_POS[4], END_POS[5],
            A_MOVE, V_MOVE
        );

        std::cout << "[INFO] Sending return movej...\n";
        mydriver.rt_interface_->addCommandToQueue(std::string(cmdbuf));
    }

    // --- Wait until reached (or timeout) ---
    {
        auto t0 = std::chrono::steady_clock::now();
        bool reached = false;

        while(keep_running.load()){
            const std::vector<double> q = state->getQActual();
            const double err = sum_abs_err(q, END_POS);

            if (err < TOL_SUMABS){
                reached = true;
                break;
            }

            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (dt > TIMEOUT_S) break;

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (!keep_running.load()){
            std::cerr << "[WARN] SIGINT received while waiting. Stopping.\n";
            mydriver.rt_interface_->addCommandToQueue("stopj(2.0)\n");
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            mydriver.halt();
            return 2;
        }

        if (!reached){
            std::cerr << "[WARN] return-to-start did not reach target within timeout (" << TIMEOUT_S << " s)\n";
        } else {
            std::cout << "[INFO] Return-to-start reached target.\n";
        }
    }

    // Clean exit
    mydriver.rt_interface_->addCommandToQueue("stopj(2.0)\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    mydriver.halt();

    std::cout << "[DONE] return_to_start finished.\n";
    return 0;
}
