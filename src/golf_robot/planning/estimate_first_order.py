#!/usr/bin/env python3
"""
Estimate simple first-order models G_j(s) = K_j / (tau_j s + 1)
for each UR10 joint, based on:

  - trajectory_sim.csv: planned joint velocities dq0..dq5 (input u)
  - kf_predictions.csv: KF-estimated joint velocities dqhat0..dqhat5 (output y)

Assumes that:
  - Both CSVs have the same number of rows N.
  - They correspond sample-by-sample to the same time stamps (your C++ code
    writes kf_predictions with one row per command step i, so this matches).

Usage:
  Just edit the file paths and DT below, then run:

    python estimate_first_order_per_joint.py
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
from math import isfinite

# === CONFIG ===
PROJECT_ROOT      = Path(__file__).resolve().parent.parent.parent.parent
TRAJ_CSV          = PROJECT_ROOT / "log" / "trajectory_sim.csv"
KF_PRED_CSV       = PROJECT_ROOT / "log" / "kf_predictions.csv"
DT                = 0.008  # [s] sample time used in streaming

# Optional: ignore first/last part of the trajectory for fitting
TRIM_START_SEC    = 0.2    # ignore first 0.2 s
TRIM_END_SEC      = 0.2    # ignore last 0.2 s

# Bounds for parameters (per joint)
K_MIN,  K_MAX  = 0.0, 10.0
TAU_MIN, TAU_MAX = 1e-3, 2.0   # seconds, adjust if you expect slower/faster


@dataclass
class FirstOrderParams:
    K: float
    tau: float


def load_trajectory_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load trajectory_sim.csv.

    Expected columns:
        t, q0..q5, dq0..dq5   (total 1 + 6 + 6 = 13 columns)

    Returns:
        t      : (N,) time array
        q_des  : (N, 6)
        dq_des : (N, 6)
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 13:
        raise ValueError(f"{path} has {data.shape[1]} columns, expected >= 13")

    t = data[:, 0]
    q_des = data[:, 1:7]
    dq_des = data[:, 7:13]
    return t, q_des, dq_des


def load_kf_predictions_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load kf_predictions.csv.

    Expected columns:
        t, qhat0..qhat5, dqhat0..dqhat5

    Returns:
        t      : (N,) time array
        q_hat  : (N, 6)
        dq_hat : (N, 6)
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 13:
        raise ValueError(f"{path} has {data.shape[1]} columns, expected >= 13")

    t = data[:, 0]
    q_hat = data[:, 1:7]
    dq_hat = data[:, 7:13]
    return t, q_hat, dq_hat


def trim_by_time(t: np.ndarray, *arrays, t_trim_start: float, t_trim_end: float):
    """Trim all arrays to [t_min + t_trim_start, t_max - t_trim_end]."""
    t0 = t[0]
    tN = t[-1]
    t_min = t0 + t_trim_start
    t_max = tN - t_trim_end
    mask = (t >= t_min) & (t <= t_max)
    trimmed = [t[mask]]
    for arr in arrays:
        trimmed.append(arr[mask, ...])
    return trimmed


def simulate_first_order(u: np.ndarray, K: float, tau: float, dt: float, y0: float) -> np.ndarray:
    """
    Simulate dy/dt = (-y + K u) / tau using forward Euler.

    Args:
        u  : (N,) input
        K  : gain
        tau: time constant (> 0)
        dt : sampling time
        y0 : initial output

    Returns:
        y  : (N,) simulated output
    """
    N = u.shape[0]
    y = np.empty_like(u)
    y[0] = y0
    if tau < TAU_MIN:
        tau = TAU_MIN
    for k in range(1, N):
        dy = (-y[k-1] + K * u[k-1]) * (dt / tau)
        y[k] = y[k-1] + dy
    return y


def fit_first_order(u: np.ndarray, y: np.ndarray, dt: float) -> FirstOrderParams:
    """
    Fit K, tau for a first-order model y_dot = (-y + K u)/tau using
    naive grid search + local refinement (no SciPy dependency).

    This is simple but robust enough for our use: we do a coarse grid search
    over tau, and for each tau we solve optimal K in closed form (linear LS).
    Then we refine tau by a small local grid search.
    """
    # Use median of y as rough bias; subtract it to fit dynamics around zero
    bias = np.median(y)
    y0 = y[0]
    y_center = y - bias

    # Helper: compute LS-optimal K for a given tau
    def best_K_for_tau(tau: float) -> float:
        if tau < TAU_MIN:
            tau = TAU_MIN
        # Build linear LS: y ≈ A * K, with A = y_sim for K = 1
        # But we can re-use the dynamic structure more explicitly:
        # We simulate with K=1, then scale.
        y1 = simulate_first_order(u, K=1.0, tau=tau, dt=dt, y0=y0 - bias)
        # Solve min || y_center - K * y1 ||^2 for K
        num = np.dot(y1, y_center)
        den = np.dot(y1, y1) + 1e-12
        K_opt = num / den
        return max(K_MIN, min(K_opt, K_MAX))

    # Coarse grid over tau
    coarse_taus = np.linspace(TAU_MIN, TAU_MAX, 30)
    best_cost = np.inf
    best_tau = coarse_taus[0]
    best_K = 0.0

    for tau in coarse_taus:
        K_candidate = best_K_for_tau(tau)
        y_sim = simulate_first_order(u, K_candidate, tau, dt, y0 - bias) + bias
        err = y_sim - y
        cost = float(np.mean(err * err))
        if cost < best_cost and isfinite(cost):
            best_cost = cost
            best_tau = tau
            best_K = K_candidate

    # Local refinement around best_tau
    local_taus = np.linspace(
        max(TAU_MIN, best_tau * 0.5),
        min(TAU_MAX, best_tau * 1.5),
        20
    )

    for tau in local_taus:
        K_candidate = best_K_for_tau(tau)
        y_sim = simulate_first_order(u, K_candidate, tau, dt, y0 - bias) + bias
        err = y_sim - y
        cost = float(np.mean(err * err))
        if cost < best_cost and isfinite(cost):
            best_cost = cost
            best_tau = tau
            best_K = K_candidate

    return FirstOrderParams(K=best_K, tau=best_tau)


def main():
    print(f"[INFO] Loading trajectory from {TRAJ_CSV}")
    t_traj, q_des, dq_des = load_trajectory_csv(TRAJ_CSV)

    print(f"[INFO] Loading KF predictions from {KF_PRED_CSV}")
    t_kf, q_hat, dq_hat = load_kf_predictions_csv(KF_PRED_CSV)

    if len(t_traj) != len(t_kf):
        print(f"[WARN] trajectory_sim has N={len(t_traj)}, kf_predictions has N={len(t_kf)}")
        N = min(len(t_traj), len(t_kf))
        t_traj = t_traj[:N]
        dq_des = dq_des[:N, :]
        t_kf   = t_kf[:N]
        dq_hat = dq_hat[:N, :]

    # Use KF time as reference
    t = t_kf.copy()

    # Optional trimming of start/end
    if TRIM_START_SEC > 0.0 or TRIM_END_SEC > 0.0:
        t, dq_des, dq_hat = trim_by_time(
            t, dq_des, dq_hat,
            t_trim_start=TRIM_START_SEC,
            t_trim_end=TRIM_END_SEC
        )

    print(f"[INFO] Using {len(t)} samples after trimming.")

    # Sanity check: compute dt from timestamps if possible
    if len(t) > 1:
        est_dt = float(np.median(np.diff(t)))
        print(f"[INFO] Estimated dt from timestamps: {est_dt:.6f} s (configured DT={DT:.6f} s)")

    # Fit per joint
    results = []
    for j in range(6):
        u_j = dq_des[:, j]
        y_j = dq_hat[:, j]

        # Remove any NaNs if they slipped in
        mask = np.isfinite(u_j) & np.isfinite(y_j)
        u_j = u_j[mask]
        y_j = y_j[mask]

        if len(u_j) < 10:
            print(f"[WARN] Not enough data for joint {j}, skipping.")
            results.append(None)
            continue

        print(f"\n[INFO] Fitting first-order model for joint {j}...")
        params = fit_first_order(u_j, y_j, dt=DT)
        results.append(params)
        print(f"  Joint {j}: K = {params.K:.4f}, tau = {params.tau:.4f} s")

    print("\n=== Summary: First-order models G_j(s) = K_j / (tau_j s + 1) ===")
    for j, p in enumerate(results):
        if p is None:
            print(f"  Joint {j}: (no fit)")
        else:
            print(f"  Joint {j}: K = {p.K:.4f}, tau = {p.tau:.4f} s")


if __name__ == "__main__":
    main()
