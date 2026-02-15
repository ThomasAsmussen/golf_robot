#!/usr/bin/env python3
"""
plot_kf_vs_plan_vs_meas_one_joint.py

Plots (for a single chosen joint):
  1) Planned vs Measured vs Kalman (KF) for joint position and velocity
  2) Difference plots: (KF - Measured) for joint position and velocity

Input files (default):
  - log/trajectory_sim.csv          : planned  (t,q0..q5,dq0..dq5)
  - log/streamed_measurements.csv   : measured (t,q0..q5,dq0..dq5)
  - log/kf_predictions.csv          : KF       (t,qhat0..qhat5,dqhat0..dqhat5)

Notes:
  - We align everything onto the MEASUREMENT timestamps using interpolation.
  - We only use the overlapping time window to avoid extrapolation.
  - Choose joint via JOINT_INDEX (0..5) or via CLI: --joint 1..6 or 0..5
"""

from pathlib import Path
import os
import glob
import time
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# -----------------------
# Config
# -----------------------
SHOW_PLOTS = True
OUT_DIR = "log"

PLANNED_CSV = "log/trajectory_sim.csv"
MEAS_GLOB   = "log/streamed_measurements.csv"   # if multiple, we'll take the latest by sort
KF_CSV      = "log/kf_predictions.csv"

# If your planner CSV needs that one-sample shift you had earlier, keep this True.
SHIFT_PLANNED_BY_ONE = True

# Default joint if not provided via CLI
JOINT_INDEX = 5  # 0..5 (q1 is 0)


# Choose interactive backend if showing
if SHOW_PLOTS:
    for bk in ("TkAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
        try:
            matplotlib.use(bk)
            break
        except Exception:
            continue
else:
    matplotlib.use("Agg")


# -----------------------
# Loaders
# -----------------------
def load_planned_csv(path):
    """t, q0..q5, dq0..dq5"""
    t, Q, dQ = [], [], []
    with open(path, "r") as f:
        _ = f.readline()
        for line in f:
            if not line.strip():
                continue
            vals = [float(x) for x in line.strip().split(",")]
            if len(vals) < 13:
                continue
            t.append(vals[0])
            Q.append(vals[1:7])
            dQ.append(vals[7:13])
    return np.asarray(t, float), np.asarray(Q, float).reshape(-1, 6), np.asarray(dQ, float).reshape(-1, 6)


def load_streamer_log_strict(path):
    """
    t, q0..q5, dq0..dq5
    - Drops header/junk.
    - Stops at first malformed row after valid rows start (common when file is still being written).
    """
    import csv

    t_list, Q_list, dQ_list = [], [], []

    def _is_header(row):
        if not row:
            return True
        s = ",".join(row).lower()
        return any(k in s for k in ("t", "time", "q0", "dq0"))

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        rows = []
        if first is not None and not _is_header(first):
            rows.append(first)
        rows.extend(list(reader))

    seen_valid = False
    for row in rows:
        if not row:
            continue
        row = [c.strip() for c in row if c.strip() != ""]
        if len(row) < 13:
            if seen_valid:
                break
            continue
        try:
            t_val  = float(row[0])
            q_vals = [float(x) for x in row[1:7]]
            dq_vals = [float(x) for x in row[7:13]]
        except ValueError:
            if seen_valid:
                break
            continue
        t_list.append(t_val)
        Q_list.append(q_vals)
        dQ_list.append(dq_vals)
        seen_valid = True

    t  = np.asarray(t_list, float)
    Q  = np.asarray(Q_list, float).reshape(-1, 6)
    dQ = np.asarray(dQ_list, float).reshape(-1, 6)
    return t, Q, dQ


def load_kf_csv(path):
    """t, qhat0..qhat5, dqhat0..dqhat5"""
    if not os.path.exists(path):
        return None, None, None
    t_list, Q_list, dQ_list = [], [], []
    with open(path, "r") as f:
        _ = f.readline()
        for line in f:
            if not line.strip():
                continue
            vals = [c.strip() for c in line.strip().split(",")]
            if len(vals) < 13:
                continue
            try:
                t_list.append(float(vals[0]))
                Q_list.append([float(x) for x in vals[1:7]])
                dQ_list.append([float(x) for x in vals[7:13]])
            except ValueError:
                continue
    if not t_list:
        return None, None, None
    return np.asarray(t_list, float), np.asarray(Q_list, float).reshape(-1, 6), np.asarray(dQ_list, float).reshape(-1, 6)


def find_latest(glob_pattern):
    files = glob.glob(glob_pattern)
    if not files:
        return None
    files.sort()
    return files[-1]


# -----------------------
# Alignment helpers
# -----------------------
def sort_unique(t, Q, dQ):
    """
    Sort by time, drop duplicate timestamps, apply same keep-mask to Q and dQ.
    """
    t = np.asarray(t, float)
    Q = np.asarray(Q, float)
    dQ = np.asarray(dQ, float)

    order = np.argsort(t)
    t_s = t[order]
    Q_s = Q[order]
    dQ_s = dQ[order]

    keep = np.ones_like(t_s, dtype=bool)
    keep[1:] = t_s[1:] > t_s[:-1]

    return t_s[keep], Q_s[keep], dQ_s[keep]


def interp_to(t_src, X_src, t_dst):
    """
    Interpolate each column of X_src (N,D) defined at t_src (N,)
    onto t_dst (M,). Returns (M,D).
    """
    t_src = np.asarray(t_src, float)
    X_src = np.asarray(X_src, float)
    t_dst = np.asarray(t_dst, float)

    # Ensure sorted unique
    order = np.argsort(t_src)
    t_s = t_src[order]
    X_s = X_src[order]
    keep = np.ones_like(t_s, dtype=bool)
    keep[1:] = t_s[1:] > t_s[:-1]
    t_s = t_s[keep]
    X_s = X_s[keep]

    Y = np.zeros((len(t_dst), X_s.shape[1]), dtype=float)
    for j in range(X_s.shape[1]):
        Y[:, j] = np.interp(t_dst, t_s, X_s[:, j])
    return Y


def overlap_window(t_meas, *t_others):
    """Return (t0,t1) overlap of all time arrays provided."""
    t0 = float(np.min(t_meas))
    t1 = float(np.max(t_meas))
    for t in t_others:
        t0 = max(t0, float(np.min(t)))
        t1 = min(t1, float(np.max(t)))
    return t0, t1


# -----------------------
# Plotting (single joint)
# -----------------------
def plot_one_joint_and_diff(
    joint_idx,
    t_meas, Q_meas, dQ_meas,
    t_plan, Q_plan, dQ_plan,
    t_kf,   Q_kf,   dQ_kf,
    out_prefix=None
):
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = out_prefix or time.strftime("%Y%m%d_%H%M%S")

    # Sort/unique each stream consistently
    t_meas, Q_meas, dQ_meas = sort_unique(t_meas, Q_meas, dQ_meas)
    t_plan, Q_plan, dQ_plan = sort_unique(t_plan, Q_plan, dQ_plan)
    t_kf,   Q_kf,   dQ_kf   = sort_unique(t_kf,   Q_kf,   dQ_kf)

    # Overlap window to avoid extrapolation
    t0, t1 = overlap_window(t_meas, t_plan, t_kf)
    if t1 <= t0:
        raise RuntimeError("No overlapping time window between meas/plan/kf")

    mask = (t_meas >= t0) & (t_meas <= t1)
    t = t_meas[mask]
    Qm = Q_meas[mask]
    dQm = dQ_meas[mask]

    # Interpolate plan + kf onto measurement timestamps
    Qp  = interp_to(t_plan, Q_plan, t)
    dQp = interp_to(t_plan, dQ_plan, t)
    Qk  = interp_to(t_kf,   Q_kf,   t)
    dQk = interp_to(t_kf,   dQ_kf,  t)

    # Pull selected joint
    q_p  = Qp[:, joint_idx]
    q_m  = Qm[:, joint_idx]
    q_k  = Qk[:, joint_idx]
    dq_p = dQp[:, joint_idx]
    dq_m = dQm[:, joint_idx]
    dq_k = dQk[:, joint_idx]

    # Diffs
    dq_pos = q_k  - q_m
    dq_vel = dq_k - dq_m

    jname = f"q{joint_idx+1}"

    # ---- Figure 1: Position triplet
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(t, q_p, "-", linewidth=2.0, label="Planned")
    ax.plot(t, q_m, ".", linewidth=0.8, label="Measured")
    ax.plot(t, q_k, "-", linewidth=1.6, label="Kalman")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{jname} (rad)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    fn_pos_triplet = os.path.join(OUT_DIR, f"kf_plan_meas_position_{jname}_{ts}.png")
    if SHOW_PLOTS:
        plt.show()
        plt.close()
        fn_pos_triplet = None
    else:
        plt.savefig(fn_pos_triplet)
        plt.close()

    # ---- Figure 1b: Velocity triplet
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(t, dq_p, "-", linewidth=2.0, label="Planned")
    ax.plot(t, dq_m, ".", linewidth=0.8, label="Measured")
    ax.plot(t, dq_k, "-", linewidth=1.6, label="Kalman")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"d{jname} (rad/s)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    fn_vel_triplet = os.path.join(OUT_DIR, f"kf_plan_meas_velocity_{jname}_{ts}.png")
    if SHOW_PLOTS:
        plt.show()
        plt.close()
        fn_vel_triplet = None
    else:
        plt.savefig(fn_vel_triplet)
        plt.close()

    # ---- Figure 2: Diff positions
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(t, dq_pos, "-", linewidth=1.8, label="Error = KF - Measured")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{jname} (rad)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    fn_pos_diff = os.path.join(OUT_DIR, f"kf_minus_meas_position_{jname}_{ts}.png")
    if SHOW_PLOTS:
        plt.show()
        plt.close()
        fn_pos_diff = None
    else:
        plt.savefig(fn_pos_diff)
        plt.close()

    # ---- Figure 2b: Diff velocities
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(t, dq_vel, "-", linewidth=1.8, label="KF - Measured")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Δd{jname} (rad/s)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    fn_vel_diff = os.path.join(OUT_DIR, f"kf_minus_meas_velocity_{jname}_{ts}.png")
    if SHOW_PLOTS:
        plt.show()
        plt.close()
        fn_vel_diff = None
    else:
        plt.savefig(fn_vel_diff)
        plt.close()

    return fn_pos_triplet, fn_vel_triplet, fn_pos_diff, fn_vel_diff


def parse_joint_arg(x: str) -> int:
    """
    Accepts 0..5 or 1..6. Returns 0..5.
    """
    v = int(x)
    if 0 <= v <= 5:
        return v
    if 1 <= v <= 6:
        return v - 1
    raise argparse.ArgumentTypeError("joint must be 0..5 or 1..6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", type=parse_joint_arg, default=JOINT_INDEX,
                    help="Joint index: 0..5 or 1..6 (default: %(default)s)")
    ap.add_argument("--planned", default=PLANNED_CSV)
    ap.add_argument("--meas", default=MEAS_GLOB)
    ap.add_argument("--kf", default=KF_CSV)
    ap.add_argument("--no-shift", action="store_true",
                    help="Disable SHIFT_PLANNED_BY_ONE")
    args = ap.parse_args()

    planned_csv = args.planned
    meas_glob   = args.meas
    kf_csv      = args.kf
    joint_idx   = args.joint
    do_shift    = (not args.no_shift) and SHIFT_PLANNED_BY_ONE

    if not os.path.exists(planned_csv):
        print(f"[ERROR] Planned CSV not found: {planned_csv}")
        return 1

    meas_path = find_latest(meas_glob)
    if meas_path is None:
        print(f"[ERROR] No measurement log found with pattern: {meas_glob}")
        return 2

    if not os.path.exists(kf_csv):
        print(f"[ERROR] KF CSV not found: {kf_csv}")
        return 3

    print(f"[INFO] Joint: {joint_idx} (q{joint_idx+1})")
    print(f"[INFO] Loading planned: {planned_csv}")
    t_plan, Q_plan, dQ_plan = load_planned_csv(planned_csv)

    if do_shift and len(Q_plan) > 2:
        Q_plan = Q_plan.copy()
        dQ_plan = dQ_plan.copy()
        Q_plan[2:, :]  = Q_plan[:-2, :]
        dQ_plan[2:, :] = dQ_plan[:-2, :]

    print(f"[INFO] Loading measured: {meas_path}")
    t_meas, Q_meas, dQ_meas = load_streamer_log_strict(meas_path)

    print(f"[INFO] Loading KF: {kf_csv}")
    t_kf, Q_kf, dQ_kf = load_kf_csv(kf_csv)
    if t_kf is None:
        print("[ERROR] KF CSV exists but no valid rows parsed.")
        return 4

    print("[INFO] Plotting (planned vs measured vs KF) + diffs (KF - meas)...")
    fns = plot_one_joint_and_diff(
        joint_idx,
        t_meas, Q_meas, dQ_meas,
        t_plan, Q_plan, dQ_plan,
        t_kf, Q_kf, dQ_kf,
    )

    print("[INFO] Done.")
    print("[INFO] Saved files (None means shown interactively):")
    for fn in fns:
        print("  -", fn)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())