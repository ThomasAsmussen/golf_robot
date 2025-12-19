"""
plot_kf_vs_planned_joints.py

Plot planned joint positions vs Kalman filter joint positions and the error.

Assumes:
 - Planned CSV:  t, q0..q5, dq0..dq5
 - KF CSV:       t, qhat0..qhat5, dqhat0..dqhat5

The script:
 - loads both CSVs
 - time-aligns them by interpolating KF joints to the planned time grid
 - plots:
     (1) planned vs KF joint positions
     (2) joint error (KF - planned)
 - saves figures into OUT_DIR (or shows them if SHOW_PLOTS=True)
"""

from pathlib import Path
import numpy as np
import matplotlib
import time
import os

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SHOW_PLOTS = True  # if False, uses non-interactive backend and saves PNGs
OUT_DIR = "log"

PLANNED_CSV = "log/trajectory_sim.csv"   # planner-produced CSV
# KF_CSV      = os.path.join(OUT_DIR, "kf_predictions.csv")  # KF log
KF_CSV = "log/streamed_measurements.csv"  # KF log

# ---------------------------------------------------------------------
# Matplotlib backend selection
# ---------------------------------------------------------------------
if SHOW_PLOTS:
    for bk in ("TkAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
        try:
            matplotlib.use(bk)
            break
        except Exception:
            continue
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------
def load_planned_csv(path):
    """Load planner CSV: t, q0..q5, dq0..dq5 -> t (N,), Q (N,6)."""
    if not os.path.exists(path):
        print(f"[ERROR] Planned CSV not found at {path}")
        return None, None

    t_list = []
    Q_list = []

    with open(path, "r") as f:
        header = f.readline()  # discard header
        for line in f:
            if not line.strip():
                continue
            vals = [x.strip() for x in line.strip().split(",")]
            if len(vals) < 7:
                # need t + 6 joints at least
                continue
            try:
                t_list.append(float(vals[0]))
                Q_list.append([float(x) for x in vals[1:7]])
            except ValueError:
                continue

    if not t_list:
        print(f"[ERROR] No valid rows found in planned CSV: {path}")
        return None, None

    t = np.asarray(t_list, dtype=float)
    Q = np.asarray(Q_list, dtype=float).reshape(-1, 6)
    return t, Q


def load_kf_csv(path):
    """
    Load Kalman filter CSV: t, qhat0..qhat5, dqhat0..dqhat5
    -> t (N,), Qhat (N,6)
    """
    if not os.path.exists(path):
        print(f"[ERROR] KF CSV not found at {path}")
        return None, None

    t_list = []
    Q_list = []

    with open(path, "r") as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            vals = [x.strip() for x in line.strip().split(",")]
            if len(vals) < 7:
                continue
            try:
                t_list.append(float(vals[0]))
                Q_list.append([float(x) for x in vals[1:7]])
            except ValueError:
                continue

    if not t_list:
        print(f"[ERROR] No valid rows found in KF CSV: {path}")
        return None, None

    t = np.asarray(t_list, dtype=float)
    Q = np.asarray(Q_list, dtype=float).reshape(-1, 6)
    return t, Q


# ---------------------------------------------------------------------
# Time alignment
# ---------------------------------------------------------------------
def align_kf_to_planned(t_plan, Q_plan, t_kf, Q_kf):
    """
    Align KF joint positions to the planned time grid via interpolation.

    Returns:
      t_common (M,), Q_plan_aligned (M,6), Q_kf_interp (M,6)
    where t_common is a subset of t_plan that lies within the overlap
    of the two time ranges, and Q_kf_interp is KF joints interpolated
    at t_common.
    """
    # Determine overlapping time window
    t_start = max(t_plan[0], t_kf[0])
    t_end   = min(t_plan[-1], t_kf[-1])

    if t_end <= t_start:
        print("[ERROR] No overlapping time window between planned and KF data")
        return None, None, None

    # Mask planned times to overlap interval
    mask = (t_plan >= t_start) & (t_plan <= t_end)
    if not np.any(mask):
        print("[ERROR] Planned times have no samples in overlap interval")
        return None, None, None

    t_common = t_plan[mask]
    Q_plan_aligned = Q_plan[mask, :]

    # Interpolate each joint of KF to t_common
    Q_kf_interp = np.zeros_like(Q_plan_aligned)
    for j in range(6):
        Q_kf_interp[:, j] = np.interp(t_common, t_kf, Q_kf[:, j])

    return t_common, Q_plan_aligned, Q_kf_interp


# ---------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------
def plot_positions_and_error(t, Q_plan, Q_kf, out_prefix=None):
    """
    Plot:
      1) Planned vs KF joint positions
      2) Position error (KF - planned)

    Returns:
      (fn_positions, fn_error) where each is the filename or None if SHOW_PLOTS.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = out_prefix or time.strftime("%Y%m%d_%H%M%S")

    # --- Joint positions: planned vs KF ---
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(6):
        ax = axes[i]
        ax.plot(t, Q_plan[:, i], "b-", label="Planned", linewidth=2)
        ax.plot(t, Q_kf[:, i], "r--", label="KF", linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Joint {i+1} (rad)")
        ax.grid(True)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    fn_positions = os.path.join(OUT_DIR, f"planned_vs_kf_joints_{ts}.png")
    if SHOW_PLOTS:
        try:
            plt.show()
        finally:
            plt.close()
        fn_positions = None
    else:
        plt.savefig(fn_positions)
        plt.close()

    # --- Joint error: KF - Planned ---
    E = Q_kf - Q_plan  # (N,6)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(6):
        ax = axes[i]
        ax.plot(t, E[:, i], "k-", linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Joint {i+1} error (rad)")
        ax.grid(True)

    plt.tight_layout()
    fn_error = os.path.join(OUT_DIR, f"kf_position_error_{ts}.png")
    if SHOW_PLOTS:
        try:
            plt.show()
        finally:
            plt.close()
        fn_error = None
    else:
        plt.savefig(fn_error)
        plt.close()

    # Basic stats to console
    rms_per_joint = np.sqrt(np.mean(E**2, axis=0))
    max_per_joint = np.max(np.abs(E), axis=0)
    print("[INFO] Joint position error stats (KF - planned):")
    for i in range(6):
        print(
            f"  Joint {i+1}: RMS={rms_per_joint[i]:.6g} rad, "
            f"max={max_per_joint[i]:.6g} rad"
        )

    return fn_positions, fn_error


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    t_plan, Q_plan = load_planned_csv(PLANNED_CSV)
    if t_plan is None:
        return 1

    t_kf, Q_kf = load_kf_csv(KF_CSV)
    if t_kf is None:
        return 2

    print(f"[INFO] Loaded planned: t={t_plan.shape}, Q={Q_plan.shape}")
    print(f"[INFO] Loaded KF:      t={t_kf.shape}, Q={Q_kf.shape}")

    t_common, Q_plan_aligned, Q_kf_interp = align_kf_to_planned(
        t_plan, Q_plan, t_kf, Q_kf
    )
    if t_common is None:
        return 3

    print(f"[INFO] Using {len(t_common)} samples in overlapping time window.")

    fn_pos, fn_err = plot_positions_and_error(t_common, Q_plan_aligned, Q_kf_interp)
    print("[INFO] Done.")
    if fn_pos:
        print(f"  Positions plot: {fn_pos}")
    if fn_err:
        print(f"  Error plot:     {fn_err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())