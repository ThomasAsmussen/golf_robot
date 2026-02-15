#!/usr/bin/env python3
"""
plot_kf_pose_errors.py

Exactly like the style of plot_streamer_results.py (interactive, no saving),
but instead of measurement data it uses TWO Kalman estimates and the planned trajectory,
and plots:

  1) TCP position error (KF - Planned): x/y/z error + ||e||  (one figure)
  2) TCP rotation error (KF - Planned): roll/pitch/yaw error + angle error (one figure)

All paths are assumed to live in the same place ("log/") like your inspiration script.
Uses CLI arguments (with the same defaults as before style).

Expected CSV formats:
  planned:   t,q0..q5,dq0..dq5
  kf files:  t,qhat0..qhat5,dqhat0..dqhat5   (header tolerated)

Example:
  python plot_kf_pose_errors.py \
      --planned log/trajectory_sim.csv \
      --kf-open log/kf_predictions_open_loop.csv \
      --kf-closed "log/kf_predictions_closed_loop copy.csv"
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from kinematics import *  # fk_ur10

# ---------------------------------------------------------
# Defaults (same style as your inspiration)
# ---------------------------------------------------------
SHOW_PLOTS = True
if SHOW_PLOTS or True:
    for bk in ("TkAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
        try:
            matplotlib.use(bk)
            break
        except Exception:
            continue
else:
    matplotlib.use("Agg")

# Defaults (same place)
PLANNED_CSV_DEFAULT = "log/trajectory_sim.csv"
KF_OPEN_DEFAULT = "log/kf_predictions_qpt_open_loop.csv"
KF_CLOSED_DEFAULT = "log/kf_predictions_qpt_closed_loop.csv"


# ---------------------------------------------------------
# Loaders
# ---------------------------------------------------------
def load_planned_csv(path):
    """planned: t, q0..q5, dq0..dq5 -> t (N,), Q (N,6), dQ (N,6)"""
    t, Q, dQ = [], [], []
    with open(path, "r") as f:
        _ = f.readline()  # header
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


def load_kf_csv(path):
    """kf: t, qhat0..qhat5, dqhat0..dqhat5 -> t (N,), Qhat (N,6), dQhat (N,6)"""
    if not os.path.exists(path):
        return None, None, None

    t_list, Q_list, dQ_list = [], [], []
    with open(path, "r") as f:
        _ = f.readline()  # header
        for line in f:
            if not line.strip():
                continue
            vals = [x.strip() for x in line.strip().split(",")]
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

    return (
        np.asarray(t_list, float),
        np.asarray(Q_list, float).reshape(-1, 6),
        np.asarray(dQ_list, float).reshape(-1, 6),
    )


# ---------------------------------------------------------
# FK helpers (copied style from your script)
# ---------------------------------------------------------
def _T_tcp_from_q(q):
    T = fk_ur10(np.asarray(q, float))
    if isinstance(T, (list, tuple)):
        T_tcp = T[-1]
    else:
        T_tcp = T
    T_tcp = np.asarray(T_tcp, float)
    if T_tcp.shape == (4, 4):
        return T_tcp
    elif T_tcp.shape == (3, 4):
        T4 = np.eye(4)
        T4[:3, :] = T_tcp
        return T4
    else:
        raise TypeError(f"Unexpected transform shape: {T_tcp.shape}")


def tcp_xyz_from_q(q):
    T = _T_tcp_from_q(q)
    return T[:3, 3].copy()


def tcp_path_from_Q(Q):
    return np.array([tcp_xyz_from_q(q) for q in np.asarray(Q, float)])


def _rot_to_rpy_xyz(R):
    R = np.asarray(R, float)
    sp = -R[2, 0]
    sp = np.clip(sp, -1.0, 1.0)
    pitch = np.arcsin(sp)

    cp = np.cos(pitch)
    if abs(cp) < 1e-9:
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw], dtype=float)


def tcp_rpy_from_Q(Q):
    Q = np.asarray(Q, float)
    rpy = np.zeros((Q.shape[0], 3), dtype=float)
    for i in range(Q.shape[0]):
        T = _T_tcp_from_q(Q[i])
        rpy[i, :] = _rot_to_rpy_xyz(T[:3, :3])
    return rpy


def tcp_R_from_Q(Q):
    Q = np.asarray(Q, float)
    R = np.zeros((Q.shape[0], 3, 3), dtype=float)
    for i in range(Q.shape[0]):
        T = _T_tcp_from_q(Q[i])
        R[i, :, :] = T[:3, :3]
    return R


def rot_angle(R_err):
    tr = float(np.trace(R_err))
    c = (tr - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return float(np.arccos(c))


# ---------------------------------------------------------
# Time alignment utilities (same idea as your script)
# ---------------------------------------------------------
def _sorted_unique_time(t, X):
    t = np.asarray(t, float)
    X = np.asarray(X)
    order = np.argsort(t)
    t_s = t[order]
    X_s = X[order]
    keep = np.ones_like(t_s, dtype=bool)
    keep[1:] = t_s[1:] > t_s[:-1]
    return t_s[keep], X_s[keep]


def _overlap_and_trim(t_a, X_a, t_b, X_b):
    """Trim both series to overlap window based on times; keeps native sampling."""
    t_a, X_a = _sorted_unique_time(t_a, X_a)
    t_b, X_b = _sorted_unique_time(t_b, X_b)

    t0 = max(t_a[0], t_b[0])
    t1 = min(t_a[-1], t_b[-1])
    if t1 <= t0:
        return None, None, None, None

    ma = (t_a >= t0) & (t_a <= t1)
    mb = (t_b >= t0) & (t_b <= t1)
    return t_a[ma], X_a[ma], t_b[mb], X_b[mb]


def _interp3(t_src, X_src, t_dst):
    out = np.zeros((len(t_dst), 3), float)
    for k in range(3):
        out[:, k] = np.interp(t_dst, t_src, X_src[:, k])
    return out


# ---------------------------------------------------------
# Error computation
# ---------------------------------------------------------
def compute_pose_errors_against_plan(t_plan, Q_plan, t_kf, Q_kf):
    """
    Returns:
      t_kf_trim (M,)
      pos_err (M,3) = P_kf - P_plan_interp
      pos_norm (M,)
      rpy_err (M,3) in radians wrapped to [-pi,pi]
      ang_err (M,) in radians (nearest-neighbor planned rotation)
    """
    # compute pose signals
    P_plan = tcp_path_from_Q(Q_plan)
    P_kf = tcp_path_from_Q(Q_kf)

    rpy_plan = tcp_rpy_from_Q(Q_plan)
    rpy_kf = tcp_rpy_from_Q(Q_kf)

    R_plan = tcp_R_from_Q(Q_plan)
    R_kf = tcp_R_from_Q(Q_kf)

    # trim to overlap
    t_plan_t, P_plan_t, t_kf_t, P_kf_t = _overlap_and_trim(t_plan, P_plan, t_kf, P_kf)
    if t_plan_t is None:
        return None

    _, rpy_plan_t, _, rpy_kf_t = _overlap_and_trim(t_plan, rpy_plan, t_kf, rpy_kf)
    _, R_plan_t, _, R_kf_t = _overlap_and_trim(t_plan, R_plan, t_kf, R_kf)

    # interpolate planned position/rpy to KF timestamps (like your script)
    P_plan_interp = _interp3(t_plan_t, P_plan_t, t_kf_t)

    rpy_plan_u = np.unwrap(rpy_plan_t, axis=0)
    rpy_kf_u = np.unwrap(rpy_kf_t, axis=0)
    rpy_plan_interp = _interp3(t_plan_t, rpy_plan_u, t_kf_t)

    pos_err = P_kf_t - P_plan_interp
    pos_norm = np.linalg.norm(pos_err, axis=1)

    rpy_err = rpy_kf_u - rpy_plan_interp
    rpy_err = (rpy_err + np.pi) % (2.0 * np.pi) - np.pi

    # rotation-angle error: nearest planned sample (robust + simple)
    t_plan_t2, R_plan_t2 = _sorted_unique_time(t_plan_t, R_plan_t)
    idx = np.searchsorted(t_plan_t2, t_kf_t)
    idx = np.clip(idx, 1, len(t_plan_t2) - 1)
    left = idx - 1
    right = idx
    choose_right = np.abs(t_plan_t2[right] - t_kf_t) < np.abs(t_plan_t2[left] - t_kf_t)
    idx_nn = np.where(choose_right, right, left)

    ang_err = np.zeros(len(t_kf_t), float)
    for i in range(len(t_kf_t)):
        # R_err = R_kf * R_plan^T
        Rerr = R_kf_t[i] @ R_plan_t2[idx_nn[i]].T
        ang_err[i] = rot_angle(Rerr)

    return t_kf_t, pos_err, pos_norm, rpy_err, ang_err


# ---------------------------------------------------------
# Plots (NO SAVING, show only)
# ---------------------------------------------------------
def plot_position_errors(t, pos_open, norm_open, pos_closed, norm_closed):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    labels = ["x error (m)", "y error (m)", "z error (m)", "||pos error|| (m)"]

    for i in range(3):
        ax = axes[i]
        ax.plot(t, pos_open[:, i], "-", linewidth=1.5, label="Open-loop: KF - Planned")
        ax.plot(t, pos_closed[:, i], "-", linewidth=1.5, label="Closed-loop: KF - Planned")
        ax.set_ylabel(labels[i])
        ax.grid(True)
        ax.legend()

    ax = axes[3]
    ax.plot(t, norm_open, "-", linewidth=1.8, label="Open-loop norm")
    ax.plot(t, norm_closed, "-", linewidth=1.8, label="Closed-loop norm")
    ax.set_ylabel(labels[3])
    ax.set_xlabel("Time (s)")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_rotation_errors(t, rpy_open, ang_open, rpy_closed, ang_closed):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    labels = ["roll error (deg)", "pitch error (deg)", "yaw error (deg)", "angle error (deg)"]

    for i in range(3):
        ax = axes[i]
        ax.plot(t, np.rad2deg(rpy_open[:, i]), "-", linewidth=1.5, label="Open-loop: KF - Planned")
        ax.plot(t, np.rad2deg(rpy_closed[:, i]), "-", linewidth=1.5, label="Closed-loop: KF - Planned")
        ax.set_ylabel(labels[i])
        ax.grid(True)
        ax.legend()

    ax = axes[3]
    ax.plot(t, np.rad2deg(ang_open), "-", linewidth=1.8, label="Open-loop angle")
    ax.plot(t, np.rad2deg(ang_closed), "-", linewidth=1.8, label="Closed-loop angle")
    ax.set_ylabel(labels[3])
    ax.set_xlabel("Time (s)")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()
    plt.close()


# ---------------------------------------------------------
# Main (args, same-place paths, no saving)
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planned", default=PLANNED_CSV_DEFAULT, help="Planned CSV (default: log/trajectory_sim.csv)")
    parser.add_argument("--kf-open", default=KF_OPEN_DEFAULT, help="Open-loop KF CSV (default: log/kf_predictions_open_loop.csv)")
    parser.add_argument("--kf-closed", default=KF_CLOSED_DEFAULT, help="Closed-loop KF CSV (default: log/kf_predictions_closed_loop.csv)")
    parser.add_argument("--shift-planned", type=int, default=1,
                        help="Shift planned arrays by 1 like your script (default: 1). Use 0 to disable.")
    args = parser.parse_args()

    if not os.path.exists(args.planned):
        print(f"[ERROR] Planned CSV not found: {args.planned}")
        return 1

    if not os.path.exists(args.kf_open):
        print(f"[ERROR] Open-loop KF CSV not found: {args.kf_open}")
        return 2

    if not os.path.exists(args.kf_closed):
        print(f"[ERROR] Closed-loop KF CSV not found: {args.kf_closed}")
        return 3

    print(f"[INFO] Loading planned CSV: {args.planned}")
    t_plan, Q_plan, dQ_plan = load_planned_csv(args.planned)

    # same "adjust planned by 1 offset" behavior as your inspiration
    if args.shift_planned != 0 and len(Q_plan) > 1:
        Q_plan[1:, :] = Q_plan[:-1, :]
        dQ_plan[1:, :] = dQ_plan[:-1, :]

    print(f"[INFO] Loading KF open-loop CSV: {args.kf_open}")
    t_open, Q_open, _ = load_kf_csv(args.kf_open)

    print(f"[INFO] Loading KF closed-loop CSV: {args.kf_closed}")
    t_closed, Q_closed, _ = load_kf_csv(args.kf_closed)

    if t_open is None or t_closed is None:
        print("[ERROR] Failed to parse one of the KF files.")
        return 4

    print("[INFO] Computing errors vs planned...")
    res_open = compute_pose_errors_against_plan(t_plan, Q_plan, t_open, Q_open)
    res_closed = compute_pose_errors_against_plan(t_plan, Q_plan, t_closed, Q_closed)
    if res_open is None or res_closed is None:
        print("[ERROR] No overlap between planned and KF times (open or closed).")
        return 5

    t1, pos1, n1, rpy1, ang1 = res_open
    t2, pos2, n2, rpy2, ang2 = res_closed

    # Put both KF results onto a common time base (use open-loop timestamps like "do the same as before")
    # by trimming to overlap and interpolating closed onto t1.
    t_common, pos1_c, _, _ = _overlap_and_trim(t1, pos1, t2, pos2)
    if t_common is None or len(t_common) < 2:
        print("[ERROR] No overlap between open-loop and closed-loop KF time ranges.")
        return 6

    # Trim open-loop to common
    mask1 = (t1 >= t_common[0]) & (t1 <= t_common[-1])
    t = t1[mask1]
    pos_open = pos1[mask1]
    n_open = n1[mask1]
    rpy_open = rpy1[mask1]
    ang_open = ang1[mask1]

    # Interpolate closed-loop onto t
    def _interpN(t_src, X_src, t_dst):
        X_src = np.asarray(X_src)
        if X_src.ndim == 1:
            return np.interp(t_dst, t_src, X_src)
        out = np.zeros((len(t_dst), X_src.shape[1]), float)
        for k in range(X_src.shape[1]):
            out[:, k] = np.interp(t_dst, t_src, X_src[:, k])
        return out

    pos_closed = _interpN(t2, pos2, t)
    n_closed = _interpN(t2, n2, t)
    rpy_closed = _interpN(t2, rpy2, t)
    ang_closed = _interpN(t2, ang2, t)

    print("[INFO] Plotting position errors (KF - Planned)...")
    plot_position_errors(t, pos_open, n_open, pos_closed, n_closed)

    print("[INFO] Plotting rotation errors (KF - Planned)...")
    plot_rotation_errors(t, rpy_open, ang_open, rpy_closed, ang_closed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())