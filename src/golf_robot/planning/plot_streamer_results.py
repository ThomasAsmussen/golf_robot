"""
plot_streamer_results.py

Plot planned vs actual trajectories saved by the planner and the C++ streamer.

This script is configuration-driven via the globals below. It will:
 - load the planned CSV (t,q0..q5,dq0..dq5)
 - find the latest streamer log in log/ named streamer_log_YYYYMMDD_HHMMSS.csv
 - compute TCP paths via FK and plot comparisons (joints, velocities, TCP path)
 - save figures into the log/ folder with timestamps

Edit the PLANNED_CSV constant or the STREAMER_LOG_GLOB if you need custom paths.
"""
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import glob
import os
from kinematics import *

# Toggle this to True to show the joint-velocity (dQ) plot interactively
# instead of saving it to disk. Default False keeps non-interactive 'Agg'
# backend so automated runs (CI / headless) won't block.
# SHOW_PLOTS = True
SHOW_PLOTS = True
ONLY_KF = True
if SHOW_PLOTS:
    # Try a sequence of common interactive backends; fall back to default
    for bk in ('TkAgg', 'Qt5Agg', 'GTK3Agg', 'WXAgg'):
        try:
            matplotlib.use(bk)
            break
        except Exception:
            continue
else:
    matplotlib.use('Agg')


# Configuration (edit as needed)
PLANNED_CSV = 'log/trajectory_sim.csv'          # planner-produced CSV
STREAMER_LOG_GLOB = 'log/streamed_measurements.csv'    # glob to find streamer logs; latest will be used
OUT_DIR = 'log'


def load_planned_csv(path):
    """Load planner CSV format: t, q0..q5, dq0..dq5 -> returns t (N,), Q (N,6), dQ (N,6)
    """
    t = []
    Q = []
    dQ = []
    with open(path, 'r') as f:
        hdr = f.readline()
        for line in f:
            if not line.strip():
                continue
            vals = [float(x) for x in line.strip().split(',')]
            if len(vals) < 13:
                continue
            t.append(vals[0])
            Q.append(vals[1:7])
            dQ.append(vals[7:13])
    return np.array(t), np.array(Q), np.array(dQ)


def load_streamer_log(path):
    """
    Load streamer log CSV with strict pairing:
    - Each kept row must have: t, q0..q5, dq0..dq5 (13 numeric cells)
    - Q_list and dQ_list are built together in the same loop
    - Any non-complete rows toward the *end* are dropped by stopping at the first bad row after data begins
    Returns: t (N,), Q (N,6), dQ (N,6)
    """
    import csv
    import numpy as np

    t_list, Q_list, dQ_list = [], [], []

    def _is_header(row):
        if not row:
            return True
        s = ",".join(row).lower()
        return any(k in s for k in ("t", "time", "q0", "dq0"))

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)

        # Peek first line to handle header
        first = next(reader, None)
        if first is not None and not _is_header(first):
            rows = [first] + list(reader)
        else:
            rows = list(reader)

        # Normalize rows: strip cells, drop trailing empties
        norm_rows = []
        for r in rows:
            if not r:
                continue
            r = [c.strip() for c in r]
            while r and r[-1] == "":
                r.pop()
            if not r:
                continue
            norm_rows.append(r)

        seen_valid = False
        for row in norm_rows:
            # Require at least 13 columns: t, 6 q's, 6 dq's
            if len(row) < 13:
                
                # If we've already started collecting valid data, stop here:
                if seen_valid:
                    break
                else:
                    # ignore leading junk/short rows before data starts
                    continue
            try:
                t_val = float(row[0])
                q_vals = [float(x) for x in row[1:7]]
                dq_vals = [float(x) for x in row[7:13]]
            except ValueError:
                # Stop if malformed appears after valid data started
                if seen_valid:
                    break
                else:
                    continue

            # Row is complete & numeric: append both Q and dQ together
            t_list.append(t_val)
            Q_list.append(q_vals)
            dQ_list.append(dq_vals)
            seen_valid = True

        

    # Convert to arrays (Q and dQ lengths are guaranteed equal here)
    t = np.asarray(t_list, dtype=float)
    Q = np.asarray(Q_list, dtype=float).reshape(-1, 6)
    dQ = np.asarray(dQ_list, dtype=float).reshape(-1, 6)
    print(f"Q shape: {Q.shape}, dQ shape: {dQ.shape}")
    print(Q)
    print(dQ)
    return t, Q, dQ


def _T_tcp_from_q(q):
    """Return 4x4 TCP transform from fk_ur10 output (which may be a list or a single matrix)."""
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


def _rot_to_rpy_xyz(R):
    """
    Convert rotation matrix to roll/pitch/yaw using XYZ (roll about x, pitch about y, yaw about z).
    Returns (roll, pitch, yaw) in radians.
    """
    R = np.asarray(R, float)
    # pitch = asin(-R[2,0]) with clamp for numerical safety
    sp = -R[2, 0]
    sp = np.clip(sp, -1.0, 1.0)
    pitch = np.arcsin(sp)

    # handle near gimbal lock
    cp = np.cos(pitch)
    if abs(cp) < 1e-9:
        # gimbal lock: roll and yaw coupled
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw], dtype=float)


def tcp_rpy_from_Q(Q):
    """Compute TCP roll/pitch/yaw for each joint configuration in Q (N,6). Returns (N,3)."""
    Q = np.asarray(Q, float)
    rpy = np.zeros((Q.shape[0], 3), dtype=float)
    for i in range(Q.shape[0]):
        T = _T_tcp_from_q(Q[i])
        R = T[:3, :3]
        rpy[i, :] = _rot_to_rpy_xyz(R)
    return rpy


def plot_tcp_orientation_error_rpy(t_plan, Q_plan, t_meas, Q_meas, out_prefix=None):
    """
    Plot (measured - planned) TCP orientation error in roll/pitch/yaw vs time (three subplots).

    We unwrap planned/measured RPY first, interpolate planned RPY onto t_meas, then subtract.
    Returns filename or None (if SHOW_PLOTS).
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = out_prefix or time.strftime('%Y%m%d_%H%M%S')

    # Basic checks
    if t_plan is None or Q_plan is None or len(t_plan) < 2 or len(Q_plan) < 2:
        print("[WARN] Not enough planned data for TCP orientation error plot")
        return None
    if t_meas is None or Q_meas is None or len(t_meas) < 2 or len(Q_meas) < 2:
        print("[WARN] Not enough measured data for TCP orientation error plot")
        return None

    # Compute TCP orientations (RPY)
    try:
        rpy_plan = tcp_rpy_from_Q(Q_plan)  # (N,3)
        rpy_meas = tcp_rpy_from_Q(Q_meas)  # (M,3)
    except Exception as e:
        print(f"[WARN] Failed to compute TCP orientations: {e}")
        return None

    # Ensure strictly increasing time for interpolation
    def _sorted_unique_time(t, X):
        t = np.asarray(t, float)
        X = np.asarray(X, float)
        order = np.argsort(t)
        t_s = t[order]
        X_s = X[order]
        keep = np.ones_like(t_s, dtype=bool)
        keep[1:] = t_s[1:] > t_s[:-1]
        return t_s[keep], X_s[keep]

    t_plan_s, rpy_plan_s = _sorted_unique_time(t_plan, rpy_plan)
    t_meas_s, rpy_meas_s = _sorted_unique_time(t_meas, rpy_meas)

    # Overlap window so we don't extrapolate
    t0 = max(t_plan_s[0], t_meas_s[0])
    t1 = min(t_plan_s[-1], t_meas_s[-1])
    if t1 <= t0:
        print("[WARN] No time overlap between planned and measured for TCP orientation error plot")
        return None

    mask = (t_meas_s >= t0) & (t_meas_s <= t1)
    t_m = t_meas_s[mask]
    rpy_m = rpy_meas_s[mask]
    if len(t_m) < 2:
        print("[WARN] Not enough overlapping samples for TCP orientation error plot")
        return None

    # Unwrap angles before interpolation/subtraction to avoid pi jumps
    rpy_plan_u = np.unwrap(rpy_plan_s, axis=0)
    rpy_meas_u = np.unwrap(rpy_m, axis=0)

    # Interpolate planned RPY onto measured timestamps
    rpy_p_interp = np.zeros_like(rpy_meas_u)
    for k in range(3):
        rpy_p_interp[:, k] = np.interp(t_m, t_plan_s, rpy_plan_u[:, k])

    err = rpy_meas_u - rpy_p_interp  # (measured - planned), unwrapped space

    # Optional: wrap error back to [-pi, pi] for readability
    err = (err + np.pi) % (2.0 * np.pi) - np.pi

    # Plot roll/pitch/yaw errors
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    labels = ["roll error (deg)", "pitch error (deg)", "yaw error (deg)"]
    for i in range(3):
        ax = axes[i]
        ax.plot(t_m, np.rad2deg(err[:, i]), '-', linewidth=1.5, label='Measured - Planned')
        ax.set_ylabel(labels[i])
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    fn = os.path.join(OUT_DIR, f'comparison_tcp_orientation_error_{ts}.png')
    if SHOW_PLOTS:
        try:
            plt.show()
        finally:
            plt.close()
        return None
    else:
        plt.savefig(fn)
        plt.close()
        return fn


def plot_tcp_orientation_comparison(t_plan, Q_plan, t_meas, Q_meas, out_prefix=None, Q_kf=None):
    """
    Plot planned vs measured TCP orientation as roll/pitch/yaw (rad) over time.
    Optionally include KF orientation if Q_kf is provided.
    Returns filename or None (if SHOW_PLOTS).
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = out_prefix or time.strftime('%Y%m%d_%H%M%S')

    if t_plan is None or Q_plan is None or len(t_plan) < 2 or len(Q_plan) < 2:
        print("[WARN] Not enough planned data for TCP orientation plot")
        return None
    if t_meas is None or Q_meas is None or len(t_meas) < 2 or len(Q_meas) < 2:
        print("[WARN] Not enough measured data for TCP orientation plot")
        return None

    # Compute RPY
    try:
        rpy_plan = tcp_rpy_from_Q(Q_plan)
        rpy_meas = tcp_rpy_from_Q(Q_meas)
        rpy_kf = tcp_rpy_from_Q(Q_kf) if Q_kf is not None and len(Q_kf) > 1 else None
    except Exception as e:
        print(f"[WARN] Failed to compute TCP orientation: {e}")
        return None

    # Unwrap to avoid pi-jumps (independently per component)
    rpy_plan_u = np.unwrap(rpy_plan, axis=0)
    rpy_meas_u = np.unwrap(rpy_meas, axis=0)
    rpy_kf_u = np.unwrap(rpy_kf, axis=0) if rpy_kf is not None else None

    labels = ["roll (rad)", "pitch (rad)", "yaw (rad)"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

    for i in range(3):
        ax = axes[i]
        ax.plot(t_plan, rpy_plan_u[:, i], 'b-', label='Planned', linewidth=2)
        ax.plot(t_meas, rpy_meas_u[:, i], 'r.', label='Measured', linewidth=0.7)
        if rpy_kf_u is not None:
            ax.plot(np.linspace(t_meas[0], t_meas[-1], len(rpy_kf_u)), rpy_kf_u[:, i],
                    color='g', linestyle='-.', label='KF', linewidth=1.5)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(labels[i])
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    fn = os.path.join(OUT_DIR, f'comparison_tcp_orientation_{ts}.png')

    if SHOW_PLOTS:
        try:
            plt.show()
        finally:
            plt.close()
        return None
    else:
        plt.savefig(fn)
        plt.close()
        return fn

def tcp_xyz_from_q(q):
    T = fk_ur10(np.asarray(q, float))
    if isinstance(T, (list, tuple)):
        T_tcp = T[-1]
    else:
        T_tcp = T
    T_tcp = np.asarray(T_tcp, float)
    if T_tcp.shape == (4,4):
        return T_tcp[:3,3].copy()
    elif T_tcp.shape == (3,4):
        return T_tcp[:3,3].copy()
    else:
        raise TypeError(f"Unexpected transform shape: {T_tcp.shape}")


def tcp_path_from_Q(Q):
    return np.array([tcp_xyz_from_q(q) for q in Q])


def plot_comparisons(t_plan, Q_plan, dQ_plan, t_meas, Q_meas, dQ_meas, out_prefix=None, Q_kf=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = out_prefix or time.strftime('%Y%m%d_%H%M%S')

    # Align times for plotting: keep native times but plots will show their own time axes

    # Joint positions
    fig, axes = plt.subplots(3,2, figsize=(12,10))
    axes = axes.flatten()
    for i in range(6):
        axes[i].plot(t_plan, Q_plan[:,i], 'b-', label='Planned', linewidth=2)
        if len(t_meas) > 0:
            axes[i].plot(t_meas, Q_meas[:,i], 'r.', label='Measured', linewidth=1)

        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel(f'Joint {i+1} (rad)')
        axes[i].legend()
        axes[i].grid(True)
    plt.tight_layout()
    fn1 = os.path.join(OUT_DIR, f'comparison_joints_{ts}.png')
    if SHOW_PLOTS:
        try:
            plt.show()
        finally:
            plt.close()
        fn1 = None
    else:
        plt.savefig(fn1)
        plt.close()

    # Joint velocities (planned vs measured via finite diff)
    fig, axes = plt.subplots(3,2, figsize=(12,10))
    axes = axes.flatten()

    for i in range(6):
        axes[i].plot(t_plan, dQ_plan[:,i], 'b-', label='Planned', linewidth=2)
        if len(t_meas) > 0:
            axes[i].plot(t_meas, dQ_meas[:,i], 'r.', label='Measured', linewidth=0.5)

        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel(f'Joint {i+1} Velocity (rad/s)')
        axes[i].legend()
        axes[i].grid(True)
    plt.tight_layout()

    # If SHOW_PLOTS is enabled, show the joint-velocity plot interactively
    # instead of saving it. Otherwise save to disk as before.
    # Save or show the velocities figure depending on SHOW_PLOTS
    if SHOW_PLOTS:
        try:
            plt.show()
        finally:
            plt.close()
        fn2 = None
    else:
        fn2 = os.path.join(OUT_DIR, f'comparison_joint_velocities_{ts}.png')
        plt.savefig(fn2)
        plt.close()

    # Joint accelerations (finite-difference of velocities)
    def compute_accelerations(t, dQ):
        if t is None or len(t) < 2 or dQ is None or len(dQ) < 2:
            return np.zeros((0,6))
        a = np.zeros_like(dQ)
        # compute per-joint derivative using numpy.gradient with non-uniform spacing
        for j in range(dQ.shape[1]):
            try:
                a[:, j] = np.gradient(dQ[:, j], t)
            except Exception:
                # fallback to simple diff/divide (last element duplicated)
                dt = np.diff(t)
                da = np.diff(dQ[:, j]) / dt
                a[:-1, j] = da
                a[-1, j] = da[-1] if len(da) > 0 else 0.0
        return a

    a_plan = compute_accelerations(t_plan, dQ_plan)
    a_meas = compute_accelerations(t_meas, dQ_meas)

    fig, axes = plt.subplots(3,2, figsize=(12,10))
    axes = axes.flatten()
    for i in range(6):
        if a_plan.size > 0:
            axes[i].plot(t_plan, a_plan[:,i], 'b-', label='Planned accel', linewidth=2)
        if a_meas.size > 0:
            axes[i].plot(t_meas, a_meas[:,i], 'r.', label='Measured accel', linewidth=1)

        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel(f'Joint {i+1} Accel (rad/s^2)')
        axes[i].legend()
        axes[i].grid(True)
    plt.tight_layout()
    fn_accel = os.path.join(OUT_DIR, f'comparison_joint_accelerations_{ts}.png')
    if SHOW_PLOTS:
        try:
            plt.show()
        finally:
            plt.close()
        fn_accel = None
    else:
        plt.savefig(fn_accel)
        plt.close()

    # TCP path comparison 3D
    try:
        P_plan = tcp_path_from_Q(Q_plan)
    except Exception:
        P_plan = None

    P_meas = None
    if len(t_meas) > 0:
        try:
            P_meas = tcp_path_from_Q(Q_meas)
        except Exception:
            P_meas = None

    P_kf = None
    if Q_kf is not None:
        try:
            P_kf = tcp_path_from_Q(Q_kf)
        except Exception:
            P_kf = None
    if P_plan is not None and P_meas is not None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(P_plan[:,0], P_plan[:,1], P_plan[:,2], 'b-', label='Planned', linewidth=2)
        ax.plot(P_meas[:,0], P_meas[:,1], P_meas[:,2], 'r--', label='Measured', linewidth=1)
        if P_kf is not None:
            ax.plot(P_kf[:,0], P_kf[:,1], P_kf[:,2], color='g', linestyle='-.', label='KF', linewidth=1.5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.scatter([P_plan[0,0]], [P_plan[0,1]], [P_plan[0,2]], color='g', s=50, label='Start', zorder=5)
        ax.scatter([P_plan[-1,0]], [P_plan[-1,1]], [P_plan[-1,2]], color='b', s=50, label='End', zorder=5)
        
        ax.legend()
        ax.set_title('TCP Path Comparison')
        plt.tight_layout()
        fn_tcp = os.path.join(OUT_DIR, f'comparison_tcp_{ts}.png')
        if SHOW_PLOTS:
            try:
                plt.show()
            finally:
                plt.close()
            fn_tcp = None
        else:
            plt.savefig(fn_tcp)
            plt.close()
    else:
        fn_tcp = None
    return fn1, fn2, fn_accel, fn_tcp


def plot_tcp_position_error_xyz(t_plan, Q_plan, t_meas, Q_meas, out_prefix=None):
    """
    Plot (measured - planned) TCP position error in x/y/z vs time (three subplots).

    We interpolate planned TCP xyz onto t_meas so the subtraction is well-defined.
    Returns filename or None (if SHOW_PLOTS).
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = out_prefix or time.strftime('%Y%m%d_%H%M%S')

    # Basic checks
    if t_plan is None or Q_plan is None or len(t_plan) < 2 or len(Q_plan) < 2:
        print("[WARN] Not enough planned data for TCP position error plot")
        return None
    if t_meas is None or Q_meas is None or len(t_meas) < 2 or len(Q_meas) < 2:
        print("[WARN] Not enough measured data for TCP position error plot")
        return None

    # Compute TCP positions
    try:
        P_plan = tcp_path_from_Q(Q_plan)  # (N,3)
        P_meas = tcp_path_from_Q(Q_meas)  # (M,3)
    except Exception as e:
        print(f"[WARN] Failed to compute TCP positions: {e}")
        return None

    # Ensure strictly increasing time for interpolation (np.interp needs ascending x)
    def _sorted_unique_time(t, P):
        t = np.asarray(t, float)
        P = np.asarray(P, float)
        order = np.argsort(t)
        t_s = t[order]
        P_s = P[order]
        # drop duplicate timestamps (keep first occurrence)
        keep = np.ones_like(t_s, dtype=bool)
        keep[1:] = t_s[1:] > t_s[:-1]
        return t_s[keep], P_s[keep]

    t_plan_s, P_plan_s = _sorted_unique_time(t_plan, P_plan)
    t_meas_s, P_meas_s = _sorted_unique_time(t_meas, P_meas)

    # Overlap window so we don't extrapolate
    t0 = max(t_plan_s[0], t_meas_s[0])
    t1 = min(t_plan_s[-1], t_meas_s[-1])
    if t1 <= t0:
        print("[WARN] No time overlap between planned and measured for TCP error plot")
        return None

    mask = (t_meas_s >= t0) & (t_meas_s <= t1)
    t_m = t_meas_s[mask]
    P_m = P_meas_s[mask]
    if len(t_m) < 2:
        print("[WARN] Not enough overlapping samples for TCP error plot")
        return None

    # Interpolate planned xyz onto measured timestamps
    P_p_interp = np.zeros_like(P_m)
    for k in range(3):
        P_p_interp[:, k] = np.interp(t_m, t_plan_s, P_plan_s[:, k])

    err = P_m - P_p_interp  # (measured - planned)

    # Plot x/y/z errors
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    labels = ["x error (m)", "y error (m)", "z error (m)"]
    for i in range(3):
        ax = axes[i]
        ax.plot(t_m, err[:, i], '-', linewidth=1.5, label='Measured - Planned')
        ax.set_ylabel(labels[i])
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    fn = os.path.join(OUT_DIR, f'comparison_tcp_position_error_{ts}.png')
    if SHOW_PLOTS:
        try:
            plt.show()
        finally:
            plt.close()
        return None
    else:
        plt.savefig(fn)
        plt.close()
        return fn


def plot_sampling_intervals(t_meas, out_prefix=None):
    """Plot distribution (histogram) of sampling intervals (diffs of t_meas).
    Saves a PNG to OUT_DIR and returns the filename (or None if not enough data).
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = out_prefix or time.strftime('%Y%m%d_%H%M%S')

    if t_meas is None or len(t_meas) < 2:
        print("[WARN] Not enough measurement times to compute sampling intervals")
        return None

    diffs = np.diff(t_meas)
    # Basic stats
    mean_dt = float(np.mean(diffs))
    median_dt = float(np.median(diffs))
    std_dt = float(np.std(diffs))
    p99 = float(np.percentile(diffs, 99))
    print(f"[INFO] Sampling intervals stats: mean={mean_dt:.6f}s median={median_dt:.6f}s std={std_dt:.6f}s 99p={p99:.6f}s")

    # Choose bins that cover the observed range (avoid zero-span)
    maxv = max(np.max(diffs), 1e-6)
    bins = np.linspace(0.0, maxv * 1.1, min(200, max(10, len(diffs)//2)))

    fig = plt.figure(figsize=(6,4))
    plt.hist(diffs, bins=bins, color='C0', edgecolor='k')
    plt.xlabel('Sampling interval (s)')
    plt.ylabel('Count')
    plt.title('Distribution of sampling intervals')
    plt.grid(True)
    # Mark the expected/planned dt (0.008s) on the histogram for reference
    target_dt = 0.008
    # plt.axvline(target_dt, color='r', linestyle='--', linewidth=2, label=f'target dt={target_dt:.3f}s')
    plt.axvline(target_dt, color='r', linestyle='--', linewidth=2, label=f'target dt={target_dt:.3f}s')
    ax = plt.gca()
    ylim = ax.get_ylim()
    y_ann = ylim[1] * 0.9
    # plt.text(target_dt, y_ann, f'{target_dt:.3f}s', color='r', ha='left', va='top', rotation=90, fontsize=9)
    plt.legend()
    fn = os.path.join(OUT_DIR, f'sampling_intervals_{ts}.png')
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()
    return fn


def load_kf_csv(path):
    """Load Kalman filter CSV: t,qhat0..qhat5,dqhat0..dqhat5 -> t (N,), Qhat (N,6), dQhat (N,6)
    """
    if not os.path.exists(path):
        return None, None, None
    t_list = []
    Q_list = []
    dQ_list = []
    with open(path, 'r') as f:
        hdr = f.readline()
        for line in f:
            if not line.strip():
                continue
            vals = [x.strip() for x in line.strip().split(',')]
            # Expect at least 13 columns: t, qhat0..5, dqhat0..5
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
    return np.asarray(t_list, float), np.asarray(Q_list, float).reshape(-1,6), np.asarray(dQ_list, float).reshape(-1,6)


def save_fd_velocities(t_meas, dQ_meas_fd, out_file='log/step.csv'):
    """Save finite-difference velocities to CSV: t, dq0, dq1, ..., dq5
    """
    if t_meas is None or dQ_meas_fd is None or len(t_meas) == 0:
        print("[WARN] Cannot save FD velocities: no data")
        return
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        # Write header
        f.write('t,dq0,dq1,dq2,dq3,dq4,dq5\n')
        # Write rows
        for i, t_val in enumerate(t_meas):
            row = [str(t_val)] + [str(dQ_meas_fd[i, j]) for j in range(6)]
            f.write(','.join(row) + '\n')
    print(f"[INFO] Saved FD velocities to {out_file}")


def plot_kf_comparisons(t_meas, Q_meas, dQ_meas, t_kf, Q_kf, dQ_kf, out_prefix=None, t_plan=None, dQ_plan=None):
    """Create two plots comparing measured joints/velocities to KF predictions.
    Adds direct finite-difference velocities from measured positions and optional planned velocities.
    Returns filenames (fn_q, fn_dq) or (None,None) if skipped.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = out_prefix or time.strftime('%Y%m%d_%H%M%S')

    fn_q = None
    fn_dq = None

    # Joint positions comparison
    if t_meas is not None and len(t_meas) > 0 and t_kf is not None and len(t_kf) > 0:
        fig, axes = plt.subplots(3,2, figsize=(12,10))
        axes = axes.flatten()
        for i in range(6):
            # measured (red dots)
            axes[i].plot(t_meas, Q_meas[:,i], 'r.', label='Measured', linewidth=1)
            # kf prediction (blue dots)
            axes[i].plot(t_kf, Q_kf[:,i], 'b.', label='KF pred', linewidth=2)
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel(f'Joint {i+1} (rad)')
            axes[i].legend()
            axes[i].grid(True)
        plt.tight_layout()
        fn_q = os.path.join(OUT_DIR, f'kf_comparison_joints_{ts}.png')
        if SHOW_PLOTS:
            try:
                plt.show()
            finally:
                plt.close()
            fn_q = None
        else:
            plt.savefig(fn_q)
            plt.close()

    # Joint velocities comparison (measured vs KF, plus finite diff and planned)
    def _finite_diff_velocities(t, Q):
        if t is None or Q is None or len(t) < 2 or len(Q) < 2:
            return None
        dQ_fd = np.zeros_like(Q)
        for j in range(Q.shape[1]):
            try:
                dQ_fd[:, j] = np.gradient(Q[:, j], t)
            except Exception:
                dt = np.diff(t)
                dq = np.diff(Q[:, j]) / dt
                dQ_fd[:-1, j] = dq
                dQ_fd[-1, j] = dq[-1] if len(dq) > 0 else 0.0
        return dQ_fd

    dQ_meas_fd = _finite_diff_velocities(t_meas, Q_meas)

    if t_meas is not None and len(t_meas) > 0 and t_kf is not None and len(t_kf) > 0:
        fig, axes = plt.subplots(3,2, figsize=(12,10))
        axes = axes.flatten()
        for i in range(6):
            # measured (red)
            axes[i].plot(t_meas, dQ_meas[:,i], 'r.', label='Measured dQ', linewidth=0.7)
            # direct finite-difference from measured positions (orange)
            if dQ_meas_fd is not None:
                axes[i].plot(t_meas, dQ_meas_fd[:,i], color='orange', linestyle='-', label='Measured FD dQ', linewidth=1.2)
            # Kalman filter prediction (blue)
            axes[i].plot(t_kf, dQ_kf[:,i], 'b.', label='KF pred dQ', linewidth=2)
            # planned velocities (green) if provided
            if t_plan is not None and dQ_plan is not None and len(t_plan) == len(dQ_plan):
                axes[i].plot(t_plan, dQ_plan[:,i], color='green', linestyle='-', label='Planned dQ', linewidth=2)
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel(f'Joint {i+1} Velocity (rad/s)')
            axes[i].legend()
            axes[i].grid(True)
        plt.tight_layout()
        fn_dq = os.path.join(OUT_DIR, f'kf_comparison_velocities_{ts}.png')
        if SHOW_PLOTS:
            try:
                plt.show()
            finally:
                plt.close()
            fn_dq = None
        else:
            plt.savefig(fn_dq)
            plt.close()

    return fn_q, fn_dq


def plot_tcp_velocity_comparison(t_plan, Q_plan, t_meas, Q_meas, out_prefix=None):
    """Compute TCP velocities from joint paths and plot vx, vy, vz, and |v| comparisons.
    Returns filename or None.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = out_prefix or time.strftime('%Y%m%d_%H%M%S')

    # Compute TCP positions
    P_plan = None
    P_meas = None
    try:
        if Q_plan is not None and len(Q_plan) > 0:
            P_plan = tcp_path_from_Q(Q_plan)
    except Exception:
        P_plan = None
    try:
        if Q_meas is not None and len(Q_meas) > 0:
            P_meas = tcp_path_from_Q(Q_meas)
    except Exception:
        P_meas = None

    # Need at least positions and time to compute velocities
    if P_plan is None or P_meas is None or t_plan is None or t_meas is None:
        return None

    def compute_velocities(t, P):
        # P: (N,3), t: (N,)
        if P is None or t is None or len(t) < 2:
            return None, None
        V = np.zeros_like(P)
        for i in range(3):
            try:
                V[:, i] = np.gradient(P[:, i], t)
            except Exception:
                # fallback to finite diff
                dt = np.diff(t)
                dv = np.diff(P[:, i], axis=0) / dt[:, None]
                V[:-1, i] = dv[:, 0]
                V[-1, i] = dv[-1, 0] if len(dv) > 0 else 0.0
        speed = np.linalg.norm(V, axis=1)
        return V, speed

    V_plan, s_plan = compute_velocities(t_plan, P_plan)
    V_meas, s_meas = compute_velocities(t_meas, P_meas)
    if V_plan is None or V_meas is None:
        return None

    # Plot vx, vy, vz, and |v|
    fig, axes = plt.subplots(2,2, figsize=(12,8))
    axes = axes.flatten()
    labels = ['vx (m/s)', 'vy (m/s)', 'vz (m/s)', '|v| (m/s)']

    for idx in range(4):
        ax = axes[idx]
        if idx < 3:
            ax.plot(t_plan, V_plan[:, idx], 'b-', label='Planned', linewidth=2)
            ax.plot(t_meas, V_meas[:, idx], 'r.', label='Measured', linewidth=0.7)
        else:
            ax.plot(t_plan, s_plan, 'b-', label='Planned', linewidth=2)
            ax.plot(t_meas, s_meas, 'r.', label='Measured', linewidth=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(labels[idx])
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    fn = os.path.join(OUT_DIR, f'comparison_tcp_velocities_{ts}.png')
    if SHOW_PLOTS:
        try:
            plt.show()
        finally:
            plt.close()
        fn = None
    else:
        plt.savefig(fn)
        plt.close()
    return fn


def find_latest_streamer_log(glob_pattern=STREAMER_LOG_GLOB):
    files = glob.glob(glob_pattern)
    if not files:
        return None
    files.sort()
    return files[-1]


def main():
    planned_path = PLANNED_CSV
    if not os.path.exists(planned_path):
        print(f"[ERROR] Planned CSV not found at {planned_path}")
        return 1

    streamer_log = find_latest_streamer_log()
    if streamer_log is None:
        print(f"[ERROR] No streamer log found with pattern {STREAMER_LOG_GLOB}")
        return 2

    print(f"[INFO] Loading planned CSV: {planned_path}")
    t_plan, Q_plan, dQ_plan = load_planned_csv(planned_path)
    print(f"[INFO] Loading streamer log: {streamer_log}")
    t_meas, Q_meas, dQ_meas = load_streamer_log(streamer_log)

    # Attempt to load KF predictions early so we can include TCP in 3D plot
    t_kf = Q_kf = dQ_kf = None
    kf_path = os.path.join(OUT_DIR, 'kf_predictions.csv')
    if os.path.exists(kf_path):
        try:
            t_kf, Q_kf, dQ_kf = load_kf_csv(kf_path)
        except Exception as e:
            print(f"[WARN] Failed to load KF predictions for TCP plot: {e}")

    print("[INFO] Plotting comparisons...")
    # Plot distribution of sampling intervals first
    sf = plot_sampling_intervals(t_meas)
    if sf:
        print(f"[INFO] Saved sampling-intervals plot: {sf}")

    # Compute and save finite-difference velocities
    def _finite_diff_velocities(t, Q):
        if t is None or Q is None or len(t) < 2 or len(Q) < 2:
            return None
        dQ_fd = np.zeros_like(Q)
        for j in range(Q.shape[1]):
            try:
                dQ_fd[:, j] = np.gradient(Q[:, j], t)
            except Exception:
                dt = np.diff(t)
                dq = np.diff(Q[:, j]) / dt
                dQ_fd[:-1, j] = dq
                dQ_fd[-1, j] = dq[-1] if len(dq) > 0 else 0.0
        return dQ_fd

    dQ_meas_fd = _finite_diff_velocities(t_meas, Q_meas)
    save_fd_velocities(t_meas, dQ_meas_fd)

    # fn_tcp_ori = plot_tcp_orientation_comparison(t_plan, Q_plan, t_meas, Q_meas, Q_kf=Q_kf if Q_kf is not None else None)
    # fn_tcp_ori = plot_tcp_orientation_comparison(t_plan, Q_plan-Q_plan, t_meas, Q_meas-Q_plan, None)
    # if fn_tcp_ori:
    #     print(f"[INFO] Saved TCP-orientation comparison: {fn_tcp_ori}")
    # Plot TCP position error (measured - planned) in x/y/z
    fn_tcp_err = plot_tcp_position_error_xyz(t_plan, Q_plan, t_meas, Q_meas)
    if fn_tcp_err:
        print(f"[INFO] Saved TCP position error plot: {fn_tcp_err}")
    fn_tcp_ori_err = plot_tcp_orientation_error_rpy(t_plan, Q_plan, t_meas, Q_meas)
    
    if fn_tcp_ori_err:
        print(f"[INFO] Saved TCP orientation error plot: {fn_tcp_ori_err}")

    f1, f2, f_accel, f_tcp = plot_comparisons(
        t_plan, Q_plan, dQ_plan,
        t_meas, Q_meas, dQ_meas,
        out_prefix=None,
        Q_kf=Q_kf if Q_kf is not None else None
    )
    print("[INFO] Saved plots:")
    print(f"  - {f1}")
    print(f"  - {f2}")
    if f_accel:
        print(f"  - {f_accel}")
    if f_tcp:
        print(f"  - {f_tcp}")

    # If Kalman filter predictions exist, plot comparisons (reuse early load if available)
    if t_kf is not None:
        print(f"[INFO] Using KF predictions for comparisons: {kf_path}")
        if t_kf is not None:
            fn_q, fn_dq = plot_kf_comparisons(
                t_meas, Q_meas, dQ_meas,
                t_kf, Q_kf, dQ_kf,
                out_prefix=None,
                t_plan=t_plan,
                dQ_plan=dQ_plan
            )
            print("[INFO] Saved KF comparison plots:")
            if fn_q:
                print(f"  - {fn_q}")
            if fn_dq:
                print(f"  - {fn_dq}")
    else:
        if os.path.exists(kf_path):
            print("[WARN] KF predictions file found but no valid rows parsed.")

    # Plot TCP velocity comparisons (vx,vy,vz,|v|)
    fn_tcp_vel = plot_tcp_velocity_comparison(t_plan, Q_plan, t_meas, Q_meas)
    if fn_tcp_vel:
        print(f"[INFO] Saved TCP-velocity comparison: {fn_tcp_vel}")

    return 0


if __name__ == '__main__':
    exit(main())
