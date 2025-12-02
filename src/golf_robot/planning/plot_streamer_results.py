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

# Toggle this to True to show the joint-velocity (dQ) plot interactively
# instead of saving it to disk. Default False keeps non-interactive 'Agg'
# backend so automated runs (CI / headless) won't block.
# SHOW_PLOTS = True
SHOW_PLOTS = True
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

import matplotlib.pyplot as plt
import time
import glob
import os
from kinematics import fk_ur10

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


def plot_comparisons(t_plan, Q_plan, dQ_plan, t_meas, Q_meas, dQ_meas, out_prefix=None):
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

    if P_plan is not None and P_meas is not None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(P_plan[:,0], P_plan[:,1], P_plan[:,2], 'b-', label='Planned', linewidth=2)
        ax.plot(P_meas[:,0], P_meas[:,1], P_meas[:,2], 'r--', label='Measured', linewidth=1)
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


def plot_kf_comparisons(t_meas, Q_meas, dQ_meas, t_kf, Q_kf, dQ_kf, out_prefix=None):
    """Create two plots comparing measured joints/velocities to KF predictions.
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

    # Joint velocities comparison
    if t_meas is not None and len(t_meas) > 0 and t_kf is not None and len(t_kf) > 0:
        fig, axes = plt.subplots(3,2, figsize=(12,10))
        axes = axes.flatten()
        for i in range(6):
            axes[i].plot(t_meas, dQ_meas[:,i], 'r.', label='Measured', linewidth=0.7)
            axes[i].plot(t_kf, dQ_kf[:,i], 'b.', label='KF pred', linewidth=2)
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

    print("[INFO] Plotting comparisons...")
    # Plot distribution of sampling intervals first
    sf = plot_sampling_intervals(t_meas)
    if sf:
        print(f"[INFO] Saved sampling-intervals plot: {sf}")

    f1, f2, f_accel, f_tcp = plot_comparisons(t_plan, Q_plan, dQ_plan, t_meas, Q_meas, dQ_meas)
    print("[INFO] Saved plots:")
    print(f"  - {f1}")
    print(f"  - {f2}")
    if f_accel:
        print(f"  - {f_accel}")
    if f_tcp:
        print(f"  - {f_tcp}")

    # If Kalman filter predictions exist, load and plot comparisons
    kf_path = os.path.join(OUT_DIR, 'kf_predictions.csv')
    if os.path.exists(kf_path):
        print(f"[INFO] Loading KF predictions: {kf_path}")
        t_kf, Q_kf, dQ_kf = load_kf_csv(kf_path)
        if t_kf is not None:
            fn_q, fn_dq = plot_kf_comparisons(t_meas, Q_meas, dQ_meas, t_kf, Q_kf, dQ_kf)
            print("[INFO] Saved KF comparison plots:")
            if fn_q:
                print(f"  - {fn_q}")
            if fn_dq:
                print(f"  - {fn_dq}")
        else:
            print("[WARN] KF predictions file found but no valid rows parsed.")

    # Plot TCP velocity comparisons (vx,vy,vz,|v|)
    fn_tcp_vel = plot_tcp_velocity_comparison(t_plan, Q_plan, t_meas, Q_meas)
    if fn_tcp_vel:
        print(f"[INFO] Saved TCP-velocity comparison: {fn_tcp_vel}")

    return 0


if __name__ == '__main__':
    exit(main())
