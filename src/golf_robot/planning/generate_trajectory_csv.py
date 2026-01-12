"""
Minimal helper: plan a trajectory (via the Python planner in `trajectory.py`) and write it to CSV.

The output CSV matches the format expected by `traj_streamer.cpp`: 
header t,q0..q5,dq0..dq5 and one row per sample.
"""
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from planning.kinematics import numeric_jacobian
from planning.trajectory import tcp_path_from_Q, generate_trajectory

# from kinematics import fk_ur10, numeric_jacobian
# from trajectory import tcp_path_from_Q, generate_trajectory

# Toggle to True to show TCP position and TCP velocity plots after planning
SAVE_PLOTS = False
SHOW_PLOTS = True
CSV_OUTPUT_PATH = 'log/trajectory_sim.csv'


def generate_trajectory_csv(impact_speed, impact_direction, ball_x_offset, ball_y_offset, path = CSV_OUTPUT_PATH):
    """
    Generate a trajectory via the planner and save it to CSV.
    
    Inputs:
    - impact_speed:     desired TCP speed at impact (m/s)
    - impact_direction: desired TCP direction at impact (3,) (will be normalized inside planner)
    - path:             output CSV file path

    outputs:
    - results: dict from generate_trajectory() containing planned trajectory data or None on failure
    """
    results = generate_trajectory(impact_speed, impact_direction, ball_x_offset=ball_x_offset, ball_y_offset=ball_y_offset)

    if results is None:
        print("[ERROR] Trajectory generation failed")
        return None
    
    # Extract planned trajectory data
    t_plan = results.get('t_plan')
    Q_all  = results.get('Q_plan')
    dQ_all = results.get('dQ_plan')
    
    # Basic validity checks
    if t_plan is None or Q_all is None or dQ_all is None:
        print("[ERROR] Planner did not return a complete trajectory")
        return None
    if len(t_plan) != len(Q_all) or len(t_plan) != len(dQ_all) or len(t_plan) == 0:
        print("[ERROR] Planner returned inconsistent lengths")
        return None
    if not np.isfinite(np.asarray(Q_all)).all() or not np.isfinite(np.asarray(dQ_all)).all():
        print("[ERROR] Planner produced non-finite values")
        return None
    
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True) # ensure directory exists

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)  # create CSV writer
        header = ['t'] + [f'q{i}' for i in range(6)] + [f'dq{i}' for i in range(6)] 
        writer.writerow(header) # write header

        # Write each trajectory sample
        for i in range(len(t_plan)):
            row = [float(t_plan[i])] + [float(v) for v in Q_all[i]] + [float(v) for v in dQ_all[i]]
            writer.writerow(row)

    print(f"[INFO] Saved trajectory CSV to: {path}")

    return results

def plot_joint_positions(t_plan, Q_plan, impact_idx=None):
    """
    Plot joint positions over time.

    Inputs:
    - t_plan:   time vector
    - Q_plan:   joint position trajectory
    """
    fig, axes = plt.subplots(6, 1, figsize=(8, 12))
    axes = axes.flatten()

    for j in range(6):
        ax = axes[j]
        ax.plot(t_plan, Q_plan[:, j], 'b-', label=f'Joint {j} Position', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Joint {j} (rad)')
        ax.grid(True)
        if impact_idx is not None:
            # ax.axvline(float(t_plan[impact_idx]), color='r', linestyle='--', alpha=0.6) # mark impact time

            ax.scatter([float(t_plan[impact_idx])], [Q_plan[impact_idx, j]], color='r', zorder=5)

    joint_pos_file = "log/planned_joint_positions.png"
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(joint_pos_file)
        print(f"[INFO] Saved joint position plot: {joint_pos_file}")

    if SHOW_PLOTS:
        plt.show()

def plot_joint_velocities(t_plan, dQ_plan, impact_idx=None):
    """
    Plot joint velocities over time.

    Inputs:
    - t_plan:   time vector
    - dQ_plan:  joint velocity trajectory
    """
    fig, axes = plt.subplots(6, 1, figsize=(8, 12))
    axes = axes.flatten()

    for j in range(6):
        ax = axes[j]
        ax.plot(t_plan, dQ_plan[:, j], 'b-', label=f'Joint {j} Velocity', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Joint {j} Velocity (rad/s)')
        ax.grid(True)
        if impact_idx is not None:
            # ax.axvline(float(t_plan[impact_idx]), color='r', linestyle='--', alpha=0.6) # mark impact time
            ax.scatter([float(t_plan[impact_idx])], [dQ_plan[impact_idx, j]], color='r', zorder=5)

    joint_vel_file = "log/planned_joint_velocities.png"
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(joint_vel_file)
        print(f"[INFO] Saved joint velocity plot: {joint_vel_file}")

    if SHOW_PLOTS:
        plt.show()

def plot_trajectory(t_plan, P_plan, tcp_vel_jac, speed_jac, impact_idx):
    """
    Plot TCP path and TCP velocities over time.

    Inputs:
    - t_plan:       time vector
    - P_plan:       TCP position trajectory
    - tcp_vel_jac:  TCP velocity trajectory via Jacobian
    - speed_jac:    TCP speed (magnitude of velocity) via Jacobian
    - impact_idx:   index of impact sample (for marking on plots)
    """

    ############ Figure 1: TCP 3D path plot ############
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(P_plan[:, 0], P_plan[:, 1], P_plan[:, 2], 'b-', linewidth=2, label='Planned')

    impact_time = float(t_plan[impact_idx])
    # Mark impact point
    if impact_idx is not None:
        ax1.scatter([P_plan[impact_idx, 0]], [P_plan[impact_idx, 1]], [P_plan[impact_idx, 2]],
                    color='r', marker='*', s=120, label='Impact')
        ax1.text(P_plan[impact_idx, 0], P_plan[impact_idx, 1], P_plan[impact_idx, 2], ' Impact', color='r')

    # Mark start point and end point
    ax1.scatter([P_plan[0, 0]], [P_plan[0, 1]], [P_plan[0, 2]], color='g', marker='o', s=80, label='Start')
    ax1.scatter([P_plan[-1, 0]], [P_plan[-1, 1]], [P_plan[-1, 2]], color='b', marker='o', s=80, label='End')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Planned TCP 3D Path')
    ax1.legend()
    ax1.axis('equal')

    tcp_path_file = f"log/planned_tcp_path.png"
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(tcp_path_file)
        print(f"[INFO] Saved TCP path plot: {tcp_path_file}")

    ############ Figure 2: TCP velocities vx,vy,vz,|v| ############
    fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes       = axes.flatten()
    labels     = ['vx (m/s)', 'vy (m/s)', 'vz (m/s)', '|v| (m/s)']

    for idx_ax in range(4):
        ax = axes[idx_ax]
        
        # Plot velocity for x, y, and z
        if idx_ax < 3:
            # Plot Jacobian-based planned TCP velocity (primary)
            ax.plot(t_plan, tcp_vel_jac[:, idx_ax], 'b-', label='Planned (J@dq)', linewidth=2)
            ax.scatter([impact_time], [tcp_vel_jac[impact_idx, idx_ax]], color='r', zorder=5)

        # Plot absolute velocity |v|
        else:
            ax.plot(t_plan, speed_jac, 'b-', label='Planned |v| (J@dq)', linewidth=2)
            ax.scatter([impact_time], [speed_jac[impact_idx]], color='r', zorder=5)

        ax.axvline(impact_time, color='r', linestyle='--', alpha=0.6) # mark impact time
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(labels[idx_ax])
        ax.grid(True)

    fig2.suptitle('Planned TCP Velocities')

    tcp_vel_file = "log/planned_tcp_velocities.png"
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(tcp_vel_file)
        print(f"[INFO] Saved TCP velocity plot: {tcp_vel_file}")

    if SHOW_PLOTS:
        plt.show()

    ############ Figure 3: TCP accelerations ax,ay,az,|a| ############
    # Compute TCP acceleration by differentiating the Jacobian-based velocity
    try:
        # numpy.gradient handles non-uniform time spacing if we pass t_plan
        tcp_acc_jac = np.gradient(tcp_vel_jac, t_plan, axis=0)
    except Exception:
        # Fallback: simple finite difference (uniform dt assumed)
        dt = np.diff(t_plan)
        dt = np.concatenate(([dt[0] if len(dt)>0 else 1.0], dt))
        tcp_acc_jac = np.zeros_like(tcp_vel_jac)
        for i in range(1, len(t_plan)):
            tcp_acc_jac[i] = (tcp_vel_jac[i] - tcp_vel_jac[i-1]) / dt[i]

    accel_mag = np.linalg.norm(tcp_acc_jac, axis=1)

    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
    axes3 = axes3.flatten()
    labels_a = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', '|a| (m/s^2)']

    for idx_ax in range(4):
        ax = axes3[idx_ax]

        if idx_ax < 3:
            ax.plot(t_plan, tcp_acc_jac[:, idx_ax], 'g-', label='Acceleration (J@ddq)', linewidth=2)
            if impact_idx is not None:
                ax.scatter([t_plan[impact_idx]], [tcp_acc_jac[impact_idx, idx_ax]], color='r', zorder=5)
        else:
            ax.plot(t_plan, accel_mag, 'g-', label='|a| (J@ddq)', linewidth=2)
            if impact_idx is not None:
                ax.scatter([t_plan[impact_idx]], [accel_mag[impact_idx]], color='r', zorder=5)

        if impact_idx is not None:
            ax.axvline(float(t_plan[impact_idx]), color='r', linestyle='--', alpha=0.6)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(labels_a[idx_ax])
        ax.grid(True)

    fig3.suptitle('Planned TCP Accelerations')
    tcp_acc_file = "log/planned_tcp_accelerations.png"
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(tcp_acc_file)
        print(f"[INFO] Saved TCP acceleration plot: {tcp_acc_file}")

    if SHOW_PLOTS:
        plt.show()


def make_angle_reachability_heatmap(
    impact_speed=2.0,
    angle_min_deg=-20.0,
    angle_max_deg=20.0,
    angle_step_deg=1.0,
    radius=0.20,
    grid_step=0.02,
    show_plot=True
):
    """
    Compute and plot a heatmap that indicates whether the robot can hit the ball
    with `impact_speed` m/s for *all* angles in [angle_min_deg, angle_max_deg]
    at each ball placement within `radius` of the start point (0,0).

    - True (1): for this (x,y) offset, every angle in the range yields a valid
      trajectory with impact speed >= impact_speed (within a small tolerance).
    - False (0): at least one angle fails (no trajectory / planner failure
      / too slow / NaNs).
    """

    # Define grid of ball offsets around (0,0)
    xs = np.arange(-radius, radius + 1e-12, grid_step)
    ys = np.arange(-radius, radius + 1e-12, grid_step)
    Nx, Ny = len(xs), len(ys)

    # Boolean heatmap: True = feasible for all angles, False = otherwise
    heat_bool = np.zeros((Nx, Ny), dtype=bool)

    # Precompute angle list
    angles_deg = np.arange(angle_min_deg, angle_max_deg + 1e-9, angle_step_deg)

    # Small tolerance for speed check
    speed_tol = 0.05  # m/s
    count_infeasible = 0
    for ix, x_off in enumerate(xs):
        for iy, y_off in enumerate(ys):
            # Keep only points within circular radius
            if x_off**2 + y_off**2 > radius**2:
                continue  # remains False

            all_ok = True
            
            for ang in angles_deg:
                print(f"Checking offset ({x_off:.2f}, {y_off:.2f}), angle {ang:.1f}°")
                # Call planner directly; it may print warnings/errors if infeasible
                results = generate_trajectory(
                    impact_speed,
                    ang,  # impact angle in degrees (same as in main())
                    ball_x_offset=x_off,
                    ball_y_offset=y_off,
                )

                
                # Any kind of failure or missing data => this angle is not OK
                if not results:
                    all_ok = False
                    break

                problem = results.get("problem", None)
                if problem == "Impact speed not feasible":
                    count_infeasible += 1
                    
                if problem is not None:
                    print(f"problem: {problem}")
                impact_idx = results.get("impact_sample_idx", None)
                Q_all      = results.get("Q_plan", None)
                dQ_all     = results.get("dQ_plan", None)

                # If any of these are missing, treat as failure
                if (
                    impact_idx is None
                    or Q_all is None
                    or dQ_all is None
                    or impact_idx < 0
                    or impact_idx >= len(Q_all)
                    or len(Q_all) != len(dQ_all)
                    or len(Q_all) == 0
                ):
                    all_ok = False
                    break

                q_imp  = Q_all[impact_idx]
                dq_imp = dQ_all[impact_idx]

                # Safety: check finiteness
                if (
                    not np.isfinite(np.asarray(q_imp)).all()
                    or not np.isfinite(np.asarray(dq_imp)).all()
                ):
                    all_ok = False
                    break

                # Compute actual TCP speed at impact using Jacobian
                J = numeric_jacobian(q_imp)
                v_tcp = J[:3, :] @ dq_imp
                speed = float(np.linalg.norm(v_tcp))

                if (not np.isfinite(speed)) or (speed < impact_speed - speed_tol):
                    all_ok = False
                    break

            heat_bool[ix, iy] = all_ok

    # Plot heatmap
    if show_plot:
        plt.figure(figsize=(6, 5))
        heat_float = heat_bool.astype(float)  # 1 = feasible, 0 = not
        extent = [xs[0], xs[-1], ys[0], ys[-1]]
        im = plt.imshow(
            heat_float.T,
            origin="lower",
            extent=extent,
            aspect="equal"
        )
        plt.colorbar(im, label="Feasible for ALL angles? (1=True, 0=False)")
        plt.title(
            f"Reachability heatmap\n"
            f"{impact_speed:.1f} m/s, angles [{angle_min_deg:.0f}, {angle_max_deg:.0f}]°"
        )
        plt.xlabel("Ball X offset [m]")
        plt.ylabel("Ball Y offset [m]")
        plt.scatter([0.0], [0.0], marker="x", color="red", label="Start point")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print(f"Total infeasible cases encountered: {count_infeasible}")
    return xs, ys, heat_bool


############### Added ################# 

def plot_tcp_path_with_waypoints(
    P_plan: np.ndarray,
    t_plan: np.ndarray | None,
    *,
    waypoints_q: list[np.ndarray] | None = None,
    fk_func=None,
    impact_sample_idx: int | None = None,
    waypoint_labels: tuple[str, ...] = ("Start", "Impact", "Post", "End"),
    save_path: str | None = None,
    save_plots: bool = True,
    show: bool = False,
):
    """
    Plot planned TCP 3D path and (optionally) overlay TCP waypoint markers computed via FK.

    Parameters
    ----------
    P_plan : (N,3) array
        Planned TCP positions sampled along the trajectory.
    t_plan : (N,) array or None
        Planned time samples (unused except for convenience; kept for signature symmetry).
    waypoints_q : list of (6,) arrays, optional
        Joint-space waypoints (e.g. [q_start, q_hit, q_post, q_end]).
        If provided, FK is used to compute their TCP positions and plot them.
    fk_func : callable, optional
        Your forward kinematics function. Must satisfy: fk_func(q)[-1][:3,3] -> (3,)
        Example: fk_ur10
        Required if waypoints_q is provided.
    impact_sample_idx : int, optional
        Index into P_plan to mark as the sampled "impact" point.
        (This is separate from the waypoint-based markers.)
    waypoint_labels : tuple of str
        Labels for waypoint markers in order. Extra waypoints get generic labels.
    save_path : str, optional
        If provided and save_plots=True, saves the figure to this path.
    save_plots : bool
        Whether to save if save_path is provided.
    show : bool
        Whether to plt.show() at the end.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes3D
    """
    import matplotlib.pyplot as plt

    P_plan = np.asarray(P_plan, dtype=float)
    if P_plan.ndim != 2 or P_plan.shape[1] != 3:
        raise ValueError(f"P_plan must be (N,3), got {P_plan.shape}")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Main planned path
    ax.plot(P_plan[:, 0], P_plan[:, 1], P_plan[:, 2], linewidth=2, label="Planned")

    # Mark sampled impact point (from the discretized plan)
    if impact_sample_idx is not None:
        i = int(impact_sample_idx)
        if i < 0 or i >= len(P_plan):
            raise IndexError(f"impact_sample_idx={i} out of range for P_plan with N={len(P_plan)}")
        ax.scatter([P_plan[i, 0]], [P_plan[i, 1]], [P_plan[i, 2]],
                   marker="*", s=120, label="Impact (sample)")
        ax.text(P_plan[i, 0], P_plan[i, 1], P_plan[i, 2], " Impact(sample)")

    # Mark start/end samples
    ax.scatter([P_plan[0, 0]], [P_plan[0, 1]], [P_plan[0, 2]],
               marker="o", s=80, label="Start (sample)")
    ax.scatter([P_plan[-1, 0]], [P_plan[-1, 1]], [P_plan[-1, 2]],
               marker="o", s=80, label="End (sample)")

    # Overlay waypoint markers (computed via FK, so they plot even if not on a sample index)
    P_wps = None
    if waypoints_q is not None:
        if fk_func is None:
            raise ValueError("fk_func must be provided when waypoints_q is provided (e.g., fk_ur10).")

        P_wps = np.array([np.asarray(fk_func(q)[-1][:3, 3], dtype=float).reshape(3,) for q in waypoints_q])

        # Marker styles for first few waypoints
        markers = ["o", "*", "^", "s", "D", "P", "X"]
        sizes   = [90, 140, 110, 90, 90, 90, 90]

        for k in range(len(P_wps)):
            label = waypoint_labels[k] if k < len(waypoint_labels) else f"WP{k}"
            m = markers[k] if k < len(markers) else "o"
            s = sizes[k] if k < len(sizes) else 90

            ax.scatter([P_wps[k, 0]], [P_wps[k, 1]], [P_wps[k, 2]],
                       marker=m, s=s, label=f"{label} (wp)")
            ax.text(P_wps[k, 0], P_wps[k, 1], P_wps[k, 2], f" {label}")

        # Optional: dashed segments between waypoints (nice for debugging)
        ax.plot(P_wps[:, 0], P_wps[:, 1], P_wps[:, 2], linestyle="--", linewidth=1.5, label="Waypoints (wp)")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Planned TCP 3D Path")
    ax.legend()
    ax.axis("equal")
    plt.tight_layout()

    if save_plots and save_path:
        plt.savefig(save_path)
        print(f"[INFO] Saved TCP path plot: {save_path}")

    if show:
        plt.show()

    return fig, ax, P_wps

############### Added ################# 

def main():
    csv_out = CSV_OUTPUT_PATH
    impact_speed = 1.5  # m/s
    impact_angle = 4.91  # desired impact angle (degrees)
    ball_x_offset = 0.0  # desired ball x offset
    ball_y_offset = 0.0  # desired ball y offset

    results = generate_trajectory_csv(impact_speed, impact_angle, ball_x_offset, ball_y_offset, csv_out)  

    if results is None:
        return 1

    # Optional TCP plots: 3D path and TCP velocities (vx,vy,vz,|v|)
    if SAVE_PLOTS or SHOW_PLOTS:
        # Extract planned trajectory data
        t_plan     = results.get('t_plan')
        Q_all      = results.get('Q_plan')
        dQ_all     = results.get('dQ_plan')
        impact_idx = results["impact_sample_idx"]


        ################## ADDED ##################

        # results from generate_trajectory(...)
        P_plan = results["P_plan"]
        t_plan = results["t_plan"]
        waypoints_q = results["waypoints"]  # [q_start, q_hit, q_post, q_end] after you add it
        impact_sample_idx = results.get("impact_sample_idx", None)

        plot_tcp_path_with_waypoints(
            P_plan,
            t_plan,
            waypoints_q=waypoints_q,
            fk_func=fk_ur10,
            impact_sample_idx=impact_sample_idx,
            save_path="log/planned_tcp_path.png",
            save_plots=SAVE_PLOTS,
            show=not SAVE_PLOTS,
        )

        ################## ADDED ##################

        plot_joint_positions(t_plan, Q_all, impact_idx)
        plot_joint_velocities(t_plan, dQ_all, impact_idx)
        P_plan = tcp_path_from_Q(Q_all)  # TCP positions from joint trajectory
        
        # Compute TCP velocities via Jacobian
        tcp_vel_jac = np.zeros_like(P_plan)
        
        for i, (q, dq) in enumerate(zip(Q_all, dQ_all)):
            J = numeric_jacobian(q)
            tcp_vel_jac[i] = J[:3, :] @ dq

        speed = np.linalg.norm(tcp_vel_jac, axis=1)
        plot_trajectory(t_plan, P_plan, tcp_vel_jac, speed, impact_idx)

    # xs, ys, heat = make_angle_reachability_heatmap(
    #     impact_speed=1.6,
    #     angle_min_deg=-5.0,
    #     angle_max_deg=5.0,
    #     angle_step_deg=2.0,
    #     radius=0.20,
    #     grid_step=0.05,
    #     show_plot=True,
    # )
    return 0


if __name__ == '__main__':
    sys.exit(main())
