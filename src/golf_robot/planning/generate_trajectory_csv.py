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
from kinematics import numeric_jacobian
from trajectory import tcp_path_from_Q, generate_trajectory

# Toggle to True to show TCP position and TCP velocity plots after planning
SAVE_PLOTS = False
SHOW_PLOTS = False
CSV_OUTPUT_PATH = 'log/trajectory_sim.csv'


def generate_trajectory_csv(impact_speed, impact_direction, path):
    """
    Generate a trajectory via the planner and save it to CSV.
    
    Inputs:
    - impact_speed:     desired TCP speed at impact (m/s)
    - impact_direction: desired TCP direction at impact (3,) (will be normalized inside planner)
    - path:             output CSV file path

    outputs:
    - results: dict from generate_trajectory() containing planned trajectory data or None on failure
    """
    results = generate_trajectory(impact_speed, impact_direction)

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


def main():
    csv_out = CSV_OUTPUT_PATH
    impact_speed = 1.2  # m/s
    impact_angle = 2  # desired impact angle (degrees)

    results = generate_trajectory_csv(impact_speed, impact_angle, csv_out)  

    if results is None:
        return 1

    # Optional TCP plots: 3D path and TCP velocities (vx,vy,vz,|v|)
    if SAVE_PLOTS or SHOW_PLOTS:
        # Extract planned trajectory data
        t_plan     = results.get('t_plan')
        Q_all      = results.get('Q_plan')
        dQ_all     = results.get('dQ_plan')
        impact_idx = results["impact_sample_idx"]

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

    return 0


if __name__ == '__main__':
    sys.exit(main())
