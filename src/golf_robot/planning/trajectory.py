"""
Trajectory generation using quintic polynomials with automatic time-scaling to satisfy joint limits and environment constraints.
"""

import numpy as np
from planning.config import Q_MIN, Q_MAX, DQ_MAX, DDQ_MAX, DT, Z_PALLET
from planning.utils import unwrap_to_seed_all, normalize
from planning.kinematics import numeric_jacobian, fk_ur10, rotation_z, pick_ik_solution, move_point_xyz
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# from config import Q_MIN, Q_MAX, DQ_MAX, DDQ_MAX, DT, Z_PALLET
# from utils import unwrap_to_seed_all, normalize
# from kinematics import numeric_jacobian, fk_ur10, rotation_z, pick_ik_solution, move_point_xyz


def quintic_coeffs(q0, dq0, ddq0, qf, dqf, ddqf, T):
    """
    Compute quintic polynomial coefficients for a joint trajectory (only one joint).

    Inputs:
        q0, dq0, ddq0: initial joint pos, vel, acc (float)
        qf, dqf, ddqf: final joint pos, vel, acc (float)
        T: time duration [s] (float)

    Outputs:
        coeffs: quintic coefficients [a0, a1, a2, a3, a4, a5] (6,)
    """
    # Precompute powers of T (time duration)
    T2 = T*T
    T3 = T**3
    T4 = T**4
    T5 = T**5

    # First three coefficients based on initial joint position, velocity, and acceleration
    a0 = q0 
    a1 = dq0 
    a2 = 0.5*ddq0

    # Construct the matrix M and vector b
    M = np.array([[ T3,    T4,     T5     ], 
                  [ 3*T2,  4*T3,   5*T4   ], 
                  [ 6*T,   12*T2,  20*T3  ]])
    
    b = np.array([ qf -(a0 + a1*T + a2*T2),   
                   dqf -(a1 + 2*a2*T)     , 
                   ddqf -(2*a2)           ])
    
    a3, a4, a5 = np.linalg.solve(M, b) # Solve for coefficients

    return np.array([a0, a1, a2, a3, a4, a5])


def eval_quintic(c, t):
    """
    Evaluate quintic polynomial and its derivatives at time t.

    Inputs:
        c: quintic coefficients [a0, a1, a2, a3, a4, a5] (6,)
        t: time [s] (float)

    Outputs:
        q, dq, ddq: joint position, velocity, and acceleration at time t (float)
    """
    a0, a1, a2, a3, a4, a5 = c  # unpack quintic coefficients

    # Evaluate position, velocity, and acceleration at time t
    q   = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    dq  = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    ddq = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3

    return q, dq, ddq


def auto_time_and_discretize(q0, dq0, ddq0, qf, dqf, ddqf, T_max=8.0,):
    """
    Optimize trajectory time T to satisfy joint limits and environment constraints for given start and end states.

    Inputs:
        q0, dq0, ddq0: initial joint pos, vel, acc (6,)
        qf, dqf, ddqf: final joint pos, vel, acc (6,)
        T_max: maximum allowed time [s]

    Outputs:
        ts: time samples [s] (N,)
        Q: joint positions [rad] (N,6)
        dQ: joint velocities [rad/s] (N,6)
        ddQ: joint accelerations [rad/s^2] (N,6)
        feasible: bool flag indicating if a feasible trajectory was found
    """

    # Boundary conditions and dt from config
    q_min   = Q_MIN
    q_max   = Q_MAX
    dq_max  = DQ_MAX
    ddq_max = DDQ_MAX
    dt      = DT

    T = 1.5             # initial time guess
    growth_gain = 1.2   # time scaling if constraints violated
    
    # Environment constraints (to avoid table, wall, floor)
    x_table = -0.25
    y_table = -0.25
    z_table =  0.2
    y_wall  = -1
    # z_floor = -0.733
    # z_floor = -0.729
    z_floor = -0.729
    z_floor += Z_PALLET

    feasible = False  # Feasibility flag
    problem = "None"  # Last problem encountered

    # Search starts with T_min and scales x1.2 until constraints are met or T_max hit
    while T <= T_max:

        # Compute quintic coefficients for each joint
        coeffs = np.vstack([quintic_coeffs(q0[i], dq0[i], ddq0[i], qf[i], dqf[i], ddqf[i], T) for i in range(6)])

        N   = int(np.floor(T / dt)) + 1  # number of samples
        ts  = np.linspace(0.0, T, N)     # time samples
        Q   = np.zeros((N, 6))           # joint positions
        dQ  = np.zeros_like(Q)           # joint velocities
        ddQ = np.zeros_like(Q)           # joint accelerations

        # Evaluate quintic at each time sample
        for i, t in enumerate(ts):
            for j in range(6):
                Q[i,j], dQ[i,j], ddQ[i,j] = eval_quintic(coeffs[j], t)

        # Check constraints
        if np.any(Q < (q_min.reshape(1, -1))):
            T *= growth_gain
            problem = "Min q violated"
            continue
        if np.any(Q > (q_max.reshape(1, -1))):
            T *= growth_gain
            problem = "Max q violated"
            continue
        if np.any(np.abs(dQ) > (dq_max.reshape(1, -1))):
            T *= growth_gain
            problem = "Max dq violated"
            continue
        if np.any(np.abs(ddQ) > (ddq_max.reshape(1, -1))):
            T *= growth_gain
            problem = "Max ddq violated"
            continue

        ok = True

        # Forward kinematics for each time step and check for collisions
        for i in range(N):
            xyz_joints = fk_ur10(Q[i])   # 9 pose matrices (4x4): joints: 0-5, and club: short, mid, long

            for j in range(2, len(xyz_joints)):  # iterate from joint 2 to end effector
                x, y, z = xyz_joints[j][:3, 3]   # extract position for current joint/end-effector

                # Check crash against table, floor, and wall
                if (x > x_table) and (y > y_table) and (z < z_table):
                    problem = f"Table crash, x={x}, y={y}, z={z}"
                    ok = False
                    break

                if (z < z_floor):
                    problem = f"Floor crash, z={z}"
                    ok = False
                    break

                if (y < y_wall):
                    problem = f"Wall crash, y={y}"
                    ok = False
                    break
            
            # Break outer loop if collision detected
            if not ok:
                break
        
        # If any collision detected, increase T and retry
        if not ok:
            # T *= growth_gain
            # continue
            return None, None, None, None, False

        feasible = True
        return ts, Q, dQ, ddQ, feasible

    print(f"[WARNING]: No feasible path found. Last problem: {problem}")
    return None, None, None, None, feasible


def joint_vel_from_tcp(q, v_lin, damping=1e-4):
    """
    Compute joint velocities dq that achieve a desired TCP velocity using
    a damped least-squares inverse of the Jacobian.

    Inputs:
        q: Joint configuration [rad] (6,)
        v_lin: Desired linear TCP velocity [m/s] (3,)
        damping (float): Regularization parameter to improve stability

    Returns:
        dq: Joint velocities [rad/s] (6,)
        v_tcp: Achieved TCP velocity vector [m/s] (3,)
        feasible (bool): True if within joint limits
    """
    dq_max = DQ_MAX                              # joint velocity limits from config
    J      = numeric_jacobian(q)                 # 6×6 geometric Jacobian
    J_lin  = J[:3, :]                            # linear velocity block
    target = np.asarray(v_lin, float).reshape(3) # desired TCP linear velocity

    # Damped least squares inverse: dq = (JᵀJ + λ²I)⁻¹ Jᵀ v
    H  = J_lin.T @ J_lin + damping * np.eye(6)
    dq = np.linalg.solve(H, J_lin.T @ target)

    # Check feasibility and compute achieved TCP velocity
    feasible = np.all(np.abs(dq) <= dq_max)
    v_tcp    = J_lin @ dq

    return dq, v_tcp, feasible


def plan_piecewise_quintic(waypoints, impact_idx, lin_velocity):
    """
    Simplified wrapper around auto_time_and_discretize.
    Inputs:
        waypoints: list of joint-space waypoints [rad] (list of (6,) arrays)
        impact: dict with impact parameters (or None)
    Returns: (segments, Q_all, dQ_all, ddQ_all)
    """
    segments    = []
    prev_end_dq = np.zeros(6)

    for seg in range(len(waypoints) - 1):
        # Initial conditions for this segment
        q0    = waypoints[seg]
        dq0   = prev_end_dq if seg > 0 else np.zeros(6)   
        ddq0  = np.zeros(6)

        # Final conditions for this segment
        qf    = unwrap_to_seed_all(waypoints[seg + 1], q0)
        # dqf dependent on whether this is the impact segment
        ddqf  = np.zeros(6)

        # If this segment ends at the impact waypoint, impose TCP velocity there
        if (seg + 1) == impact_idx:  
            dqf, v_tcp, feasible = joint_vel_from_tcp(qf, lin_velocity)

            # Check feasibility and warn if not feasible
            if not feasible:
                print("[WARNING] Impact joint velocity is not feasible within joint limits.")
                print(f"  Desired TCP velocity: {lin_velocity}, possible: {v_tcp}")
                return None, None, None, None
        else:
            dqf = np.zeros(6)

        # print(f"ddq0: {ddq0}, ddqf: {ddqf}")
        # Given start and end conditions, plan the segment with time-scaling
        ts, Q, DQ, DDQ, feasible = auto_time_and_discretize(q0, dq0, ddq0, qf, dqf, ddqf, T_max=5.0)

        if not feasible:
            print(f"[ERROR] Could not find feasible trajectory for segment {seg} to {seg+1}.")
            return None, None, None, None
        
        segments.append({'ts': ts, 'Q': Q, 'DQ': DQ, 'DDQ': DDQ})
        prev_end_dq = DQ[-1].copy()

    # Start with the first segment (include all points)
    Q_blocks   = [segments[0]['Q']]
    dQ_blocks  = [segments[0]['DQ']]
    ddQ_blocks = [segments[0]['DDQ']]

    # Append remaining segments, skipping duplicate first row
    for seg in segments[1:]:
        Q_blocks.append(seg['Q'][1:])
        dQ_blocks.append(seg['DQ'][1:])
        ddQ_blocks.append(seg['DDQ'][1:])


    # Stack into continuous trajectories
    Q_all   = np.vstack(Q_blocks)
    dQ_all  = np.vstack(dQ_blocks)
    ddQ_all = np.vstack(ddQ_blocks)

    return segments, Q_all, dQ_all, ddQ_all


def max_speed_at_pose(q, v_dir):
    """
    Compute maximum feasible TCP speed in a given direction at joint configuration q.

    Inputs:
        q: Joint configuration at impact [rad] (6,)
        v_dir: Desired TCP direction (unit vector) (3,)

    Outputs:
        v_mag_max: Maximum feasible TCP speed [m/s] (float)
    """
    joint_vel_max = DQ_MAX               # joint velocity limits from config
    eps           = 1e-12                # small threshold to avoid divide-by-zero
    J             = numeric_jacobian(q)  # 6×6 geometric Jacobian
    J_lin         = J[:3, :]             # linear velocity block

    dq_dir = np.linalg.pinv(J_lin) @ v_dir      # joint velocities for 1 m/s TCP in v_dir
    idx    = np.where(np.abs(dq_dir) > eps)[0]  # joint indices that contribute to this direction

    # If no joints contribute, direction is unreachable
    if idx.size == 0:
        v_mag_max = 0.0
        dq_at_limit = np.zeros(6)

    else:
        # Scale so the first joint to hit its limit defines the maximum
        per_joint_scale = joint_vel_max[idx] / np.abs(dq_dir[idx]) 
        scaling         = np.min(per_joint_scale)

        dq_at_limit = dq_dir * scaling
        v_tcp_max   = J_lin @ dq_at_limit
        v_mag_max   = np.linalg.norm(v_tcp_max)

    return v_mag_max


def orth(b):
    # quick orthonormalize: make x = z × y
    z, y = b[:,2], b[:,1]
    x = np.cross(z, y)
    x /= np.linalg.norm(x)
    y = np.cross(x, z)
    return np.column_stack([x, y, z])

def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

def impact_joint_config_from_direction(q_hit_ref, impact_dir_xy, ball_center=[0.0, 0.0, 0.0]):

    # Rotation matrix (original TCP) points in x->+Y, y->-Z, z->-X.
    # We want to rotate the TCP rotation matrix so x->+X, y->+Y in the base frame.

    T_ref = fk_ur10(q_hit_ref)[-1]
    p_ref = T_ref[:3, 3]
    R_ref = T_ref[:3, :3]

    tcp_to_world = np.array([[  0,  1,  0],
                             [  0,  0,  -1],
                             [ -1,  0,  0]])
    
    R_ref = R_ref @ tcp_to_world
    v     = impact_dir_xy.copy()

    if v.size == 2:
        v = np.array([v[0], v[1], 0.0])
    
    v[2] = 0.0                
    n    = np.linalg.norm(v)

    if n < 1e-12:
        print("[ERROR] Impact direction has zero xy component.")
        return None
    
    v         = v / n  # normalize
    theta_tar = np.arctan2(v[1], v[0])                # target angle in xy-plane
    theta_ref = np.arctan2(R_ref[1, 0], R_ref[0, 0])  # Reference zero angle
    theta_ref += np.deg2rad(0.0)  # debug offset
    dtheta    = wrap_pi(theta_tar - theta_ref)                 # angle difference

    R_des = rotation_z(dtheta) @ R_ref  # desired rotation matrix in world frame

    r0_xy = p_ref[:2] - ball_center[:2]  # vector from ball center to TCP in xy-plane
    
    Rz2   = np.array([[ np.cos(dtheta), -np.sin(dtheta)],
                      [ np.sin(dtheta),  np.cos(dtheta)]]) # rotation in xy-plane

    p_hit_xy = ball_center[:2] + Rz2 @ r0_xy  # desired TCP position in xy-plane
    p_des   = np.array([p_hit_xy[0], p_hit_xy[1], p_ref[2]])  # desired TCP position (keep z from ref)

    T_des = np.eye(4)                       # initialize desired TCP transform
    T_des[:3, :3] = R_des @ tcp_to_world.T  # back to original TCP frame
    T_des[:3, 3]  = p_des                   # set desired TCP position

    Q, errs = pick_ik_solution(T_des, q_hit_ref)

    # print(Q)
    # print(q_hit_ref)
    if errs == 1 or Q is None:
        print("[WARNING] No IK solution found for desired impact pose.")
        return None
    
    if np.any(Q < Q_MIN) or np.any(Q > Q_MAX):
        print("[WARNING] IK solution is out of joint limits.")
        print(f"Required joint angles: {Q}")
        print(f"Joint limits: {Q_MIN} to {Q_MAX}")
        return None

    return Q


def score_trajectory(Q, DQ, DDQ, dt, dq_max, ddq_max):
    """
    Mixed smoothness/effort score described above.
    Returns a scalar score (lower is better).
    """
    N = len(Q)
    # Precompute J at each sample
    Js = [numeric_jacobian(Q[i]) for i in range(N)]  # each 6x6
    # Approximate Jdot*dq via central differences
    Jdot_dq = [np.zeros(3) for _ in range(N)]
    for i in range(1, N-1):
        Jp = Js[i+1][:3, :]
        Jm = Js[i-1][:3, :]
        Jdot_dq[i] = ((Jp - Jm) / (2*dt)) @ DQ[i]   # (3,)

    Wdq  = np.diag(1.0 / np.asarray(dq_max,  float))
    Wddq = np.diag(1.0 / np.asarray(ddq_max, float))

    alpha, beta, gamma = 0.3, 0.3, 0.05

    J_total = 0.0
    for i in range(N):
        ddq_n = Wddq @ DDQ[i]
        dq_n  = Wdq  @ DQ[i]

        a_tcp = Js[i][:3, :] @ DDQ[i] + Jdot_dq[i]              # (3,)
        tau_proxy = Js[i][:3, :].T @ a_tcp                      # (6,)

        term_j   = float(ddq_n @ ddq_n)
        term_tcp = float(a_tcp @ a_tcp)
        term_tau = float(tau_proxy @ tau_proxy)
        term_v   = float(dq_n @ dq_n)

        J_total += dt * (term_j + alpha*term_tcp + beta*term_tau + gamma*term_v)

    return J_total


def gaussian_windows(N, center_idx, half_width_s, dt, sigma_scale=0.6):
    half_w = max(1, int(round(half_width_s / dt)))
    i = np.arange(N)
    sigma = max(1e-9, sigma_scale * half_w)
    w = np.exp(-0.5 * ((i - center_idx) / sigma)**2)

    w[(i < center_idx - 3*half_w) | (i > center_idx + 3*half_w)] = 0.0
    return w


def score_abs_joint_accel(Q, DQ, DDQ, dt, impact_idx, ddq_max, window_half_width_s=0.05):
    """
    Score based on absolute joint accelerations, with Gaussian windowing around impact.

    Inputs:
        Q, DQ, DDQ: joint positions, velocities, accelerations (N,6)
        dt: time step [s] (float)
        impact_idx: index of impact sample (int)
        ddq_max: joint acceleration limits (6,)
        window_half_width_s: half-width of Gaussian window [s] (float)
        lambda_T: weight for time penalty (float)
        T: total trajectory time [s] (float)
    """
    N = len(DDQ)
    w = gaussian_windows(N, impact_idx, window_half_width_s, dt)  # (N,)

    l1 = np.sum(np.abs(DDQ) / ddq_max.reshape(1, -1), axis=1)  # (N,
    
    num = np.sum(w * l1)  # scalar
    denom = np.sum(w) + 1e-9  # scalar to avoid divide

    J = num / denom
    return J



def generate_trajectory(impact_speed, impact_angle, ball_x_offset, ball_y_offset):
    """
    Generate a joint-space trajectory for the UR10 robot to achieve a specified
    TCP impact speed and direction at a designated waypoint.

    Inputs:
        impact_speed: Desired TCP speed at impact (m/s)
        impact_angle: Desired TCP direction angle at impact (0 degrees = +X, 90 degrees = +Y) (degrees)
        
    Outputs:
        results: Dictionary containing planned trajectory data:
            - 't_plan': time samples (N,)
            - 'P_plan': TCP positions (N, 3)
            - 'Q_plan': joint positions (N, 6)
            - 'dQ_plan': joint velocities (N, 6)
            - 'ddQ_plan': joint accelerations (N, 6)
            - 'segments': list of trajectory segments with details
            - 'waypoints': list of joint waypoints used
            - 'impact_sample_idx': index of impact sample in trajectory
            - 'tcp_vel_impact': actual TCP velocity vector at impact (3,)
            - 'achieved_impact_speed': magnitude of TCP velocity at impact (float)
    """
    # defaults
    dt = DT

    # Hard-coded waypoints for golf swing
    # q_start = np.array([np.deg2rad(-144.36), np.deg2rad(-162.82), np.deg2rad(-51.80),  np.deg2rad(66.92), np.deg2rad(42.60), np.deg2rad(6.51)])
    # q_exit  = np.array([np.deg2rad(-87.65),  np.deg2rad(-135.05), np.deg2rad(-108.29), np.deg2rad(85.72), np.deg2rad(4.39),  np.deg2rad(-12.23)])
    # q0_hit   = np.array([np.deg2rad(-127.23), np.deg2rad(-153.93), np.deg2rad(-100.71), np.deg2rad(74.59), np.deg2rad(35.18), np.deg2rad(28.72)])
    q0_start = np.array([np.deg2rad(-144.36), np.deg2rad(-162.82), np.deg2rad(-51.80),  np.deg2rad(66.92), np.deg2rad(42.60), np.deg2rad(6.51)])
    q0_end  = np.array([np.deg2rad(-87.65),  np.deg2rad(-135.05), np.deg2rad(-108.29), np.deg2rad(85.72), np.deg2rad(4.39),  np.deg2rad(-12.23)])
    
    # q0_hit  = np.array([-2.18480298, -2.68658532, -1.75772109,  1.30184109,  0.61400683,  0.50125856])  # reference impact joint config (+X direction)
    q0_hit  = np.array([-2.11202641, -2.45037247, -1.67584054,  0.95906874,  0.53322783,  0.36131151])  # From 8th Jan
    # q0_hit    = np.array([-2.24881973, -2.4917556,  -1.57452762,  0.90943969,  1.01899601,  0.34730125])
    q0_hit  = move_point_xyz(ball_x_offset, ball_y_offset, 0.01, q0_hit, q0_hit)[0]  # unwrap to near reference
    # possible_start_points = []
    # possible_end_points   = []
    # for x in range(5):
    #     for y in range(5):
    #         for z in range(5):
    #             possible_start_points.append(move_point_xyz(-0.6 + 0.05*x, -0.1 + 0.05*y, 0.15 + 0.03*z, q0_hit, q0_start)[0])
    #             possible_end_points.append(move_point_xyz(0.4 + 0.05*x, -0.1 + 0.05*y, 0.15 + 0.03*z, q0_hit, q0_end)[0])
        
    possible_start_points = [move_point_xyz(-0.4, 0.0, 0.25, q0_hit, q0_start)[0]
    ]

    possible_end_points = [move_point_xyz(0.5, 0.0, 0.25, q0_hit, q0_end)[0]
                        #    move_point_xyz(0.5, 0.1, 0.25, q0_hit, q0_end)[0],
                        #    move_point_xyz(0.4, -0.1, 0.25, q0_hit, q0_end)[0],
                        #    move_point_xyz(0.4, 0.0, 0.30, q0_hit, q0_end)[0],
                        #    move_point_xyz(0.4, 0.0, 0.20, q0_hit, q0_end)[0]
    ]

    # q_start, _ = move_point_xyz(-0.4, 0.0, 0.25, q0_hit, q0_start) # 60 cm behind and 25 cm above ball
    # q_end, _   = move_point_xyz( 0.5, 0.0, 0.25, q0_hit, q0_end) # 60 cm in front and 25 cm above ball

    # q_start = np.array([-2.35718127, -2.84501045, -0.88769778,  1.27293301,  0.61129035, -0.03746998]) # 60 cm behind and 25 cm above ball
    # q_exit = np.array([-0.74857402, -2.44563742, -1.97906305,  0.87441754, -1.14358243,  0.7267085]) # 60 cm in front and 25 cm above ball
    # q0_hit  = np.array([-2.18480298, -2.68658532, -1.75772109,  1.30184109,  0.61400683,  0.50125856])  # reference impact joint config (+X direction)

    # Compute impact joint configuration from desired direction
    impact_direction = np.array([np.cos(np.deg2rad(impact_angle)), np.sin(np.deg2rad(impact_angle)), 0.0])
    
    ball_center = fk_ur10(q0_hit)[-1][:3, 3] + np.array([0.02133, 0.0, 0.0])  # TCP position at zero angle + offset
    q_hit = impact_joint_config_from_direction(q0_hit, impact_direction, ball_center=ball_center)
    q_hit = move_point_xyz(0.0, 0.0, 0.0, q_hit, q_hit)[0]  # unwrap to near reference
    # waypoints = [q_start, q_hit, q_end]
    # print("Q_HIT:")
    # print(q_hit)
    v_dir_base   = normalize(impact_direction)  # unit vector of desired direction
    lin_velocity = v_dir_base * impact_speed    # desired TCP linear velocity at impact
    impact_idx   = 1                            # index of impact waypoint (second in a 3-waypoint plan)

    # Check feasibility at impact
    max_speed = max_speed_at_pose(q_hit, v_dir_base)

    if max_speed < impact_speed:
        print("[WARNING] Desired impact speed is not kinematically feasible at impact pose.")
        print(f"  Desired TCP speed: {impact_speed:.2f} m/s, possible: {max_speed:.2f} m/s")
        print(f"  Impact direction (base frame): {v_dir_base}")
        results = {"t_plan": None,
                   "P_plan": None,
                   "Q_plan": None,
                   "dQ_plan": None,
                   "ddQ_plan": None,
                   "segments": None,
                   "waypoints": None,   
                   "problem": "Impact speed not feasible"}
        return results

    best_segment = None
    best_Q_all   = None
    best_dQ_all  = None
    best_ddQ_all = None
    found_any    = False

    for q_start in possible_start_points:
        for q_end in possible_end_points:
            waypoints = [q_start, q_hit, q_end]
            segments_i, Q_all_i, dQ_all_i, ddQ_all_i = plan_piecewise_quintic(
                waypoints, impact_idx, lin_velocity
            )

            if Q_all_i is not None:
                # First feasible configuration found – stop searching
                best_segment = segments_i
                best_Q_all   = Q_all_i
                best_dQ_all  = dQ_all_i
                best_ddQ_all = ddQ_all_i
                waypoints_best = waypoints
                found_any = True
                break  # break inner loop

        if found_any:
            break  # break outer loop

    if not found_any:
        print("[ERROR] Trajectory planning failed.")
        results = {
            "t_plan": None,
            "P_plan": None,
            "Q_plan": None,
            "dQ_plan": None,
            "ddQ_plan": None,
            "segments": None,
            "waypoints": None,
            "problem": "Trajectory planning failed for all start/end candidates.",
        }
        return results

    # Use the first feasible trajectory we found
    segments = best_segment
    Q_all    = best_Q_all
    dQ_all   = best_dQ_all
    ddQ_all  = best_ddQ_all
    waypoints = waypoints_best

    # Impact is feasible; proceed with planning
    # segments, Q_all, dQ_all, ddQ_all = plan_piecewise_quintic(waypoints, impact_idx, lin_velocity)

    if Q_all is None:
        print("[ERROR] Trajectory planning failed.")
        results = {"t_plan": None,
                   "P_plan": None,
                   "Q_plan": None,
                   "dQ_plan": None,
                   "ddQ_plan": None,
                   "segments": None,
                   "waypoints": None,   
                   "problem": "Trajectory planning failed for waypoints. Try new waypoints."}
        return results

    t_plan = np.arange(len(Q_all)) * dt  # uniform time steps
    P_plan = tcp_path_from_Q(Q_all)      # TCP positions throughout trajectory

    # Wrap the results in a dictionary for return
    results = {
        't_plan': t_plan,
        'P_plan': P_plan,
        'Q_plan': Q_all,
        'dQ_plan': dQ_all,
        'ddQ_plan': ddQ_all,
        'segments': segments,
        'waypoints': waypoints
    }

    # Compute actual TCP velocity at impact for verification
    impact_sample_idx = len(segments[0]['Q']) - 1
    q_impact          = Q_all[impact_sample_idx]
    dq_impact         = dQ_all[impact_sample_idx]
    J                 = numeric_jacobian(q_impact)
    tcp_vel_impact    = J[:3, :] @ dq_impact

    results['impact_sample_idx']     = impact_sample_idx
    results['tcp_vel_impact']        = tcp_vel_impact
    results['achieved_impact_speed'] = float(np.linalg.norm(tcp_vel_impact))

    # Print joint limits used
    # print("[INFO] Joint limits used for planning:")
    # print(f"  - Max joint velocity (rad/s): {np.round(DQ_MAX, 3).tolist()}")
    # print(f"  - Max joint acceleration (rad/s^2): {np.round(DDQ_MAX, 3).tolist()}\n")

    # Report impact verification
    print("[IMPACT VERIFICATION]")
    print(f"  - Requested impact speed: {impact_speed:.4f} m/s")
    print(f"  - Achieved impact speed:  {results['achieved_impact_speed']:.4f} m/s")
    print(f"  - Achievement ratio:      {results['achieved_impact_speed']/impact_speed*100:.1f}%")
    print(f"  - Impact joint velocities (rad/s):                  {np.round(dq_impact, 3).tolist()}")
    print(f"  - Max joint velocity throughout trajectory (rad/s): {np.round(np.max(np.abs(dQ_all), axis=0), 3).tolist()}\n")
          
    return results


def tcp_path_from_Q(Q: np.ndarray) -> np.ndarray:
    """
    Compute TCP XYZ positions for every row in Q (N,6) -> (N,3).
    Input:
        Q: joint positions for entire trajectory [rad] (N,6)
    Output:
        P: TCP positions in xyz throughout trajectory [m] (N,3)
    """
    return np.array([fk_ur10(q)[-1][:3, 3] for q in Q])


if __name__ == '__main__':
    # Test impact_joint_config_from_direction function
    q0_hit = np.array([-2.18480298, -2.68658532, -1.75772109,  1.30184109,  0.61400683,  0.50125856])

    angles = np.linspace(-90, 90, 51)
    tcp_pos = []
    tcp_seed = np.array(fk_ur10(q0_hit)[-1][:3, 3])
    ball_center = tcp_seed + np.array([0.02133, 0.0, 0.0])

    for angle in angles:
        v_dir = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0.0])
        q_new = impact_joint_config_from_direction(q0_hit, v_dir, ball_center=ball_center)
        if q_new is None:
            continue

        tcp_pos.append(fk_ur10(q_new)[-1][:3, 3])

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    tcp_pos = np.array(tcp_pos)
    ax.plot(tcp_pos[:,0], tcp_pos[:,1], tcp_pos[:,2], marker='o')
    ax.scatter(tcp_seed[0], tcp_seed[1], tcp_seed[2], color='r', s=100, label='Seed TCP Position')
    ax.scatter(ball_center[0], ball_center[1], ball_center[2], color='g', s=100, label='Ball Center')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('TCP Positions for Varying Impact Directions')
    ax.axis('equal')
    plt.show()
