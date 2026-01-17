"""
Kinematics for the UR10 robot (with a mounted golf club).
"""

import numpy as np

try:
    from planning.config import A, D, Z_CLUB, Q_MIN, Q_MAX, X_CLUB_LONG, Y_CLUB_LONG, X_CLUB_SHORT, Y_CLUB_SHORT, CLUB_MID
    from planning.utils import rotm_from_rpy, T_from_xyz_rpy, unwrap_to_seed
except ImportError:
    from config import A, D, Z_CLUB, Q_MIN, Q_MAX, X_CLUB_LONG, Y_CLUB_LONG, X_CLUB_SHORT, Y_CLUB_SHORT, CLUB_MID
    from utils import rotm_from_rpy, T_from_xyz_rpy, unwrap_to_seed

def fk_ur10(q):
    """
    Forward kinematics for the UR10 robot.

    Input:
        q: Joint angles [rad] (6,)

    Output:
        List of 9 transformation matrices, 6 for each joint frame, 
        and 3 for the tool (golf club head) (9x4x4).

    A and D are the Denavit-Hartenberg parameters defined in config.py
    """
    c, s = np.cos, np.sin                # shorthand cosine and sine functions
    q1, q2, q3, q4, q5, q6 = q.tolist()  # unpack joint angles

    # Define transformation matrices between consecutive joints
    T12 = np.array([[ c(q1),  0,  s(q1),  0         ],
                    [ s(q1),  0, -c(q1),  0         ],
                    [     0,  1,     0 ,  D[0]      ],
                    [     0,  0,     0 ,  1         ]])
    
    T23 = np.array([[ c(q2), -s(q2),  0,  A[1]*c(q2)],
                    [ s(q2),  c(q2),  0,  A[1]*s(q2)],
                    [     0,      0,  1,  0         ],
                    [     0,      0,  0,  1         ]])
    
    T34 = np.array([[ c(q3), -s(q3),  0,  A[2]*c(q3)],
                    [ s(q3),  c(q3),  0,  A[2]*s(q3)],
                    [     0,      0,  1,  0         ],
                    [     0,      0,  0,  1         ]])
    
    T45 = np.array([[ c(q4),  0,  s(q4),  0         ],
                    [ s(q4),  0, -c(q4),  0         ],
                    [     0,  1,      0,  D[3]      ],
                    [     0,  0,      0,  1         ]])
    
    T56 = np.array([[ c(q5),  0, -s(q5),  0         ],
                    [ s(q5),  0,  c(q5),  0         ],
                    [     0, -1,      0,  D[4]      ],
                    [     0,  0,      0,  1         ]])

    T67 = np.array([[ c(q6), -s(q6),  0,  0         ],
                    [ s(q6),  c(q6),  0,  0         ],
                    [     0,      0,  1,  D[5]      ],
                    [     0,      0,  0,  1         ]])


    # Transformation matrices for tool (golf club head) positions from joint 6
    T7_short = np.array([[ 1,  0,  0,  X_CLUB_SHORT ],
                         [ 0,  1,  0,  Y_CLUB_SHORT ],
                         [ 0,  0,  1,  Z_CLUB       ],
                         [ 0,  0,  0,  1            ]])
    
    T7_long  = np.array([[ 1,  0,  0,  X_CLUB_LONG  ],
                         [ 0,  1,  0,  Y_CLUB_LONG  ],
                         [ 0,  0,  1,  Z_CLUB       ],
                         [ 0,  0,  0,  1            ]])

    T7_mid   = np.array([[ 1,  0,  0,  (X_CLUB_LONG+X_CLUB_SHORT)/2],
                         [ 0,  1,  0,  (Y_CLUB_LONG+Y_CLUB_SHORT)/2],
                         [ 0,  0,  1,  Z_CLUB                      ],
                         [ 0,  0,  0,  1                           ]])     

    # Compute cumulative transformation matrices
    T01 = np.eye(4) 
    T02 = T01 @ T12 
    T03 = T02 @ T23
    T04 = T03 @ T34 
    T05 = T04 @ T45 
    T06 = T05 @ T56 
    T07 = T06 @ T67

    # Transformation matrices for tool (golf club head) positions from last joint
    T0_SHORT = T07 @ T7_short
    T0_LONG  = T07 @ T7_long
    T0_MID   = T07 @ T7_mid

    return [T01,T02,T03,T04,T05,T06,T07,T0_SHORT,T0_LONG,T0_MID]


def rpy_from_R_zyx(R):
    """
    Convert rotation matrix -> roll, pitch, yaw using ZYX convention:
      R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Returns (roll, pitch, yaw) in radians.
    """
    R = np.asarray(R, float)

    # pitch = asin(-R[2,0]) in the standard ZYX extraction
    sp = -R[2, 0]
    sp = np.clip(sp, -1.0, 1.0)
    pitch = np.arcsin(sp)

    # handle gimbal lock near +-90deg pitch
    cp = np.cos(pitch)
    if abs(cp) < 1e-8:
        # Gimbal lock: yaw and roll are coupled. Choose roll=0 and solve yaw from R[0,1], R[1,1]
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw  = np.arctan2(R[1, 0], R[0, 0])

    return roll, pitch, yaw


def tcp_rpy_from_Q(Q):
    """Compute TCP roll/pitch/yaw for each q in Q -> array (N,3)"""
    rpy = np.zeros((len(Q), 3), dtype=float)
    for i, q in enumerate(Q):
        _, R = tcp_pose_from_q(q)
        rpy[i, :] = rpy_from_R_zyx(R)
    return rpy


def ur10_inv_kin(T):
    """
    Semi-analytic UR10 IK with tool-z offset.
    Returns up to 8 branches (2 base, 2 wrist flips, 2 elbows).

    Inputs:
        T:    TCP pose in base frame (4x4)

    Outputs:
        Q:    Joint solutions [rad] (Nx6), N is number of solutions (0..8).
        errs: 0 if within reach, 1 if shoulder-wrist distance > |a2|+|a3|.
    """
    EPS = 1e-12

    # Remove the tool translation defined in frame 6 (same orientation as R_mid)
    T_adj = np.array(T, dtype=float)
    T_adj[:3, 3] = T[:3, 3] - T[:3, :3] @ CLUB_MID

    a = A.copy()
    d = D.copy()

    # --- Joint 1 (base yaw) from xy geometry with wrist offset d5 along TCP z ---
    a1 = -T_adj[1, 3] + T_adj[1, 2] * d[5]
    b1 =  T_adj[0, 3] - T_adj[0, 2] * d[5]
    disc = a1*a1 + b1*b1 - d[3]*d[3]
    if disc < -1e-12:
        return np.empty((0, 6)), np.empty((0,), int)

    k  = np.arctan2(np.sqrt(max(disc, 0.0)), d[3])
    th = np.arctan2(b1, a1)
    v1_candidates = np.array([th + k, th - k])

    sols, errs = [], []

    z_t   = T_adj[:3, 2]            # TCP z-axis in base frame
    p_tcp = T_adj[:3, 3]            # TCP position

    for v1 in v1_candidates:
        # n1: direction orthogonal to base radius at angle v1 (shoulder circle tangent)
        n1 = np.array([np.sin(v1), -np.cos(v1), 0.0])

        # Wrist axis candidate from cross(n1, z_t); try both flips ±n5
        n5 = np.cross(n1, z_t)
        n5 = n5 / np.linalg.norm(n5) if np.linalg.norm(n5) > 1e-6 else np.array([0.0, 0.0, 1.0])

        for flip in (1.0, -1.0):
            n5f = n5 * flip

            # Key points
            p3 = p_tcp - d[5] * z_t + d[4] * n5f          # wrist center
            p1 = d[3] * n1 + np.array([0.0, 0.0, d[0]])   # shoulder
            l  = p3 - p1
            L  = np.linalg.norm(l)
            lhat = l / (L if L > EPS else EPS)

            # Reach feasibility (triangle inequality on upper/lower arm)
            A2, A3 = abs(a[1]), abs(a[2])
            reach_err = int(L > A2 + A3 + 1e-12)

            # Elbow angle v3 from law of cosines
            cos_v3 = np.clip((A2*A2 + A3*A3 - L*L) / (2.0 * A2 * A3), -1.0, 1.0)
            v3_abs = np.pi - np.arccos(cos_v3)

            # Shoulder pitch v2: elevation + triangle split
            proj   = np.array([-np.cos(v1), -np.sin(v1), 0.0])   # radial dir. in base xy
            phi    = np.arctan2(lhat[2], proj @ lhat)
            cos_t  = np.clip((A2*A2 - A3*A3 + L*L) / (2.0 * A2 * max(L, EPS)), -1.0, 1.0)

            for elbow in (1, -1):  # +: elbow-down, -: elbow-up (convention-dependent)
                v3 =  v3_abs if elbow == 1 else -v3_abs
                v2 = -np.arccos(cos_t) - phi if elbow == 1 else  np.arccos(cos_t) - phi

                # Wrist pitch v4 to align forearm to n5f
                v4 = np.arctan2(-n5f[2], n5f @ proj) - v3 - v2 + np.pi / 2.0

                # Wrist yaw v5 to align TCP z with (n1, z_t) plane
                v5 = np.arctan2(np.dot(np.cross(-n5f, n1), z_t),
                                float(np.dot(n1, z_t)))

                # Tool roll v6 about TCP z to align x/y
                v6 = np.arctan2(float(n5f @ T_adj[:3, 0]),
                                float(n5f @ T_adj[:3, 1]))

                sols.append(np.array([v1, v2, v3, v4, v5, v6]))
                errs.append(reach_err)

    if not sols:
        return np.empty((0, 6)), np.empty((0,), int)

    return np.vstack(sols), np.array(errs, dtype=int)


def tcp_pose_from_q(q):
    """
    Returns (p, R) where:
      p is (3,) TCP position
      R is (3,3) rotation matrix of TCP
    """
    T = fk_ur10(np.asarray(q, float))
    T_tcp = T[-1] if isinstance(T, (list, tuple)) else T
    T_tcp = np.asarray(T_tcp, float)

    if T_tcp.shape == (4, 4):
        p = T_tcp[:3, 3].copy()
        R = T_tcp[:3, :3].copy()
    elif T_tcp.shape == (3, 4):
        p = T_tcp[:3, 3].copy()
        R = T_tcp[:3, :3].copy()
    else:
        raise TypeError(f"Unexpected transform shape: {T_tcp.shape}")

    return p, R




def pick_ik_solution(T, q_seed):
    """
    Pick the best IK solution for the UR10 given a seed configuration.

    Inputs:
        T:      Desired TCP pose in base frame (4x4)
        q_seed: Seed joint configuration [rad] (6,)

    Outputs:
        q:      Chosen joint configuration [rad] (6,)
        err:    IK error code (0 if within reach, 1 if shoulder-wrist distance > |a2|+|a3|).
                None if no valid solution within joint limits.
    """
    q_min = Q_MIN
    q_max = Q_MAX

    solutions, errors = ur10_inv_kin(T)

    if solutions.shape[0]==0: 
        return None, None
    
    cands=[]; e2=[]

    for q, e in zip(solutions, errors):
        q2 = unwrap_to_seed(q, q_seed)

        if np.all(q2>=q_min) and np.all(q2<=q_max):
            cands.append(q2); e2.append(e)

    if not cands: 
        return None, None
    
    if q_seed is None: 
        return cands[0], e2[0]
    
    i = int(np.argmin([np.linalg.norm(q - q_seed) for q in cands]))
    return cands[i], e2[i]



def numeric_jacobian(q, eps=1e-6):
    """
    Compute the 6x6 geometric Jacobian of the UR10 numerically using forward finite differences.

    The Jacobian relates joint velocities dq to the spatial end-effector twist [v; ω]:
        [v; ω] = J(q) * dq

    where:
        v = translational velocity of the TCP (world frame)
        ω = angular velocity of the TCP (world frame)

    Inputs:
        q:   Joint configuration [rad] (6,)
        eps: Small perturbation [rad] used for numerical differentiation.

    Outputs:
        J:   Numerical Jacobian in the world frame.

    Note: This is a first order finite difference approximation.
    
    """

    # forward kinematics
    T0 = fk_ur10(q)[-1]        # 4×4 transform of TCP in world frame
    p0 = T0[:3, 3]             # position vector (x,y,z)
    R0 = T0[:3, :3]            # rotation matrix (3x3)

    # Initialize Jacobian matrix
    J = np.zeros((6, 6))

    # Finite difference each joint
    for i in range(6):
        # Perturb only joint i
        q_perturbed = q.copy()
        q_perturbed[i] += eps

        # Forward kinematics at perturbed configuration
        Ti = fk_ur10(q_perturbed)[-1]
        pi = Ti[:3, 3]
        Ri = Ti[:3, :3]

        # Linear velocity part
        v_i = (pi - p0) / eps

        # Angular velocity part
        dR = Ri @ R0.T 
        w_i = np.array([ dR[2,1] - dR[1,2],
                         dR[0,2] - dR[2,0],
                         dR[1,0] - dR[0,1]]) / (2 * eps)

        # Stack [v_i; w_i] into column i
        J[:, i] = np.hstack((v_i, w_i))

    return J


def rotation_z(theta):
    """
    Rotation matrix for a rotation about the z-axis by angle theta [rad].

    Input:
    - theta: Rotation angle [rad]

    Output:
    - Rz:    Rotation matrix (3x3)
    """
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[ c, -s, 0],
                   [ s,  c, 0],
                   [ 0,  0, 1]])
    return Rz


def move_point_xyz(x_change, y_change, z_change, q_init_pos, q_init_ori):
    """
    Move the TCP by specified changes in x, y, z from the initial joint configuration.
    
    Inputs:
      - x_change: Change in x position [m]
      - y_change: Change in y position [m]
      - z_change: Change in z position [m]
      - q_init_pos:   Initial joint position configuration [rad] (6,)
      - q_init_ori:   Initial joint orientation configuration [rad] (6,)

    Outputs:
      - q_new:    New joint configuration [rad] (6,)
      - info:     IK error code (0 if within reach, 1 if shoulder-wrist distance > |a2|+|a3|).
    """
    T_init_pos = fk_ur10(q_init_pos)[-1]
    x_init, y_init, z_init = T_init_pos[:3,3]
    T_new = T_init_pos.copy()
    T_init_ori = fk_ur10(q_init_ori)[-1]
    R_init_ori = T_init_ori[:3,:3]
    T_new[:3,:3] = R_init_ori
    T_new[0,3] = x_init + x_change
    T_new[1,3] = y_init + y_change
    T_new[2,3] = z_init + z_change
    q_new, info = pick_ik_solution(T_new, q_init_pos)
    return q_new, info