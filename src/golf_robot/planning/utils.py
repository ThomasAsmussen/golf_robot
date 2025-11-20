import time
import numpy as np

def unwrap_to_seed(q, q_seed):
    """
    Unwrap joint angles q to be as close as possible to seed configuration by adding or subtracting multiples of 2π.

    Input:
        q:      Joint angles [rad] (6,)
        q_seed: Seed joint angles [rad] (6,)

    Output:
        q_unwrapped: Unwrapped joint angles [rad] (6,)
    """
    if q_seed is None: return q 

    q  = np.asarray(q, float).copy()
    q_seed = np.asarray(q_seed, float)

    for i in range(6):
        d = q[i] - q_seed[i]
        q[i] -= 2*np.pi * np.round(d / (2*np.pi))

    return q

def unwrap_to_seed_all(qf, q0):
    out = np.asarray(qf, float).copy()
    for i in range(6):
        d = out[i] - q0[i]
        out[i] -= 2*np.pi * np.round(d / (2*np.pi))
    return out

def rotm_from_rpy(roll, pitch, yaw):
    """
    Rotation matrix from roll, pitch, yaw angles [rad].
    ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)

    Inputs:
        roll:   Rotation about x-axis [rad]
        pitch:  Rotation about y-axis [rad]
        yaw:    Rotation about z-axis [rad]

    Output:
        R:      Rotation matrix (3x3)
    """
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])

    return Rz @ Ry @ Rx

def T_from_xyz_rpy(x, y, z, roll, pitch, yaw):
    """
    Homogeneous transformation matrix from position (x,y,z) and roll, pitch, yaw angles [rad].
    ZYX convention for rotation matrix.

    Inputs:
        x, y, z:       Position coordinates
        roll, pitch, yaw:  Roll, pitch, yaw angles [rad]
        
    Output:
        T:             Homogeneous transformation matrix (4x4)
    """
    T = np.eye(4)
    T[:3,:3] = rotm_from_rpy(roll, pitch, yaw)
    T[:3, 3] = [x, y, z]
    return T


# jitter-tight timing for servoj loop
def spin_until(t_deadline):
    while True:
        now = time.perf_counter()
        if t_deadline - now <= 2e-4:  # 0.2 ms
            break
        time.sleep(2e-4)
    while time.perf_counter() < t_deadline:
        pass

def normalize(v, eps=1e-12):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / (n if n > eps else 1.0)
