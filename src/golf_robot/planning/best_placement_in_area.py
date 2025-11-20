#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ball placement search (XY within 5 cm) with NO Z-velocity at impact.

Now prints the joint configuration (q_hit_best) for the best XY placement.
"""

import numpy as np
from typing import Tuple, Optional, Dict

from kinematics import fk_ur10, numeric_jacobian
try:
    from kinematics import pick_ik_solution
    HAVE_CLOSED_FORM_IK = True
except Exception:
    HAVE_CLOSED_FORM_IK = False

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
Q_HIT_SEED = np.array([
    np.deg2rad(-127.23), np.deg2rad(-153.93), np.deg2rad(-100.71),
    np.deg2rad(74.59),   np.deg2rad(35.18),   np.deg2rad(-150.02)
], dtype=float)

JOINT_VEL_MAX = np.array([3.0]*6, dtype=float)   # [rad/s]
RADIUS    = 0.5                                 # 5 cm search radius
GRID_STEP = 0.01                                 # 1 cm resolution
N_DIRS    = 72                                   # directions on YZ plane
EPS       = 1e-10

# differential IK params
DIK_LAMBDA = 1e-3
DIK_POS_W = 1.0
DIK_ORI_W = 0.0
DIK_MAX_ITERS = 150
DIK_STEP_SCALE = 0.8

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def tcp_from_q(q: np.ndarray) -> np.ndarray:
    T = fk_ur10(np.asarray(q, float))
    if isinstance(T, (list, tuple)):
        T = T[-1]
    return np.asarray(T, float)

def update_tcp_xy_keep_z_ori(T_ref: np.ndarray, x: float, y: float) -> np.ndarray:
    T2 = T_ref.copy()
    T2[0, 3] = x
    T2[1, 3] = y
    return T2

def pinv_damped(J: np.ndarray, lam: float) -> np.ndarray:
    return J.T @ np.linalg.inv(J @ J.T + (lam**2)*np.eye(J.shape[0]))

def delta_pose(T_now: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """Return position error (ignoring orientation)"""
    return T_target[:3, 3] - T_now[:3, 3]

def differential_ik_xy(T_target: np.ndarray, q_seed: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """Simple position-only damped least-squares IK (orientation fixed)."""
    q = q_seed.copy()
    for _ in range(DIK_MAX_ITERS):
        T_now = tcp_from_q(q)
        dp = delta_pose(T_now, T_target)
        if np.linalg.norm(dp) < 1e-5:
            return q, "converged"
        J = numeric_jacobian(q)
        dq = pinv_damped(J[:3,:], DIK_LAMBDA) @ (dp * DIK_POS_W)
        q += DIK_STEP_SCALE * dq
    return None, "no_convergence"


def try_ik(T_target: np.ndarray, q_seed: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    if HAVE_CLOSED_FORM_IK:
        try:
            q_sol = pick_ik_solution(T_target, seed=q_seed)
            if q_sol is not None:
                return np.asarray(q_sol, float).reshape(6), "closed_form_ok"
        except Exception:
            pass
    return differential_ik_xy(T_target, q_seed)

def max_speed_in_direction(J_lin: np.ndarray, u: np.ndarray, dq_max: np.ndarray) -> float:
    u = np.asarray(u, float)
    if np.linalg.norm(u) < EPS:
        return 0.0
    u /= np.linalg.norm(u)
    dq_dir = pinv_damped(J_lin, 1e-6) @ u
    idx = np.where(np.abs(dq_dir) > 1e-12)[0]
    if len(idx) == 0:
        return 0.0
    per_joint_scale = dq_max[idx] / np.abs(dq_dir[idx])
    scaling = float(np.min(per_joint_scale))
    v_tcp = J_lin @ (dq_dir * scaling)
    return float(np.linalg.norm(v_tcp))

def min_of_max_speeds_yz(J_lin: np.ndarray, dq_max: np.ndarray, n_dirs: int = 72) -> float:
    worst = np.inf
    for k in range(n_dirs):
        theta = 2*np.pi * (k/n_dirs)
        u = np.array([np.cos(theta), np.sin(theta), 0.0])  # u_z = 0
        s = max_speed_in_direction(J_lin, u, dq_max)
        if s < worst:
            worst = s
    return float(worst) if np.isfinite(worst) else 0.0

# ------------------------------------------------------------
# Search function
# ------------------------------------------------------------
def search_best_ball_xy(
    q_hit_seed: np.ndarray,
    dq_max: np.ndarray,
    radius: float,
    step: float,
    n_dirs: int = 72
) -> Tuple[Tuple[float, float], float, Dict]:
    T0 = tcp_from_q(q_hit_seed)
    x0, y0, z0 = T0[0,3], T0[1,3], T0[2,3]
    xs = np.arange(-radius, radius + 1e-12, step)
    ys = np.arange(-radius, radius + 1e-12, step)

    best_xy = (x0, y0)
    best_q = q_hit_seed.copy()
    best_score = -np.inf
    heat: Dict[Tuple[float,float], float] = {}
    ik_fail_reasons: Dict[str, int] = {}
    tested = solved = 0
    seed = q_hit_seed.copy()

    for dx in xs:
        for dy in ys:
            if dx**2 + dy**2 > radius**2:
                continue
            tested += 1
            xt, yt = x0 + dx, y0 + dy
            Tt = update_tcp_xy_keep_z_ori(T0, xt, yt)
            q_hit, reason = try_ik(Tt, seed)
            if q_hit is None:
                ik_fail_reasons[reason] = ik_fail_reasons.get(reason, 0) + 1
                heat[(xt, yt)] = np.nan
                continue
            solved += 1
            seed = q_hit
            J = numeric_jacobian(q_hit)
            J_lin = J[:3,:]
            score = min_of_max_speeds_yz(J_lin, dq_max, n_dirs)
            heat[(xt, yt)] = score
            if score > best_score:
                best_score = score
                best_xy = (xt, yt)
                best_q = q_hit.copy()

    stats = {
        "tested": tested,
        "ik_solved": solved,
        "ik_fail_reasons": ik_fail_reasons,
        "origin_xyz": (x0, y0, z0),
        "best_q": best_q
    }
    return best_xy, best_score, {"heatmap": heat, "stats": stats}

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    best_xy, best_score, extra = search_best_ball_xy(
        q_hit_seed=Q_HIT_SEED,
        dq_max=JOINT_VEL_MAX,
        radius=RADIUS,
        step=GRID_STEP,
        n_dirs=N_DIRS
    )

    s = extra["stats"]
    print("\n[Ball Placement Search Result]")
    print(f"Origin XY: ({s['origin_xyz'][0]:.4f}, {s['origin_xyz'][1]:.4f}) m")
    print(f"Best XY:   ({best_xy[0]:.4f}, {best_xy[1]:.4f}) m")
    if np.isfinite(best_score):
        print(f"Best score (min-of-max over YZ dirs): {best_score:.6f} m/s")
    else:
        print("Best score: no feasible placements")
    print(f"Grid tested: {s['tested']} positions, IK solved: {s['ik_solved']}")
    if s['ik_fail_reasons']:
        print("IK failures by reason:", s['ik_fail_reasons'])

    # Print the best joint configuration
    q_best = s["best_q"]
    print("\nBest joint configuration (radians):")
    print(np.round(q_best, 6))
    print("Best joint configuration (degrees):")
    print(np.round(np.rad2deg(q_best), 3))
