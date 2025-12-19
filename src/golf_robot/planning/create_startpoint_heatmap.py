#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ball placement grid search in a 0.5 x 0.5 m area around the origin, computing
the *minimum* achievable TCP speed over directions with angles between -20 and 20
degrees (in the XY plane).

Now prints the joint configuration (q_hit_best) for the best XY placement and
(optionally) plots a heatmap of this min-over-directions speed.
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
Q_HIT_SEED = np.array([-2.18539977, -2.44831034, -1.85033569,  1.15618144,  0.61460362,  0.50125766], dtype=float)
# Q_HIT_SEED = np.array([
#     np.deg2rad(-127.23), np.deg2rad(-153.93), np.deg2rad(-100.71),
#     np.deg2rad(74.59),   np.deg2rad(35.18),   np.deg2rad(-150.02)
# ], dtype=float)

JOINT_VEL_MAX = np.array([3.0]*6, dtype=float)  # [rad/s]

HALF_WIDTH = 0.5        # 1.0 m x 1.0 m area around origin -> +/- 0.5 m
GRID_STEP  = 0.01        # 1 cm resolution

# Angular search range for TCP direction in XY plane
ANGLE_MIN_DEG = -20.0
ANGLE_MAX_DEG =  20.0
N_ANGLES      = 81       # e.g. ~0.5 deg steps

EPS = 1e-10

# differential IK params
DIK_LAMBDA     = 1e-3
DIK_POS_W      = 1.0
DIK_ORI_W      = 0.0
DIK_MAX_ITERS  = 150
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
    """Return position error (ignoring orientation)."""
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
        dq = pinv_damped(J[:3, :], DIK_LAMBDA) @ (dp * DIK_POS_W)
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
    """
    Given a linear Jacobian J_lin (3x6), a desired direction u (3,),
    and joint speed limits dq_max (6,), compute how large the TCP speed
    can be along u without violating dq_max.
    """
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

def min_speed_over_angle_range(
    J_lin: np.ndarray,
    dq_max: np.ndarray,
    angle_min_deg: float = ANGLE_MIN_DEG,
    angle_max_deg: float = ANGLE_MAX_DEG,
    n_angles: int = N_ANGLES
) -> float:
    """
    For a given J_lin, compute the *minimum* over angles of the maximum
    achievable TCP speed, for directions in the XY plane whose angle is
    between angle_min_deg and angle_max_deg (degrees, from +X towards +Y).

    I.e. for each direction u(theta), compute max_speed_in_direction(J,u),
    then return min_theta max_speed(theta).
    """
    angles = np.deg2rad(np.linspace(angle_min_deg, angle_max_deg, n_angles))
    worst = np.inf
    for th in angles:
        u = np.array([np.cos(th), np.sin(th), 0.0])  # XY plane
        s = max_speed_in_direction(J_lin, u, dq_max)
        if s < worst:
            worst = s
    if not np.isfinite(worst):
        return 0.0
    return float(worst)

# ------------------------------------------------------------
# Grid search function
# ------------------------------------------------------------
def grid_search_ball_xy_angle_range(
    q_hit_seed: np.ndarray,
    dq_max: np.ndarray,
    half_width: float,
    step: float,
    angle_min_deg: float = ANGLE_MIN_DEG,
    angle_max_deg: float = ANGLE_MAX_DEG,
    n_angles: int = N_ANGLES
) -> Tuple[Tuple[float, float], float, Dict]:
    """
    Grid-search ball XY position in a square of side 2*half_width around
    the origin TCP XY position. For each position, compute the *minimum*
    over directions (in [angle_min_deg, angle_max_deg]) of the maximum
    achievable TCP speed.
    """
    T0 = tcp_from_q(q_hit_seed)
    x0, y0, z0 = T0[0, 3], T0[1, 3], T0[2, 3]

    xs_off = np.arange(-half_width, half_width + 1e-12, step)
    ys_off = np.arange(-half_width, half_width + 1e-12, step)

    xs_abs = x0 + xs_off
    ys_abs = y0 + ys_off

    Nx = len(xs_abs)
    Ny = len(ys_abs)

    # Heatmap array: heat[i, j] corresponds to xs_abs[i], ys_abs[j]
    heat = np.full((Nx, Ny), np.nan, dtype=float)

    best_xy = (x0, y0)
    best_q = q_hit_seed.copy()
    best_score = -np.inf  # we still want the *largest* minimum

    ik_fail_reasons: Dict[str, int] = {}
    tested = solved = 0
    seed = q_hit_seed.copy()

    for ix, xt in enumerate(xs_abs):
        for iy, yt in enumerate(ys_abs):
            tested += 1
            Tt = update_tcp_xy_keep_z_ori(T0, xt, yt)
            q_hit, reason = try_ik(Tt, seed)
            if q_hit is None:
                ik_fail_reasons[reason] = ik_fail_reasons.get(reason, 0) + 1
                heat[ix, iy] = np.nan
                continue

            solved += 1
            seed = q_hit

            J = numeric_jacobian(q_hit)
            J_lin = J[:3, :]

            score = min_speed_over_angle_range(
                J_lin, dq_max,
                angle_min_deg=angle_min_deg,
                angle_max_deg=angle_max_deg,
                n_angles=n_angles
            )
            heat[ix, iy] = score

            # We want the placement whose *worst* direction is as good as possible
            if score > best_score:
                best_score = score
                best_xy = (xt, yt)
                best_q = q_hit.copy()

    stats = {
        "tested": tested,
        "ik_solved": solved,
        "ik_fail_reasons": ik_fail_reasons,
        "origin_xyz": (x0, y0, z0),
        "best_q": best_q,
        "xs_abs": xs_abs,
        "ys_abs": ys_abs,
        "angle_min_deg": angle_min_deg,
        "angle_max_deg": angle_max_deg,
    }
    return best_xy, best_score, {"heatmap": heat, "stats": stats}

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    best_xy, best_score, extra = grid_search_ball_xy_angle_range(
        q_hit_seed=Q_HIT_SEED,
        dq_max=JOINT_VEL_MAX,
        half_width=HALF_WIDTH,
        step=GRID_STEP,
        angle_min_deg=ANGLE_MIN_DEG,
        angle_max_deg=ANGLE_MAX_DEG,
        n_angles=N_ANGLES
    )

    s = extra["stats"]
    heat = extra["heatmap"]
    xs = s["xs_abs"]
    ys = s["ys_abs"]

    print("\n[Ball Placement Grid Search Result]")
    print(f"Origin XY: ({s['origin_xyz'][0]:.4f}, {s['origin_xyz'][1]:.4f}) m")
    print(f"Best XY:   ({best_xy[0]:.4f}, {best_xy[1]:.4f}) m")
    if np.isfinite(best_score):
        print(
            f"Best score (max over placements of "
            f"min over angles [{ANGLE_MIN_DEG}, {ANGLE_MAX_DEG}] deg): "
            f"{best_score:.6f} m/s"
        )
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

    # Optional: plot heatmap (requires matplotlib)
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        extent = [xs[0], xs[-1], ys[0], ys[-1]]
        plt.imshow(
            heat.T, origin="lower", extent=extent, aspect="equal"
        )
        plt.colorbar(label="Min TCP speed over angles [m/s]")
        plt.scatter([best_xy[0]], [best_xy[1]], marker="x")
        plt.title(f"Min max-achievable TCP speed (angles {ANGLE_MIN_DEG}° to {ANGLE_MAX_DEG}°)")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.show()
    except ImportError:
        print("\nmatplotlib not installed; skipping heatmap plot.")
