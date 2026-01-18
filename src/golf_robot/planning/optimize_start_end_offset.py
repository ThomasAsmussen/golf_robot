"""
optimize_start_end_offsets.py

Optimize start/end placement (as TCP offsets relative to the impact pose)
to minimize maximum absolute joint acceleration over the full planned trajectory.

Assumptions:
- You already have these functions available (same file or importable):
    - move_point_xyz
    - fk_ur10
    - normalize
    - impact_joint_config_from_direction
    - max_speed_at_pose
    - plan_piecewise_quintic
- And constants:
    - DT

This script:
1) Builds q_hit from (impact_speed, impact_angle, ball offsets)
2) Samples start/end offsets (dx,dy,dz) relative to q_hit
3) Plans using plan_piecewise_quintic
4) Scores feasibility + peak abs ddq (and optional time penalty)
5) Returns best found offsets and saves a JSON report

Run:
  python optimize_start_end_offsets.py
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import argparse
import os
from datetime import datetime


# ------------------------------------------------------------
# Import your project functions
# ------------------------------------------------------------
try:
    from planning.utils import normalize
    from planning.kinematics import fk_ur10, move_point_xyz
    from planning.config import DT
    # These must exist in the same module you edited earlier, or import them from there:
    # from planning.your_planner_module import impact_joint_config_from_direction, max_speed_at_pose, plan_piecewise_quintic
except ImportError:
    from utils import normalize
    from kinematics import fk_ur10, move_point_xyz
    from config import DT
    # from your_planner_module import impact_joint_config_from_direction, max_speed_at_pose, plan_piecewise_quintic

# IMPORTANT:
# Adjust this import to where these functions actually live in your repo.
from trajectory import (
    impact_joint_config_from_direction,
    max_speed_at_pose,
    plan_piecewise_quintic,
    plan_segment_start_to_hit,
    plan_segment_hit_to_end,
)

# ------------------------------------------------------------
# User-configurable settings
# ------------------------------------------------------------

# Impact / ball conditions you want to optimize for
IMPACT_SPEED = 1.614      # m/
IMPACT_ANGLE_DEG = 1.29   # degrees
BALL_X_OFFSET = 0.0      # m
BALL_Y_OFFSET = 0.0      # m

# ------------------------------------------------------------
# Optimization mode
# ------------------------------------------------------------
# "both"       : optimize start+end together (2 segments planned)
# "start_only" : optimize ONLY start offsets by planning ONLY segment 0 (start->hit)
# "end_only"   : optimize ONLY end offsets by planning ONLY segment 1 (hit->end)
OPT_MODE = "start_only"   # <-- set this

# Fixed offsets (used for the side you are NOT optimizing)
# Set these to your known-good defaults (or previous best).
FIXED_START_OFF = np.array([-0.36, 0.20, 0.236], dtype=float)
FIXED_END_OFF   = np.array([ 0.683, 0.127, 0.11], dtype=float)

# ------------------------------------------------------------

# Seeds for IK / move_point_xyz (your existing hard-coded seeds)
Q0_START_SEED = np.array([np.deg2rad(-144.36), np.deg2rad(-162.82), np.deg2rad(-51.80),
                          np.deg2rad(66.92), np.deg2rad(42.60), np.deg2rad(6.51)])
Q0_END_SEED   = np.array([np.deg2rad(-87.65),  np.deg2rad(-135.05), np.deg2rad(-108.29),
                          np.deg2rad(85.72), np.deg2rad(4.39), np.deg2rad(-12.23)])

# Reference q_hit seed (your typical one)
Q0_HIT_REF = np.array([-2.11202641, -2.45037247, -1.67584054,  0.95906874,  0.53322783,  0.36131151])

# Search space for start/end offsets (meters), relative to q_hit
# These are TCP offsets in base frame passed into move_point_xyz(dx,dy,dz,...)
START_BOUNDS = dict(
    dx=(-0.60, -0.20),   # behind ball (negative x)
    dy=(-0.30,  0.30),
    dz=( 0.05,  0.50),
)

END_BOUNDS = dict(
    dx=( 0.20,  0.75),   # after hit (positive x)
    dy=(-0.20,  0.20),
    dz=( 0.00,  0.35),
)

# Planner knobs
T_MAX_DEFAULT = 5.0
T_MAX_IMPACT = 6.0

# If 0.0 => will prefer larger segment times (lowest accel but slow).
# If >0 => trades off time vs accel.
TIME_PENALTY_DEFAULT = 0.2
TIME_PENALTY_IMPACT  = 0.2

# Optimization budget
N_COARSE = 80          # random candidates in coarse search
N_REFINE = 80          # random candidates around best found
REFINE_SIGMA = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])  # std for [sx,sy,sz, ex,ey,ez] in meters

# Reproducibility
RNG_SEED = 7

# Output
OUT_DIR = Path("log")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _sample_uniform(bounds: Dict[str, Tuple[float, float]], rng: np.random.Generator) -> np.ndarray:
    dx = rng.uniform(*bounds["dx"])
    dy = rng.uniform(*bounds["dy"])
    dz = rng.uniform(*bounds["dz"])
    return np.array([dx, dy, dz], dtype=float)


def _clamp(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(v, lo), hi)


def _bounds_to_arrays(bounds: Dict[str, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.array([bounds["dx"][0], bounds["dy"][0], bounds["dz"][0]], dtype=float)
    hi = np.array([bounds["dx"][1], bounds["dy"][1], bounds["dz"][1]], dtype=float)
    return lo, hi


def peak_abs_ddq(ddQ_all: np.ndarray) -> float:
    """Peak absolute acceleration across all joints and all time."""
    return float(np.max(np.abs(ddQ_all)))


def objective(peak_ddq: float, total_T: float,
              time_penalty_total: float = 0.0) -> float:
    """Scalar objective (lower is better)."""
    return float(peak_ddq + time_penalty_total * total_T)


@dataclass
class CandidateResult:
    ok: bool
    sx: float
    sy: float
    sz: float
    ex: float
    ey: float
    ez: float
    peak_ddq: float = math.inf
    total_T: float = math.inf
    J: float = math.inf
    meta: Optional[Dict[str, Any]] = None


# ------------------------------------------------------------
# Core evaluation
# ------------------------------------------------------------

def build_q_hit(
    impact_speed: float,
    impact_angle_deg: float,
    ball_x_offset: float,
    ball_y_offset: float,
    q0_hit_ref: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Your same procedure:
    - move q0_hit_ref for ball offset
    - compute q_hit from direction using impact_joint_config_from_direction
    - optionally small z tweak
    """
    # you had these offsets baked in; keep consistent if needed
    # q0_hit = move_point_xyz(ball_x_offset + 0.005, ball_y_offset + 0.02, 0.00, q0_hit_ref, q0_hit_ref)[0]
    q0_hit, info = move_point_xyz(ball_x_offset, ball_y_offset + 0.01, 0.00, q0_hit_ref, q0_hit_ref)
    if q0_hit is None:
        return None
    q0_hit = np.asarray(q0_hit, float)
    if q0_hit.shape != (6,):
        return None

    impact_direction = np.array([math.cos(math.radians(impact_angle_deg)),
                                 math.sin(math.radians(impact_angle_deg)),
                                 0.0], dtype=float)
    v_dir_base = normalize(impact_direction)

    # ball center as you did
    ball_center = fk_ur10(q0_hit)[-1][:3, 3] + np.array([0.02133, 0.0, 0.0])
    q_hit = impact_joint_config_from_direction(q0_hit, v_dir_base, ball_center=ball_center)
    if q_hit is None:
        return None

    # q_hit = move_point_xyz(0.0, 0.0, 0.005, q_hit, q_hit)[0]
    return q_hit


def evaluate_candidate(
    q_hit: np.ndarray,
    impact_speed: float,
    impact_angle_deg: float,
    start_off: np.ndarray,
    end_off: np.ndarray,
    *,
    opt_mode: str,
) -> CandidateResult:
    """
    Mode-aware evaluation:
      - both       : plans both segments via plan_piecewise_quintic (current behavior)
      - start_only : plans ONLY segment 0 via plan_segment_start_to_hit
      - end_only   : plans ONLY segment 1 via plan_segment_hit_to_end
    """
    opt_mode = opt_mode.lower().strip()
    sx, sy, sz = map(float, start_off)
    ex, ey, ez = map(float, end_off)

    impact_direction = np.array([
        math.cos(math.radians(impact_angle_deg)),
        math.sin(math.radians(impact_angle_deg)),
        0.0
    ], dtype=float)
    v_dir_base   = normalize(impact_direction)
    lin_velocity = v_dir_base * impact_speed

    # Feasibility at hit pose (same as generate_trajectory)
    max_speed = max_speed_at_pose(q_hit, v_dir_base)
    if max_speed < impact_speed:
        return CandidateResult(
            ok=False, sx=sx, sy=sy, sz=sz, ex=ex, ey=ey, ez=ez,
            meta={"problem": "Impact speed not feasible at q_hit", "max_speed": float(max_speed)}
        )

    # Build only what we need depending on mode
    if opt_mode == "start_only":
        # q_start varies, q_end fixed but NOT planned
        q_start, info = move_point_xyz(sx, sy, sz, q_hit, Q0_START_SEED)
        if q_start is None:
            return CandidateResult(
                ok=False, sx=sx, sy=sy, sz=sz, ex=ex, ey=ey, ez=ez,
                meta={"problem": "IK failed for start offset", "start_offset": [sx, sy, sz], "ik_info": info}
            )
        q_start = np.asarray(q_start, float)
        if q_start.shape != (6,):
            return CandidateResult(ok=False, sx=sx, sy=sy, sz=sz, ex=ex, ey=ey, ez=ez,
                                meta={"problem": "Bad q_start shape", "shape": str(q_start.shape)})
        
        seg0, Q0, dQ0, ddQ0 = plan_segment_start_to_hit(
            q_start, q_hit, lin_velocity,
            T_max=T_MAX_IMPACT,                # this segment ends at impact -> use impact knobs
            time_penalty=TIME_PENALTY_IMPACT,
        )
        if Q0 is None or ddQ0 is None or seg0 is None:
            return CandidateResult(ok=False, sx=sx, sy=sy, sz=sz, ex=ex, ey=ey, ez=ez,
                                   meta={"problem": "Planner returned None (seg0)"})

        peak = peak_abs_ddq(ddQ0)
        total_T = float(seg0.get("T", seg0["ts"][-1]))

        # Objective is only segment 0 (pre-impact)
        print(f"peak: {peak}, total_T: {total_T}")
        J = objective(peak, total_T, time_penalty_total=TIME_PENALTY_DEFAULT)

        return CandidateResult(
            ok=True, sx=sx, sy=sy, sz=sz, ex=ex, ey=ey, ez=ez,
            peak_ddq=peak, total_T=total_T, J=J,
            meta={"mode": "start_only", "T0": total_T, "N0": int(Q0.shape[0])}
        )

    elif opt_mode == "end_only":
        # q_end varies, q_start fixed but NOT planned
        q_end, info = move_point_xyz(ex, ey, ez, q_hit, Q0_END_SEED)
        if q_end is None:
            return CandidateResult(
                ok=False, sx=sx, sy=sy, sz=sz, ex=ex, ey=ey, ez=ez,
                meta={"problem": "IK failed for end offset", "end_offset": [ex, ey, ez], "ik_info": info}
            )
        q_end = np.asarray(q_end, float)
        if q_end.shape != (6,):
            return CandidateResult(ok=False, sx=sx, sy=sy, sz=sz, ex=ex, ey=ey, ez=ez,
                                meta={"problem": "Bad q_end shape", "shape": str(q_end.shape)})
        seg1, Q1, dQ1, ddQ1 = plan_segment_hit_to_end(
            q_hit, q_end, lin_velocity,
            T_max=T_MAX_DEFAULT,               # post-impact segment -> use default knobs (tune if you prefer)
            time_penalty=TIME_PENALTY_DEFAULT,
        )
        if Q1 is None or ddQ1 is None or seg1 is None:
            return CandidateResult(ok=False, sx=sx, sy=sy, sz=sz, ex=ex, ey=ey, ez=ez,
                                   meta={"problem": "Planner returned None (seg1)"})

        peak = peak_abs_ddq(ddQ1)
        total_T = float(seg1.get("T", seg1["ts"][-1]))

        # Objective is only segment 1
        J = objective(peak, total_T, time_penalty_total=TIME_PENALTY_DEFAULT)

        return CandidateResult(
            ok=True, sx=sx, sy=sy, sz=sz, ex=ex, ey=ey, ez=ez,
            peak_ddq=peak, total_T=total_T, J=J,
            meta={"mode": "end_only", "T1": total_T, "N1": int(Q1.shape[0])}
        )

    elif opt_mode == "both":
        # Current behavior: plan both segments
        q_start, info_start = move_point_xyz(sx, sy, sz, q_hit, Q0_START_SEED)
        if q_start is None:
            return CandidateResult(
                ok=False, sx=sx, sy=sy, sz=sz, ex=ex, ey=ey, ez=ez,
                meta={"problem": "IK failed for start offset", "start_offset": [sx, sy, sz], "info": info_start}
            )
        q_start = np.asarray(q_start, float)
        if q_start.shape != (6,):
            return CandidateResult(ok=False, sx=sx, sy=sy, sz=sz, ex=ex, ey=ey, ez=ez,
                                meta={"problem": "Bad q_start shape", "shape": str(q_start.shape)})
        q_end, info_end = move_point_xyz(ex, ey, ez, q_hit, Q0_END_SEED)
        if q_end is None:
            return CandidateResult(
                ok=False, sx=sx, sy=sy, sz=sz, ex=ex, ey=ey, ez=ez,
                meta={"problem": "IK failed for end offset", "end_offset": [ex, ey, ez], "info": info_end}
            )
        q_end = np.asarray(q_end, float)
        if q_end.shape != (6,):
            return CandidateResult(ok=False, sx=sx, sy=sy, sz=sz, ex=ex, ey=ey, ez=ez,
                                meta={"problem": "Bad q_end shape", "shape": str(q_end.shape)})


        impact_idx = 1
        waypoints = [q_start, q_hit, q_end]

        segments, Q_all, dQ_all, ddQ_all = plan_piecewise_quintic(
            waypoints, impact_idx, lin_velocity,
            T_max_default=T_MAX_DEFAULT,
            T_max_impact=T_MAX_IMPACT,
            time_penalty_default=TIME_PENALTY_DEFAULT,
            time_penalty_impact=TIME_PENALTY_IMPACT,
        )
        if Q_all is None or ddQ_all is None or segments is None:
            return CandidateResult(ok=False, sx=sx, sy=sy, sz=sz, ex=ex, ey=ey, ez=ez,
                                   meta={"problem": "Planner returned None"})

        peak = peak_abs_ddq(ddQ_all)
        total_T = float(sum(float(seg.get("T", seg["ts"][-1])) for seg in segments))
        J = objective(peak, total_T, time_penalty_total=TIME_PENALTY_DEFAULT)

        return CandidateResult(
            ok=True, sx=sx, sy=sy, sz=sz, ex=ex, ey=ey, ez=ez,
            peak_ddq=peak, total_T=total_T, J=J,
            meta={"mode": "both",
                  "segments_T": [float(seg.get("T", seg["ts"][-1])) for seg in segments],
                  "N": int(Q_all.shape[0])}
        )

    else:
        raise ValueError(f"Unknown opt_mode: {opt_mode!r}")


def iter_speed_angle_grid(
    speeds: List[float],
    angles_deg: List[float],
) -> List[Tuple[float, float]]:
    return [(float(s), float(a)) for s in speeds for a in angles_deg]


def shard_items(items: List[Any], shard_idx: int, num_shards: int) -> List[Any]:
    # stride sharding (stable, no need to know chunk sizes)
    # shard_idx in [0, num_shards-1]
    return items[shard_idx::num_shards]


def make_rng_seed(base_seed: int, *, speed: float, angle: float, bx: float, by: float) -> int:
    # deterministic per condition so parallelism doesn’t change results
    key = f"{base_seed}|{speed:.6f}|{angle:.6f}|{bx:.6f}|{by:.6f}"
    return (abs(hash(key)) % (2**31 - 1)) + 1


# ------------------------------------------------------------
# Optimization loop
# ------------------------------------------------------------

def optimize_start_end_for_condition(
    impact_speed: float,
    impact_angle_deg: float,
    ball_x_offset: float,
    ball_y_offset: float,
    rng_seed: int = RNG_SEED,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(rng_seed))

    q_hit = build_q_hit(impact_speed, impact_angle_deg, ball_x_offset, ball_y_offset, Q0_HIT_REF)
    if q_hit is None:
        return {"ok": False, "problem": "Failed to build q_hit (IK) for this condition."}

    start_lo, start_hi = _bounds_to_arrays(START_BOUNDS)
    end_lo, end_hi = _bounds_to_arrays(END_BOUNDS)

    best: Optional[CandidateResult] = None
    n_ok = 0

    def consider(res: CandidateResult):
        nonlocal best, n_ok
        if res.ok:
            n_ok += 1
            if best is None or res.J < best.J:
                best = res

    t0 = time.time()

    opt_mode = OPT_MODE.lower().strip()

    # ---- coarse search (mode-aware)
    for _ in range(N_COARSE):
        if opt_mode == "start_only":
            s = _sample_uniform(START_BOUNDS, rng)
            e = FIXED_END_OFF.copy()
        elif opt_mode == "end_only":
            s = FIXED_START_OFF.copy()
            e = _sample_uniform(END_BOUNDS, rng)
        else:  # both
            s = _sample_uniform(START_BOUNDS, rng)
            e = _sample_uniform(END_BOUNDS, rng)

        res = evaluate_candidate(q_hit, impact_speed, impact_angle_deg, s, e, opt_mode=OPT_MODE)
        consider(res)


    if best is None:
        return {
            "ok": False,
            "problem": "No feasible candidates found in coarse search.",
            "n_ok": n_ok,
        }

    # ---- local refinement around best
    best_vec = np.array([best.sx, best.sy, best.sz, best.ex, best.ey, best.ez], dtype=float)

    lo6 = np.concatenate([start_lo, end_lo])
    hi6 = np.concatenate([start_hi, end_hi])

    for _ in range(N_REFINE):
        noise = rng.normal(0.0, 1.0, size=6) * REFINE_SIGMA

        if opt_mode == "start_only":
            noise[3:] = 0.0
        elif opt_mode == "end_only":
            noise[:3] = 0.0

        x = best_vec + noise
        x = _clamp(x, lo6, hi6)

        s = x[:3]
        e = x[3:]
        res = evaluate_candidate(q_hit, impact_speed, impact_angle_deg, s, e, opt_mode=OPT_MODE)
        consider(res)

    dt_s = time.time() - t0

    assert best is not None

    return {
        "ok": True,
        "impact_speed": float(impact_speed),
        "impact_angle_deg": float(impact_angle_deg),
        "ball_x_offset": float(ball_x_offset),
        "ball_y_offset": float(ball_y_offset),
        "best": {
            "start_off": [best.sx, best.sy, best.sz],
            "end_off": [best.ex, best.ey, best.ez],
            "peak_abs_ddq": float(best.peak_ddq),
            "total_T": float(best.total_T),
            "J": float(best.J),
            "meta": best.meta or {},
        },
        "search": {
            "N_COARSE": int(N_COARSE),
            "N_REFINE": int(N_REFINE),
            "n_ok": int(n_ok),
            "wall_time_s": float(dt_s),
            "bounds": {"start": START_BOUNDS, "end": END_BOUNDS},
            "planner": {
                "T_MAX_DEFAULT": float(T_MAX_DEFAULT),
                "T_MAX_IMPACT": float(T_MAX_IMPACT),
                "TIME_PENALTY_DEFAULT": float(TIME_PENALTY_DEFAULT),
                "TIME_PENALTY_IMPACT": float(TIME_PENALTY_IMPACT),
            }
        }
    }


def iter_ball_offset_grid(
    x_min: float, x_max: float, nx: int,
    y_min: float, y_max: float, ny: int,
) -> List[Tuple[float, float]]:
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    return [(float(x), float(y)) for x in xs for y in ys]

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

# if __name__ == "__main__":
#     stamp = time.strftime("%Y%m%d_%H%M%S")

#     # ---------------------------
#     # Grid definition (EDIT THIS)
#     # ---------------------------
#     # These are OFFSETS applied via move_point_xyz in build_q_hit.
#     # If you want “ball start in meters in base frame”, you should convert that to offsets
#     # consistently with however your q0_hit_ref is defined.
#     GRID_X_MIN, GRID_X_MAX, GRID_NX = 0.2, 0.2, 21
#     GRID_Y_MIN, GRID_Y_MAX, GRID_NY = 0.1, 0.1, 11

#     grid = iter_ball_offset_grid(GRID_X_MIN, GRID_X_MAX, GRID_NX,
#                                  GRID_Y_MIN, GRID_Y_MAX, GRID_NY)

#     out_jsonl = OUT_DIR / f"start_end_opt_grid_{stamp}.jsonl"

#     # Optional: write a header/meta line (nice for provenance)
#     append_jsonl(out_jsonl, {
#         "type": "meta",
#         "time": stamp,
#         "impact_speed": float(IMPACT_SPEED),
#         "impact_angle_deg": float(IMPACT_ANGLE_DEG),
#         "grid": {
#             "x": {"min": GRID_X_MIN, "max": GRID_X_MAX, "n": GRID_NX},
#             "y": {"min": GRID_Y_MIN, "max": GRID_Y_MAX, "n": GRID_NY},
#         },
#         "bounds": {"start": START_BOUNDS, "end": END_BOUNDS},
#         "planner": {
#             "T_MAX_DEFAULT": float(T_MAX_DEFAULT),
#             "T_MAX_IMPACT": float(T_MAX_IMPACT),
#             "TIME_PENALTY_DEFAULT": float(TIME_PENALTY_DEFAULT),
#             "TIME_PENALTY_IMPACT": float(TIME_PENALTY_IMPACT),
#         },
#         "budget": {"N_COARSE": int(N_COARSE), "N_REFINE": int(N_REFINE), "RNG_SEED": int(RNG_SEED)},
#     })

#     print(f"[INFO] Grid size: {len(grid)}")
#     print(f"[INFO] Writing JSONL to: {out_jsonl}")

#     n_ok = 0
#     n_fail = 0

#     for i, (bx, by) in enumerate(grid):
#         # Run optimization for this ball offset
#         res = optimize_start_end_for_condition(
#             IMPACT_SPEED, IMPACT_ANGLE_DEG, bx, by
#         )

#         # Convert into a compact JSONL “record” your trajectory generator can use
#         record: Dict[str, Any] = {
#             "type": "record",
#             "idx": int(i),
#             "impact_speed": float(IMPACT_SPEED),
#             "impact_angle_deg": float(IMPACT_ANGLE_DEG),
#             "ball_x_offset": float(bx),
#             "ball_y_offset": float(by),
#             "ok": bool(res.get("ok", False)),
#         }

#         if record["ok"]:
#             b = res["best"]
#             record.update({
#                 # the thing you actually want to reuse later:
#                 "start_off": b["start_off"],   # [sx, sy, sz]
#                 "end_off": b["end_off"],       # [ex, ey, ez]
#                 "peak_abs_ddq": float(b["peak_abs_ddq"]),
#                 "total_T": float(b["total_T"]),
#                 "J": float(b["J"]),
#                 # optional provenance/debug
#                 "meta": b.get("meta", {}),
#             })
#             n_ok += 1
#             print(f"[{i+1:4d}/{len(grid)}] ok  bx={bx:+.3f} by={by:+.3f}  J={record['J']:.4f}")
#         else:
#             record["problem"] = res.get("problem", "unknown")
#             n_fail += 1
#             print(f"[{i+1:4d}/{len(grid)}] FAIL bx={bx:+.3f} by={by:+.3f}  ({record['problem']})")

#         append_jsonl(out_jsonl, record)

#     print("[DONE]")
#     print(f"  ok:   {n_ok}")
#     print(f"  fail: {n_fail}")
#     print(f"  wrote: {out_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # define your speed/angle grid on the CLI (simple “min/max/n” form)
    parser.add_argument("--speed-min", type=float, required=True)
    parser.add_argument("--speed-max", type=float, required=True)
    parser.add_argument("--speed-n", type=int, required=True)

    parser.add_argument("--angle-min", type=float, required=True)
    parser.add_argument("--angle-max", type=float, required=True)
    parser.add_argument("--angle-n", type=int, required=True)

    # sharding
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)

    # output
    parser.add_argument("--out-dir", type=str, default="log")
    parser.add_argument("--tag", type=str, default="")  # optional label
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = (args.tag + "_") if args.tag else ""

    # Build grids
    speeds = np.linspace(args.speed_min, args.speed_max, args.speed_n).tolist()
    angles = np.linspace(args.angle_min, args.angle_max, args.angle_n).tolist()
    sa_grid = iter_speed_angle_grid(speeds, angles)

    GRID_X_MIN, GRID_X_MAX, GRID_NX = -0.18, 0.18, 14
    GRID_Y_MIN, GRID_Y_MAX, GRID_NY = -0.9, 0.9, 7

    ball_grid = iter_ball_offset_grid(GRID_X_MIN, GRID_X_MAX, GRID_NX,
                                 GRID_Y_MIN, GRID_Y_MAX, GRID_NY)


    # Take this shard’s slice
    shard_idx = int(args.shard_idx)
    num_shards = int(args.num_shards)
    sa_conditions = sa_grid
    my_sa = shard_items(sa_conditions, shard_idx, num_shards)

    if len(my_sa) == 0:
        print(f"[INFO] shard {shard_idx} got 0 (speed,angle) pairs. num_shards too large?")
        raise SystemExit(0)

    # If you set num_shards == len(sa_conditions), this will be exactly 1.
    if len(my_sa) != 1:
        print(f"[WARN] shard {shard_idx} got {len(my_sa)} speed/angle pairs. "
            f"For exactly 1 per job, set num_shards={len(sa_conditions)}.")

    speed, angle = my_sa[0]

    print(f"[INFO] This job runs speed={speed:.4f}, angle={angle:+.3f}")
    print(f"[INFO] Ball grid size: {len(ball_grid)}")

    out_jsonl = OUT_DIR / f"{tag}start_end_opt_{stamp}_v{speed:.3f}_a{angle:+.2f}_shard{shard_idx:04d}of{num_shards:04d}.jsonl"

    append_jsonl(out_jsonl, {
        "type": "meta",
        "time": stamp,
        "tag": tag[:-1] if tag else "",
        "opt_mode": OPT_MODE,
        "planner": {
            "T_MAX_DEFAULT": float(T_MAX_DEFAULT),
            "T_MAX_IMPACT": float(T_MAX_IMPACT),
            "TIME_PENALTY_DEFAULT": float(TIME_PENALTY_DEFAULT),
            "TIME_PENALTY_IMPACT": float(TIME_PENALTY_IMPACT),
        },
        "budget": {"N_COARSE": int(N_COARSE), "N_REFINE": int(N_REFINE), "RNG_SEED": int(RNG_SEED)},
        "grid": {
            "speed": {"min": args.speed_min, "max": args.speed_max, "n": args.speed_n},
            "angle": {"min": args.angle_min, "max": args.angle_max, "n": args.angle_n},
            "ball_x": {"min": GRID_X_MIN, "max": GRID_X_MAX, "n": GRID_NX},
            "ball_y": {"min": GRID_Y_MIN, "max": GRID_Y_MAX, "n": GRID_NY},
            "n_total_conditions": len(sa_conditions) * len(ball_grid),
            "shard_idx": shard_idx,
            "num_shards": num_shards,
            "n_this_shard": len(my_sa) * len(ball_grid),
        }
    })

    print(f"[INFO] total conditions: {len(sa_conditions) * len(ball_grid)}")
    print(f"[INFO] shard {shard_idx}/{num_shards} conditions: {len(my_sa) * len(ball_grid)}")
    print(f"[INFO] writing: {out_jsonl}")

    n_ok = 0
    n_fail = 0

    for j, (bx, by) in enumerate(ball_grid):
        seed = make_rng_seed(RNG_SEED, speed=speed, angle=angle, bx=bx, by=by)

        res = optimize_start_end_for_condition(speed, angle, bx, by, rng_seed=seed)

        record = {
            "type": "record",
            "shard_idx": shard_idx,
            "speed_angle_idx": shard_idx,
            "ball_idx": int(j),
            "impact_speed": float(speed),
            "impact_angle_deg": float(angle),
            "ball_x_offset": float(bx),
            "ball_y_offset": float(by),
            "rng_seed": int(seed),
            "ok": bool(res.get("ok", False)),
        }

        if record["ok"]:
            b = res["best"]
            record.update({
                "start_off": b["start_off"],
                "end_off": b["end_off"],
                "peak_abs_ddq": float(b["peak_abs_ddq"]),
                "total_T": float(b["total_T"]),
                "J": float(b["J"]),
                "meta": b.get("meta", {}),
            })
            n_ok += 1
            print(f"[{j+1:4d}/{len(ball_grid)}] ok  v={speed:.4f} a={angle:+.3f}  J={record['J']:.4f}")
        else:
            record["problem"] = res.get("problem", "unknown")
            n_fail += 1
            print(f"[{j+1:4d}/{len(ball_grid)}] FAIL v={speed:.4f} a={angle:+.3f} ({record['problem']})")
        append_jsonl(out_jsonl, record)

    print("[DONE]")
    print(f"  ok:   {n_ok}")
    print(f"  fail: {n_fail}")
    print(f"  wrote: {out_jsonl}")
