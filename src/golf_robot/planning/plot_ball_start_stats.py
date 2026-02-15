#!/usr/bin/env python3
"""
plot_ball_grid_startoff_cost.py

Read optimization JSONL (merged shards), and plot:
- cost J vs (ball_x_offset, ball_y_offset) for a chosen (speed, angle)
- start_off distribution (x,y) colored by cost for a chosen (speed, angle)
- per-ball-cell summary (median start_off x,y; best cost)
- cost J vs (speed, angle) for a chosen (ball_x, ball_y)

Assumes JSONL lines like:
  {"type":"meta", ...}
  {"type":"record", "impact_speed":..., "impact_angle_deg":..., "ball_x_offset":..., "ball_y_offset":...,
   "start_off":[x,y,z], "J":..., "ok":true, ...}

Usage examples:
  python plot_ball_grid_startoff_cost.py --jsonl speed_angle_grid_start_end_all.jsonl

  # Pick the (speed,angle) slice you want to study
  python plot_ball_grid_startoff_cost.py --jsonl speed_angle_grid_start_end_all.jsonl --speed 1.8 --angle 2.0

  # Or pick the (ball_x, ball_y) slice for the speed-angle heatmap
  python plot_ball_grid_startoff_cost.py --jsonl speed_angle_grid_start_end_all.jsonl --ball-x 0.2 --ball-y 0.1

Outputs:
  Writes PNGs into --outdir (default: ./plots_<timestamp>/)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime


# ---------------------------
# IO
# ---------------------------

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


@dataclass
class Rec:
    speed: float
    angle: float
    bx: float
    by: float
    ok: bool
    sx: float
    sy: float
    sz: float
    J: float
    peak_abs_ddq: float
    total_T: float


def load_records(jsonl_path: Path, only_ok: bool = True) -> Tuple[list[Rec], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    out: list[Rec] = []

    for obj in iter_jsonl(jsonl_path):
        t = obj.get("type")
        if t == "meta":
            meta = obj
            continue
        if t != "record":
            continue

        ok = bool(obj.get("ok", False))
        if only_ok and not ok:
            continue

        start_off = obj.get("start_off", None)
        if not (isinstance(start_off, list) and len(start_off) == 3):
            continue

        speed = float(obj.get("impact_speed", math.nan))
        angle = float(obj.get("impact_angle_deg", math.nan))
        bx = float(obj.get("ball_x_offset", math.nan))
        by = float(obj.get("ball_y_offset", math.nan))
        J = float(obj.get("J", obj.get("peak_abs_ddq", math.nan)))
        peak = float(obj.get("peak_abs_ddq", math.nan))
        total_T = float(obj.get("total_T", math.nan))

        if not (np.isfinite(speed) and np.isfinite(angle) and np.isfinite(bx) and np.isfinite(by) and np.isfinite(J)):
            continue

        out.append(
            Rec(
                speed=speed, angle=angle, bx=bx, by=by, ok=ok,
                sx=float(start_off[0]), sy=float(start_off[1]), sz=float(start_off[2]),
                J=J, peak_abs_ddq=peak, total_T=total_T
            )
        )

    return out, meta


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------
# Helpers
# ---------------------------

def unique_sorted(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    return np.array(sorted(set(vals.tolist())))


def nearest_value(vals: np.ndarray, target: float) -> float:
    vals = np.asarray(vals)
    idx = int(np.argmin(np.abs(vals - target)))
    return float(vals[idx])


def group_min_J_by_keys(
    bx: np.ndarray, by: np.ndarray, J: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each unique (bx,by), compute min J.
    Returns: unique_bx, unique_by, grid_minJ with shape (len(by), len(bx)).
    """
    ux = unique_sorted(bx)
    uy = unique_sorted(by)
    grid = np.full((len(uy), len(ux)), np.nan, dtype=float)

    # map value->index (works if exact floats are consistent; if not, binning is needed)
    ix = {v: i for i, v in enumerate(ux.tolist())}
    iy = {v: i for i, v in enumerate(uy.tolist())}

    for x, y, j in zip(bx, by, J):
        xi = ix.get(float(x), None)
        yi = iy.get(float(y), None)
        if xi is None or yi is None:
            continue
        cur = grid[yi, xi]
        if not np.isfinite(cur) or j < cur:
            grid[yi, xi] = j

    return ux, uy, grid


def group_median_startoff_and_minJ(
    bx: np.ndarray, by: np.ndarray, sx: np.ndarray, sy: np.ndarray, J: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each unique (bx,by):
      - median start_off x,y
      - min J
    Returns:
      ux, uy, grid_med_sx, grid_med_sy, grid_minJ  (all shaped (len(uy), len(ux)))
    """
    ux = unique_sorted(bx)
    uy = unique_sorted(by)
    ix = {v: i for i, v in enumerate(ux.tolist())}
    iy = {v: i for i, v in enumerate(uy.tolist())}

    # collect lists per cell
    cells = [[[] for _ in range(len(ux))] for __ in range(len(uy))]
    cellsJ = [[[] for _ in range(len(ux))] for __ in range(len(uy))]

    for x, y, xs, ys, j in zip(bx, by, sx, sy, J):
        xi = ix.get(float(x), None)
        yi = iy.get(float(y), None)
        if xi is None or yi is None:
            continue
        cells[yi][xi].append((xs, ys))
        cellsJ[yi][xi].append(j)

    grid_sx = np.full((len(uy), len(ux)), np.nan, dtype=float)
    grid_sy = np.full((len(uy), len(ux)), np.nan, dtype=float)
    grid_minJ = np.full((len(uy), len(ux)), np.nan, dtype=float)

    for yi in range(len(uy)):
        for xi in range(len(ux)):
            pts = cells[yi][xi]
            js = cellsJ[yi][xi]
            if not pts:
                continue
            arr = np.array(pts, dtype=float)
            grid_sx[yi, xi] = float(np.median(arr[:, 0]))
            grid_sy[yi, xi] = float(np.median(arr[:, 1]))
            grid_minJ[yi, xi] = float(np.min(np.array(js, dtype=float)))

    return ux, uy, grid_sx, grid_sy, grid_minJ


def plot_heatmap_ball(grid: np.ndarray, ux: np.ndarray, uy: np.ndarray, title: str, outpath: Path):
    plt.figure()
    # extent: [xmin,xmax,ymin,ymax] for imshow; show y increasing upward
    extent = [ux[0], ux[-1], uy[0], uy[-1]]
    plt.imshow(grid, origin="lower", aspect="auto", extent=extent)
    plt.colorbar(label="min cost J")
    plt.xlabel("ball_x_offset")
    plt.ylabel("ball_y_offset")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_scatter_startxy(sx: np.ndarray, sy: np.ndarray, J: np.ndarray, title: str, outpath: Path):
    plt.figure()
    plt.scatter(sx, sy, c=J, s=18)
    plt.colorbar(label="cost J")
    plt.xlabel("start_off x")
    plt.ylabel("start_off y")
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_quiver_ball(ux: np.ndarray, uy: np.ndarray, grid_sx: np.ndarray, grid_sy: np.ndarray, grid_c: np.ndarray,
                     title: str, outpath: Path):
    """
    Quiver over (ball_x, ball_y) showing median start_off (x,y).
    Arrow color = best cost J.
    """
    # make mesh of ball coordinates
    X, Y = np.meshgrid(ux, uy)
    U = grid_sx
    V = grid_sy

    # mask missing
    mask = np.isfinite(U) & np.isfinite(V) & np.isfinite(grid_c)

    plt.figure()
    # Use scatter as color legend (since quiver colormap support is finicky across backends)
    sc = plt.scatter(X[mask], Y[mask], c=grid_c[mask], s=28)
    plt.colorbar(sc, label="min cost J")

    # scale arrows so they are visible (data-dependent)
    # We plot arrows anchored at ball positions, pointing towards median start_off (x,y) direction in offset-space.
    plt.quiver(
        X[mask], Y[mask],
        U[mask], V[mask],
        angles="xy", scale_units="xy", scale=1.0, width=0.003
    )

    plt.xlabel("ball_x_offset")
    plt.ylabel("ball_y_offset")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_heatmap_speed_angle(grid: np.ndarray, us: np.ndarray, ua: np.ndarray, title: str, outpath: Path):
    plt.figure()
    extent = [us[0], us[-1], ua[0], ua[-1]]
    plt.imshow(grid, origin="lower", aspect="auto", extent=extent)
    plt.colorbar(label="min cost J")
    plt.xlabel("impact_speed")
    plt.ylabel("impact_angle_deg")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def minJ_grid_speed_angle_for_ball(recs: list[Rec], bx0: float, by0: float, tol: float = 1e-9):
    sel = [r for r in recs if abs(r.bx - bx0) <= tol and abs(r.by - by0) <= tol]
    if not sel:
        return None

    speeds = unique_sorted(np.array([r.speed for r in sel]))
    angles = unique_sorted(np.array([r.angle for r in sel]))
    grid = np.full((len(angles), len(speeds)), np.nan, dtype=float)

    ispeed = {v: i for i, v in enumerate(speeds.tolist())}
    iangle = {v: i for i, v in enumerate(angles.tolist())}

    for r in sel:
        xi = ispeed[r.speed]
        yi = iangle[r.angle]
        cur = grid[yi, xi]
        if not np.isfinite(cur) or r.J < cur:
            grid[yi, xi] = r.J

    return speeds, angles, grid


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True, help="Path to merged JSONL (or single shard JSONL).")
    ap.add_argument("--outdir", type=str, default="", help="Output directory (default: ./plots_<timestamp>/).")
    ap.add_argument("--only-ok", action="store_true", help="Use only ok=true records (recommended).")
    ap.add_argument("--speed", type=float, default=np.nan, help="Speed slice (nearest available will be used).")
    ap.add_argument("--angle", type=float, default=np.nan, help="Angle slice (nearest available will be used).")
    ap.add_argument("--ball-x", type=float, default=np.nan, help="Ball-x slice for speed-angle heatmap (nearest used).")
    ap.add_argument("--ball-y", type=float, default=np.nan, help="Ball-y slice for speed-angle heatmap (nearest used).")
    ap.add_argument("--tol", type=float, default=1e-9, help="Float matching tolerance for ball offsets.")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    recs, meta = load_records(jsonl_path, only_ok=args.only_ok or False)

    if not recs:
        raise SystemExit("No records loaded (check --only-ok / file path).")

    outdir = Path(args.outdir) if args.outdir else Path(f"plots_{ts()}")
    outdir.mkdir(parents=True, exist_ok=True)

    # arrays
    speed_all = np.array([r.speed for r in recs], dtype=float)
    angle_all = np.array([r.angle for r in recs], dtype=float)
    bx_all = np.array([r.bx for r in recs], dtype=float)
    by_all = np.array([r.by for r in recs], dtype=float)
    sx_all = np.array([r.sx for r in recs], dtype=float)
    sy_all = np.array([r.sy for r in recs], dtype=float)
    J_all = np.array([r.J for r in recs], dtype=float)

    speeds = unique_sorted(speed_all)
    angles = unique_sorted(angle_all)
    bxs = unique_sorted(bx_all)
    bys = unique_sorted(by_all)

    # Choose default slice values if not provided
    if not np.isfinite(args.speed):
        args.speed = float(np.median(speeds))
    if not np.isfinite(args.angle):
        args.angle = float(np.median(angles))
    if not np.isfinite(args.ball_x):
        args.ball_x = float(np.median(bxs))
    if not np.isfinite(args.ball_y):
        args.ball_y = float(np.median(bys))

    speed0 = nearest_value(speeds, args.speed)
    angle0 = nearest_value(angles, args.angle)
    bx0 = nearest_value(bxs, args.ball_x)
    by0 = nearest_value(bys, args.ball_y)

    # -----------------------
    # (A) Ball grid plots for fixed (speed, angle)
    # -----------------------
    sel_sa = np.array([(abs(r.speed - speed0) < 1e-12 and abs(r.angle - angle0) < 1e-12) for r in recs], dtype=bool)
    if np.any(sel_sa):
        bx = bx_all[sel_sa]
        by = by_all[sel_sa]
        sx = sx_all[sel_sa]
        sy = sy_all[sel_sa]
        J = J_all[sel_sa]

        ux, uy, grid_minJ = group_min_J_by_keys(bx, by, J)
        title = f"Best (min) cost over ball start grid @ speed={speed0:.3f}, angle={angle0:.3f}"
        plot_heatmap_ball(grid_minJ, ux, uy, title, outdir / f"ballgrid_minJ_speed{speed0:.3f}_angle{angle0:.3f}.png")

        title = f"Start_off (x,y) colored by cost @ speed={speed0:.3f}, angle={angle0:.3f}"
        plot_scatter_startxy(sx, sy, J, title, outdir / f"startxy_scatter_cost_speed{speed0:.3f}_angle{angle0:.3f}.png")

        ux, uy, grid_med_sx, grid_med_sy, grid_minJ2 = group_median_startoff_and_minJ(bx, by, sx, sy, J)
        title = f"Median start_off (x,y) per ball cell (quiver), colored by best cost @ speed={speed0:.3f}, angle={angle0:.3f}"
        plot_quiver_ball(ux, uy, grid_med_sx, grid_med_sy, grid_minJ2, title,
                         outdir / f"ballgrid_quiver_medStart_cost_speed{speed0:.3f}_angle{angle0:.3f}.png")

        # Also show a heatmap of median start magnitude if useful
        mag = np.sqrt(grid_med_sx**2 + grid_med_sy**2)
        plot_heatmap_ball(mag, ux, uy,
                          f"Median |start_off(x,y)| per ball cell @ speed={speed0:.3f}, angle={angle0:.3f}",
                          outdir / f"ballgrid_medStartMag_speed{speed0:.3f}_angle{angle0:.3f}.png")

    # -----------------------
    # (B) Speed-angle heatmap for fixed (ball_x, ball_y)
    # -----------------------
    sa_grid = minJ_grid_speed_angle_for_ball(recs, bx0, by0, tol=args.tol)
    if sa_grid is not None:
        us, ua, grid = sa_grid
        title = f"Best (min) cost over (speed, angle) @ ball=({bx0:.3f}, {by0:.3f})"
        plot_heatmap_speed_angle(grid, us, ua, title, outdir / f"speed_angle_minJ_ballx{bx0:.3f}_bally{by0:.3f}.png")

    # quick info file
    info = outdir / "info.txt"
    with info.open("w", encoding="utf-8") as f:
        f.write(f"jsonl: {jsonl_path}\n")
        f.write(f"n_records: {len(recs)}\n")
        f.write(f"chosen speed/angle: {speed0}, {angle0}\n")
        f.write(f"chosen ball_x/ball_y: {bx0}, {by0}\n")
        if meta:
            f.write("\nmeta (last seen):\n")
            f.write(json.dumps(meta, indent=2))
            f.write("\n")

    print(f"[DONE] Wrote plots to: {outdir.resolve()}")
    print(f"[DONE] Info: {info.resolve()}")


if __name__ == "__main__":
    main()
