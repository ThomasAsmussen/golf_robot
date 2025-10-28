#!/usr/bin/env python3
"""
Green mesh with a circular cup opening and tiny thickness to avoid Qhull coplanarity.
- Top surface at z=0 (optional funnel annulus).
- Bottom surface at z=-thickness (no funnel).
- Any triangle whose vertex or edge intersects the cup radius+margin is removed.
"""

import argparse, os, sys
from dataclasses import dataclass
import numpy as np

@dataclass
class Params:
    sx: float = 6.0
    sy: float = 4.0
    nx: int = 240
    ny: int = 160
    hole_x: float = 2.0
    hole_y: float = 0.0
    hole_r: float = 0.055
    hole_margin: float = 0.003   # extra clearance to guarantee a clean opening
    thickness: float = 0.02      # bottom layer offset (meters)
    funnel_on: bool = True
    funnel_width: float = 0.03
    funnel_depth: float = 0.005
    out: str = "green_with_cup.obj"

def _cos_smoothstep01(u):
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(u, 0.0, 1.0)))

def build_grid(p: Params):
    xs = np.linspace(-p.sx/2, p.sx/2, p.nx)
    ys = np.linspace(-p.sy/2, p.sy/2, p.ny)
    xx, yy = np.meshgrid(xs, ys)
    zz_top = np.zeros_like(xx)

    if p.funnel_on and p.funnel_width > 0 and p.funnel_depth > 0:
        dx, dy = xx - p.hole_x, yy - p.hole_y
        rr = np.sqrt(dx*dx + dy*dy)
        inner = p.hole_r
        outer = p.hole_r + p.funnel_width
        annulus = (rr >= inner) & (rr <= outer)
        u = (rr - inner) / max(outer - inner, 1e-9)
        prof = _cos_smoothstep01(1.0 - u)          # 1 at inner rim -> 0 at outer
        zz_top = zz_top - (annulus * (p.funnel_depth * prof))

    zz_bot = zz_top - p.thickness  # copy top (including any funnel) downward
    # If you prefer bottom perfectly flat, uncomment:
    # zz_bot[:] = -p.thickness
    return xx, yy, zz_top, zz_bot

def _seg_circle_intersect(ax, ay, bx, by, cx, cy, r):
    # translate to circle frame
    ax -= cx; ay -= cy; bx -= cx; by -= cy
    r2 = r*r
    # endpoint inside?
    if ax*ax + ay*ay <= r2 or bx*bx + by*by <= r2:
        return True
    # project origin onto segment
    abx, aby = bx - ax, by - ay
    ab2 = abx*abx + aby*aby
    if ab2 <= 1e-15:
        return False
    t = -(ax*abx + ay*aby) / ab2
    if t <= 0 or t >= 1:
        return False
    qx = ax + t*abx
    qy = ay + t*aby
    return (qx*qx + qy*qy) <= r2

def _tri_intersects_circle(ax, ay, bx, by, cx, cy, cx0, cy0, r):
    r2 = r*r
    if (ax-cx0)**2 + (ay-cy0)**2 <= r2: return True
    if (bx-cx0)**2 + (by-cy0)**2 <= r2: return True
    if (cx-cx0)**2 + (cy-cy0)**2 <= r2: return True
    if _seg_circle_intersect(ax, ay, bx, by, cx0, cy0, r): return True
    if _seg_circle_intersect(bx, by, cx, cy, cx0, cy0, r): return True
    if _seg_circle_intersect(cx, cy, ax, ay, cx0, cy0, r): return True
    return False

def build_faces(xx, yy, hole_x, hole_y, hole_r_eff):
    ny, nx = xx.shape
    def vid(i, j): return i * nx + j
    faces = []
    dropped = 0
    for i in range(ny - 1):
        for j in range(nx - 1):
            a = vid(i, j);     b = vid(i, j+1)
            c = vid(i+1, j);   d = vid(i+1, j+1)
            ax, ay = xx.flat[a], yy.flat[a]
            bx, by = xx.flat[b], yy.flat[b]
            cx_, cy_ = xx.flat[c], yy.flat[c]
            dx, dy = xx.flat[d], yy.flat[d]
            # tri1: (a,b,d)
            if not _tri_intersects_circle(ax, ay, bx, by, dx, dy, hole_x, hole_y, hole_r_eff):
                faces.append((a, b, d))
            else:
                dropped += 1
            # tri2: (a,d,c)
            if not _tri_intersects_circle(ax, ay, dx, dy, cx_, cy_, hole_x, hole_y, hole_r_eff):
                faces.append((a, d, c))
            else:
                dropped += 1
    return faces, dropped

def write_obj(path, top_verts, bot_verts, faces_top):
    """
    Write OBJ:
      - top vertices then bottom vertices
      - top faces as-is
      - bottom faces with reversed winding and index offset
    """
    n_top = top_verts.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# green with cup opening, extruded\n")
        for v in top_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for v in bot_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # top faces
        for (a, b, c) in faces_top:
            f.write(f"f {a+1} {b+1} {c+1}\n")
        # bottom faces (reverse winding)
        for (a, b, c) in faces_top:
            f.write(f"f {c+1+n_top} {b+1+n_top} {a+1+n_top}\n")

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Generate a slightly thick green mesh with a circular opening.")
    ap.add_argument("--out", type=str, default="green_with_cup.obj")
    ap.add_argument("--sx", type=float, default=6.0)
    ap.add_argument("--sy", type=float, default=4.0)
    ap.add_argument("--nx", type=int, default=240)
    ap.add_argument("--ny", type=int, default=160)
    ap.add_argument("--hole-x", type=float, default=2.0)
    ap.add_argument("--hole-y", type=float, default=0.0)
    ap.add_argument("--hole-r", type=float, default=0.055)
    ap.add_argument("--hole-margin", type=float, default=0.003)
    ap.add_argument("--thickness", type=float, default=0.02)
    ap.add_argument("--funnel-on", action="store_true")
    ap.add_argument("--no-funnel", action="store_true")
    ap.add_argument("--funnel-width", type=float, default=0.03)
    ap.add_argument("--funnel-depth", type=float, default=0.005)
    args = ap.parse_args(argv)

    if args.nx < 3 or args.ny < 3:
        print("nx and ny must be >= 3", file=sys.stderr); return 2
    funnel_on = True
    if args.funnel_on and args.no_funnel: funnel_on = True
    elif args.funnel_on: funnel_on = True
    elif args.no_funnel: funnel_on = False

    p = Params(
        sx=args.sx, sy=args.sy, nx=args.nx, ny=args.ny,
        hole_x=args.hole_x, hole_y=args.hole_y, hole_r=args.hole_r,
        hole_margin=args.hole_margin, thickness=args.thickness,
        funnel_on=funnel_on, funnel_width=args.funnel_width, funnel_depth=args.funnel_depth,
        out=args.out
    )

    xx, yy, zz_top, zz_bot = build_grid(p)
    faces_top, dropped = build_faces(xx, yy, p.hole_x, p.hole_y, p.hole_r + p.hole_margin)

    # pack vertices
    top_verts = np.stack([xx, yy, zz_top], axis=-1).reshape(-1, 3)
    bot_verts = np.stack([xx, yy, zz_bot], axis=-1).reshape(-1, 3)

    os.makedirs(os.path.dirname(p.out) or ".", exist_ok=True)
    write_obj(p.out, top_verts, bot_verts, faces_top)

    print(f"Wrote OBJ: {p.out}")
    print(f"  top verts: {top_verts.shape[0]:,}, bottom verts: {bot_verts.shape[0]:,}")
    print(f"  faces per layer: {len(faces_top):,} (dropped near hole: {dropped:,})")
    print(f"  thickness: {p.thickness} m, hole_r: {p.hole_r} m, margin: {p.hole_margin} m")
    print(f"  funnel: {'ON' if p.funnel_on else 'OFF'} (width={p.funnel_width}, depth={p.funnel_depth})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
