#!/usr/bin/env python3
"""
Generate a hollow tube (cup wall) as an OBJ mesh:
- Inner radius r_in
- Wall thickness t  (outer radius = r_in + t)
- Height H (top at z=0, bottom at z=-H)
- N segments around

The mesh is open at the top (like a real cup rim) and open at the bottom.
Use a separate primitive (or mesh) for the bottom disc in MuJoCo.
"""

import argparse
import math
import os
import sys

def ring_xy(r, N):
    return [(r*math.cos(2*math.pi*k/N), r*math.sin(2*math.pi*k/N)) for k in range(N)]

def write_obj(path, verts, faces):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# cup tube (hollow cylinder)\n")
        for x,y,z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a,b,c in faces:
            # OBJ is 1-based
            f.write(f"f {a+1} {b+1} {c+1}\n")

def build_tube(r_in=0.054, t=0.004, H=0.11, N=64):
    r_out = r_in + t
    # Vertex order (top z=0, bottom z=-H):
    # outer top ring, inner top ring, outer bottom ring, inner bottom ring
    ot = [(x, y, 0.0)     for (x,y) in ring_xy(r_out, N)]
    it = [(x, y, 0.0)     for (x,y) in ring_xy(r_in,  N)]
    ob = [(x, y, -H)      for (x,y) in ring_xy(r_out, N)]
    ib = [(x, y, -H)      for (x,y) in ring_xy(r_in,  N)]
    verts = ot + it + ob + ib

    def idx(base, k): return base + k
    faces = []

    # OUTER wall (CCW when viewed from outside)
    # ot[k] -> ot[k+1] -> ob[k+1],  ot[k] -> ob[k+1] -> ob[k]
    for k in range(N):
        k1 = (k+1) % N
        a = idx(0, k)          # ot
        b = idx(0, k1)
        c = idx(0+N+N, k1)     # ob
        d = idx(0+N+N, k)
        faces.append((a, b, c))
        faces.append((a, c, d))

    # INNER wall (faces inward; reverse winding relative to outer)
    # it[k] -> ib[k+1] -> it[k+1],  it[k] -> ib[k] -> ib[k+1]
    IT = N      # inner top base
    OB = 2*N    # outer bottom base
    IB = 3*N    # inner bottom base
    for k in range(N):
        k1 = (k+1) % N
        it_k  = idx(IT, k)
        it_k1 = idx(IT, k1)
        ib_k  = idx(IB, k)
        ib_k1 = idx(IB, k1)
        faces.append((it_k, ib_k1, it_k1))
        faces.append((it_k, ib_k,  ib_k1))

    return verts, faces

def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate hollow cup tube OBJ.")
    ap.add_argument("--out", type=str, default="cup_tube.obj")
    ap.add_argument("--r-in", type=float, default=0.054, help="Inner radius (m)")
    ap.add_argument("--thickness", type=float, default=0.004, help="Wall thickness (m)")
    ap.add_argument("--height", type=float, default=0.11, help="Tube height (m)")
    ap.add_argument("--segments", type=int, default=64, help="Circumferential segments")
    args = ap.parse_args(argv)

    verts, faces = build_tube(args.r_in, args.thickness, args.height, args.segments)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_obj(args.out, verts, faces)
    print(f"Wrote OBJ: {args.out}")
    print(f"  verts: {len(verts)} | faces: {len(faces)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
