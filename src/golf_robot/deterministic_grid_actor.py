from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # allow pure-numpy usage too
    torch = None
    nn = object


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


@dataclass
class HoleGrid:
    gx: np.ndarray  # shape [nx]
    gy: np.ndarray  # shape [ny]
    grid: np.ndarray  # shape [ny, nx, 2]  (action0, action1)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return float(self.gx[0]), float(self.gx[-1]), float(self.gy[0]), float(self.gy[-1])


class DeterministicGridActor(nn.Module):
    """
    Torch-friendly actor (nn.Module) with deterministic forward().
    You can also call .act_numpy(...) without torch.

    forward() expects either:
      A) state tensor shaped [B, state_dim] where you provide hole_id separately, OR
      B) (hole_id, ball_start_x, ball_start_y) via act_numpy().
    """
    def __init__(self, hole_grids: Dict[int, HoleGrid]):
        super().__init__()
        self.hole_grids = hole_grids

    # -----------------------
    # Core interpolation
    # -----------------------
    @staticmethod
    def _bilinear_action(hg: HoleGrid, x: float, y: float) -> np.ndarray:
        gx, gy, G = hg.gx, hg.gy, hg.grid
        x0, x1, y0, y1 = hg.bounds

        # clamp to bounds
        x = _clamp(x, x0, x1)
        y = _clamp(y, y0, y1)

        # find cell indices i (x), j (y) such that gx[i] <= x <= gx[i+1]
        # handle edge at max by clamping i/j to last cell
        i = int(np.searchsorted(gx, x, side="right") - 1)
        j = int(np.searchsorted(gy, y, side="right") - 1)
        i = max(0, min(i, len(gx) - 2))
        j = max(0, min(j, len(gy) - 2))

        xL, xR = gx[i], gx[i + 1]
        yB, yT = gy[j], gy[j + 1]

        # normalized cell coordinates
        tx = 0.0 if xR == xL else (x - xL) / (xR - xL)
        ty = 0.0 if yT == yB else (y - yB) / (yT - yB)

        # grid corners (note: j is y index, i is x index)
        a00 = G[j,     i    ]  # (xL, yB)
        a10 = G[j,     i + 1]  # (xR, yB)
        a01 = G[j + 1, i    ]  # (xL, yT)
        a11 = G[j + 1, i + 1]  # (xR, yT)

        # bilinear interpolation
        a0 = (1.0 - tx) * a00 + tx * a10
        a1 = (1.0 - tx) * a01 + tx * a11
        a  = (1.0 - ty) * a0  + ty * a1
        return a.astype(np.float32)

    # -----------------------
    # User-facing API
    # -----------------------
    def act_numpy(self, hole_id: int, ball_start_x: float, ball_start_y: float) -> np.ndarray:
        if hole_id not in self.hole_grids:
            raise KeyError(f"hole_id={hole_id} not in grids. Available: {sorted(self.hole_grids.keys())}")
        hg = self.hole_grids[hole_id]
        return self._bilinear_action(hg, float(ball_start_x), float(ball_start_y))

    def forward(self, state, hole_id: Optional[int] = None):
        """
        If you want torch usage: actor(state, hole_id=2) -> action tensor [B,2].

        - state is expected to contain ball_start_x, ball_start_y as the first two entries.
          (Adjust indexing below if your state differs.)
        """
        if torch is None:
            raise RuntimeError("PyTorch not available; use act_numpy().")
        if hole_id is None:
            raise ValueError("Provide hole_id=... for forward().")

        # state: [B, state_dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)

        bsx = state[:, 0].detach().cpu().numpy()
        bsy = state[:, 1].detach().cpu().numpy()

        out = np.stack([self.act_numpy(int(hole_id), float(x), float(y)) for x, y in zip(bsx, bsy)], axis=0)
        return torch.from_numpy(out).to(state.device)


def load_grids_npz(npz_path: str | Path) -> Dict[int, HoleGrid]:
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=False)
    hole_ids = sorted({int(k.split("_")[0].replace("hole","")) for k in data.files if k.endswith("_grid")})
    hole_grids: Dict[int, HoleGrid] = {}
    for h in hole_ids:
        gx = data[f"hole{h}_gx"]
        gy = data[f"hole{h}_gy"]
        grid = data[f"hole{h}_grid"]
        hole_grids[h] = HoleGrid(gx=gx, gy=gy, grid=grid)
    return hole_grids


def make_actor_from_npz(npz_path: str | Path) -> DeterministicGridActor:
    return DeterministicGridActor(load_grids_npz(npz_path))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str, required=True)
    p.add_argument("--hole", type=int, required=True)
    p.add_argument("--x", type=float, required=True)
    p.add_argument("--y", type=float, required=True)
    args = p.parse_args()

    actor = make_actor_from_npz(args.npz)
    a = actor.act_numpy(args.hole, args.x, args.y)
    print("action_norm:", a.tolist())
