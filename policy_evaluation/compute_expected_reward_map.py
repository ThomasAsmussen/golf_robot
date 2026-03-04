import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

try:
    from scipy.io import savemat
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# =========================
# CONFIG (EDIT THESE)
# =========================
TAG = "golf_world_tmp_14811_6393a6c6a3f147638d204c186ccfc8b1"
K = 8

STATE_DIM = 4
ACTION_DIM = 2
HIDDEN_DIM = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Grid ranges (meters) ----
BALL_X_MIN, BALL_X_MAX = -0.172, 0.19
BALL_Y_MIN, BALL_Y_MAX = -0.077, 0.107
NX, NY = 140, 90

# ---- Action unscale (set from rl_cfg["model"]) ----
SPEED_LOW, SPEED_HIGH = 1.0, 2.0       # <-- EDIT
ANGLE_LOW, ANGLE_HIGH = -20.0, 20.0    # <-- EDIT

# ---- Hole config ----
HOLE_CONFIG_PATH = "hole_config.yaml"  # holes 1-3 will be used

# ---- CEM settings ----
CEM_ITERS = 2
CEM_POP = 384
CEM_ELITE_FRAC = 0.16
ACTION_LOW, ACTION_HIGH = -1.0, 1.0
INIT_STD = 0.75
MIN_STD = 0.15

STATE_BATCH = 2048  # bigger = faster if memory allows

OUT_DATA_DIR = "computed_maps"
OUT_PREFIX = "planned_maps"
# =========================


# =========================
# Bounds (match rl_common.py scaling)
# =========================
MIN_HOLE_X, MAX_HOLE_X = 3.0, 4.0
MIN_HOLE_Y, MAX_HOLE_Y = -0.5, 0.5

MIN_BALL_X, MAX_BALL_X = -0.5, 0.5
MIN_BALL_Y, MAX_BALL_Y = -0.5, 0.5


def scale_to_unit(x, lo, hi):
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def squash_to_range(x_unit, lo, hi):
    return lo + 0.5 * (x_unit + 1.0) * (hi - lo)


def build_state_norm_batch(ball_xy_batch, hole_xy):
    """
    ball_xy_batch: [B,2] meters
    hole_xy: [2] meters (torch)
    returns [B,4] normalized [-1,1]
    """
    bx = ball_xy_batch[:, 0]
    by = ball_xy_batch[:, 1]
    hx = hole_xy[0].expand_as(bx)
    hy = hole_xy[1].expand_as(by)

    bx_s = scale_to_unit(bx, MIN_BALL_X, MAX_BALL_X)
    by_s = scale_to_unit(by, MIN_BALL_Y, MAX_BALL_Y)
    hx_s = scale_to_unit(hx, MIN_HOLE_X, MAX_HOLE_X)
    hy_s = scale_to_unit(hy, MIN_HOLE_Y, MAX_HOLE_Y)

    return torch.stack([bx_s, by_s, hx_s, hy_s], dim=1).to(torch.float32)


class Critic(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


def autodetect_pth_dir(tag: str) -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here, here / "rl_model", here / "models", here / "model",
        Path.cwd(), Path.cwd() / "rl_model", Path.cwd() / "models", Path.cwd() / "model",
    ]
    for d in candidates:
        if (d / f"ddpg_critic1_{tag}_h0.pth").is_file() and (d / f"ddpg_critic2_{tag}_h0.pth").is_file():
            return d
    tried = "\n".join(str(x) for x in candidates)
    raise FileNotFoundError(
        f"Could not find critic checkpoints for TAG={tag}\nTried:\n{tried}\n"
        "Put the .pth files next to this script or in ./model, ./models, or ./rl_model."
    )


def load_ensemble_double_critics(pth_dir: Path, tag, k, state_dim, action_dim, hidden_dim, device):
    critics1, critics2 = [], []
    for h in range(k):
        f1 = pth_dir / f"ddpg_critic1_{tag}_h{h}.pth"
        f2 = pth_dir / f"ddpg_critic2_{tag}_h{h}.pth"
        if not f1.is_file() or not f2.is_file():
            raise FileNotFoundError(f"Missing files for head {h}:\n  {f1}\n  {f2}")

        c1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        c2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        c1.load_state_dict(torch.load(str(f1), map_location=device))
        c2.load_state_dict(torch.load(str(f2), map_location=device))
        c1.eval(); c2.eval()
        critics1.append(c1)
        critics2.append(c2)
    return critics1, critics2


@torch.no_grad()
def score_median_minQ(critics1, critics2, s_flat, a_flat):
    vals = []
    for c1, c2 in zip(critics1, critics2):
        q1 = c1(s_flat, a_flat).squeeze(-1)
        q2 = c2(s_flat, a_flat).squeeze(-1)
        vals.append(torch.minimum(q1, q2))
    v = torch.stack(vals, dim=0)      # [K, N]
    return v.median(dim=0).values     # [N]


@torch.no_grad()
def cem_plan_action_batched(
    s_batch, score_fn, action_dim, device,
    cem_iters=2, cem_pop=256, cem_elite_frac=0.2,
    action_low=-1.0, action_high=1.0,
    init_std=0.4, min_std=0.1
):
    """
    Batched CEM over B states.
    Returns:
      best_a: [B, action_dim]
      best_score: [B]
    """
    B = s_batch.shape[0]
    elite_n = max(1, int(round(cem_pop * cem_elite_frac)))

    mu = torch.zeros(B, action_dim, device=device, dtype=torch.float32)
    std = torch.ones(B, action_dim, device=device, dtype=torch.float32) * float(init_std)

    best_a = torch.zeros(B, action_dim, device=device, dtype=torch.float32)
    best_score = torch.full((B,), -float("inf"), device=device, dtype=torch.float32)

    for _ in range(int(cem_iters)):
        a = mu[:, None, :] + std[:, None, :] * torch.randn(B, cem_pop, action_dim, device=device)
        a = torch.clamp(a, float(action_low), float(action_high))

        a_flat = a.reshape(B * cem_pop, action_dim)
        s_flat = s_batch[:, None, :].expand(B, cem_pop, s_batch.shape[1]).reshape(B * cem_pop, s_batch.shape[1])

        scores_flat = score_fn(s_flat, a_flat)
        scores = scores_flat.reshape(B, cem_pop)

        topk_vals, topk_idx = torch.topk(scores, k=elite_n, dim=1, largest=True)
        elite_a = a.gather(1, topk_idx[:, :, None].expand(B, elite_n, action_dim))

        mu = elite_a.mean(dim=1)
        if elite_n > 1:
            std = elite_a.std(dim=1).clamp(min=float(min_std))
        else:
            std = (std * 0.9).clamp(min=float(min_std))

        iter_best = topk_vals[:, 0]
        iter_best_a = elite_a[:, 0, :]
        improved = iter_best > best_score
        best_score = torch.where(improved, iter_best, best_score)
        best_a = torch.where(improved[:, None], iter_best_a, best_a)

    return best_a, best_score


def load_holes_from_yaml(path: str):
    p = Path(path)
    if not p.is_file():
        p = Path(__file__).resolve().parent / path
    if not p.is_file():
        raise FileNotFoundError(f"Could not find {path}. Put it next to this script or set HOLE_CONFIG_PATH.")
    if not HAVE_YAML:
        raise RuntimeError("PyYAML not installed. Install with: pip install pyyaml")

    holes_raw = yaml.safe_load(p.read_text())
    holes = {}
    for k, v in holes_raw.items():
        holes[int(k)] = (float(v["x"]), float(v["y"]))
    return holes, str(p)


def main():
    os.makedirs(OUT_DATA_DIR, exist_ok=True)
    device = torch.device(DEVICE)
    print("DEVICE =", device)

    holes_all, hole_cfg_path = load_holes_from_yaml(HOLE_CONFIG_PATH)
    print("Loaded holes from:", hole_cfg_path)

    # only holes 1-3
    holes = {k: holes_all[k] for k in [1, 2, 3] if k in holes_all}
    print("Using holes:", holes)

    pth_dir = autodetect_pth_dir(TAG)
    print("Using PTH_DIR =", str(pth_dir))

    critics1, critics2 = load_ensemble_double_critics(
        pth_dir, TAG, K, STATE_DIM, ACTION_DIM, HIDDEN_DIM, device
    )
    print(f"Loaded {K} heads.")

    xlin = np.linspace(BALL_X_MIN, BALL_X_MAX, NX, dtype=np.float32)
    ylin = np.linspace(BALL_Y_MIN, BALL_Y_MAX, NY, dtype=np.float32)
    X, Y = np.meshgrid(xlin, ylin)
    grid_xy = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)
    N = grid_xy.shape[0]
    print(f"Grid states: {N} (= {NX} x {NY})")

    def score_fn(s_flat, a_flat):
        return score_median_minQ(critics1, critics2, s_flat, a_flat)

    for hole_id, (hx, hy) in holes.items():
        hole_name = f"hole{hole_id}"
        hole_xy = torch.tensor([hx, hy], device=device, dtype=torch.float32)
        print(f"\n=== {hole_name}: hole_xy={[hx, hy]} ===")

        V_all = np.zeros((N,), dtype=np.float32)
        A_all = np.zeros((N, ACTION_DIM), dtype=np.float32)

        for start in range(0, N, STATE_BATCH):
            end = min(N, start + STATE_BATCH)
            bxy = torch.tensor(grid_xy[start:end], device=device, dtype=torch.float32)
            s_batch = build_state_norm_batch(bxy, hole_xy)

            best_a, best_v = cem_plan_action_batched(
                s_batch, score_fn, ACTION_DIM, device,
                cem_iters=CEM_ITERS,
                cem_pop=CEM_POP,
                cem_elite_frac=CEM_ELITE_FRAC,
                action_low=ACTION_LOW,
                action_high=ACTION_HIGH,
                init_std=INIT_STD,
                min_std=MIN_STD
            )

            V_all[start:end] = best_v.detach().cpu().numpy()
            A_all[start:end, :] = best_a.detach().cpu().numpy()

            if (start // STATE_BATCH) % 10 == 0:
                print(f"{hole_name}: {start}-{end}/{N}")

        V_map = V_all.reshape(NY, NX)
        a_norm_map = A_all.reshape(NY, NX, ACTION_DIM)

        speed_map = squash_to_range(a_norm_map[:, :, 0], SPEED_LOW, SPEED_HIGH)
        angle_map = squash_to_range(a_norm_map[:, :, 1], ANGLE_LOW, ANGLE_HIGH)

        out_npz = Path(OUT_DATA_DIR) / f"{OUT_PREFIX}_{hole_name}.npz"
        np.savez(
            out_npz,
            xlin=xlin, ylin=ylin,
            V_map=V_map,
            a_norm_map=a_norm_map,
            speed_map=speed_map,
            angle_map=angle_map,
            hole_xy=np.array([hx, hy], dtype=np.float32),
            tag=TAG, K=K,
            SPEED_LOW=SPEED_LOW, SPEED_HIGH=SPEED_HIGH,
            ANGLE_LOW=ANGLE_LOW, ANGLE_HIGH=ANGLE_HIGH,
            grid_bounds=np.array([BALL_X_MIN, BALL_X_MAX, BALL_Y_MIN, BALL_Y_MAX], dtype=np.float32),
            cem=np.array([CEM_ITERS, CEM_POP, CEM_ELITE_FRAC, INIT_STD, MIN_STD], dtype=np.float32),
        )
        print("Saved:", out_npz)

        if HAVE_SCIPY:
            out_mat = Path(OUT_DATA_DIR) / f"{OUT_PREFIX}_{hole_name}.mat"
            savemat(
                out_mat,
                {
                    "xlin": xlin, "ylin": ylin,
                    "V_map": V_map,
                    "a_norm_map": a_norm_map,
                    "speed_map": speed_map,
                    "angle_map": angle_map,
                    "hole_xy": np.array([hx, hy], dtype=np.float32),
                    "tag": TAG, "K": K,
                }
            )
            print("Saved:", out_mat)

    print("\nDONE. Computed data saved in:", OUT_DATA_DIR)


if __name__ == "__main__":
    main()