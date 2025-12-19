"""
test_contextual_bandit_agent.py

Load a trained Actor (contextual bandit) and evaluate it in the simulator.

Usage examples:
  # Basic test (1000 eval episodes)
  python src/golf_robot/reinforcement_learning/test_contextual_bandit_agent.py \
    --actor models/rl/ddpg/ddpg_actor_local_run.pth \
    --episodes 1000

  # Test with up to 3 discs and save per-episode CSV
  python src/golf_robot/reinforcement_learning/test_contextual_bandit_agent.py \
    --actor models/rl/ddpg/ddpg_actor_local_run.pth \
    --episodes 2000 \
    --max-discs 3 \
    --csv log/eval_results.csv

Notes:
- Expects your usual config files at:
    configs/mujoco_config.yaml
    configs/rl_config.yaml
- Expects simulator entrypoint:
    simulation/run_sim_rl.py  (providing run_sim(angle_deg, speed, [hole_x,hole_y], mujoco_cfg, disc_positions))
"""

import os
import sys
import csv
import yaml
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Noise + ranges (same defaults)
# -----------------------------
# SPEED_NOISE_STD      = 0.1
# ANGLE_NOISE_STD      = 0.1
# BALL_OBS_NOISE_STD   = 0.002
# HOLE_OBS_NOISE_STD   = 0.002

SPEED_NOISE_STD      = 0.0
ANGLE_NOISE_STD      = 0.0
BALL_OBS_NOISE_STD   = 0.0
HOLE_OBS_NOISE_STD   = 0.0

MIN_HOLE_X = 3.0
MAX_HOLE_X = 4.0
MIN_HOLE_Y = -0.5
MAX_HOLE_Y = 0.5
MIN_BALL_X = -0.5
MAX_BALL_X = 0.5
MIN_BALL_Y = -0.5
MAX_BALL_Y = 0.5


# -----------------------------
# Model (same as training)
# -----------------------------
class Actor(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


# -----------------------------
# Helpers: scaling + encoding
# -----------------------------
def squash_to_range(x, lo, hi):
    """Squash value in [-1,1] to [lo,hi]."""
    return lo + 0.5 * (x + 1.0) * (hi - lo)

def get_sim_input(a_norm, speed_low, speed_high, angle_low, angle_high):
    speed_norm, angle_norm = float(a_norm[0]), float(a_norm[1])
    speed     = squash_to_range(speed_norm,  speed_low,  speed_high)
    angle_deg = squash_to_range(angle_norm, angle_low, angle_high)
    return speed, angle_deg

def scale(x, lo, hi):
    """Scale x in [lo,hi] to [-1,1]."""
    return 2.0 * (x - lo) / (hi - lo) - 1.0

def scale_state_vec(state_vec):
    """
    Layout:
    [ ball_x, ball_y,
      hole_x, hole_y,
      disc1_x, disc1_y, disc1_present,
      ...
    ]
    """
    ball_x, ball_y = state_vec[0], state_vec[1]
    hole_x, hole_y = state_vec[2], state_vec[3]

    ball_x_scaled = scale(ball_x, MIN_BALL_X, MAX_BALL_X)
    ball_y_scaled = scale(ball_y, MIN_BALL_Y, MAX_BALL_Y)
    hole_x_scaled = scale(hole_x, MIN_HOLE_X, MAX_HOLE_X)
    hole_y_scaled = scale(hole_y, MIN_HOLE_Y, MAX_HOLE_Y)

    disc_data = state_vec[4:]
    disc_scaled = []
    for i in range(0, len(disc_data), 3):
        disc_x, disc_y, disc_present = disc_data[i], disc_data[i + 1], disc_data[i + 2]
        if disc_present == 0:
            disc_x_scaled, disc_y_scaled = -1.0, -1.0
        else:
            # same convention as your training script
            disc_x_scaled = scale(disc_x, MIN_HOLE_X - 2.0, MAX_HOLE_X)
            disc_y_scaled = scale(disc_y, MIN_HOLE_Y, MAX_HOLE_Y)
        disc_scaled.extend([disc_x_scaled, disc_y_scaled, disc_present])

    return np.array([ball_x_scaled, ball_y_scaled, hole_x_scaled, hole_y_scaled] + disc_scaled, dtype=np.float32)

def encode_state_with_discs(ball_start_obs, hole_pos_obs, disc_positions, max_num_discs):
    default_value = 0.0
    disc_coords = []
    for i in range(max_num_discs):
        if i < len(disc_positions):
            x, y = disc_positions[i]
            disc_placed = 1
        else:
            x, y = default_value, default_value
            disc_placed = 0
        disc_coords.extend([x, y, disc_placed])

    return np.concatenate([ball_start_obs, hole_pos_obs, np.array(disc_coords, dtype=np.float32)], axis=0)

def random_hole_in_rectangle(x_min=MIN_HOLE_X, x_max=MAX_HOLE_X, y_min=MIN_HOLE_Y, y_max=MAX_HOLE_Y):
    return np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)

def generate_disc_positions(max_num_discs, x_min, x_max, y_min, y_max, hole_xy):
    hole_x, hole_y = hole_xy
    num_discs = np.random.randint(0, max_num_discs + 1)
    disc_positions = []
    min_dist_from_objects = 0.25

    x_lo = x_min + min_dist_from_objects
    x_hi = x_max - min_dist_from_objects
    y_lo = y_min + min_dist_from_objects
    y_hi = y_max - min_dist_from_objects

    for _ in range(num_discs):
        placed = False
        for _ in range(100):
            x = np.random.uniform(x_lo, x_hi)
            y = np.random.uniform(y_lo, y_hi)

            if np.hypot(x - hole_x, y - hole_y) < min_dist_from_objects:
                continue

            if any(np.hypot(x - dx, y - dy) < min_dist_from_objects for (dx, dy) in disc_positions):
                continue

            disc_positions.append((x, y))
            placed = True
            break

        if not placed:
            raise RuntimeError("Could not place all discs without overlap.")

    disc_positions.sort(key=lambda p: np.hypot(p[0] - hole_x, p[1] - hole_y))
    return disc_positions

def compute_reward(ball_end_pos, hole_pos, in_hole, in_hole_reward=3.0, distance_scale=0.5):
    if int(in_hole) == 1:
        return float(in_hole_reward)
    final_dist = float(np.linalg.norm(ball_end_pos - hole_pos))
    return float(np.exp(-distance_scale * final_dist))


# -----------------------------
# Agent wrapper
# -----------------------------
class BanditAgent:
    def __init__(self, actor: Actor, device: torch.device, rl_cfg: dict):
        self.actor = actor
        self.device = device
        self.speed_low  = rl_cfg["model"]["speed_low"]
        self.speed_high = rl_cfg["model"]["speed_high"]
        self.angle_low  = rl_cfg["model"]["angle_low"]
        self.angle_high = rl_cfg["model"]["angle_high"]

    @torch.no_grad()
    def act(self, state_vec_scaled: np.ndarray) -> tuple[float, float]:
        s = torch.tensor(state_vec_scaled, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_norm = self.actor(s).squeeze(0).cpu().numpy()
        speed, angle_deg = get_sim_input(a_norm, self.speed_low, self.speed_high, self.angle_low, self.angle_high)
        return float(speed), float(angle_deg)


def load_actor(actor_path: Path, rl_cfg: dict) -> tuple[Actor, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(
        state_dim=rl_cfg["model"]["state_dim"],
        action_dim=rl_cfg["model"]["action_dim"],
        hidden=rl_cfg["model"]["hidden_dim"],
    ).to(device)
    sd = torch.load(actor_path, map_location=device)
    actor.load_state_dict(sd)
    actor.eval()
    return actor, device


# -----------------------------
# Evaluation loop
# -----------------------------
def evaluate(
    agent: BanditAgent,
    run_sim_fn,
    mujoco_cfg: dict,
    rl_cfg: dict,
    episodes: int,
    max_discs: int,
    seed: int | None,
    apply_actuator_noise: bool,
    apply_obs_noise: bool,
):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    in_hole_reward = rl_cfg["reward"]["in_hole_reward"]
    distance_scale = rl_cfg["reward"]["distance_scale"]

    successes = 0
    rewards = []
    distances = []

    per_episode = []

    for ep in range(episodes):
        # Sample context
        ball_start = np.random.uniform(-0.5, 0.5, size=(2,))
        ball_start_obs = ball_start.copy()
        if apply_obs_noise:
            ball_start_obs = ball_start + np.random.normal(0, BALL_OBS_NOISE_STD, size=(2,))

        mujoco_cfg["ball"]["start_pos"]     = [float(ball_start[0]),     float(ball_start[1]),     0.02135]
        mujoco_cfg["ball"]["obs_start_pos"] = [float(ball_start_obs[0]), float(ball_start_obs[1]), 0.02135]

        hx, hy = random_hole_in_rectangle()
        hole_pos = np.array([hx, hy], dtype=np.float32)
        hole_pos_obs = hole_pos.copy()
        if apply_obs_noise:
            hole_pos_obs = hole_pos + np.random.normal(0, HOLE_OBS_NOISE_STD, size=(2,))

        disc_positions = generate_disc_positions(
            max_num_discs=max_discs,
            x_min=hx - 2.0,
            x_max=hx,
            y_min=MIN_HOLE_Y,
            y_max=MAX_HOLE_Y,
            hole_xy=hole_pos,
        )

        state_vec = encode_state_with_discs(ball_start_obs, hole_pos_obs, disc_positions, max_num_discs=5)
        state_scaled = scale_state_vec(state_vec)

        # Policy action
        speed, angle_deg = agent.act(state_scaled)

        # Actuator noise (same style as training)
        if apply_actuator_noise:
            speed     = float(np.clip(speed + np.random.normal(0, SPEED_NOISE_STD), agent.speed_low,  agent.speed_high))
            angle_deg = float(np.clip(angle_deg + np.random.normal(0, ANGLE_NOISE_STD), agent.angle_low, agent.angle_high))

        # Simulate
        out = run_sim_fn(angle_deg, speed, [float(hx), float(hy)], mujoco_cfg, disc_positions)
        if out is None:
            # treat failure as a bad episode
            ball_x, ball_y, in_hole, trajectory = float("nan"), float("nan"), 0, None
            reward = 0.0
            dist = float("inf")
        else:
            ball_x, ball_y, in_hole, trajectory = out
            ball_end = np.array([ball_x, ball_y], dtype=np.float32)
            reward = compute_reward(ball_end, hole_pos, in_hole, in_hole_reward=in_hole_reward, distance_scale=distance_scale)
            dist = float(np.linalg.norm(ball_end - hole_pos))

        successes += int(in_hole == 1)
        rewards.append(float(reward))
        distances.append(float(dist))

        per_episode.append({
            "episode": ep,
            "hole_x": float(hx),
            "hole_y": float(hy),
            "ball0_x": float(ball_start[0]),
            "ball0_y": float(ball_start[1]),
            "speed": float(speed),
            "angle_deg": float(angle_deg),
            "in_hole": int(in_hole == 1),
            "reward": float(reward),
            "dist_to_hole": float(dist),
            "num_discs": int(len(disc_positions)),
        })

    rewards_np = np.array(rewards, dtype=np.float32)
    dist_np = np.array(distances, dtype=np.float32)

    summary = {
        "episodes": episodes,
        "success_rate": float(successes / max(1, episodes)),
        "avg_reward": float(np.nanmean(rewards_np)),
        "avg_distance": float(np.nanmean(dist_np)),
        "median_distance": float(np.nanmedian(dist_np)),
        "p90_distance": float(np.nanpercentile(dist_np, 90)),
        "p99_distance": float(np.nanpercentile(dist_np, 99)),
    }
    return summary, per_episode


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", type=str, required=False, help="Path to saved actor .pth")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--max-discs", type=int, default=0, help="Max discs to place in eval (0..5 typical)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-act-noise", action="store_true", help="Disable actuator noise during eval")
    ap.add_argument("--no-obs-noise", action="store_true", help="Disable observation noise during eval")
    ap.add_argument("--csv", type=str, default=None, help="Optional path to save per-episode CSV")
    args = ap.parse_args()

    # Locate repo + configs (mirrors your training script layout)
    here = Path(__file__).resolve().parent
    project_root = here.parents[2]

    mujoco_config_path = project_root / "configs" / "mujoco_config.yaml"
    rl_config_path     = project_root / "configs" / "rl_config.yaml"

    with open(mujoco_config_path, "r") as f:
        mujoco_cfg = yaml.safe_load(f)
    with open(rl_config_path, "r") as f:
        rl_cfg = yaml.safe_load(f)

    # Import simulator
    here    = Path(__file__).resolve().parent
    sim_dir = here.parent / "simulation"
    sys.path.append(str(sim_dir))

    from run_sim_rl import run_sim  # noqa: E402

    # Load actor + build agent
    actor_path = project_root / "models" / "rl" / "bandit" / f"ddpg_actor_{rl_cfg["training"]["model_name"]}"
    actor, device = load_actor(actor_path, rl_cfg)
    agent = BanditAgent(actor, device, rl_cfg)

    summary, per_episode = evaluate(
        agent=agent,
        run_sim_fn=run_sim,
        mujoco_cfg=mujoco_cfg,
        rl_cfg=rl_cfg,
        episodes=args.episodes,
        max_discs=0,
        seed=args.seed,
        apply_actuator_noise=(not args.no_act_noise),
        apply_obs_noise=(not args.no_obs_noise),
    )

    print("========== EVAL SUMMARY ==========")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:>16}: {v:.4f}")
        else:
            print(f"{k:>16}: {v}")
    print("==================================")

    if args.csv:
        csv_path = Path(args.csv)
        write_csv(per_episode, csv_path)
        print(f"Wrote per-episode results to: {csv_path}")


if __name__ == "__main__":
    main()
