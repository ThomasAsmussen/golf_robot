"""
rl_common.py

Reusable components for golf-ball contextual RL experiments.

This module is intentionally algorithm-agnostic: it contains utilities for
state/action scaling, reward shaping, trajectory meta extraction, simple replay
buffers, JSONL episode logging, and environment parameter sampling helpers.

Expected conventions (from your sim script):
- trajectory rows are [t, x, y] (time in seconds, x/y in meters)
- hole_pos is [x, y] (meters)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import time


# =========================================================
# Noise + bounds (keep these centralized and consistent)
# =========================================================

# Environment / measurement noise
SPEED_NOISE_STD: float = 0.05
ANGLE_NOISE_STD: float = 0.1
BALL_OBS_NOISE_STD: float = 0.002
HOLE_OBS_NOISE_STD: float = 0.002

MAX_DISCS = 3

# Sampling ranges
MIN_HOLE_X: float = 3.0
MAX_HOLE_X: float = 4.0
MIN_HOLE_Y: float = -0.5
MAX_HOLE_Y: float = 0.5

MIN_BALL_X: float = -0.5
MAX_BALL_X: float = 0.5
MIN_BALL_Y: float = -0.5
MAX_BALL_Y: float = 0.5


# =========================================================
# Basic scaling helpers
# =========================================================

def squash_to_range(x: float | np.ndarray, lo: float, hi: float) -> float | np.ndarray:
    """Squash x in [-1, 1] to [lo, hi]."""
    return lo + 0.5 * (x + 1.0) * (hi - lo)


def scale_to_unit(x: float | np.ndarray, lo: float, hi: float) -> float | np.ndarray:
    """Scale x in [lo, hi] to [-1, 1]."""
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def unscale_from_unit(x_unit: float | np.ndarray, lo: float, hi: float) -> float | np.ndarray:
    """Inverse of scale_to_unit: map [-1, 1] back to [lo, hi]."""
    return squash_to_range(x_unit, lo, hi)


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
class Actor(nn.Module):
    """
    Policy π(s) -> a, deterministic, used as contextual bandit policy.
    """
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


class Critic(nn.Module):
    """
    Critic Q(s, a) ≈ E[r | s,a], purely single-step / bandit.
    """
    def __init__(self, state_dim=2, action_dim=2, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# ---------------------------------------------------------
# Loaders (for continue_training or separate evaluation)
# ---------------------------------------------------------
def load_actor(model_path, rl_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim  = rl_cfg["model"]["state_dim"]
    action_dim = rl_cfg["model"]["action_dim"]
    hidden_dim = rl_cfg["model"]["hidden_dim"]

    actor = Actor(state_dim, action_dim, hidden_dim).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()
    return actor, device


def load_critic(model_path, rl_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim  = rl_cfg["model"]["state_dim"]
    action_dim = rl_cfg["model"]["action_dim"]
    hidden_dim = rl_cfg["model"]["hidden_dim"]

    critic = Critic(state_dim, action_dim, hidden_dim).to(device)
    critic.load_state_dict(torch.load(model_path, map_location=device))
    critic.eval()
    return critic, device


def find_latest_ddpg_checkpoint(model_dir: Path, prefix: str | None = None):
    """
    Find newest (actor, critic) checkpoint pair.
    If prefix is given, only match files containing that prefix.
    """
    actor_files = list(model_dir.glob("ddpg_actor_*.pth"))

    if prefix is not None:
        actor_files = [
            f for f in actor_files
            if prefix in f.stem
        ]

    if not actor_files:
        raise FileNotFoundError("No matching actor checkpoints found.")

    # newest by modification time
    actor_file = max(actor_files, key=lambda f: f.stat().st_mtime)

    # infer critic path
    critic_file = actor_file.with_name(
        actor_file.name.replace("ddpg_actor_", "ddpg_critic_")
    )

    if not critic_file.exists():
        raise FileNotFoundError(f"Matching critic not found for {actor_file.name}")

    return actor_file, critic_file

# =========================================================
# JSON helpers for robust logging
# =========================================================

def json_default(obj: Any) -> Any:
    """JSON serializer for numpy + torch-ish objects."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return str(obj)


class EpisodeLoggerJsonl:
    """
    Append-only JSONL logger with fsync for robustness (real-robot friendly).

    Usage:
        logger = EpisodeLoggerJsonl(Path("episodes.jsonl"))
        logger.log({...})
        logger.close()
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", buffering=1)

    def log(self, record: Dict[str, Any]) -> None:
        self.f.write(json.dumps(record, default=json_default) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def get_length(self) -> int:
        """Return number of logged episodes (lines)."""
        self.f.flush()
        with open(self.path, "r") as fr:
            return sum(1 for _ in fr)

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass

def load_replay_from_jsonl(
    jsonl_path: Path,
    replay_buffer_big: "ReplayBuffer",
    replay_buffer_recent: "ReplayBuffer",
    max_recent: int = 1000,
    state_key: str = "state_norm",
    action_key: str = "action_norm",
    reward_key: str = "reward",              # only used as fallback now
    used_key: str = "used_for_training",
    *,
    reward_cfg: Optional[Dict[str, Any]] = None,
    # Where to read the ingredients from:
    ball_end_key: str = "ball_end_pos",
    hole_key: str = "hole_pos_obs",
    in_hole_key: str = "in_hole",
) -> int:
    """
    Load JSONL episodes into replay buffers.

    IMPORTANT: reward is recomputed from JSONL values using compute_reward()
    (ball end position, hole position, in_hole, and optional meta), rather than
    using the stored reward field. The stored reward is only used as a fallback
    if the needed fields are missing.

    reward_cfg should typically be rl_cfg["reward"] and may include:
      distance_scale, in_hole_reward, w_distance, optimal_speed,
      dist_at_hole_scale, optimal_speed_scale, use_meta_if_available, etc.

    Returns number of loaded transitions added to the *big* buffer.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        return 0

    text = jsonl_path.read_text().strip()
    if not text:
        return 0

    # Default reward config (matches your compute_reward defaults if omitted)
    reward_cfg = dict(reward_cfg or {})

    lines = text.splitlines()
    eps: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            eps.append(json.loads(line))
        except json.JSONDecodeError:
            # Skip malformed lines rather than crashing training
            continue

    def _get_ball_end(ep: Dict[str, Any]) -> Optional[np.ndarray]:
        # primary
        v = ep.get(ball_end_key, None)
        # common alternatives
        if v is None:
            v = ep.get("ball_final_pos", None)
        if v is None:
            v = ep.get("ball_end_xy", None)
        if v is None:
            return None
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.size < 2:
            return None
        return arr[:2]

    def _get_hole(ep: Dict[str, Any]) -> Optional[np.ndarray]:
        v = ep.get(hole_key, None)
        if v is None:
            v = ep.get("hole_pos", None)         # sometimes you may log the true hole pos
        if v is None:
            v = ep.get("hole_xy", None)
        if v is None:
            return None
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.size < 2:
            return None
        return arr[:2]

    def _get_meta(ep: Dict[str, Any]) -> Dict[str, Any]:
        # You log dist_at_hole/speed_at_hole at top-level; also allow nested "meta"
        meta = {}
        if isinstance(ep.get("meta", None), dict):
            meta.update(ep["meta"])
        if "dist_at_hole" in ep:
            meta["dist_at_hole"] = ep.get("dist_at_hole")
        if "speed_at_hole" in ep:
            meta["speed_at_hole"] = ep.get("speed_at_hole")
        if "out_of_bounds" in ep:
            meta["out_of_bounds"] = ep.get("out_of_bounds")
        return meta

    def _recompute_reward(ep: Dict[str, Any]) -> Tuple[float, bool]:
        """
        Returns (reward, used_fallback).
        """
        ball_end = _get_ball_end(ep)
        hole_xy = _get_hole(ep)
        in_hole = ep.get(in_hole_key, False)

        if ball_end is None or hole_xy is None:
            # Fallback to stored reward if we can't recompute
            r_fallback = float(ep.get(reward_key, 0.0))
            return r_fallback, True

        meta = _get_meta(ep)
        r = compute_reward(
            ball_end_xy=ball_end,
            hole_xy=hole_xy,
            in_hole=in_hole,
            meta=meta,
            is_out_of_bounds=meta.get("out_of_bounds", False),
            # pass through your tuning knobs if provided
            in_hole_reward=float(reward_cfg.get("in_hole_reward", 3.0)),
            distance_scale=float(reward_cfg.get("distance_scale", 0.5)),
            w_distance=float(reward_cfg.get("w_distance", 0.6)),
            optimal_speed=float(reward_cfg.get("optimal_speed", 0.65)),
            dist_at_hole_scale=float(reward_cfg.get("dist_at_hole_scale", 5.0)),
            optimal_speed_scale=float(reward_cfg.get("optimal_speed_scale", 5.0)),
            use_meta_if_available=bool(reward_cfg.get("use_meta_if_available", True)),
        )
        return float(r), False

    loaded = 0
    fallback_count = 0

    # Fill big buffer
    for ep in eps:
        if not ep.get(used_key, True):
            continue

        s = torch.tensor(ep[state_key], dtype=torch.float32)
        a = torch.tensor(ep[action_key], dtype=torch.float32)
        r, used_fallback = _recompute_reward(ep)
        fallback_count += int(used_fallback)

        replay_buffer_big.add(s, a, r)
        loaded += 1

    # Fill recent buffer
    replay_buffer_recent.clear()
    for ep in eps[-max_recent:]:
        if not ep.get(used_key, True):
            continue

        s = torch.tensor(ep[state_key], dtype=torch.float32)
        a = torch.tensor(ep[action_key], dtype=torch.float32)
        r, _ = _recompute_reward(ep)

        replay_buffer_recent.add(s, a, r)

    if fallback_count > 0:
        print(
            f"[load_replay_from_jsonl] Warning: recompute_reward fell back to stored "
            f"'{reward_key}' for {fallback_count}/{loaded} loaded episodes "
            f"(missing ball_end/hole in JSONL)."
        )

    return loaded



# =========================================================
# Replay buffers (bandit-friendly, but useful elsewhere too)
# =========================================================

class ReplayBuffer:
    """
    Minimal replay buffer for (state, action, reward).

    - state, action are torch tensors (typically on CPU when stored)
    - reward is float
    """
    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        self.ptr = 0
        self.data: List[Tuple[torch.Tensor, torch.Tensor, float]] = []

    def __len__(self) -> int:
        return len(self.data)

    def clear(self) -> None:
        self.ptr = 0
        self.data.clear()

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: float) -> None:
        if len(self.data) < self.capacity:
            self.data.append((state, action, float(reward)))
        else:
            self.data[self.ptr] = (state, action, float(reward))
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.data) == 0:
            raise RuntimeError("ReplayBuffer is empty")
        idx = torch.randint(0, len(self.data), (batch_size,))
        states, actions, rewards = zip(*[self.data[i] for i in idx])
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1),
        )


def sample_mixed(
    recent_buf: ReplayBuffer,
    all_buf: ReplayBuffer,
    recent_ratio: float,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a batch as a mixture of recent + all experiences.
    Useful across many off-policy algorithms.
    """
    if len(all_buf) == 0:
        raise RuntimeError("All-buffer is empty; cannot sample.")

    n_recent_avail = len(recent_buf)
    num_recent = int(round(batch_size * float(recent_ratio)))
    num_recent = min(num_recent, n_recent_avail)
    num_all = batch_size - num_recent

    states_a, actions_a, rewards_a = all_buf.sample(num_all)

    if num_recent > 0:
        states_r, actions_r, rewards_r = recent_buf.sample(num_recent)
        states = torch.cat([states_r, states_a], dim=0)
        actions = torch.cat([actions_r, actions_a], dim=0)
        rewards = torch.cat([rewards_r, rewards_a], dim=0)
    else:
        states, actions, rewards = states_a, actions_a, rewards_a

    return states, actions, rewards


# =========================================================
# Trajectory meta extraction
# =========================================================

@dataclass(frozen=True)
class TrajectoryMeta:
    dist_at_hole: float
    speed_at_hole: float
    closest_x_index: int


def meta_from_trajectory_xy(
    trajectory: np.ndarray,
    hole_pos_xy: Sequence[float],
) -> TrajectoryMeta:
    """
    Compute:
      - dist_at_hole: distance in xy at the sample where x is closest to hole_x
      - speed_at_hole: xy speed immediately BEFORE that sample using backward difference
    Expects trajectory rows: [t, x, y].
    """
    traj = np.asarray(trajectory)
    if traj.ndim != 2 or traj.shape[1] < 3:
        raise ValueError(f"trajectory must be (N,3+) with [t,x,y], got shape {traj.shape}")

    hole_pos_xy = np.asarray(hole_pos_xy, dtype=float).reshape(2,)
    hole_x = float(hole_pos_xy[0])

    # column mapping: [t, x, y]
    closest_x_index = int(np.argmin(np.abs(traj[:, 1] - hole_x)))

    dist_at_hole = float(np.linalg.norm(traj[closest_x_index, 1:3] - hole_pos_xy))

    if closest_x_index == 0:
        raise ValueError("closest_x_index is 0; cannot compute backward-difference speed")

    dt = float(traj[closest_x_index, 0] - traj[closest_x_index - 1, 0])
    if dt <= 0:
        raise ValueError(f"Non-positive dt={dt} at index {closest_x_index}")

    dxy = traj[closest_x_index, 1:3] - traj[closest_x_index - 1, 1:3]
    speed_at_hole = float(np.linalg.norm(dxy) / dt)

    return TrajectoryMeta(
        dist_at_hole=dist_at_hole,
        speed_at_hole=speed_at_hole,
        closest_x_index=closest_x_index,
    )


# =========================================================
# Reward shaping (kept generic: algorithms can plug this in)
# =========================================================

def compute_reward(
    ball_end_xy: Sequence[float],
    hole_xy: Sequence[float],
    in_hole: bool | int,
    meta: Optional[Dict[str, Any]] = None,
    *,
    in_hole_reward: float = 3.0,
    distance_scale: float = 0.5,
    w_distance: float = 0.6,
    optimal_speed: float = 0.65,
    dist_at_hole_scale: float = 5.0,
    optimal_speed_scale: float = 5.0,
    use_meta_if_available: bool = True,
    is_out_of_bounds: bool = False,
    multiple_dist_at_hole = None

) -> float:
    """
    Single-step reward.

    - If in_hole: return in_hole_reward.
    - Else: if meta provides dist_at_hole and speed_at_hole, combine two shaped terms.
    - Else: exponential shaping of final distance.

    meta convention:
        meta["dist_at_hole"], meta["speed_at_hole"]
    """
    # print(in_hole_reward)
    if bool(in_hole):
        reward = float(in_hole_reward)
        # print("in hole reward:", reward)
        return reward

    if is_out_of_bounds:
        reward = 0.0
        # print("out of bounds reward: ", reward)
        return reward


    ball_end_xy = np.asarray(ball_end_xy, dtype=float).reshape(2,)
    hole_xy = np.asarray(hole_xy, dtype=float).reshape(2,)

    if use_meta_if_available and isinstance(meta, dict):
        if multiple_dist_at_hole is not None:
            dist_at_hole = multiple_dist_at_hole
        else:
            dist_at_hole = meta.get("dist_at_hole", None)

        speed_at_hole = meta.get("speed_at_hole", None)
        if dist_at_hole is not None and speed_at_hole is not None:
            dist_at_hole = float(dist_at_hole)
            speed_at_hole = float(speed_at_hole)

            # print(f"Dist at hole: {dist_at_hole:.4f} m, Speed at hole: {speed_at_hole:.4f} m/s")
            # You can tune these shaping terms globally
            dist_term = np.exp(-dist_at_hole_scale * dist_at_hole)
            speed_term = np.exp(-optimal_speed_scale * abs(speed_at_hole - optimal_speed))  # target ~0.65 m/s near hole
            w_speed = 1.0 - w_distance
            reward = float(w_distance * dist_term + w_speed * speed_term)
            # print(f"Reward from meta: {reward:.4f} (dist_term: {dist_term:.4f}, speed_term: {speed_term:.4f})")
            return reward

    final_dist = float(np.linalg.norm(ball_end_xy - hole_xy))
    reward = float(np.exp(-float(distance_scale) * final_dist))
    # print(f"Reward from final distance {final_dist:.4f} m: {reward:.4f}")
    return reward


# =========================================================
# Disc placement + state encoding
# =========================================================

def generate_disc_positions(
    max_num_discs: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    hole_xy: Sequence[float],
    *,
    min_dist_from_objects: float = 0.25,
    max_tries_per_disc: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[float, float]]:
    """
    Randomly place discs in a rectangle, avoiding overlap and the hole.

    Returns list[(x,y)] sorted by distance to hole.
    """
    if max_num_discs <= 0:
        return []
    
    rng = rng or np.random.default_rng()
    hole_x, hole_y = float(hole_xy[0]), float(hole_xy[1])
    p_full = 0.8  # probability of max_num_discs


    if rng.random() < p_full: # use max discs
        num_discs = max_num_discs
    else: # uniformly sample fewer discs
        num_discs = int(rng.integers(0, max_num_discs + 1))
    discs: List[Tuple[float, float]] = []

    x_lo = x_min + min_dist_from_objects
    x_hi = x_max - min_dist_from_objects
    y_lo = y_min + min_dist_from_objects
    y_hi = y_max - min_dist_from_objects

    for _ in range(num_discs):
        placed = False
        for _ in range(max_tries_per_disc):
            x = float(rng.uniform(x_lo, x_hi))
            y = float(rng.uniform(y_lo, y_hi))

            # too close to hole
            if np.hypot(x - hole_x, y - hole_y) < min_dist_from_objects:
                continue

            # too close to other discs
            if any(np.hypot(x - dx, y - dy) < min_dist_from_objects for dx, dy in discs):
                continue

            discs.append((x, y))
            placed = True
            break

        if not placed:
            raise RuntimeError("Could not place all discs without overlap.")

    discs.sort(key=lambda p: float(np.hypot(p[0] - hole_x, p[1] - hole_y)))
    return discs


def encode_state_with_discs(
    ball_xy_obs: Sequence[float],
    hole_xy_obs: Sequence[float],
    disc_positions: Sequence[Sequence[float]],
    max_num_discs: int,
    *,
    default_value: float = 0.0,
) -> np.ndarray:
    """
    State layout:
        [ ball_x, ball_y,
          hole_x, hole_y,
          disc1_x, disc1_y, disc1_present,
          ...
          discN_x, discN_y, discN_present ]
    """
    ball_xy_obs = np.asarray(ball_xy_obs, dtype=float).reshape(2,)
    hole_xy_obs = np.asarray(hole_xy_obs, dtype=float).reshape(2,)

    disc_coords: List[float] = []
    for i in range(int(max_num_discs)):
        if i < len(disc_positions):
            x, y = float(disc_positions[i][0]), float(disc_positions[i][1])
            present = 1.0
        else:
            x, y = float(default_value), float(default_value)
            present = 0.0
        disc_coords.extend([x, y, present])

    return np.concatenate([ball_xy_obs, hole_xy_obs, np.array(disc_coords, dtype=float)])


def scale_state_vec(state_vec: Sequence[float]) -> np.ndarray:
    """
    Scale state vector components to [-1, 1].

    Assumes state layout from encode_state_with_discs.
    """
    s = np.asarray(state_vec, dtype=float)
    if s.shape[0] < 4:
        raise ValueError("state_vec must contain at least ball(2) + hole(2) components")

    ball_x, ball_y = s[0], s[1]
    hole_x, hole_y = s[2], s[3]

    ball_x_scaled = scale_to_unit(ball_x, MIN_BALL_X, MAX_BALL_X)
    ball_y_scaled = scale_to_unit(ball_y, MIN_BALL_Y, MAX_BALL_Y)
    hole_x_scaled = scale_to_unit(hole_x, MIN_HOLE_X, MAX_HOLE_X)
    hole_y_scaled = scale_to_unit(hole_y, MIN_HOLE_Y, MAX_HOLE_Y)

    disc_data = s[4:]
    disc_scaled: List[float] = []
    for i in range(0, len(disc_data), 3):
        disc_x = disc_data[i]
        disc_y = disc_data[i + 1]
        present = disc_data[i + 2]

        if present == 0:
            disc_x_scaled, disc_y_scaled = -1.0, -1.0
        else:
            # heuristic disc bounds (match your current code)
            disc_x_scaled = scale_to_unit(disc_x, MIN_HOLE_X - 2.0, MAX_HOLE_X)
            disc_y_scaled = scale_to_unit(disc_y, MIN_HOLE_Y, MAX_HOLE_Y)

        disc_scaled.extend([disc_x_scaled, disc_y_scaled, present])

    return np.array([ball_x_scaled, ball_y_scaled, hole_x_scaled, hole_y_scaled] + disc_scaled, dtype=float)


# =========================================================
# Context sampling (sim) — algorithm independent
# =========================================================

def random_hole_in_rectangle(
    *,
    x_min: float = MIN_HOLE_X,
    x_max: float = MAX_HOLE_X,
    y_min: float = MIN_HOLE_Y,
    y_max: float = MAX_HOLE_Y,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    rng = rng or np.random.default_rng()
    return float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max))


def sim_init_parameters(
    mujoco_cfg: Dict[str, Any],
    max_num_discs: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]], float, float, np.ndarray]:
    """
    Sample:
      - ball_start_obs (2,)
      - hole_pos_obs (2,)
      - disc_positions list[(x,y)]
      - hole_x, hole_y
      - hole_pos (2,) (true, un-noised)

    Also writes start positions into mujoco_cfg for convenience.
    """
    rng = rng or np.random.default_rng()

    ball_start = rng.uniform(MIN_BALL_X, MAX_BALL_X, size=(2,))
    ball_start_obs = ball_start + rng.normal(0.0, BALL_OBS_NOISE_STD, size=(2,))

    mujoco_cfg.setdefault("ball", {})
    mujoco_cfg["ball"]["start_pos"] = [float(ball_start[0]), float(ball_start[1]), 0.02135]
    mujoco_cfg["ball"]["obs_start_pos"] = [float(ball_start_obs[0]), float(ball_start_obs[1]), 0.02135]

    x, y = random_hole_in_rectangle(rng=rng)
    hole_pos = np.array([x, y], dtype=float)
    hole_pos_obs = hole_pos + rng.normal(0.0, HOLE_OBS_NOISE_STD, size=(2,))

    disc_positions = generate_disc_positions(
        max_num_discs,
        x - 2.0, x,
        MIN_HOLE_Y, MAX_HOLE_Y,
        hole_xy=hole_pos,
        rng=rng,
    )

    return ball_start_obs, hole_pos_obs, disc_positions, float(x), float(y), hole_pos


# =========================================================
# Action mapping (normalized -> physical)
# =========================================================

def action_to_speed_angle(
    a_norm: Sequence[float],
    speed_low: float,
    speed_high: float,
    angle_low: float,
    angle_high: float,
) -> Tuple[float, float]:
    """
    Map normalized action a_norm=[speed_norm, angle_norm] in [-1,1]^2 to:
      - speed in [speed_low, speed_high]
      - angle_deg in [angle_low, angle_high]
    """
    a = np.asarray(a_norm, dtype=float).reshape(2,)
    speed_norm, angle_norm = float(a[0]), float(a[1])
    speed = float(squash_to_range(speed_norm, speed_low, speed_high))
    angle_deg = float(squash_to_range(angle_norm, angle_low, angle_high))
    return speed, angle_deg

# ---------------------------------------------------------
# Helper: map normalized actions -> (speed, angle_deg)
# ---------------------------------------------------------
def get_input_parameters(a_norm, speed_low, speed_high, angle_low, angle_high):
    speed_norm, angle_norm = a_norm
    speed     = squash_to_range(speed_norm,  speed_low,  speed_high)
    angle_deg = squash_to_range(angle_norm, angle_low, angle_high)
    return speed, angle_deg


def get_hole_positions():
    here = Path(__file__).resolve().parent
    config_dir = here.parents[1] / "configs"
    with open(config_dir / "hole_config.yaml", "r") as f:
        hole_positions = yaml.safe_load(f)
    return hole_positions


# ---------------------------------------------------------
# Evaluation (greedy policy, contextual bandit)
# ---------------------------------------------------------
def evaluation_policy_short(
    actor,
    device,
    mujoco_cfg,
    rl_cfg,
    num_episodes,
    max_num_discs=5,
    env_step=None, 
    env_type="sim",
    input_func=None,
    big_episode_logger=None,
):
    
    if env_step is None:
        raise ValueError("evaluation_policy_short() requires env_step.")
    
    speed_low  = rl_cfg["model"]["speed_low"]
    speed_high = rl_cfg["model"]["speed_high"]
    angle_low  = rl_cfg["model"]["angle_low"]
    angle_high = rl_cfg["model"]["angle_high"]

    if env_type == "real":
        print("[EVAL] Real-world evaluation mode started.")
        num_episodes = 3

        here = Path(__file__).resolve().parent
        project_root       = here.parents[1]
        episode_log_path = project_root / "log" / "real_episodes_eval" / "episode_logger_eval.jsonl"
        episode_logger = EpisodeLoggerJsonl(episode_log_path)
    
    # hole_positions = get_hole_positions()

    successes          = 0
    rewards            = []
    distances_to_hole  = []
    # max_num_discs = 5
    actor.eval()
    with torch.no_grad():
        for i in range(num_episodes):
            # New random context each eval episode
            if env_type == "sim":
                ball_start_obs, hole_pos_obs, disc_positions, x, y, hole_pos = sim_init_parameters(mujoco_cfg, max_num_discs)
            
            if env_type == "real":
                ball_start_obs, hole_pos_obs, disc_positions, chosen_hole = input_func(camera_index=4, chosen_hole=i+1)
                hole_pos = np.array(hole_pos_obs)
    
            state_vec = encode_state_with_discs(
                ball_start_obs, hole_pos_obs, disc_positions, max_num_discs=5
            )

            # state_vec = np.concatenate([ball_start_obs, hole_pos_obs])  # No discs for now

            state_vec = scale_state_vec(state_vec)

            state = torch.tensor(
                state_vec, dtype=torch.float32, device=device
            ).unsqueeze(0)

            a_norm = actor(state).squeeze(0).cpu().numpy()
            speed, angle_deg = get_input_parameters(
                a_norm, speed_low, speed_high, angle_low, angle_high
            )

            # Apply same actuator noise as during training
            if env_type == "sim":
                speed_noise     = np.random.normal(0, SPEED_NOISE_STD)
                angle_deg_noise = np.random.normal(0, ANGLE_NOISE_STD)
                speed     = np.clip(speed + speed_noise,     speed_low,  speed_high)
                angle_deg = np.clip(angle_deg + angle_deg_noise, angle_low, angle_high)


                ball_x, ball_y, in_hole, meta = env_step(
                    angle_deg, speed, [x, y], mujoco_cfg, disc_positions
                )
            if env_type == "real":
                result = env_step(impact_velocity=speed, swing_angle=angle_deg, ball_start_position=ball_start_obs, planner="quintic", check_rtt=True, chosen_hole=chosen_hole)
                ball_x, ball_y, in_hole, meta = result

            reward = compute_reward(
                ball_end_xy=np.array([ball_x, ball_y]),
                hole_xy=hole_pos,
                in_hole=in_hole,
                meta=meta,
                is_out_of_bounds=meta["out_of_bounds"],
                distance_scale=rl_cfg["reward"]["distance_scale"],
                in_hole_reward=rl_cfg["reward"]["in_hole_reward"],
                w_distance=rl_cfg["reward"]["w_distance"],
                optimal_speed=rl_cfg["reward"]["optimal_speed"],
                dist_at_hole_scale=rl_cfg["reward"]["dist_at_hole_scale"],
                optimal_speed_scale=rl_cfg["reward"]["optimal_speed_scale"],
            )
            if env_type == "real":
                logging_record = {
                    "episode": i,
                    "time": time.time(),
                    "used_for_training": meta["used_for_training"],
                    "ball_start_obs": ball_start_obs,
                    "hole_pos_obs": hole_pos_obs,
                    "disc_positions": disc_positions,
                    "state_norm": state_vec,
                    "action_norm": a_norm,
                    "speed": speed,
                    "angle_deg": angle_deg,
                    "ball_final_pos": [ball_x, ball_y],
                    "in_hole": bool(in_hole),
                    "out_of_bounds": meta["out_of_bounds"],
                    "reward": reward,
                    "chosen_hole": chosen_hole,
                    "dist_at_hole": meta.get("dist_at_hole", None),
                    "speed_at_hole": meta.get("speed_at_hole", None),
                    "exploring": False,
                }
                episode_logger.log(logging_record)
            if meta["used_for_training"]:
                big_episode_logger.log(logging_record)
            rewards.append(reward)
            successes += int(in_hole == 1)
            distance_to_hole = np.linalg.norm(np.array([ball_x, ball_y]) - hole_pos)
            distances_to_hole.append(distance_to_hole)

    avg_distance_to_hole = float(np.mean(distances_to_hole)) if distances_to_hole else 0.0
    actor.train()
    return successes / num_episodes, float(np.mean(rewards)), avg_distance_to_hole


if __name__ == "__main__":
    print(np.array([get_hole_positions()[1]["x"], get_hole_positions()[1]["y"]]))