import os
from pathlib import Path
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import uuid
import json
import time

# ---------------------------------------------------------
# Global noise parameters (environment / measurement noise)
# ---------------------------------------------------------
SPEED_NOISE_STD      = 0.1
ANGLE_NOISE_STD      = 0.1
BALL_OBS_NOISE_STD   = 0.002
HOLE_OBS_NOISE_STD   = 0.002
MIN_HOLE_X           = 3.0
MAX_HOLE_X           = 4.0
MIN_HOLE_Y           = -0.5
MAX_HOLE_Y           = 0.5
MIN_BALL_X           = -0.5
MAX_BALL_X           = 0.5
MIN_BALL_Y           = -0.5
MAX_BALL_Y           = 0.5


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def squash_to_range(x, lo, hi):
    """Squash a value in [-1, 1] to [lo, hi]."""
    return lo + 0.5 * (x + 1.0) * (hi - lo)


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
    

def load_replay_from_jsonl(
        jsonl_path: Path,
        replay_buffer_big,
        replay_buffer_recent,
        max_recent: int = 1000,
):
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        return 0

    loaded = 0
    lines = jsonl_path.read_text().strip().splitlines()

    if not lines:
        return 0
    
    for line in lines:
        ep = json.loads(line)

        if not ep.get("used_for_training", True):
            continue

        s = torch.tensor(ep["state_norm"], dtype=torch.float32)
        a = torch.tensor(ep["action_norm"], dtype=torch.float32)
        r = float(ep["reward"])

        replay_buffer_big.add(s, a, r)
        loaded += 1

    replay_buffer_recent.clear()
    for line in lines[-max_recent:]:
        ep = json.loads(line)
        if not ep.get("used_for_training", True):
            continue

        s = torch.tensor(ep["state_norm"], dtype=torch.float32)
        a = torch.tensor(ep["action_norm"], dtype=torch.float32)
        r = float(ep["reward"])
        replay_buffer_recent.add(s, a, r)
    
    return loaded



class EpisodeLoggerJson:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", buffering=1)

    def log(self, record: dict):
        print(f"Logging episode to {self.path}: episode {record.get('episode', 'N/A')}")
        self.f.write(json.dumps(record, default=_json_default) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.f.close()


def _json_default(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    # last resort: stringify (or raise)
    return str(obj)

# ---------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------
class ReplayBuffer:
    """
    Stores (state, action, reward) tuples for a contextual bandit.
    No next_state, no done.
    """
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.ptr = 0
        self.full = False
        self.data = []

    def add(self, state, action, reward):
        if len(self.data) < self.capacity:
            self.data.append((state, action, reward))
        else:
            self.data[self.ptr] = (state, action, reward)
            self.full = True

        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        idx = torch.randint(0, len(self.data), (batch_size,))
        states, actions, rewards = zip(*[self.data[i] for i in idx])

        return (
            torch.stack(states),
            torch.stack(actions),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1),
        )

    def clear(self):
        self.ptr = 0
        self.full = False
        self.data = []





def sample_mixed(recent_buf, all_buf, recent_ratio, batch_size):
    n_recent_avail = len(recent_buf.data)
    n_all_avail    = len(all_buf.data)

    if n_all_avail == 0:
        raise RuntimeError("All-buffer is empty; cannot sample.")

    # target split
    num_recent = int(round(batch_size * recent_ratio))
    num_all    = batch_size - num_recent

    # clamp to availability (and fall back to all_buf)
    num_recent = min(num_recent, n_recent_avail)
    num_all    = batch_size - num_recent  # fill remainder from all_buf

    states_a, actions_a, rewards_a = all_buf.sample(num_all)

    if num_recent > 0:
        states_r, actions_r, rewards_r = recent_buf.sample(num_recent)
        states  = torch.cat([states_r,  states_a],  dim=0)
        actions = torch.cat([actions_r, actions_a], dim=0)
        rewards = torch.cat([rewards_r, rewards_a], dim=0)
    else:
        states, actions, rewards = states_a, actions_a, rewards_a

    return states, actions, rewards



def scale(x, lo, hi):
    """Scale x in [lo, hi] to [-1, 1]."""
    return 2.0 * (x - lo) / (hi - lo) - 1.0

def scale_state_vec(state_vec):
    """
    Scale state vector components to [-1, 1].
    Assumes state vector layout:
    [ ball_x, ball_y,
      hole_x, hole_y,
      disc1_x, disc1_y, disc1_present,
      ...
      discN_x, discN_y, discN_present ]
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
        disc_x, disc_y, disc_present = disc_data[i], disc_data[i+1], disc_data[i+2]
        if disc_present == 0:
            disc_x_scaled, disc_y_scaled = -1.0, -1.0  # Default position for absent discs
        else:
            disc_x_scaled = scale(disc_x, MIN_HOLE_X - 2.0, MAX_HOLE_X)
            disc_y_scaled = scale(disc_y, MIN_HOLE_Y, MAX_HOLE_Y)
        disc_scaled.extend([disc_x_scaled, disc_y_scaled, disc_present])

    return np.array([ball_x_scaled, ball_y_scaled, hole_x_scaled, hole_y_scaled] + disc_scaled)
# ---------------------------------------------------------
# Reward and state encoding
# ---------------------------------------------------------
def compute_reward(ball_end_pos, hole_pos, in_hole, meta, in_hole_reward=3.0, distance_scale=0.5):
    """
    Single-step reward.
    - If in_hole: large positive reward.
    - Else: distance-based shaping.
    """
    if in_hole:
        return in_hole_reward

    dist_at_hole = meta.get("dist_at_hole", None)
    speed_at_hole = meta.get("speed_at_hole", None)
    if dist_at_hole is not None and speed_at_hole is not None:
        dist_at_hole_reward = np.exp(-5 * dist_at_hole)
        speed_at_hole_reward = np.exp(-5 * np.abs(speed_at_hole - 0.5))
        return 0.5 * dist_at_hole_reward + 0.5 * speed_at_hole_reward
    
    final_dist = np.linalg.norm(ball_end_pos - hole_pos)
    final_dist_reward = np.exp(-distance_scale * final_dist)
    return final_dist_reward


def generate_disc_positions(max_num_discs, x_min, x_max, y_min, y_max, hole_xy):
    """
    Randomly place discs around the hole, no overlap and not too close to hole.
    """
    hole_x, hole_y = hole_xy
    num_discs = np.random.randint(0, max_num_discs + 1)
    disc_positions = []
    min_dist_from_objects = 0.25

    x_lo = x_min + min_dist_from_objects
    x_hi = x_max - min_dist_from_objects
    y_lo = y_min + min_dist_from_objects
    y_hi = y_max - min_dist_from_objects

    max_tries_per_disc = 100

    for _ in range(num_discs):
        placed = False
        for _ in range(max_tries_per_disc):
            x = np.random.uniform(x_lo, x_hi)
            y = np.random.uniform(y_lo, y_hi)

            if np.hypot(x - hole_x, y - hole_y) < min_dist_from_objects:
                continue

            too_close = False
            for (dx, dy) in disc_positions:
                if np.hypot(x - dx, y - dy) < min_dist_from_objects:
                    too_close = True
                    break

            if too_close:
                continue

            disc_positions.append((x, y))
            placed = True
            break

        if not placed:
            raise RuntimeError("Could not place all discs without overlap.")

    disc_positions.sort(key=lambda p: np.hypot(p[0] - hole_x, p[1] - hole_y))
    return disc_positions


def encode_state_with_discs(ball_start_obs, hole_pos_obs, disc_positions, max_num_discs):
    """
    State layout:

    [ ball_x, ball_y,
      hole_x, hole_y,
      disc1_x, disc1_y, disc1_present,
      ...
      discN_x, discN_y, discN_present ]
    """
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

    return np.concatenate([ball_start_obs, hole_pos_obs, np.array(disc_coords)])



def sim_init_parameters(mujoco_cfg, max_num_discs):
    # max_num_discs = np.min([5, episode // 3000])  # Increase discs over time
    # print(f"Episode {episode + 1}: max_num_discs = {max_num_discs}")
    ball_start = np.random.uniform(-0.5, 0.5, size=(2,))
    # ball_start = np.array([0, 0])  # FIXED START FOR DEBUGGING
    ball_start_obs = ball_start + np.random.normal(0, BALL_OBS_NOISE_STD, size=(2,))
    mujoco_cfg["ball"]["start_pos"]     = [ball_start[0],     ball_start[1],     0.02135]
    mujoco_cfg["ball"]["obs_start_pos"] = [ball_start_obs[0], ball_start_obs[1], 0.02135]

    x, y = random_hole_in_rectangle(x_min=MIN_HOLE_X, x_max=MAX_HOLE_X, y_min=MIN_HOLE_Y, y_max=MAX_HOLE_Y)
    hole_pos = np.array([x, y])
    hole_pos_obs = hole_pos + np.random.normal(0, HOLE_OBS_NOISE_STD, size=(2,))

    disc_positions = generate_disc_positions(
        max_num_discs, x - 2.0, x, MIN_HOLE_Y, MAX_HOLE_Y, hole_xy=hole_pos
    ) # No discs for now

    return ball_start_obs, hole_pos_obs, disc_positions, x, y, hole_pos


# ---------------------------------------------------------
# Training loop (Contextual Bandit)
# ---------------------------------------------------------
def training(rl_cfg, mujoco_cfg, project_root, continue_training=False, input_func=None, env_step=None, env_type="sim", tmp_name=None, camera_index_start=None):
    """
    Contextual bandit:
      - context = (ball_start_obs, hole_pos_obs, discs)
      - action  = (speed, angle)
      - reward  = f(final ball position, hole position, in_hole)

    No bootstrapping, no next_state.
    Critic learns Q(s,a) ≈ E[r | s,a].
    Actor learns to maximize Q(s, π(s)).
    """
    if env_step is None:
        raise ValueError("training() requires env_step (sim or real environment function).")

    episodes   = rl_cfg["training"]["episodes"]
    batch_size = rl_cfg["training"]["batch_size"]
    actor_lr   = rl_cfg["training"]["actor_lr"]
    critic_lr  = rl_cfg["training"]["critic_lr"]
    noise_std  = rl_cfg["training"]["noise_std"]     # policy exploration noise
    grad_steps = rl_cfg["training"]["grad_steps"]

    state_dim  = rl_cfg["model"]["state_dim"]
    action_dim = rl_cfg["model"]["action_dim"]
    hidden_dim = rl_cfg["model"]["hidden_dim"]
    speed_low  = rl_cfg["model"]["speed_low"]
    speed_high = rl_cfg["model"]["speed_high"]
    angle_low  = rl_cfg["model"]["angle_low"]
    angle_high = rl_cfg["model"]["angle_high"]

    in_hole_reward = rl_cfg["reward"]["in_hole_reward"]
    distance_scale = rl_cfg["reward"]["distance_scale"]

    # Linear schedule for exploration noise (policy noise)
    noise_std_start = noise_std
    noise_std_end   = 0.05

    if env_type == "sim":
        use_wandb = rl_cfg["training"]["use_wandb"]
    else:
        use_wandb = False

    model_name = rl_cfg["training"].get("model_name", None)

    # -------------------------
    # Initialize models
    # -------------------------
    if continue_training:
        actor, device = load_actor(
            model_path=project_root / "models" / "rl" / "ddpg" / f"ddpg_actor_{model_name}",
            rl_cfg=rl_cfg,
        )
        critic, _ = load_critic(
            model_path=project_root / "models" / "rl" / "ddpg" / f"ddpg_critic_{model_name}",
            rl_cfg=rl_cfg,
        )
        print("Continuing training from model:", model_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        critic = Critic(state_dim, action_dim, hidden_dim).to(device)

    print(f"Using device: {device}")

    actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    replay_buffer_big = ReplayBuffer(capacity=rl_cfg["training"]["replay_buffer_capacity"])
    replay_buffer_recent = ReplayBuffer(1000)  # Smaller buffer for recent experiences

    if use_wandb:
        wandb.watch(actor,  log="gradients", log_freq=100)
        wandb.watch(critic, log="gradients", log_freq=100)
        run_name = wandb.run.name.replace("-", "_")
    else:
        run_name = "local_run"

    model_dir = project_root / "models" / "rl" / "ddpg"
    model_dir.mkdir(parents=True, exist_ok=True)

    actor.train()
    critic.train()
    log_dict = {}
    last_success_rate = 0.0
    last_last_success_rate = 0.0
    # For now we don't actually place discs, but the state format reserves 5.
    
    max_num_discs = 0
    stage_start_episode = 0
    noise_std_stage_start = noise_std

    episode_logger = None
    episode_log_path = None

    if env_type == "real":
        episode_log_path = project_root / "log" / "real_episodes" / "logging_test.jsonl"
        episode_logger = EpisodeLoggerJson(episode_log_path)

        loaded_n = load_replay_from_jsonl(
            episode_log_path,
            replay_buffer_big,
            replay_buffer_recent,
            max_recent=1000,
        )

        if loaded_n > 0:
            print(f"Loaded {loaded_n} episodes from {episode_log_path} into replay buffer.")
        else:
            print(f"No existing episode log found at {episode_log_path}. Starting fresh.")
    
    for episode in range(episodes):
        # -------------------------------------------------
        # Sample a context (ball start + hole + discs)
        # -------------------------------------------------
        if env_type == "sim":

            if last_success_rate > 0.8 and last_last_success_rate > 0.8 and False:
                max_num_discs = min(1, max_num_discs + 1)
                last_success_rate = 0.0
                last_last_success_rate = 0.0
                noise_std = 0.2
                noise_std_stage_start = noise_std
                stage_start_episode = episode
                replay_buffer_recent.clear()
            
            ball_start_obs, hole_pos_obs, disc_positions, x, y, hole_pos = sim_init_parameters(mujoco_cfg, max_num_discs)

        if env_type == "real":
            ball_start_obs, hole_pos_obs, disc_positions, chosen_hole = input_func(camera_index=camera_index_start)


        state_vec = encode_state_with_discs(ball_start_obs, hole_pos_obs, disc_positions, 5)
        # state_vec = np.concatenate([ball_start_obs, hole_pos_obs])  # No discs for now

        state_norm = scale_state_vec(state_vec)

        s = torch.tensor(state_norm, dtype=torch.float32).to(device)
        # critic_loss_value = None
        # actor_loss_value = None

        # -------------------------------------------------
        # Policy: a = π(s) + exploration_noise
        # -------------------------------------------------
        with torch.no_grad():
            a_norm = actor(s.unsqueeze(0)).squeeze(0)
            noise = torch.normal(
                mean=torch.zeros_like(a_norm),
                std=noise_std * torch.ones_like(a_norm),
            )
        a_noisy = torch.clamp(a_norm + noise, -1.0, 1.0)


        speed, angle_deg = get_input_parameters(
            a_noisy.cpu().numpy(), speed_low, speed_high, angle_low, angle_high
        )

        # Environment / actuator noise
        if env_type == "sim":
            speed_noise      = np.random.normal(0, SPEED_NOISE_STD)
            angle_deg_noise  = np.random.normal(0, ANGLE_NOISE_STD)
            speed     = np.clip(speed + speed_noise,     speed_low, speed_high)
            angle_deg = np.clip(angle_deg + angle_deg_noise, angle_low, angle_high)

        # -------------------------------------------------
        # One-step environment: simulate and get reward
        # -------------------------------------------------
        if env_type == "sim":
            result = env_step(angle_deg, speed, [x, y], mujoco_cfg, disc_positions)
        if env_type == "real":
            result = env_step(impact_velocity=speed, swing_angle=angle_deg, ball_start_position=ball_start_obs, planner="quintic", check_rtt=True, chosen_hole=chosen_hole)

        if result is None:
            if env_type == "sim":
                result = env_step(angle_deg, speed, [x, y], mujoco_cfg, disc_positions)
            if env_type == "real":
                result = env_step(impact_velocity=speed, swing_angle=angle_deg, ball_start_position=ball_start_obs, planner="quintic", check_rtt=True, chosen_hole=chosen_hole)

            if result is None:
                print(
                    f"Episode {episode + 1}: Simulation failed again. "
                    f"Bad action: speed={speed}, angle={angle_deg}"
                )
                print(f"  Hole Position: x={x:.4f}, y={y:.4f}")
                if rl_cfg["training"]["error_hard_stop"] and not mujoco_cfg["sim"]["render"]:
                    raise RuntimeError("Simulation failed twice — aborting training.")
                else:
                    continue

        ball_x, ball_y, in_hole, meta = result

        reward = compute_reward(
            ball_end_pos=np.array([ball_x, ball_y]),
            hole_pos=hole_pos_obs,
            in_hole=in_hole,
            meta=meta,
            distance_scale=distance_scale,
            in_hole_reward=in_hole_reward,
        )

        if env_type == "real" and isinstance(trajectory, dict):
            meta = trajectory
            used_for_training = bool(meta.get("used_for_training", True))
            out_of_bounds = bool(meta.get("out_of_bounds", False))

            if not used_for_training:
                print("Episode discarded by user; not adding to replay buffer.")
                continue
            else:
                print(f"Storing episode")
                logging_dict = {
                    "episode": episode,
                    "time": time.time(),
                    "used_for_training": used_for_training,
                    "ball_start_obs": ball_start_obs.tolist(),
                    "hole_pos_obs": hole_pos_obs.tolist(),
                    "disc_positions": disc_positions,
                    "state_norm": state_norm.tolist(),
                    "action_norm": a_noisy.cpu().numpy().tolist(),
                    "speed": speed,
                    "angle_deg": angle_deg,
                    "ball_final_pos": [ball_x, ball_y],
                    "in_hole": in_hole,
                    "out_of_bounds": out_of_bounds,
                    "reward": reward,
                    "chosen_hole": chosen_hole,
                    "ball_end": [ball_x, ball_y],
                }
                print(f"Logging episode data: {logging_dict}")
                episode_logger.log(logging_dict)

        # Store (s,a,r) in buffer (contextual bandit)
        s_train = s.detach().cpu()
        a_train = a_noisy.detach().cpu()
        replay_buffer_big.add(s_train, a_train, reward)
        replay_buffer_recent.add(s_train, a_train, reward)


        # -------------------------------------------------
        # TD-style supervised update: Q(s,a) -> r
        # -------------------------------------------------
        if len(replay_buffer_big.data) >= batch_size:
            print("Updating networks...")
            for _ in range(grad_steps):
                states_b, actions_b, rewards_b = sample_mixed(
                    replay_buffer_recent,
                    replay_buffer_big,
                    recent_ratio=0.7,
                    batch_size=batch_size,
                )

                states_b  = states_b.to(device)
                actions_b = actions_b.to(device)
                rewards_b = rewards_b.to(device)

                # Contextual bandit: target is just reward
                td_target = rewards_b

                q_pred = critic(states_b, actions_b)
                critic_loss = F.mse_loss(q_pred, td_target)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Actor tries to maximize Q(s, π(s))
                a_for_actor = actor(states_b)
                actor_loss  = -critic(states_b, a_for_actor).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

            critic_loss_value = critic_loss.item()
            actor_loss_value  = actor_loss.item()

            if env_type == "real":
                if tmp_name is not None: 
                    torch.save(actor.state_dict(),  model_dir / f"ddpg_actor_{tmp_name}.pth")
                    torch.save(critic.state_dict(), model_dir / f"ddpg_critic_{tmp_name}.pth")
                else: 
                    print("No tmp_name provided; models not saved during real robot training.")
        if env_type == "real":
            key = input("Continue training? ").lower()
            if key != "y":
                print("Training aborted by user.")
                sys.exit(0)


        # -------------------------------------------------
        # Logging / prints
        # -------------------------------------------------
        if rl_cfg["training"]["do_prints"]:
            print("========================================")
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward:.4f}")
            dist_to_hole = np.linalg.norm(np.array([ball_x, ball_y]) - hole_pos)
            print(f"  Distance to Hole: {dist_to_hole:.4f}, In Hole: {in_hole}")

        # Periodic evaluation of greedy policy (no exploration noise)
        if (episode) % rl_cfg["training"]["eval_interval"] == 24 and env_type == "sim":
            (
                success_rate_eval,
                avg_reward_eval,
                avg_distance_to_hole_eval,
            ) = evaluation_policy_short(
                actor,
                device,
                mujoco_cfg,
                rl_cfg,
                num_episodes=rl_cfg["training"]["eval_episodes"],
                max_num_discs=max_num_discs,
                env_step=env_step,
                env_type=env_type,
            )

            # Linearly decay policy exploration noise
            stage_len = max(1, episode - stage_start_episode)
            
            if max_num_discs >= 3:
                horizon = 6000
            else:
                horizon = 20000

            frac = min(1.0, stage_len / horizon)
            noise_std = noise_std_stage_start + frac * (noise_std_end - noise_std_stage_start)
            # frac = episode / max(1, episodes)
            # noise_std = noise_std_start + frac * (noise_std_end - noise_std_start)
            last_last_success_rate = last_success_rate
            last_success_rate = success_rate_eval
            print(
                f"[EVAL] Success Rate after {episode + 1} episodes: "
                f"{success_rate_eval:.2f}, Avg Reward: {avg_reward_eval:.3f}"
            )

            if use_wandb:
                log_dict["success_rate"]          = success_rate_eval
                log_dict["avg_reward"]            = avg_reward_eval
                log_dict["avg_distance_to_hole"]  = avg_distance_to_hole_eval
                log_dict["noise_std"]             = noise_std

        if use_wandb:
            distance_to_hole = np.linalg.norm(np.array([ball_x, ball_y]) - hole_pos)
            log_dict["reward"]            = reward
            log_dict["distance_to_hole"]  = distance_to_hole
            log_dict["max_num_discs"]     = max_num_discs
            # if critic_loss_value is not None:
            #     log_dict["critic_loss"] = critic_loss_value
            # if actor_loss_value is not None:
            #     log_dict["actor_loss"]  = actor_loss_value

            wandb.log(log_dict, step=episode)
        

    # -------------------------------------------------
    # Final evaluation
    # -------------------------------------------------
    if env_type == "sim":
        (final_success_rate, final_avg_reward, final_avg_distance_to_hole,) = evaluation_policy_short(
            actor,
            device,
            mujoco_cfg,
            rl_cfg,
            100,
            max_num_discs=max_num_discs,
            env_step=env_step,
            env_type=env_type,
        )

    if use_wandb:
        wandb.log({"final_avg_reward":           final_avg_reward})
        wandb.log({"final_success_rate":         final_success_rate})
        wandb.log({"final_avg_distance_to_hole": final_avg_distance_to_hole})
        torch.save(actor.state_dict(),  model_dir / f"ddpg_actor_{run_name}.pth")
        torch.save(critic.state_dict(), model_dir / f"ddpg_critic_{run_name}.pth")
        print("Sweep complete. Models saved.")
    
    if episode_logger is not None:
        episode_logger.close()

    elif tmp_name is not None:
        torch.save(actor.state_dict(),  model_dir / f"ddpg_actor_{tmp_name}.pth")
        torch.save(critic.state_dict(), model_dir / f"ddpg_critic_{tmp_name}.pth")
        print("Training complete. Models saved.")


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


# ---------------------------------------------------------
# Helper: map normalized actions -> (speed, angle_deg)
# ---------------------------------------------------------
def get_input_parameters(a_norm, speed_low, speed_high, angle_low, angle_high):
    speed_norm, angle_norm = a_norm
    speed     = squash_to_range(speed_norm,  speed_low,  speed_high)
    angle_deg = squash_to_range(angle_norm, angle_low, angle_high)
    return speed, angle_deg


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
    env_type="sim"
):
    
    if env_step is None:
        raise ValueError("evaluation_policy_short() requires env_step.")
    
    speed_low  = rl_cfg["model"]["speed_low"]
    speed_high = rl_cfg["model"]["speed_high"]
    angle_low  = rl_cfg["model"]["angle_low"]
    angle_high = rl_cfg["model"]["angle_high"]

    successes          = 0
    rewards            = []
    distances_to_hole  = []
    # max_num_discs = 5
    actor.eval()
    with torch.no_grad():
        for _ in range(num_episodes):
            # New random context each eval episode
            if env_type == "sim":
                ball_start_obs, hole_pos_obs, disc_positions, x, y, hole_pos = sim_init_parameters(mujoco_cfg, max_num_discs)
            
            if env_type == "real":
                ball_start_obs, hole_pos_obs, disc_positions = real_init_parameters()

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

            reward = compute_reward(
                ball_end_pos=np.array([ball_x, ball_y]),
                hole_pos=hole_pos,
                in_hole=in_hole,
                meta=meta,
                distance_scale=rl_cfg["reward"]["distance_scale"],
                in_hole_reward=rl_cfg["reward"]["in_hole_reward"],
            )

            rewards.append(reward)
            successes += int(in_hole == 1)
            distance_to_hole = np.linalg.norm(np.array([ball_x, ball_y]) - hole_pos)
            distances_to_hole.append(distance_to_hole)

    avg_distance_to_hole = float(np.mean(distances_to_hole)) if distances_to_hole else 0.0
    actor.train()
    return successes / num_episodes, float(np.mean(rewards)), avg_distance_to_hole


# ---------------------------------------------------------
# Context sampling helper
# ---------------------------------------------------------
def random_hole_in_rectangle(x_min=3.0, x_max=5.0, y_min=-0.5, y_max=0.5):
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    return x, y


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    here    = Path(__file__).resolve().parent
    sim_dir = here / "simulation"
    sys.path.append(str(sim_dir))

    project_root        = here.parents[1]
    mujoco_config_path  = project_root / "configs" / "mujoco_config.yaml"
    rl_config_path      = project_root / "configs" / "rl_config.yaml"

    with open(mujoco_config_path, "r") as f:
        mujoco_cfg = yaml.safe_load(f)

    with open(rl_config_path, "r") as f:
        rl_cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Select environment: SIM vs REAL
    # ------------------------------------------------------------------
    env_type = rl_cfg["training"].get("env_type", "sim")

    if env_type == "sim":
        # Import simulator entry point
        from run_sim_rl import run_sim as env_step

        # Temporary XML path (sim only)
        tmp_name     = f"golf_world_tmp_{os.getpid()}_{uuid.uuid4().hex}.xml"
        tmp_xml_path = project_root / "models" / "mujoco" / tmp_name
        mujoco_cfg["sim"]["xml_path"] = str(tmp_xml_path)
    elif env_type == "real":
        """
        For the real robot, you should implement a function with the same
        signature as run_sim:

            def run_real(angle_deg, speed, hole_xy, env_cfg, disc_positions):
                ...
                return ball_x, ball_y, in_hole, trajectory

        and place it e.g. in run_real_rl.py next to run_sim_rl.py
        """
        from run_real_rl import run_real as env_step

        # For real robot, the mujoco_cfg may be unused or used only for
        # shared parameters; we do NOT create a temporary XML.
        tmp_xml_path = None
    else:
        raise ValueError(f"Unknown env_type: {env_type} (expected 'sim' or 'real')")

    # Optional: wandb sweeps
    if rl_cfg["training"]["use_wandb"]:
        sweep_config = {
            "actor_lr":          rl_cfg["training"]["actor_lr"],
            "critic_lr":         rl_cfg["training"]["critic_lr"],
            "noise_std":         rl_cfg["training"]["noise_std"],
            "hidden_dim":        rl_cfg["model"]["hidden_dim"],
            "batch_size":        rl_cfg["training"]["batch_size"],
            "grad_steps":        rl_cfg["training"]["grad_steps"],
        }

        wandb.init(
            project="rl_golf_contextual_bandit",
            config={
                **sweep_config,
                "rl_config":     rl_cfg,
                "mujoco_config": mujoco_cfg,
            },
        )

        cfg = wandb.config
        rl_cfg["reward"]["distance_scale"]   = cfg.get("distance_scale", rl_cfg["reward"]["distance_scale"])
        rl_cfg["reward"]["in_hole_reward"]   = cfg.get("in_hole_reward", rl_cfg["reward"]["in_hole_reward"])

    # Train contextual bandit policy
    training(
        rl_cfg,
        mujoco_cfg,
        project_root,
        continue_training=rl_cfg["training"]["continue_training"],
        env_step=env_step,
        env_type=env_type,
        tmp_name=tmp_name if env_type == "real" else None,
    )

    # Clean up temporary XML if it exists (sim only)
    if env_type == "sim" and tmp_xml_path is not None:
        try:
            os.remove(tmp_xml_path)
        except FileNotFoundError:
            pass
