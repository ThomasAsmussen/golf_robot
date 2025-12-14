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

# ---------------------------------------------------------
# Global noise parameters (environment / measurement noise)
# ---------------------------------------------------------
SPEED_NOISE_STD      = 0.0
ANGLE_NOISE_STD      = 0.0
BALL_OBS_NOISE_STD   = 0.0
HOLE_OBS_NOISE_STD   = 0.0


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


# ---------------------------------------------------------
# Reward and state encoding
# ---------------------------------------------------------
def compute_reward(ball_end_pos, hole_pos, in_hole, trajectory):
    """
    Single-step reward.
    - If in_hole: large positive reward.
    - Else: distance-based shaping.
    """
    if in_hole:
        return 3.0

    final_dist = np.linalg.norm(ball_end_pos - hole_pos)
    final_dist_reward = np.exp(-0.5 * final_dist)
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


# ---------------------------------------------------------
# Normalizer
# ---------------------------------------------------------
class Normalizer:
    """
    Online normalization of state features.
    """
    def __init__(self, size, eps=1e-8):
        self.size = size
        self.mean = np.zeros(size)
        self.var = np.ones(size)
        self.count = eps

    def update(self, x):
        x = np.asarray(x)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += delta * (x - self.mean)

    def std(self):
        return np.sqrt(self.var / self.count)

    def normalize(self, x):
        return (x - self.mean) / (self.std() + 1e-6)


# ---------------------------------------------------------
# Training loop (Contextual Bandit)
# ---------------------------------------------------------
def training(rl_cfg, mujoco_cfg, project_root, continue_training=False):
    """
    Contextual bandit:
      - context = (ball_start_obs, hole_pos_obs, discs)
      - action  = (speed, angle)
      - reward  = f(final ball position, hole position, in_hole)

    No bootstrapping, no next_state.
    Critic learns Q(s,a) ≈ E[r | s,a].
    Actor learns to maximize Q(s, π(s)).
    """
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

    normalizer = Normalizer(size=state_dim)

    # Linear schedule for exploration noise (policy noise)
    noise_std_start = noise_std
    noise_std_end   = 0.05

    use_wandb = rl_cfg["training"]["use_wandb"]
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
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        critic = Critic(state_dim, action_dim, hidden_dim).to(device)

    print(f"Using device: {device}")

    actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    replay_buffer = ReplayBuffer(capacity=rl_cfg["training"]["replay_buffer_capacity"])

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

    # For now we don't actually place discs, but the state format reserves 5.
    

    for episode in range(episodes):
        # -------------------------------------------------
        # Sample a context (ball start + hole + discs)
        # -------------------------------------------------
        max_num_discs = np.min([5, episode // 3000])  # Increase discs over time
        # print(f"Episode {episode + 1}: max_num_discs = {max_num_discs}")
        ball_start = np.random.uniform(-0.5, 0.5, size=(2,))
        # ball_start = np.array([0, 0])  # FIXED START FOR DEBUGGING
        ball_start_obs = ball_start + np.random.normal(0, BALL_OBS_NOISE_STD, size=(2,))
        mujoco_cfg["ball"]["start_pos"]     = [ball_start[0],     ball_start[1],     0.02135]
        mujoco_cfg["ball"]["obs_start_pos"] = [ball_start_obs[0], ball_start_obs[1], 0.02135]

        x, y = random_hole_in_rectangle(x_min=3.0, x_max=5.0, y_min=-0.5, y_max=0.5)
        hole_pos = np.array([x, y])
        hole_pos_obs = hole_pos + np.random.normal(0, HOLE_OBS_NOISE_STD, size=(2,))

        disc_positions = generate_disc_positions(
            max_num_discs, x - 2.0, x, -0.5, 0.5, hole_xy=hole_pos
        ) # No discs for now

        state_vec = encode_state_with_discs(ball_start_obs, hole_pos_obs, disc_positions, 5)
        # state_vec = np.concatenate([ball_start_obs, hole_pos_obs])  # No discs for now
        normalizer.update(state_vec)
        state_norm = normalizer.normalize(state_vec)

        s = torch.tensor(state_norm, dtype=torch.float32).to(device)
        critic_loss_value = None
        actor_loss_value = None

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

        speed, angle_deg = get_sim_input(
            a_noisy.cpu().numpy(), speed_low, speed_high, angle_low, angle_high
        )

        # Environment / actuator noise
        speed_noise      = np.random.normal(0, SPEED_NOISE_STD)
        angle_deg_noise  = np.random.normal(0, ANGLE_NOISE_STD)
        speed     = np.clip(speed + speed_noise,     speed_low, speed_high)
        angle_deg = np.clip(angle_deg + angle_deg_noise, angle_low, angle_high)

        # -------------------------------------------------
        # One-step environment: simulate and get reward
        # -------------------------------------------------
        result = run_sim(angle_deg, speed, [x, y], mujoco_cfg, disc_positions)

        if result is None:
            result = run_sim(angle_deg, speed, [x, y], mujoco_cfg, disc_positions)
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

        ball_x, ball_y, in_hole, trajectory = result

        reward = compute_reward(
            ball_end_pos=np.array([ball_x, ball_y]),
            hole_pos=hole_pos,
            in_hole=in_hole,
            trajectory=trajectory,
        )

        # Store (s,a,r) in buffer (contextual bandit)
        s_train = s.detach().cpu()
        a_train = a_noisy.detach().cpu()
        replay_buffer.add(s_train, a_train, reward)

        # -------------------------------------------------
        # TD-style supervised update: Q(s,a) -> r
        # -------------------------------------------------
        if len(replay_buffer.data) >= batch_size:
            for _ in range(grad_steps):
                states_b, actions_b, rewards_b = replay_buffer.sample(batch_size)

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

        # -------------------------------------------------
        # Logging / prints
        # -------------------------------------------------
        if rl_cfg["training"]["do_prints"]:
            print("========================================")
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward:.4f}")
            dist_to_hole = np.linalg.norm(np.array([ball_x, ball_y]) - hole_pos)
            print(f"  Distance to Hole: {dist_to_hole:.4f}, In Hole: {in_hole}")

        # Periodic evaluation of greedy policy (no exploration noise)
        if (episode) % rl_cfg["training"]["eval_interval"] == 24:
            (
                success_rate_eval,
                avg_reward_eval,
                avg_distance_to_hole_eval,
            ) = evaluation_policy_short(
                actor,
                device,
                mujoco_cfg,
                project_root,
                rl_cfg,
                num_episodes=rl_cfg["training"]["eval_episodes"],
                normalizer=normalizer,
                max_num_discs=max_num_discs,
            )

            # Linearly decay policy exploration noise
            frac = episode / max(1, episodes)
            noise_std = noise_std_start + frac * (noise_std_end - noise_std_start)

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
            if critic_loss_value is not None:
                log_dict["critic_loss"] = critic_loss_value
            if actor_loss_value is not None:
                log_dict["actor_loss"]  = actor_loss_value

            wandb.log(log_dict, step=episode)

    # -------------------------------------------------
    # Final evaluation
    # -------------------------------------------------
    (
        final_success_rate,
        final_avg_reward,
        final_avg_distance_to_hole,
    ) = evaluation_policy_short(
        actor,
        device,
        mujoco_cfg,
        project_root,
        rl_cfg,
        100,
        normalizer=normalizer,
        max_num_discs=max_num_discs,
    )

    if use_wandb:
        wandb.log({"final_avg_reward":           final_avg_reward})
        wandb.log({"final_success_rate":         final_success_rate})
        wandb.log({"final_avg_distance_to_hole": final_avg_distance_to_hole})

    print("Sweep complete")
    torch.save(actor.state_dict(),  model_dir / f"ddpg_actor_{run_name}.pth")
    torch.save(critic.state_dict(), model_dir / f"ddpg_critic_{run_name}.pth")
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
def get_sim_input(a_norm, speed_low, speed_high, angle_low, angle_high):
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
    project_root,
    rl_cfg,
    num_episodes,
    normalizer=None,
    max_num_discs=5,
):
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
            ball_start     = np.random.uniform(-0.5, 0.5, size=(2,))

            # ball_start = np.array([0, 0])  # FIXED START FOR DEBUGGING
            ball_start_obs = ball_start + np.random.normal(0, BALL_OBS_NOISE_STD, size=(2,))

            mujoco_cfg["ball"]["start_pos"]     = [ball_start[0],     ball_start[1],     0.02135]
            mujoco_cfg["ball"]["obs_start_pos"] = [ball_start_obs[0], ball_start_obs[1], 0.02135]

            x, y = random_hole_in_rectangle(x_min=3.0, x_max=5.0, y_min=-0.5, y_max=0.5)
            hole_pos = np.array([x, y])
            hole_pos_obs = hole_pos + np.random.normal(0, HOLE_OBS_NOISE_STD, size=(2,))

            disc_positions = generate_disc_positions(
                max_num_discs, x - 2.0, x, -0.5, 0.5, hole_xy=hole_pos
            )

            state_vec = encode_state_with_discs(
                ball_start_obs, hole_pos_obs, disc_positions, max_num_discs=5
            )

            # state_vec = np.concatenate([ball_start_obs, hole_pos_obs])  # No discs for now

            if normalizer is not None:
                state_vec = normalizer.normalize(state_vec)

            state = torch.tensor(
                state_vec, dtype=torch.float32, device=device
            ).unsqueeze(0)

            a_norm = actor(state).squeeze(0).cpu().numpy()
            speed, angle_deg = get_sim_input(
                a_norm, speed_low, speed_high, angle_low, angle_high
            )

            # Apply same actuator noise as during training
            speed_noise     = np.random.normal(0, SPEED_NOISE_STD)
            angle_deg_noise = np.random.normal(0, ANGLE_NOISE_STD)
            speed     = np.clip(speed + speed_noise,     speed_low,  speed_high)
            angle_deg = np.clip(angle_deg + angle_deg_noise, angle_low, angle_high)

            ball_x, ball_y, in_hole, trajectory = run_sim(
                angle_deg, speed, [x, y], mujoco_cfg, disc_positions
            )

            reward = compute_reward(
                ball_end_pos=np.array([ball_x, ball_y]),
                hole_pos=hole_pos,
                in_hole=in_hole,
                trajectory=trajectory,
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
    sim_dir = here.parent / "simulation"
    sys.path.append(str(sim_dir))

    # Import simulator entry point
    from run_sim_rl import run_sim

    project_root        = here.parents[2]
    mujoco_config_path  = project_root / "configs" / "mujoco_config.yaml"
    rl_config_path      = project_root / "configs" / "rl_config.yaml"

    with open(mujoco_config_path, "r") as f:
        mujoco_cfg = yaml.safe_load(f)

    with open(rl_config_path, "r") as f:
        rl_cfg = yaml.safe_load(f)

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
        rl_cfg["training"]["actor_lr"]   = cfg.actor_lr
        rl_cfg["training"]["critic_lr"]  = cfg.critic_lr
        rl_cfg["training"]["noise_std"]  = cfg.noise_std
        rl_cfg["model"]["hidden_dim"]    = cfg.hidden_dim
        rl_cfg["training"]["batch_size"] = cfg.batch_size
        rl_cfg["training"]["grad_steps"] = cfg.grad_steps

    # Temporary XML path
    tmp_name     = f"golf_world_tmp_{os.getpid()}_{uuid.uuid4().hex}.xml"
    tmp_xml_path = project_root / "models" / "mujoco" / tmp_name
    mujoco_cfg["sim"]["xml_path"] = str(tmp_xml_path)

    # Train contextual bandit policy
    training(
        rl_cfg,
        mujoco_cfg,
        project_root,
        continue_training=rl_cfg["training"]["continue_training"],
    )

    # Clean up temporary XML if it exists
    try:
        os.remove(tmp_xml_path)
    except FileNotFoundError:
        pass
