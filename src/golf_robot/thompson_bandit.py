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
import time

from rl_common import *


# ---------------------------------------------------------
# Small helper: Ensemble actor for evaluation (mean of heads)
# ---------------------------------------------------------
class EnsembleMeanActor(nn.Module):
    """
    Wrap a list of deterministic Actors and return the mean action.
    This is useful for evaluation (greedy-ish) without picking a random head.
    """
    def __init__(self, actors):
        super().__init__()
        self.actors = nn.ModuleList(actors)

    def forward(self, state):
        # state: [B, state_dim]
        outs = [a(state) for a in self.actors]
        return torch.stack(outs, dim=0).mean(dim=0)


# ---------------------------------------------------------
# Helper: Find newest Thompson checkpoints (head0 as reference)
# ---------------------------------------------------------
def find_latest_thompson_checkpoint(model_dir: Path, prefix: str | None = None):
    """
    Find newest checkpoint set by looking for head0 actor file:
      ddpg_actor_<something>_h0.pth
    If prefix is given, only match files containing that prefix.
    Returns: (actor_h0_path, critic_h0_path, tag_stem)
    where tag_stem is the stem with "_h0" stripped, so you can load all heads.
    """
    actor_files = list(model_dir.glob("ddpg_actor_*_h0.pth"))
    if prefix is not None:
        actor_files = [f for f in actor_files if prefix in f.stem]
    if not actor_files:
        raise FileNotFoundError("No matching Thompson (head0) actor checkpoints found.")

    actor_h0 = max(actor_files, key=lambda f: f.stat().st_mtime)
    critic_h0 = actor_h0.with_name(actor_h0.name.replace("ddpg_actor_", "ddpg_critic_"))

    if not critic_h0.exists():
        raise FileNotFoundError(f"Matching critic not found for {actor_h0.name}")

    # Strip "_h0" from stem to get the "tag"
    tag_stem = actor_h0.stem.replace("_h0", "")
    return actor_h0, critic_h0, tag_stem


def load_thompson_heads(model_dir: Path, rl_cfg, tag_stem: str, device):
    """
    Load all heads for Thompson:
      expects files:
        ddpg_actor_<tag>_h{k}.pth
        ddpg_critic_<tag>_h{k}.pth
    where <tag> is tag_stem without "ddpg_actor_".
    """
    K = rl_cfg["training"].get("num_heads", 8)
    state_dim = rl_cfg["model"]["state_dim"]
    action_dim = rl_cfg["model"]["action_dim"]
    hidden_dim = rl_cfg["model"]["hidden_dim"]

    # tag_stem includes "ddpg_actor_..." stem name; we will use it directly
    # Example tag_stem: "ddpg_actor_runname"
    # We need its suffix after "ddpg_actor_" to construct filenames.
    if not tag_stem.startswith("ddpg_actor_"):
        raise ValueError(f"Unexpected tag_stem format: {tag_stem}")

    tag = tag_stem.replace("ddpg_actor_", "", 1)

    actors = []
    critics = []
    for k in range(K):
        actor_path = model_dir / f"ddpg_actor_{tag}_h{k}.pth"
        critic_path = model_dir / f"ddpg_critic_{tag}_h{k}.pth"

        if not actor_path.exists() or not critic_path.exists():
            raise FileNotFoundError(
                f"Missing Thompson head files for k={k}:\n  {actor_path}\n  {critic_path}"
            )

        actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        critic = Critic(state_dim, action_dim, hidden_dim).to(device)

        actor.load_state_dict(torch.load(actor_path, map_location=device))
        critic.load_state_dict(torch.load(critic_path, map_location=device))

        actor.train()
        critic.train()

        actors.append(actor)
        critics.append(critic)

    return actors, critics, tag


def save_thompson_heads(model_dir: Path, tag: str, actors, critics):
    """
    Save heads to:
      ddpg_actor_<tag>_h{k}.pth
      ddpg_critic_<tag>_h{k}.pth
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    for k, (a, c) in enumerate(zip(actors, critics)):
        torch.save(a.state_dict(), model_dir / f"ddpg_actor_{tag}_h{k}.pth")
        torch.save(c.state_dict(), model_dir / f"ddpg_critic_{tag}_h{k}.pth")


# ---------------------------------------------------------
# Training loop (Thompson Sampling via bootstrapped heads)
# ---------------------------------------------------------
def training(
    rl_cfg,
    mujoco_cfg,
    project_root,
    continue_training=False,
    input_func=None,
    env_step=None,
    env_type="sim",
    tmp_name=None,
    camera_index_start=None,
):
    """
    Thompson Sampling (bootstrapped ensemble) contextual bandit for golf robot.

    - context = (ball_start_obs, hole_pos_obs, discs)
    - action  = (speed, angle) continuous (normalized to [-1,1]^2 inside network)
    - reward  = compute_reward(...) based on terminal outcome / shaped distance

    One-step bandit:
      - each episode: sample a head k, act with actor_k(s)
      - training: each head is trained on a bootstrap mask of minibatches
    """
    if env_step is None:
        raise ValueError("training() requires env_step (sim or real environment function).")

    episodes   = rl_cfg["training"]["episodes"]
    batch_size = rl_cfg["training"]["batch_size"]
    actor_lr   = rl_cfg["training"]["actor_lr"]
    critic_lr  = rl_cfg["training"]["critic_lr"]
    grad_steps = rl_cfg["training"]["grad_steps"]

    # Thompson controls
    K = rl_cfg["training"].get("num_heads", 8)
    bootstrap_p = rl_cfg["training"].get("bootstrap_p", 0.8)  # per-head mask prob
    ts_action_noise_std = rl_cfg["training"].get("thompson_action_noise_std", 0.0)  # optional tiny noise

    # Old exploration noise in the script (now mostly unnecessary)
    # Keep it but default it to tiny in config, or we override below.
    noise_std  = rl_cfg["training"]["noise_std"]

    state_dim  = rl_cfg["model"]["state_dim"]
    action_dim = rl_cfg["model"]["action_dim"]
    hidden_dim = rl_cfg["model"]["hidden_dim"]
    speed_low  = rl_cfg["model"]["speed_low"]
    speed_high = rl_cfg["model"]["speed_high"]
    angle_low  = rl_cfg["model"]["angle_low"]
    angle_high = rl_cfg["model"]["angle_high"]

    in_hole_reward = rl_cfg["reward"]["in_hole_reward"]
    distance_scale = rl_cfg["reward"]["distance_scale"]
    w_distance     = rl_cfg["reward"]["w_distance"]
    optimal_speed  = rl_cfg["reward"]["optimal_speed"]
    dist_at_hole_scale = rl_cfg["reward"]["dist_at_hole_scale"]
    optimal_speed_scale = rl_cfg["reward"]["optimal_speed_scale"]

    hole_positions = get_hole_positions()

    # With Thompson, you generally do NOT want large additive Gaussian action noise.
    # Keep it but make it tiny so it doesn't fight the exploration mechanism.
    noise_std = min(float(noise_std), 0.05)

    if env_type == "sim":
        use_wandb = rl_cfg["training"]["use_wandb"]
    else:
        use_wandb = False

    model_name = rl_cfg["training"].get("model_name", None)

    # -------------------------
    # Initialize models (K heads)
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dir = project_root / "models" / "rl" / "bandit"
    model_dir.mkdir(parents=True, exist_ok=True)

    tag_for_saving = None

    if continue_training:
        # Load the newest Thompson checkpoint set (based on head0 timestamp),
        # or load a specific tag if model_name is provided.
        if env_type == "real":
            # For real: load absolutely newest by mtime on head0 (optionally filter by model_name prefix)
            prefix = model_name if model_name else None
            actor_h0, critic_h0, tag_stem = find_latest_thompson_checkpoint(model_dir, prefix=prefix)
            print("Loading latest Thompson checkpoints:")
            print(f"  Actor0 : {actor_h0.name}")
            print(f"  Critic0: {critic_h0.name}")
            actors, critics, loaded_tag = load_thompson_heads(model_dir, rl_cfg, tag_stem, device)
            tag_for_saving = loaded_tag
        else:
            # Sim: if model_name is given, treat it as the tag directly
            if model_name is None:
                actor_h0, critic_h0, tag_stem = find_latest_thompson_checkpoint(model_dir, prefix=None)
                print("Continuing training from latest Thompson checkpoints:")
                print(f"  Actor0 : {actor_h0.name}")
                print(f"  Critic0: {critic_h0.name}")
                actors, critics, loaded_tag = load_thompson_heads(model_dir, rl_cfg, tag_stem, device)
                tag_for_saving = loaded_tag
            else:
                # expected filenames: ddpg_actor_<model_name>_h0.pth etc
                tag_for_saving = model_name
                # Construct a "tag_stem" that matches our loader expectations
                tag_stem = f"ddpg_actor_{model_name}"
                print(f"Continuing training from tag: {model_name}")
                actors, critics, _ = load_thompson_heads(model_dir, rl_cfg, tag_stem, device)

    else:
        # Fresh init
        actors  = [Actor(state_dim, action_dim, hidden_dim).to(device) for _ in range(K)]
        critics = [Critic(state_dim, action_dim, hidden_dim).to(device) for _ in range(K)]
        # Tag for saving: use wandb run name or local_run-like
        tag_for_saving = None

    actor_optimizers  = [torch.optim.SGD(a.parameters(), lr=actor_lr) for a in actors]
    critic_optimizers = [torch.optim.SGD(c.parameters(), lr=critic_lr) for c in critics]

    replay_buffer_big = ReplayBuffer(capacity=rl_cfg["training"]["replay_buffer_capacity"])
    replay_buffer_recent = ReplayBuffer(1000)

    if use_wandb:
        # Watch just head0 to avoid noisy logs
        wandb.watch(actors[0],  log="gradients", log_freq=100)
        wandb.watch(critics[0], log="gradients", log_freq=100)
        run_name = wandb.run.name.replace("-", "_")
    else:
        run_name = "local_run"

    # Decide a stable tag to save
    # If continuing, keep existing tag; else new tag = run_name (or tmp_name for real if provided)
    if tag_for_saving is None:
        if env_type == "real" and tmp_name is not None:
            tag_for_saving = tmp_name
        else:
            tag_for_saving = run_name

    # For evaluation we’ll use the mean actor across heads
    eval_actor = EnsembleMeanActor(actors).to(device)

    # Logging
    log_dict = {}
    last_success_rate = 0.0
    last_last_success_rate = 0.0

    max_num_discs = 2
    stage_start_episode = 0

    episode_logger = None
    episode_log_path = None

    if env_type == "real":
        episode_log_path = project_root / "log" / "real_episodes" / "episode_logger.jsonl"
        episode_logger = EpisodeLoggerJsonl(episode_log_path)

        loaded_n = load_replay_from_jsonl(
            episode_log_path,
            replay_buffer_big,
            replay_buffer_recent,
            max_recent=1000,
            reward_cfg=rl_cfg.get("reward", None),
        )

        if loaded_n > 0:
            print(f"Loaded {loaded_n} episodes from {episode_log_path} into replay buffer.")
        else:
            print(f"No existing episode log found at {episode_log_path}. Starting fresh.")

    for episode in range(episodes):
        # Periodic evaluation of greedy-ish policy
        if env_type == "real" and episode_logger.get_length() % 10000 == 0:
            (
                success_rate_eval,
                avg_reward_eval,
                avg_distance_to_hole_eval,
            ) = evaluation_policy_short(
                eval_actor,
                device,
                mujoco_cfg=None,
                rl_cfg=rl_cfg,
                num_episodes=3,
                env_step=env_step,
                env_type=env_type,
                input_func=input_func,
                big_episode_logger=episode_logger,
            )
            print(
                f"[EVAL] Success Rate after {episode + 1} episodes: "
                f"{success_rate_eval:.2f}, Avg Reward: {avg_reward_eval:.3f}"
            )

        # -------------------------------------------------
        # Sample a context (ball start + hole + discs)
        # -------------------------------------------------
        if env_type == "sim":
            if last_success_rate > 0.9 and last_last_success_rate > 0.9 and False:
                max_num_discs = min(MAX_DISCS, max_num_discs + 1)
                last_success_rate = 0.0
                last_last_success_rate = 0.0
                stage_start_episode = episode
                replay_buffer_recent.clear()

            ball_start_obs, hole_pos_obs, disc_positions, x, y, hole_pos = sim_init_parameters(
                mujoco_cfg, max_num_discs
            )

        if env_type == "real":
            ball_start_obs, hole_pos_obs, disc_positions, chosen_hole = input_func(
                camera_index=camera_index_start
            )

        # -------------------------------------------------
        # Encode + scale state
        # -------------------------------------------------
        # NOTE: keeping your original training state encoding (no engineered aug here),
        # because you asked to only change where Thompson needs to be.
        state_vec = encode_state_with_discs(ball_start_obs, hole_pos_obs, disc_positions, 5)
        state_norm = scale_state_vec(state_vec)
        s = torch.tensor(state_norm, dtype=torch.float32, device=device)

        # -------------------------------------------------
        # Thompson action selection: sample head k, act with actor_k
        # -------------------------------------------------
        head = int(np.random.randint(K))

        with torch.no_grad():
            a_norm = actors[head](s.unsqueeze(0)).squeeze(0)

            # Optional: tiny additive noise (not required)
            if ts_action_noise_std > 0:
                a_norm = a_norm + torch.normal(
                    mean=torch.zeros_like(a_norm),
                    std=ts_action_noise_std * torch.ones_like(a_norm),
                )

            # Optional: keep the old noise_std but clipped tiny
            if noise_std > 0:
                a_norm = a_norm + torch.normal(
                    mean=torch.zeros_like(a_norm),
                    std=noise_std * torch.ones_like(a_norm),
                )

        a_noisy = torch.clamp(a_norm, -1.0, 1.0)

        speed, angle_deg = get_input_parameters(
            a_noisy.detach().cpu().numpy(), speed_low, speed_high, angle_low, angle_high
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
        else:
            result = env_step(
                impact_velocity=speed,
                swing_angle=angle_deg,
                ball_start_position=ball_start_obs,
                planner="quintic",
                check_rtt=True,
                chosen_hole=chosen_hole,
            )

        if result is None:
            # retry once
            if env_type == "sim":
                result = env_step(angle_deg, speed, [x, y], mujoco_cfg, disc_positions)
            else:
                result = env_step(
                    impact_velocity=speed,
                    swing_angle=angle_deg,
                    ball_start_position=ball_start_obs,
                    planner="quintic",
                    check_rtt=True,
                    chosen_hole=chosen_hole,
                )

            if result is None:
                print(
                    f"Episode {episode + 1}: Simulation failed twice. "
                    f"Bad action: speed={speed}, angle={angle_deg}"
                )
                if rl_cfg["training"]["error_hard_stop"] and env_type == "sim" and not mujoco_cfg["sim"]["render"]:
                    raise RuntimeError("Simulation failed twice — aborting training.")
                else:
                    continue

        ball_x, ball_y, in_hole, meta = result

        if env_type == "sim":
            meta = meta_from_trajectory_xy(meta, hole_pos_obs)

        is_out_of_bounds = False
        using_all_holes = False

        if env_type == "real" and isinstance(meta, dict):
            is_out_of_bounds = bool(meta.get("out_of_bounds", False))
            if meta.get("wrong_hole", None) is not None:
                chosen_hole = meta["wrong_hole"]
                hole_pos_obs = np.array([get_hole_positions()[chosen_hole]["x"], get_hole_positions()[chosen_hole]["y"]])
            elif not in_hole:
                using_all_holes = True
                hole1 = np.array([get_hole_positions()[1]["x"], get_hole_positions()[1]["y"]])
                hole2 = np.array([get_hole_positions()[2]["x"], get_hole_positions()[2]["y"]])
                hole3 = np.array([get_hole_positions()[3]["x"], get_hole_positions()[3]["y"]])

        if using_all_holes:
            rewards = []
            for i, hole_pos_obs_try in enumerate([hole1, hole2, hole3]):
                dist_at_hole = meta.get("dist_at_hole", None)
                if dist_at_hole is not None:
                    dist_at_hole = float(dist_at_hole[i])
                reward_try = compute_reward(
                    ball_end_xy=np.array([ball_x, ball_y]),
                    hole_xy=hole_pos_obs_try,
                    in_hole=in_hole,
                    meta=meta,
                    is_out_of_bounds=is_out_of_bounds,
                    in_hole_reward=in_hole_reward,
                    distance_scale=distance_scale,
                    w_distance=w_distance,
                    optimal_speed=optimal_speed,
                    dist_at_hole_scale=dist_at_hole_scale,
                    optimal_speed_scale=optimal_speed_scale,
                    multiple_dist_at_hole=dist_at_hole,
                )
                rewards.append(reward_try)
            reward = float(max(rewards))  # for printing/logging only
        else:
            reward = compute_reward(
                ball_end_xy=np.array([ball_x, ball_y]),
                hole_xy=hole_pos_obs,
                in_hole=in_hole,
                meta=meta,
                is_out_of_bounds=is_out_of_bounds,
                in_hole_reward=in_hole_reward,
                distance_scale=distance_scale,
                w_distance=w_distance,
                optimal_speed=optimal_speed,
                dist_at_hole_scale=dist_at_hole_scale,
                optimal_speed_scale=optimal_speed_scale,
            )

        # -------------------------------------------------
        # Real-robot logging
        # -------------------------------------------------
        if env_type == "real" and isinstance(meta, dict):
            used_for_training = bool(meta.get("used_for_training", True))
            out_of_bounds = bool(meta.get("out_of_bounds", False))

            if not used_for_training:
                print("Episode discarded by user; not adding to replay buffer.")
                continue
            else:
                print("Storing episode")
                logging_dict = {
                    "episode": episode,
                    "time": time.time(),
                    "used_for_training": used_for_training,
                    "ball_start_obs": ball_start_obs.tolist(),
                    "hole_pos_obs": hole_pos_obs.tolist(),
                    "disc_positions": disc_positions,
                    "state_norm": state_norm.tolist(),
                    "action_norm": a_noisy.detach().cpu().numpy().tolist(),
                    "speed": speed,
                    "angle_deg": angle_deg,
                    "ball_final_pos": [ball_x, ball_y],
                    "in_hole": bool(in_hole),
                    "out_of_bounds": out_of_bounds,
                    "reward": float(reward),
                    "chosen_hole": chosen_hole,
                    "dist_at_hole": meta.get("dist_at_hole", None),
                    "speed_at_hole": meta.get("speed_at_hole", None),
                    "exploring": True,
                    "thompson_head": head,
                }
                episode_logger.log(logging_dict)

        # -------------------------------------------------
        # Store transition(s) in replay buffer
        # -------------------------------------------------
        s_train = s.detach().cpu()
        a_train = a_noisy.detach().cpu()

        if using_all_holes:
            # store all three as separate samples
            for r in rewards:
                replay_buffer_big.add(s_train, a_train, float(r))
                replay_buffer_recent.add(s_train, a_train, float(r))
        else:
            replay_buffer_big.add(s_train, a_train, float(reward))
            replay_buffer_recent.add(s_train, a_train, float(reward))

        # -------------------------------------------------
        # Thompson training: bootstrapped updates per head
        # -------------------------------------------------
        if len(replay_buffer_big.data) >= batch_size:
            if env_type == "real":
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

                td_target = rewards_b  # one-step bandit target

                # Update each head with its own bootstrap mask
                for k in range(K):
                    mask = (torch.rand(td_target.shape[0], device=device) < float(bootstrap_p))
                    if int(mask.sum().item()) < 2:
                        continue

                    sb = states_b[mask]
                    ab = actions_b[mask]
                    rb = td_target[mask]

                    # Critic: Q_k(s,a) -> r
                    q_pred = critics[k](sb, ab)
                    critic_loss = F.mse_loss(q_pred, rb)

                    critic_optimizers[k].zero_grad()
                    critic_loss.backward()
                    critic_optimizers[k].step()

                    # Actor: maximize its own critic
                    a_for_actor = actors[k](sb)
                    actor_loss = -critics[k](sb, a_for_actor).mean()

                    actor_optimizers[k].zero_grad()
                    actor_loss.backward()
                    actor_optimizers[k].step()

            # Save periodically for real robot
            if env_type == "real":
                if tmp_name is not None:
                    save_thompson_heads(model_dir, tag_for_saving, actors, critics)
                else:
                    print("No tmp_name provided; models not saved during real robot training.")

        if env_type == "real" and isinstance(meta, dict):
            if not meta.get("continue_training", True):
                print("Training aborted by user.")
                break

        # -------------------------------------------------
        # Prints
        # -------------------------------------------------
        if rl_cfg["training"]["do_prints"]:
            print("========================================")
            print(f"Episode {episode + 1}/{episodes}, Reward: {float(reward):.4f}, TS head: {head}")
            if env_type == "sim":
                dist_to_hole = np.linalg.norm(np.array([ball_x, ball_y]) - hole_pos)
                print(f"  Distance to Hole: {dist_to_hole:.4f}, In Hole: {bool(in_hole)}")

        # -------------------------------------------------
        # Periodic eval (sim)
        # -------------------------------------------------
        if (episode) % rl_cfg["training"]["eval_interval"] == 24 and env_type == "sim":
            (
                success_rate_eval,
                avg_reward_eval,
                avg_distance_to_hole_eval,
            ) = evaluation_policy_short(
                eval_actor,
                device,
                mujoco_cfg,
                rl_cfg,
                num_episodes=rl_cfg["training"]["eval_episodes"],
                max_num_discs=max_num_discs,
                env_step=env_step,
                env_type=env_type,
            )

            last_last_success_rate = last_success_rate
            last_success_rate = success_rate_eval

            print(
                f"[EVAL] Success Rate after {episode + 1} episodes: "
                f"{success_rate_eval:.2f}, Avg Reward: {avg_reward_eval:.3f}"
            )

            if use_wandb:
                log_dict["success_rate"] = success_rate_eval
                log_dict["avg_reward"] = avg_reward_eval
                log_dict["avg_distance_to_hole"] = avg_distance_to_hole_eval
                log_dict["bootstrap_p"] = float(bootstrap_p)
                log_dict["K_heads"] = int(K)

        # wandb per-step logging
        if use_wandb:
            if env_type == "sim":
                distance_to_hole = np.linalg.norm(np.array([ball_x, ball_y]) - hole_pos)
                log_dict["distance_to_hole"] = float(distance_to_hole)
            log_dict["reward"] = float(reward)
            log_dict["max_num_discs"] = int(max_num_discs)
            log_dict["thompson_head"] = int(head)
            wandb.log(log_dict, step=episode)

    # -------------------------------------------------
    # Final evaluation + saving
    # -------------------------------------------------
    if env_type == "sim":
        final_success_rate, final_avg_reward, final_avg_distance_to_hole = evaluation_policy_short(
            eval_actor,
            device,
            mujoco_cfg,
            rl_cfg,
            100,
            max_num_discs=max_num_discs,
            env_step=env_step,
            env_type=env_type,
        )

        print(
            f"[FINAL EVAL] success={final_success_rate:.2f}, "
            f"avg_reward={final_avg_reward:.3f}, avg_dist={final_avg_distance_to_hole:.3f}"
        )

    # Save heads
    save_thompson_heads(model_dir, tag_for_saving, actors, critics)

    if use_wandb:
        if env_type == "sim":
            wandb.log({"final_avg_reward": final_avg_reward})
            wandb.log({"final_success_rate": final_success_rate})
            wandb.log({"final_avg_distance_to_hole": final_avg_distance_to_hole})
        print("Run complete. Thompson models saved.")

    if episode_logger is not None:
        episode_logger.close()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    sim_dir = here / "simulation"
    sys.path.append(str(sim_dir))

    project_root = here.parents[1]
    mujoco_config_path = project_root / "configs" / "mujoco_config.yaml"
    rl_config_path = project_root / "configs" / "rl_config.yaml"

    with open(mujoco_config_path, "r") as f:
        mujoco_cfg = yaml.safe_load(f)

    with open(rl_config_path, "r") as f:
        rl_cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Select environment: SIM vs REAL
    # ------------------------------------------------------------------
    env_type = rl_cfg["training"].get("env_type", "sim")

    if env_type == "sim":
        from run_sim_rl import run_sim as env_step

        tmp_name = f"golf_world_tmp_{os.getpid()}_{uuid.uuid4().hex}.xml"
        tmp_xml_path = project_root / "models" / "mujoco" / tmp_name
        mujoco_cfg["sim"]["xml_path"] = str(tmp_xml_path)

    elif env_type == "real":
        from run_real_rl import run_real as env_step
        tmp_xml_path = None

    else:
        raise ValueError(f"Unknown env_type: {env_type} (expected 'sim' or 'real')")

    # Optional: wandb sweeps
    if rl_cfg["training"]["use_wandb"]:
        sweep_config = {
            "actor_lr": rl_cfg["training"]["actor_lr"],
            "critic_lr": rl_cfg["training"]["critic_lr"],
            "noise_std": rl_cfg["training"]["noise_std"],
            "hidden_dim": rl_cfg["model"]["hidden_dim"],
            "batch_size": rl_cfg["training"]["batch_size"],
            "grad_steps": rl_cfg["training"]["grad_steps"],
            "num_heads": rl_cfg["training"].get("num_heads", 8),
            "bootstrap_p": rl_cfg["training"].get("bootstrap_p", 0.8),
        }

        wandb.init(
            project="rl_golf_thompson_bandit",
            config={
                **sweep_config,
                "rl_config": rl_cfg,
                "mujoco_config": mujoco_cfg,
            },
        )

        cfg = wandb.config
        # Keep your reward knobs
        rl_cfg["reward"]["distance_scale"] = cfg.get("distance_scale", rl_cfg["reward"]["distance_scale"])
        rl_cfg["reward"]["in_hole_reward"] = cfg.get("in_hole_reward", rl_cfg["reward"]["in_hole_reward"])
        rl_cfg["reward"]["w_distance"] = cfg.get("w_distance", rl_cfg["reward"]["w_distance"])
        rl_cfg["reward"]["optimal_speed"] = cfg.get("optimal_speed", rl_cfg["reward"]["optimal_speed"])
        rl_cfg["reward"]["dist_at_hole_scale"] = cfg.get("dist_at_hole_scale", rl_cfg["reward"]["dist_at_hole_scale"])
        rl_cfg["reward"]["optimal_speed_scale"] = cfg.get("optimal_speed_scale", rl_cfg["reward"]["optimal_speed_scale"])

    # Train policy
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
