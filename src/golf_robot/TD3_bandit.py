# td3_bandit.py
import os
from pathlib import Path
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import uuid
import time

from rl_common import *  # DO NOT CHANGE (expects Actor, QNetwork, etc.)


# ---------------------------------------------------------
# TD3 checkpoint helpers (local to keep changes self-contained)
# ---------------------------------------------------------
def _load_td3_actor(model_path: Path, rl_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim  = rl_cfg["model"]["state_dim"]
    action_dim = rl_cfg["model"]["action_dim"]
    hidden_dim = rl_cfg["model"]["hidden_dim"]
    actor = Actor(state_dim, action_dim, hidden_dim).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()
    return actor, device


def _load_q(model_path: Path, rl_cfg, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim  = rl_cfg["model"]["state_dim"]
    action_dim = rl_cfg["model"]["action_dim"]
    hidden_dim = rl_cfg["model"]["hidden_dim"]
    qnet = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    qnet.load_state_dict(torch.load(model_path, map_location=device))
    qnet.eval()
    return qnet


def find_latest_td3_checkpoint(model_dir: Path, prefix: str | None = None):
    """
    Find newest TD3 checkpoint trio: (actor, q1, q2).
    If prefix is given, only match files containing that prefix.
    """
    actor_files = list(model_dir.glob("td3_actor_*.pth"))
    if prefix is not None:
        actor_files = [f for f in actor_files if prefix in f.stem]

    if not actor_files:
        raise FileNotFoundError("No matching TD3 actor checkpoints found.")

    actor_file = max(actor_files, key=lambda f: f.stat().st_mtime)
    q1_file = actor_file.with_name(actor_file.name.replace("td3_actor_", "td3_q1_"))
    q2_file = actor_file.with_name(actor_file.name.replace("td3_actor_", "td3_q2_"))

    if not q1_file.exists():
        raise FileNotFoundError(f"Matching q1 not found for {actor_file.name}")
    if not q2_file.exists():
        raise FileNotFoundError(f"Matching q2 not found for {actor_file.name}")

    return actor_file, q1_file, q2_file


# ---------------------------------------------------------
# Polyak averaging helper (TD3)
# ---------------------------------------------------------
@torch.no_grad()
def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


# ---------------------------------------------------------
# Training loop (TD3, single-step / contextual bandit)
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
    TD3-style training loop for golf robot (single-step contextual bandit):
      - context = (ball_start_obs, hole_pos_obs, discs)
      - action  = (speed, angle) in normalized [-1,1]^2 then mapped to physical
      - reward  = f(final ball position, hole position, in_hole)

    Bandit note:
      - No next_state, no bootstrapping, so critic targets are simply rewards.
      - TD3 still makes sense as "DDPG + twin critics + delayed actor updates"
        with exploration noise for data collection.
    """
    if env_step is None:
        raise ValueError("training() requires env_step (sim or real environment function).")

    episodes   = rl_cfg["training"]["episodes"]
    batch_size = rl_cfg["training"]["batch_size"]
    actor_lr   = rl_cfg["training"]["actor_lr"]
    critic_lr  = rl_cfg["training"]["critic_lr"]
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
    w_distance     = rl_cfg["reward"]["w_distance"]
    optimal_speed  = rl_cfg["reward"]["optimal_speed"]
    dist_at_hole_scale = rl_cfg["reward"]["dist_at_hole_scale"]
    optimal_speed_scale = rl_cfg["reward"]["optimal_speed_scale"]

    # Wandb only in sim by default
    use_wandb = rl_cfg["training"]["use_wandb"] if env_type == "sim" else False
    model_name = rl_cfg["training"].get("model_name", None)

    # TD3 knobs (safe defaults if not in YAML)
    td3_cfg = rl_cfg["training"].get("td3", {})
    exploration_noise = float(td3_cfg.get("exploration_noise", 0.10))  # action noise during collection
    policy_delay      = int(td3_cfg.get("policy_delay", 2))            # actor update every N critic updates
    tau               = float(td3_cfg.get("tau", 0.005))               # target smoothing (polyak)
    # Target policy smoothing is irrelevant for pure bandit targets, but we keep targets anyway.

    # -------------------------
    # Initialize models (TD3)
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = project_root / "models" / "rl" / "bandit"
    model_dir.mkdir(parents=True, exist_ok=True)

    if continue_training and env_type == "real":
        actor_path, q1_path, q2_path = find_latest_td3_checkpoint(model_dir, prefix=None)
        print("Loading latest TD3 checkpoints:")
        print(f"  Actor: {actor_path.name}")
        print(f"  Q1   : {q1_path.name}")
        print(f"  Q2   : {q2_path.name}")

        actor, device = _load_td3_actor(actor_path, rl_cfg)
        q1 = _load_q(q1_path, rl_cfg, device=device)
        q2 = _load_q(q2_path, rl_cfg, device=device)

        actor.train()
        q1.train()
        q2.train()

    elif continue_training:
        if model_name is None:
            raise ValueError("continue_training=True but rl_cfg['training']['model_name'] is None")

        actor_path = model_dir / f"td3_actor_{model_name}"
        q1_path    = model_dir / f"td3_q1_{model_name}"
        q2_path    = model_dir / f"td3_q2_{model_name}"

        actor, device = _load_td3_actor(actor_path, rl_cfg)
        q1 = _load_q(q1_path, rl_cfg, device=device)
        q2 = _load_q(q2_path, rl_cfg, device=device)

        print("Continuing TD3 training from model:", model_name)
        actor.train()
        q1.train()
        q2.train()

    else:
        actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        q1    = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        q2    = QNetwork(state_dim, action_dim, hidden_dim).to(device)

    # Target networks (kept to be faithful to TD3 style)
    actor_t = Actor(state_dim, action_dim, hidden_dim).to(device)
    q1_t    = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    q2_t    = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    actor_t.load_state_dict(actor.state_dict())
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())
    actor_t.eval()
    q1_t.eval()
    q2_t.eval()

    print(f"Using device: {device}")

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    q1_optimizer    = torch.optim.Adam(q1.parameters(), lr=critic_lr)
    q2_optimizer    = torch.optim.Adam(q2.parameters(), lr=critic_lr)

    replay_buffer_big = ReplayBuffer(capacity=rl_cfg["training"]["replay_buffer_capacity"])
    replay_buffer_recent = ReplayBuffer(1000)

    if use_wandb:
        wandb.watch(actor, log="gradients", log_freq=100)
        wandb.watch(q1,    log="gradients", log_freq=100)
        wandb.watch(q2,    log="gradients", log_freq=100)
        run_name = wandb.run.name.replace("-", "_")
    else:
        run_name = "local_run"

    log_dict = {}
    last_success_rate = 0.0
    last_last_success_rate = 0.0

    max_num_discs = rl_cfg["training"]["max_num_discs"]
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
        )

        if loaded_n > 0:
            print(f"Loaded {loaded_n} episodes from {episode_log_path} into replay buffer.")
        else:
            print(f"No existing episode log found at {episode_log_path}. Starting fresh.")

    # ---------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------
    global_grad_step = 0

    for episode in range(episodes):

        # Periodic eval (real): keep your existing trigger logic
        if env_type == "real" and episode_logger.get_length() % 10000 == 0:
            (
                success_rate_eval,
                avg_reward_eval,
                avg_distance_to_hole_eval,
            ) = evaluation_policy_short(
                actor,
                device,
                mujoco_cfg=None,
                rl_cfg=rl_cfg,
                num_episodes=3,
                env_step=env_step,
                env_type=env_type,
                input_func=input_func,
                big_episode_logger=episode_logger
            )

            print(
                f"[EVAL] Success Rate after {episode + 1} episodes: "
                f"{success_rate_eval:.2f}, Avg Reward: {avg_reward_eval:.3f}"
            )

        # Sample context
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
        else:
            ball_start_obs, hole_pos_obs, disc_positions, chosen_hole = input_func(
                camera_index=camera_index_start
            )

        MAX_DISCS_FEATS = 5  # must match encode_state_with_discs call

        state_vec = encode_state_with_discs(ball_start_obs, hole_pos_obs, disc_positions, max_num_discs=MAX_DISCS_FEATS)
        state_norm = scale_state_vec(state_vec)  # 19 states

        if rl_cfg["model"]["state_dim"] == 37:  # 37 states from augmented states
            aug = augment_state_features(state_vec, max_num_discs=MAX_DISCS_FEATS, dist_clip=5.0)
            aug_norm = scale_aug_features(aug, max_num_discs=MAX_DISCS_FEATS, dist_clip=5.0)
            state_norm = np.concatenate([state_norm, aug_norm], axis=0)

        s = torch.tensor(state_norm, dtype=torch.float32, device=device)

        # -------------------------------------------------
        # TD3 policy action: deterministic + exploration noise in [-1,1]^2
        # -------------------------------------------------
        with torch.no_grad():
            a = actor(s.unsqueeze(0)).squeeze(0)  # in [-1,1]
            noise = torch.randn_like(a) * exploration_noise
            a_noisy = torch.clamp(a + noise, -1.0, 1.0)

        speed, angle_deg = get_input_parameters(
            a_noisy.cpu().numpy(), speed_low, speed_high, angle_low, angle_high
        )

        # Keep actuator noise for sim (optional, your original behavior)
        if env_type == "sim":
            speed_noise     = np.random.normal(0, SPEED_NOISE_STD)
            angle_deg_noise = np.random.normal(0, ANGLE_NOISE_STD)
            speed     = np.clip(speed + speed_noise,     speed_low,  speed_high)
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
                    f"Episode {episode + 1}: Environment failed twice. "
                    f"Bad action: speed={speed}, angle={angle_deg}"
                )
                if env_type == "sim":
                    print(f"  Hole Position: x={x:.4f}, y={y:.4f}")
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
                hole_pos_obs = np.array(
                    [get_hole_positions()[chosen_hole]["x"], get_hole_positions()[chosen_hole]["y"]]
                )
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
                    dist_at_hole_i = float(dist_at_hole[i])
                else:
                    dist_at_hole_i = None

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
                    multiple_dist_at_hole=dist_at_hole_i,
                )
                rewards.append(reward_try)
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

        # Real logging behavior kept (unchanged, but now action comes from TD3)
        if env_type == "real" and isinstance(meta, dict):
            used_for_training = bool(meta.get("used_for_training", True))
            out_of_bounds = bool(meta.get("out_of_bounds", False))

            if not used_for_training:
                print("Episode discarded by user; not adding to replay buffer.")
                continue
            else:
                print("Storing episode")
                if using_all_holes:
                    for i, hole_pos_obs_try in enumerate([hole1, hole2, hole3]):
                        logging_dict = {
                            "episode": episode,
                            "time": time.time(),
                            "used_for_training": used_for_training,
                            "ball_start_obs": ball_start_obs.tolist(),
                            "hole_pos_obs": hole_pos_obs_try.tolist(),
                            "disc_positions": disc_positions,
                            "state_norm": state_norm.tolist(),
                            "action_norm": a_noisy.detach().cpu().numpy().tolist(),
                            "speed": speed,
                            "angle_deg": angle_deg,
                            "ball_final_pos": [ball_x, ball_y],
                            "in_hole": in_hole,
                            "out_of_bounds": out_of_bounds,
                            "reward": rewards[i],
                            "chosen_hole": i + 1,
                            "dist_at_hole": meta["dist_at_hole"][i] if meta.get("dist_at_hole", None) is not None else None,
                            "speed_at_hole": meta.get("speed_at_hole", None),
                            "exploring": True,
                        }
                        episode_logger.log(logging_dict)
                else:
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
                        "in_hole": in_hole,
                        "out_of_bounds": out_of_bounds,
                        "reward": reward,
                        "chosen_hole": chosen_hole,
                        "dist_at_hole": meta.get("dist_at_hole", None),
                        "speed_at_hole": meta.get("speed_at_hole", None),
                        "exploring": True,
                    }
                    episode_logger.log(logging_dict)

        # Store transition
        s_train = s.detach().cpu()
        a_train = a_noisy.detach().cpu()

        if using_all_holes:
            for i in range(3):
                replay_buffer_big.add(s_train, a_train, rewards[i])
                replay_buffer_recent.add(s_train, a_train, rewards[i])
        else:
            replay_buffer_big.add(s_train, a_train, reward)
            replay_buffer_recent.add(s_train, a_train, reward)

        # -------------------------------------------------
        # TD3 update (bandit)
        # -------------------------------------------------
        critic_loss_value = None
        actor_loss_value = None

        if len(replay_buffer_big.data) >= batch_size:
            if env_type == "real":
                print("Updating networks...")

            # TD3 delayed policy update counter
            for g in range(grad_steps):
                global_grad_step += 1

                states_b, actions_b, rewards_b = sample_mixed(
                    replay_buffer_recent,
                    replay_buffer_big,
                    recent_ratio=0.7,
                    batch_size=batch_size,
                )

                states_b  = states_b.to(device)
                actions_b = actions_b.to(device)
                rewards_b = rewards_b.to(device)

                # Bandit target: y = r
                y = rewards_b

                # Critic updates (twin)
                q1_pred = q1(states_b, actions_b)
                q2_pred = q2(states_b, actions_b)
                q1_loss = F.mse_loss(q1_pred, y)
                q2_loss = F.mse_loss(q2_pred, y)

                q1_optimizer.zero_grad()
                q1_loss.backward()
                q1_optimizer.step()

                q2_optimizer.zero_grad()
                q2_loss.backward()
                q2_optimizer.step()

                # Delayed actor update
                if (global_grad_step % policy_delay) == 0:
                    a_pi = actor(states_b)
                    q_pi = torch.min(q1(states_b, a_pi), q2(states_b, a_pi))
                    actor_loss = (-q_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # Soft-update target nets (kept TD3-style)
                    soft_update(actor_t, actor, tau)
                    soft_update(q1_t, q1, tau)
                    soft_update(q2_t, q2, tau)

            critic_loss_value = 0.5 * (q1_loss.item() + q2_loss.item())
            actor_loss_value  = actor_loss.item() if "actor_loss" in locals() else None

            # Save during real training
            if env_type == "real":
                if tmp_name is not None:
                    torch.save(actor.state_dict(), model_dir / f"td3_actor_{tmp_name}.pth")
                    torch.save(q1.state_dict(),    model_dir / f"td3_q1_{tmp_name}.pth")
                    torch.save(q2.state_dict(),    model_dir / f"td3_q2_{tmp_name}.pth")
                else:
                    print("No tmp_name provided; models not saved during real robot training.")

        if env_type == "real" and isinstance(meta, dict):
            if not meta.get("continue_training", True):
                print("Training aborted by user.")
                break

        # -------------------------------------------------
        # Prints / logging
        # -------------------------------------------------
        if rl_cfg["training"]["do_prints"]:
            print("========================================")
            if using_all_holes:
                best_r = float(max(rewards)) if rewards else 0.0
                print(f"Episode {episode + 1}/{episodes}, Reward(best): {best_r:.4f}")
            else:
                print(f"Episode {episode + 1}/{episodes}, Reward: {reward:.4f}")

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

            last_last_success_rate = last_success_rate
            last_success_rate = success_rate_eval
            print(
                f"[EVAL] Success Rate after {episode + 1} episodes: "
                f"{success_rate_eval:.2f}, Avg Reward: {avg_reward_eval:.3f}"
            )

            if use_wandb:
                log_dict["success_rate"]         = success_rate_eval
                log_dict["avg_reward"]           = avg_reward_eval
                log_dict["avg_distance_to_hole"] = avg_distance_to_hole_eval

        if use_wandb:
            if env_type == "sim":
                distance_to_hole = np.linalg.norm(np.array([ball_x, ball_y]) - hole_pos)
            else:
                distance_to_hole = float(np.linalg.norm(np.array([ball_x, ball_y]) - np.array(hole_pos_obs)))

            log_dict["reward"]           = float(max(rewards)) if using_all_holes else float(reward)
            log_dict["distance_to_hole"] = distance_to_hole
            log_dict["max_num_discs"]    = max_num_discs
            if critic_loss_value is not None:
                log_dict["critic_loss"] = critic_loss_value
            if actor_loss_value is not None:
                log_dict["actor_loss"]  = float(actor_loss_value)

            wandb.log(log_dict, step=episode)

    # -------------------------------------------------
    # Final evaluation + save
    # -------------------------------------------------
    if env_type == "sim":
        final_success_rate, final_avg_reward, final_avg_distance_to_hole = evaluation_policy_short(
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
            wandb.log({"final_avg_reward": final_avg_reward})
            wandb.log({"final_success_rate": final_success_rate})
            wandb.log({"final_avg_distance_to_hole": final_avg_distance_to_hole})

            torch.save(actor.state_dict(), model_dir / f"td3_actor_{run_name}.pth")
            torch.save(q1.state_dict(),    model_dir / f"td3_q1_{run_name}.pth")
            torch.save(q2.state_dict(),    model_dir / f"td3_q2_{run_name}.pth")
            print("Sweep complete. TD3 models saved.")

    if episode_logger is not None:
        episode_logger.close()

    elif tmp_name is not None:
        torch.save(actor.state_dict(), model_dir / f"td3_actor_{tmp_name}.pth")
        torch.save(q1.state_dict(),    model_dir / f"td3_q1_{tmp_name}.pth")
        torch.save(q2.state_dict(),    model_dir / f"td3_q2_{tmp_name}.pth")
        print("Training complete. TD3 models saved.")


# ---------------------------------------------------------
# Main (same structure as your SAC script)
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

    env_type = rl_cfg["training"].get("env_type", "sim")

    if env_type == "sim":
        from run_sim_rl import run_sim as env_step

        tmp_name     = f"golf_world_tmp_{os.getpid()}_{uuid.uuid4().hex}.xml"
        tmp_xml_path = project_root / "models" / "mujoco" / tmp_name
        mujoco_cfg["sim"]["xml_path"] = str(tmp_xml_path)

    elif env_type == "real":
        from run_real_rl import run_real as env_step
        tmp_xml_path = None
        tmp_name = f"real_td3_{os.getpid()}_{uuid.uuid4().hex}"
    else:
        raise ValueError(f"Unknown env_type: {env_type} (expected 'sim' or 'real')")

    # Optional: wandb sweeps
    if rl_cfg["training"]["use_wandb"]:
        sweep_config = {
            "actor_lr":          rl_cfg["training"]["actor_lr"],
            "critic_lr":         rl_cfg["training"]["critic_lr"],
            "hidden_dim":        rl_cfg["model"]["hidden_dim"],
            "batch_size":        rl_cfg["training"]["batch_size"],
            "grad_steps":        rl_cfg["training"]["grad_steps"],
        }
        
        project_name = rl_cfg["training"].get("project_name", "rl_golf_wandb")
        wandb.init(
            project=project_name, 
            group="td3",  
            config={
                **sweep_config,
                "rl_config":     rl_cfg,
                "mujoco_config": mujoco_cfg,
            },
        )

        cfg = wandb.config
        rl_cfg["reward"]["distance_scale"]      = cfg.get("distance_scale", rl_cfg["reward"]["distance_scale"])
        rl_cfg["reward"]["in_hole_reward"]      = cfg.get("in_hole_reward", rl_cfg["reward"]["in_hole_reward"])
        rl_cfg["reward"]["w_distance"]          = cfg.get("w_distance", rl_cfg["reward"]["w_distance"])
        rl_cfg["reward"]["optimal_speed"]       = cfg.get("optimal_speed", rl_cfg["reward"]["optimal_speed"])
        rl_cfg["reward"]["dist_at_hole_scale"]  = cfg.get("dist_at_hole_scale", rl_cfg["reward"]["dist_at_hole_scale"])
        rl_cfg["reward"]["optimal_speed_scale"] = cfg.get("optimal_speed_scale", rl_cfg["reward"]["optimal_speed_scale"])

    training(
        rl_cfg,
        mujoco_cfg,
        project_root,
        continue_training=rl_cfg["training"]["continue_training"],
        env_step=env_step,
        env_type=env_type,
        tmp_name=tmp_name if env_type == "real" else None,
    )

    if env_type == "sim" and tmp_xml_path is not None:
        try:
            os.remove(tmp_xml_path)
        except FileNotFoundError:
            pass
