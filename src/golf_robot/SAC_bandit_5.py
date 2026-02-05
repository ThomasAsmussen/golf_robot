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

from rl_common_5_no_noise import *  # expects SACActor and QNetwork to exist in rl_common.py


# ---------------------------------------------------------
# SAC checkpoint helpers (local to keep changes self-contained)
# ---------------------------------------------------------
def _load_sac_actor(model_path: Path, rl_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim  = rl_cfg["model"]["state_dim"]
    action_dim = rl_cfg["model"]["action_dim"]
    hidden_dim = rl_cfg["model"]["hidden_dim"]
    actor = SACActor(state_dim, action_dim, hidden_dim).to(device)
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


def find_latest_sac_checkpoint(model_dir: Path, prefix: str | None = None):
    """
    Find newest SAC checkpoint trio: (actor, q1, q2) and optional log_alpha.
    If prefix is given, only match files containing that prefix.
    """
    actor_files = list(model_dir.glob("sac_actor_*.pth"))
    if prefix is not None:
        actor_files = [f for f in actor_files if prefix in f.stem]

    if not actor_files:
        raise FileNotFoundError("No matching SAC actor checkpoints found.")

    actor_file = max(actor_files, key=lambda f: f.stat().st_mtime)

    q1_file = actor_file.with_name(actor_file.name.replace("sac_actor_", "sac_q1_"))
    q2_file = actor_file.with_name(actor_file.name.replace("sac_actor_", "sac_q2_"))
    alpha_file = actor_file.with_name(actor_file.name.replace("sac_actor_", "sac_log_alpha_"))

    if not q1_file.exists():
        raise FileNotFoundError(f"Matching q1 not found for {actor_file.name}")
    if not q2_file.exists():
        raise FileNotFoundError(f"Matching q2 not found for {actor_file.name}")
    
    print(f"loaded sac actor from {actor_file}")

    return actor_file, q1_file, q2_file, (alpha_file if alpha_file.exists() else None)


# ---------------------------------------------------------
# Training loop (SAC, single-step / contextual bandit)
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
    SAC training loop for golf robot (single-step contextual bandit):
      - context = (ball_start_obs, hole_pos_obs, discs)
      - action  = (speed, angle) in normalized [-1,1]^2 then mapped to physical
      - reward  = f(final ball position, hole position, in_hole)

    No next_state, no bootstrapping. Critics regress Q(s,a) ≈ E[r|s,a].
    Actor maximizes E[Q - alpha*logpi] (equivalently minimize alpha*logpi - Q).
    """
    if env_step is None:
        raise ValueError("training() requires env_step (sim or real environment function).")

    print("Starting training with SAC Bandit...")
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

    hole_positions = get_hole_positions()

    # Wandb only in sim by default
    use_wandb = rl_cfg["training"]["use_wandb"] if env_type == "sim" else False
    model_name = rl_cfg["training"].get("model_name", None)

    # -------------------------
    # Initialize models (SAC)
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = project_root / "models" / "rl" / "sac"
    model_dir.mkdir(parents=True, exist_ok=True)

    if continue_training and env_type == "real":
        actor_path, q1_path, q2_path, alpha_path = find_latest_sac_checkpoint(model_dir, prefix=None)
        print("Loading latest SAC checkpoints:")
        print(f"  Actor: {actor_path.name}")
        print(f"  Q1   : {q1_path.name}")
        print(f"  Q2   : {q2_path.name}")
        if alpha_path is not None:
            print(f"  Alpha: {alpha_path.name}")

        actor, device = _load_sac_actor(actor_path, rl_cfg)
        q1 = _load_q(q1_path, rl_cfg, device=device)
        q2 = _load_q(q2_path, rl_cfg, device=device)

        # log_alpha (optional)
        if alpha_path is not None:
            log_alpha = torch.load(alpha_path, map_location=device)
            if not torch.is_tensor(log_alpha):
                log_alpha = torch.tensor(float(log_alpha))
            log_alpha = log_alpha.to(device).detach().clone().requires_grad_(True)
        else:
            log_alpha = torch.tensor(np.log(rl_cfg["sac"].get("alpha_init", 0.2)), device=device, requires_grad=True)

        actor.train()
        q1.train()
        q2.train()

    elif continue_training:
        # Expect explicit model_name (kept similar to old behavior)
        if model_name is None:
            raise ValueError("continue_training=True but rl_cfg['training']['model_name'] is None")

        actor_path = model_dir / f"sac_actor_{model_name}"
        q1_path    = model_dir / f"sac_q1_{model_name}"
        q2_path    = model_dir / f"sac_q2_{model_name}"
        alpha_path = model_dir / f"sac_log_alpha_{model_name}"

        actor, device = _load_sac_actor(actor_path, rl_cfg)
        q1 = _load_q(q1_path, rl_cfg, device=device)
        q2 = _load_q(q2_path, rl_cfg, device=device)

        if alpha_path.exists():
            log_alpha = torch.load(alpha_path, map_location=device)
            if not torch.is_tensor(log_alpha):
                log_alpha = torch.tensor(float(log_alpha))
            log_alpha = log_alpha.to(device).detach().clone().requires_grad_(True)
        else:
            log_alpha = torch.tensor(np.log(rl_cfg["sac"].get("alpha_init", 0.2)), device=device, requires_grad=True)


        print("Continuing SAC training from model:", model_name)
        actor.train()
        q1.train()
        q2.train()

    else:
        actor = SACActor(state_dim, action_dim, hidden_dim).to(device)
        q1    = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        q2    = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        alpha0 = rl_cfg.get("sac", {}).get("alpha_init", 0.2)
        log_alpha = torch.tensor(np.log(alpha0), device=device, requires_grad=True)


    print(f"Using device: {device}")

    # SAC optimizers (Adam is strongly recommended)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    q1_optimizer    = torch.optim.Adam(q1.parameters(), lr=critic_lr)
    q2_optimizer    = torch.optim.Adam(q2.parameters(), lr=critic_lr)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=actor_lr * rl_cfg["sac"].get("alpha_lr_mult", 1.0))

    # Entropy target
    target_entropy = rl_cfg["sac"].get("target_entropy", -action_dim)

    replay_buffer_big = ReplayBuffer(capacity=rl_cfg["training"]["replay_buffer_capacity"])
    replay_buffer_recent = ReplayBuffer(1000)

    if use_wandb:
        # wandb.watch(actor, log="gradients", log_freq=100)
        # wandb.watch(q1,    log="gradients", log_freq=100)
        # wandb.watch(q2,    log="gradients", log_freq=100)
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
        episode_log_path = project_root / "log" / "sac" / "real_episodes" / "episode_logger.jsonl"
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
    for episode in range(episodes):

        # Periodic eval (real): keep your existing trigger logic
        if env_type == "real" and episode_logger.get_length() % 1 == 0:
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
            original_hole_num = chosen_hole

        MAX_DISCS_FEATS = 5  # must match encode_state_with_discs call

        state_vec = encode_state_with_discs(ball_start_obs, hole_pos_obs, disc_positions, max_num_discs=0)
        state_norm = scale_state_vec(state_vec) # 19 states
        
        if rl_cfg["model"]["state_dim"] == 37: # 37 states from augmented states
            aug = augment_state_features(state_vec, max_num_discs=MAX_DISCS_FEATS, dist_clip=5.0)
            aug_norm = scale_aug_features(aug, max_num_discs=MAX_DISCS_FEATS, dist_clip=5.0)

            state_norm = np.concatenate([state_norm, aug_norm], axis=0)

        s = torch.tensor(state_norm, dtype=torch.float32, device=device)

        # -------------------------------------------------
        # SAC policy action: sample a_norm in [-1,1]^2
        # -------------------------------------------------
        with torch.no_grad():
            a_noisy, _, _ = actor.sample(s.unsqueeze(0))
            a_noisy = a_noisy.squeeze(0)

        speed, angle_deg = get_input_parameters(
            a_noisy.cpu().numpy(), speed_low, speed_high, angle_low, angle_high
        )

        # Keep actuator noise for sim (optional)
        if env_type == "sim":
            speed_noise     = np.random.normal(0, SPEED_NOISE_STD)
            angle_deg_noise = np.random.normal(0, ANGLE_NOISE_STD)
            speed     = np.clip(speed + speed_noise,     speed_low,  speed_high)
            angle_deg = np.clip(angle_deg + angle_deg_noise, angle_low, angle_high)

        # -------------------------------------------------
        # One-step environment: simulate and get reward
        # -------------------------------------------------
        if env_type == "sim":
            disc_positions = [(2.0, -0.3), (2.1, 0.0), (2.0, 0.3), (2.4, -0.2), (2.4, 0.2)] # with 5 static discs
            # disc_positions = [(2.0, -0.3), (2.1, 0.0), (2.0, 0.3)] # with 3 static discs
            # disc_positions = [(2.1, 0.0)] # with 1 static disc
            # disc_positions = [] # with 0 discs
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
                dist_at_hole = meta["dist_at_hole"]
                speed_at_hole = meta["speed_at_hole"]
                if dist_at_hole is not None and dist_at_hole[i] is not None:
                    dist_at_hole = float(dist_at_hole[i])
                    speed_at_hole = float(speed_at_hole[i])
                else:
                    continue

                hole_pos_try = hole_pos_obs_try
                reward_try = compute_reward(
                    ball_end_xy=np.array([ball_x, ball_y]),
                    hole_xy=hole_pos_try,
                    in_hole=in_hole,
                    meta=meta,
                    is_out_of_bounds=is_out_of_bounds,
                    in_hole_reward=in_hole_reward,
                    distance_scale=distance_scale,
                    w_distance=w_distance,
                    optimal_speed=optimal_speed,
                    dist_at_hole_scale=dist_at_hole_scale,
                    optimal_speed_scale=optimal_speed_scale,
                    multiple_dist_at_hole = dist_at_hole,
                    multiple_speed_at_hole = speed_at_hole
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

        # Real logging behavior kept

        if env_type == "real" and isinstance(meta, dict):
            used_for_training = bool(meta.get("used_for_training", True))
            out_of_bounds = bool(meta.get("out_of_bounds", False))

            if not used_for_training:
                print("Episode discarded by user; not adding to replay buffer.")
                continue
            else:
                print(f"Storing episode")
                if using_all_holes:
                    reward_count = 0
                    for i, hole_pos_obs_try in enumerate([hole1, hole2, hole3]):
                        if meta["dist_at_hole"] is not None:
                            dist_at_hole = meta["dist_at_hole"][i]
                            speed_at_hole = meta["speed_at_hole"][i]
                        if dist_at_hole is None:
                            continue
                        
                        hole_pos_try = hole_pos_obs_try
                        reward = rewards[reward_count]
                        reward_count += 1
                        logging_dict = {
                            "episode": episode,
                            "time": time.time(),
                            "used_for_training": used_for_training,
                            "ball_start_obs": ball_start_obs.tolist(),
                            "hole_pos_obs": hole_pos_try.tolist(),
                            "disc_positions": disc_positions,
                            "state_norm": state_norm.tolist(),
                            "action_norm": a_noisy.cpu().numpy().tolist(),
                            "speed": speed,
                            "angle_deg": angle_deg,
                            "ball_final_pos": [ball_x, ball_y],
                            "in_hole": in_hole,
                            "out_of_bounds": out_of_bounds,
                            "reward": reward,
                            "chosen_hole": i+1,
                            "aimed_hole": original_hole_num,
                            "dist_at_hole": dist_at_hole,
                            "speed_at_hole": speed_at_hole,
                            "exploring": True,
                        }
                        # print(f"Logging episode data: {logging_dict}")
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
                        "action_norm": a_noisy.cpu().numpy().tolist(),
                        "speed": speed,
                        "angle_deg": angle_deg,
                        "ball_final_pos": [ball_x, ball_y],
                        "in_hole": in_hole,
                        "out_of_bounds": out_of_bounds,
                        "reward": reward,
                        "chosen_hole": chosen_hole,
                        "aimed_hole": original_hole_num,
                        "dist_at_hole": meta["dist_at_hole"],
                        "speed_at_hole": meta["speed_at_hole"],
                        "exploring": True,
                    }
                    # print(f"Logging episode data: {logging_dict}")
                    episode_logger.log(logging_dict)

        # Store transition
        s_train = s.detach().cpu()
        a_train = a_noisy.detach().cpu()

        if using_all_holes:
            for i, hole_pos_obs_try in enumerate([hole1, hole2, hole3]):
                if meta["dist_at_hole"] is not None:
                    dist_at_hole = meta["dist_at_hole"][i]
                    speed_at_hole = meta["speed_at_hole"][i]
                if dist_at_hole is None:
                    continue
                replay_buffer_big.add(s_train, a_train, rewards[i])
                replay_buffer_recent.add(s_train, a_train, rewards[i])
                # -------------------------------------------------
        else:
        # Store (s,a,r) in buffer 
            replay_buffer_big.add(s_train, a_train, reward)
            replay_buffer_recent.add(s_train, a_train, reward)

        # -------------------------------------------------
        # SAC update (single-step)
        # -------------------------------------------------
        critic_loss_value = None
        actor_loss_value = None

        if len(replay_buffer_big.data) >= batch_size:
            if env_type == "real":
                print("Updating networks...")

            for _ in range(grad_steps):
                states_b, actions_b, rewards_b = sample_mixed_zero_mean(
                    replay_buffer_recent,
                    replay_buffer_big,
                    recent_ratio=0.7,
                    batch_size=batch_size,
                    ball_xy_idx=(0,1)
                )

                states_b  = states_b.to(device)
                actions_b = actions_b.to(device)
                rewards_b = rewards_b.to(device)

                # Critic targets are just rewards (bandit)
                y = rewards_b

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

                # Actor update
                a_new, logp, _ = actor.sample(states_b)
                q_new = torch.min(q1(states_b, a_new), q2(states_b, a_new))

                alpha = log_alpha.exp().detach()
                actor_loss = (alpha * logp - q_new).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Temperature update (auto alpha)
                alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()

                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

            critic_loss_value = 0.5 * (q1_loss.item() + q2_loss.item())
            actor_loss_value  = actor_loss.item()

            # Save during real training
            if env_type == "real":
                if tmp_name is not None:
                    torch.save(actor.state_dict(), model_dir / f"sac_actor_{tmp_name}.pth")
                    torch.save(q1.state_dict(),    model_dir / f"sac_q1_{tmp_name}.pth")
                    torch.save(q2.state_dict(),    model_dir / f"sac_q2_{tmp_name}.pth")
                    torch.save(log_alpha.detach().cpu(), model_dir / f"sac_log_alpha_{tmp_name}.pth")
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
                # show best of the three
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
                log_dict["alpha"]                = float(log_alpha.exp().detach().cpu().item())
                wandb.log(log_dict, step=episode)

        # if use_wandb:
        #     if env_type == "sim":
        #         distance_to_hole = np.linalg.norm(np.array([ball_x, ball_y]) - hole_pos)
        #     else:
        #         # best effort (hole_pos not defined in real path)
        #         distance_to_hole = float(np.linalg.norm(np.array([ball_x, ball_y]) - np.array(hole_pos_obs)))

        #     log_dict["reward"]           = float(max(rewards)) if using_all_holes else float(reward)
        #     log_dict["distance_to_hole"] = distance_to_hole
        #     log_dict["max_num_discs"]    = max_num_discs
        #     if critic_loss_value is not None:
        #         log_dict["critic_loss"] = critic_loss_value
        #     if actor_loss_value is not None:
        #         log_dict["actor_loss"]  = actor_loss_value

        #     wandb.log(log_dict, step=episode)

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

            torch.save(actor.state_dict(), model_dir / f"sac_actor_{run_name}.pth")
            torch.save(q1.state_dict(),    model_dir / f"sac_q1_{run_name}.pth")
            torch.save(q2.state_dict(),    model_dir / f"sac_q2_{run_name}.pth")
            torch.save(log_alpha.detach().cpu(), model_dir / f"sac_log_alpha_{run_name}.pth")
            print("Sweep complete. SAC models saved.")

    if episode_logger is not None:
        episode_logger.close()

    elif tmp_name is not None:
        torch.save(actor.state_dict(), model_dir / f"sac_actor_{tmp_name}.pth")
        torch.save(q1.state_dict(),    model_dir / f"sac_q1_{tmp_name}.pth")
        torch.save(q2.state_dict(),    model_dir / f"sac_q2_{tmp_name}.pth")
        torch.save(log_alpha.detach().cpu(), model_dir / f"sac_log_alpha_{tmp_name}.pth")
        print("Training complete. SAC models saved.")


# ---------------------------------------------------------
# Main (kept the same structure)
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
        tmp_name = f"real_sac_{os.getpid()}_{uuid.uuid4().hex}"
    else:
        raise ValueError(f"Unknown env_type: {env_type} (expected 'sim' or 'real')")

    # Optional: wandb sweeps
    if rl_cfg["training"]["use_wandb"]:

        project_name = rl_cfg["training"].get("project_name", "rl_golf_wandb")
        wandb.init()

        cfg = wandb.config
        rl_cfg["reward"]["distance_scale"]      = cfg.get("distance_scale", rl_cfg["reward"]["distance_scale"])
        rl_cfg["reward"]["in_hole_reward"]      = cfg.get("in_hole_reward", rl_cfg["reward"]["in_hole_reward"])
        rl_cfg["reward"]["w_distance"]          = cfg.get("w_distance", rl_cfg["reward"]["w_distance"])
        rl_cfg["reward"]["optimal_speed"]       = cfg.get("optimal_speed", rl_cfg["reward"]["optimal_speed"])
        rl_cfg["reward"]["dist_at_hole_scale"]  = cfg.get("dist_at_hole_scale", rl_cfg["reward"]["dist_at_hole_scale"])
        rl_cfg["reward"]["optimal_speed_scale"] = cfg.get("optimal_speed_scale", rl_cfg["reward"]["optimal_speed_scale"])
        rl_cfg["training"]["actor_lr"]        = cfg["actor_lr"]
        rl_cfg["training"]["critic_lr"]       = cfg["critic_lr"]
        rl_cfg["sac"]["target_entropy"]     = cfg["target_entropy"]
        rl_cfg["sac"]["alpha_init"]         = cfg["alpha_init"]
        rl_cfg["sac"]["alpha_lr_mult"]      = cfg["alpha_lr_mult"]


    try:
        training(
            rl_cfg,
            mujoco_cfg,
            project_root,
            continue_training=rl_cfg["training"]["continue_training"],
            env_step=env_step,
            env_type=env_type,
            tmp_name=tmp_name if env_type == "real" else None,
        )
    finally:
        if rl_cfg["training"]["use_wandb"]:
            wandb.finish()

    if env_type == "sim" and tmp_xml_path is not None:
        try:
            os.remove(tmp_xml_path)
        except FileNotFoundError:
            pass
