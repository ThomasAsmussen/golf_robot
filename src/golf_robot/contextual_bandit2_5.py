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
from rl_common_5_no_noise import *


# ---------------------------------------------------------
# Training loop 
# ---------------------------------------------------------
def training(rl_cfg, mujoco_cfg, project_root, continue_training=False, input_func=None, env_step=None, env_type="sim", tmp_name=None, camera_index_start=None):
    """
    Training loop for golf robot:
      - context = (ball_start_obs, hole_pos_obs, discs)
      - action  = (speed, angle)
      - reward  = f(final ball position, hole position, in_hole)

    No bootstrapping, no next_state.
    Critic learns Q(s,a) ≈ E[r | s,a].
    Actor learns to maximize Q(s, π(s)).
    """
    if env_step is None:
        raise ValueError("training() requires env_step (sim or real environment function).")

    print("Starting training with Contextual Bandit 2...")
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
    w_distance     = rl_cfg["reward"]["w_distance"]
    optimal_speed  = rl_cfg["reward"]["optimal_speed"]
    dist_at_hole_scale = rl_cfg["reward"]["dist_at_hole_scale"]
    optimal_speed_scale = rl_cfg["reward"]["optimal_speed_scale"]

    hole_positions = get_hole_positions()

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
    if continue_training and env_type == "real":

        model_dir = project_root / "models" / "rl" / "bandit"

        actor_path, critic_path = find_latest_ddpg_checkpoint(
            model_dir,
            prefix=None,   # or None to load absolutely newest
        )

        print(f"Loading latest checkpoints:")
        print(f"  Actor : {actor_path.name}")
        print(f"  Critic: {critic_path.name}")

        actor, device = load_actor(actor_path, rl_cfg)
        critic, _     = load_critic(critic_path, rl_cfg)

    elif continue_training:
        actor, device = load_actor(
            model_path=project_root / "models" / "rl" / "bandit" / f"ddpg_actor_{model_name}",
            rl_cfg=rl_cfg,
        )
        critic, _ = load_critic(
            model_path=project_root / "models" / "rl" / "bandit" / f"ddpg_critic_{model_name}",
            rl_cfg=rl_cfg,
        )
        print("Continuing training from model:", model_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        critic = Critic(state_dim, action_dim, hidden_dim).to(device)

    print(f"Using device: {device}")

    actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=actor_lr, weight_decay=1e-5)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr, weight_decay=1e-5)
    # actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=actor_lr)
    # critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    # actor_optimizer  = torch.optim.SGD(actor.parameters(),  lr=actor_lr)
    # critic_optimizer = torch.optim.SGD(critic.parameters(), lr=critic_lr)

    replay_buffer_big = ReplayBuffer(capacity=rl_cfg["training"]["replay_buffer_capacity"])
    replay_buffer_recent = ReplayBuffer(1000)  # Smaller buffer for recent experiences

    if use_wandb:
        wandb.watch(actor,  log="gradients", log_freq=100)
        wandb.watch(critic, log="gradients", log_freq=100)
        run_name = wandb.run.name.replace("-", "_")
    else:
        run_name = "local_run"

    model_dir = project_root / "models" / "rl" / "bandit"
    model_dir.mkdir(parents=True, exist_ok=True)

    actor.train()
    critic.train()
    log_dict = {}
    last_success_rate = 0.0
    last_last_success_rate = 0.0
    # For now we don't actually place discs, but the state format reserves 5.
    
    max_num_discs = rl_cfg["training"]["max_num_discs"]
    stage_start_episode = 0
    noise_std_stage_start = noise_std

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
    
    for episode in range(episodes):
        # -------------------------------------------------
        # Sample a context (ball start + hole + discs)
        # -------------------------------------------------
                # Periodic evaluation of greedy policy (no exploration noise)
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
        if env_type == "sim":

            if last_success_rate > 0.9 and last_last_success_rate > 0.9 and False:
                max_num_discs = min(MAX_DISCS, max_num_discs + 1)
                last_success_rate = 0.0
                last_last_success_rate = 0.0
                noise_std = 0.15
                noise_std_stage_start = noise_std
                stage_start_episode = episode
                replay_buffer_recent.clear()
            
            ball_start_obs, hole_pos_obs, disc_positions, x, y, hole_pos = sim_init_parameters(mujoco_cfg, max_num_discs)

        if env_type == "real":
            ball_start_obs, hole_pos_obs, disc_positions, chosen_hole = input_func(camera_index=camera_index_start)
            original_hole_num = chosen_hole


        state_vec = encode_state_with_discs(ball_start_obs, hole_pos_obs, disc_positions, 0)
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
            disc_positions = [(2.0, -0.3), (2.1, 0.0), (2.0, 0.3), (2.4, -0.2), (2.4, 0.2)] # with 5 static discs
            # disc_positions = [(2.0, -0.3), (2.1, 0.0), (2.0, 0.3)] # with 3 static discs
            # disc_positions = [(2.1, 0.0)] # with 1 static disc
            # disc_positions = [] # with 0 discs
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

        if env_type == "sim":
            meta = meta_from_trajectory_xy(meta, hole_pos_obs)

        is_out_of_bounds = False
        using_all_holes = False
        if env_type == "real" and isinstance(meta, dict):
            is_out_of_bounds = bool(meta.get("out_of_bounds", False))
            if meta["wrong_hole"] is not None:
                chosen_hole = meta["wrong_hole"]
                hole_pos_obs = np.array([get_hole_positions()[chosen_hole]["x"], get_hole_positions()[chosen_hole]["y"]])
            elif not in_hole and not is_out_of_bounds and not meta["on_green"]:
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

        if env_type == "real" and isinstance(meta, dict):
            used_for_training = bool(meta.get("used_for_training", True))
            out_of_bounds = bool(meta.get("out_of_bounds", False))

            if not used_for_training:
                print("Episode discarded by user; not adding to replay buffer.")
                continue_training_ = meta.get("continue_training", True)
                if not continue_training_:
                    print("Training aborted by user.")
                    break   # 👈 clean exit

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
        # TD-style supervised update: Q(s,a) -> r
        # -------------------------------------------------
        if len(replay_buffer_big.data) >= batch_size:
            update_actor = True
            if env_type == "real":
                print("Updating networks...")
                update_actor = False
                if episode_logger.get_length() % 10 == 0:
                    print(f"Updating actor")
                    update_actor = True

            for _ in range(grad_steps):
                states_b, actions_b, rewards_b = sample_mixed_zero_mean(
                    replay_buffer_recent,
                    replay_buffer_big,
                    recent_ratio=0.7,
                    batch_size=batch_size,
                    ball_xy_idx=(0, 1),
                )

                bxby_mean = states_b[:, [0,1]].mean(dim=0).cpu().numpy()
                print("batch mean ball_xy (norm):", bxby_mean)


                states_b  = states_b.to(device)
                actions_b = actions_b.to(device)
                rewards_b = rewards_b.to(device)

                
                td_target = rewards_b

                q_pred = critic(states_b, actions_b)
                # critic_loss = F.mse_loss(q_pred, td_target)
                critic_loss = F.smooth_l1_loss(q_pred, td_target)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Actor tries to maximize Q(s, π(s))
                if update_actor:
                    a_for_actor = actor(states_b)
                    actor_loss  = -critic(states_b, a_for_actor).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

            # critic_loss_value = critic_loss.item()
            # actor_loss_value  = actor_loss.item()

            if env_type == "real":
                if tmp_name is not None: 
                    torch.save(actor.state_dict(),  model_dir / f"ddpg_actor_{tmp_name}.pth")
                    torch.save(critic.state_dict(), model_dir / f"ddpg_critic_{tmp_name}.pth")
                else: 
                    print("No tmp_name provided; models not saved during real robot training.")
        if env_type == "real":
            if not meta.get("continue_training", True):
                print("Training aborted by user.")
                break   # 👈 clean exit


        # -------------------------------------------------
        # Logging / prints
        # -------------------------------------------------
        if rl_cfg["training"]["do_prints"]:
            print("========================================")
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward:.4f}")
            dist_to_hole = np.linalg.norm(np.array([ball_x, ball_y]) - hole_pos)
            print(f"  Distance to Hole: {dist_to_hole:.4f}, In Hole: {in_hole}")


        if (episode) % rl_cfg["training"]["eval_interval"] == 2 and env_type == "sim":
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
                horizon = episodes

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

        project_name = rl_cfg["training"].get("project_name", "rl_golf_wandb")
        wandb.init(
            project=project_name,
            group = "ddpg-1step",
            config={
                **sweep_config,
                "rl_config":     rl_cfg,
                "mujoco_config": mujoco_cfg,
            },
        )

        cfg = wandb.config
        rl_cfg["reward"]["distance_scale"]   = cfg.get("distance_scale", rl_cfg["reward"]["distance_scale"])
        rl_cfg["reward"]["in_hole_reward"]   = cfg.get("in_hole_reward", rl_cfg["reward"]["in_hole_reward"])
        rl_cfg["reward"]["w_distance"]       = cfg.get("w_distance", rl_cfg["reward"]["w_distance"])
        rl_cfg["reward"]["optimal_speed"]    = cfg.get("optimal_speed", rl_cfg["reward"]["optimal_speed"])
        rl_cfg["reward"]["dist_at_hole_scale"] = cfg.get("dist_at_hole_scale", rl_cfg["reward"]["dist_at_hole_scale"])
        rl_cfg["reward"]["optimal_speed_scale"] = cfg.get("optimal_speed_scale", rl_cfg["reward"]["optimal_speed_scale"])
        rl_cfg["training"]["actor_lr"]      = float(cfg["actor_lr"])
        rl_cfg["training"]["critic_lr"]     = float(cfg["critic_lr"])
        rl_cfg["training"]["noise_std"]     = float(cfg["noise_std"])

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
