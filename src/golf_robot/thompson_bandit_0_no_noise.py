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
# import time
# import cv2

from rl_common_0_no_noise import *


# =========================================================
# Algorithm: Bootstrapped Thompson + CEM planning + Double Critic per head
# =========================================================
# - K heads (bootstrapped)
# - Each head has TWO critics: Q1_k, Q2_k
# - Thompson action selection: sample head k, plan with score(a)=min(Q1_k, Q2_k)
# - Train both critics by supervised regression to reward
# - Evaluation: wrap a planner as an "actor" so evaluation_policy_short() works


# ---------------------------------------------------------
# CEM planner
# ---------------------------------------------------------
def cem_plan_action(
    s,
    score_fn,
    action_dim,
    device,
    cem_iters=2,
    cem_pop=256,
    cem_elite_frac=0.2,
    action_low=-1.0,
    action_high=1.0,
    init_std=0.4,
    min_std=0.1, #0.1
    init_mu=None,
    return_stats=False,
):
    elite_n = max(1, int(round(cem_pop * cem_elite_frac)))

    if init_mu is None:
        mu = torch.zeros(action_dim, device=device)
    else:
        mu = init_mu.detach().clone().to(device)

    std = torch.ones(action_dim, device=device) * float(init_std)

    s_batch = s.unsqueeze(0).repeat(cem_pop, 1)

    best_a = None
    best_score = None

    for _ in range(int(cem_iters)):
        a = mu + std * torch.randn(cem_pop, action_dim, device=device)
        a = torch.clamp(a, float(action_low), float(action_high))

        scores = score_fn(s_batch, a)  # [N]
        topk = torch.topk(scores, k=elite_n, largest=True)
        elite_a = a[topk.indices]

        mu = elite_a.mean(dim=0)
        std = elite_a.std(dim=0).clamp(min=float(min_std))

        if best_score is None or topk.values[0].item() > best_score:
            best_score = topk.values[0].item()
            best_a = elite_a[0].detach()
    
    if return_stats:
        return best_a, {"cem_mu": mu.detach(), "cem_std": std.detach(), "best_score": float(best_score)}
    return best_a


# ---------------------------------------------------------
# Planner wrappers for evaluation_policy_short()
# ---------------------------------------------------------
class MeanPlannerActor(nn.Module):
    """
    Deterministic eval actor:
      - Plans using mean across heads of min(Q1,Q2)
      - Returns normalized action in [-1,1]^2
    """
    def __init__(self, critics1, critics2, rl_cfg, device):
        super().__init__()
        self.critics1 = nn.ModuleList(critics1)
        self.critics2 = nn.ModuleList(critics2)
        self.rl_cfg = rl_cfg
        self.device = device

    @torch.no_grad()
    def forward(self, state):
        # state: [B, state_dim], eval uses B=1
        # print(state)
        s = state.squeeze(0)

        cem_pop = self.rl_cfg["training"].get("cem_pop_eval", self.rl_cfg["training"].get("cem_pop", 256))
        cem_iters = self.rl_cfg["training"].get("cem_iters_eval", self.rl_cfg["training"].get("cem_iters", 2))
        cem_elite_frac = self.rl_cfg["training"].get("cem_elite_frac", 0.2)
        cem_init_std = self.rl_cfg["training"].get("cem_init_std", 0.4)
        cem_min_std = self.rl_cfg["training"].get("cem_min_std", 0.1) #0.1
        # print(f"cem_pop: {cem_pop}, cem_iters: {cem_iters}, cem_elite_frac: {cem_elite_frac}, cem_init_std: {cem_init_std}, cem_min_std: {cem_min_std}")

        action_dim = self.rl_cfg["model"]["action_dim"]

        @torch.no_grad()
        def score_fn(s_batch, a_batch):
            # mean_k min(Q1_k, Q2_k)
            vals = []
            for c1, c2 in zip(self.critics1, self.critics2):
                q1 = c1(s_batch, a_batch).squeeze(-1)
                q2 = c2(s_batch, a_batch).squeeze(-1)
                vals.append(torch.minimum(q1, q2))
            v = torch.stack(vals, dim=0).mean(dim=0)
            return v

        a = cem_plan_action(
            s=s,
            score_fn=score_fn,
            action_dim=action_dim,
            device=self.device,
            cem_iters=cem_iters,
            cem_pop=cem_pop,
            cem_elite_frac=cem_elite_frac,
            init_std=cem_init_std,
            min_std=cem_min_std,
        )
        return a.unsqueeze(0)


# ---------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------
def find_latest_doublecritic_checkpoint(model_dir: Path, prefix: str | None = None):
    critic_files = list(model_dir.glob("ddpg_critic1_*_h0.pth"))
    if prefix is not None:
        critic_files = [f for f in critic_files if prefix in f.stem]
    if not critic_files:
        raise FileNotFoundError("No matching double-critic (head0) checkpoints found.")
    critic1_h0 = max(critic_files, key=lambda f: f.stat().st_mtime)
    tag_stem = critic1_h0.stem.replace("_h0", "")
    return critic1_h0, tag_stem


def load_doublecritics(model_dir: Path, rl_cfg, tag_stem: str, device):
    K = rl_cfg["training"].get("num_heads", 8)
    state_dim = rl_cfg["model"]["state_dim"]
    action_dim = rl_cfg["model"]["action_dim"]
    hidden_dim = rl_cfg["model"]["hidden_dim"]

    if not tag_stem.startswith("ddpg_critic1_"):
        raise ValueError(f"Unexpected tag_stem format: {tag_stem}")
    tag = tag_stem.replace("ddpg_critic1_", "", 1)

    critics1, critics2 = [], []
    for k in range(K):
        c1_path = model_dir / f"ddpg_critic1_{tag}_h{k}.pth"
        c2_path = model_dir / f"ddpg_critic2_{tag}_h{k}.pth"
        if not c1_path.exists() or not c2_path.exists():
            raise FileNotFoundError(f"Missing critic files for head {k}:\n  {c1_path}\n  {c2_path}")
        c1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        c2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        c1.load_state_dict(torch.load(c1_path, map_location=device))
        c2.load_state_dict(torch.load(c2_path, map_location=device))
        c1.train(); c2.train()
        critics1.append(c1); critics2.append(c2)
    print(f"Loaded double-critic models from tag: {tag}")
    return critics1, critics2, tag


def save_doublecritics(model_dir: Path, tag: str, critics1, critics2):
    model_dir.mkdir(parents=True, exist_ok=True)
    for k, (c1, c2) in enumerate(zip(critics1, critics2)):
        torch.save(c1.state_dict(), model_dir / f"ddpg_critic1_{tag}_h{k}.pth")
        torch.save(c2.state_dict(), model_dir / f"ddpg_critic2_{tag}_h{k}.pth")


# Wandb helper
def maybe_init_wandb(rl_cfg, mujoco_cfg):
    if not rl_cfg["training"].get("use_wandb", False):
        return
    if wandb.run is not None:
        return
    run_id = rl_cfg["training"].get("wandb_run_id")
    if not run_id:
        raise ValueError(
            "wandb_run_id must be explicitly set in rl_config.yaml when use_wandb=True"
        )

    wandb.init(
        project=rl_cfg["training"].get("project_name", "rl_golf_no_name"),
        id=run_id,
        resume="allow",
        group="dqn-bts",
        config={
            "rl_config": rl_cfg,
            "mujoco_config": mujoco_cfg,
        },
    )

# ---------------------------------------------------------
# Training loop
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
    cap=None,
):
    if env_type == "real":
        maybe_init_wandb(rl_cfg, mujoco_cfg)
    if env_step is None:
        raise ValueError("training() requires env_step.")

    print("Starting training with Thompson + CEM + Double Critic per head")
    episodes   = rl_cfg["training"]["episodes"]
    batch_size = rl_cfg["training"]["batch_size"]
    critic_lr  = rl_cfg["training"]["critic_lr"]
    grad_steps = rl_cfg["training"]["grad_steps"]

    K = rl_cfg["training"].get("num_heads", 8)
    bootstrap_p = rl_cfg["training"].get("bootstrap_p", 0.8)

    cem_pop = rl_cfg["training"].get("cem_pop", 256)
    cem_iters = rl_cfg["training"].get("cem_iters", 1)
    cem_elite_frac = rl_cfg["training"].get("cem_elite_frac", 0.1)
    cem_init_std = rl_cfg["training"].get("cem_init_std", 0.4)
    cem_min_std = rl_cfg["training"].get("cem_min_std", 0.1) #0.1
    use_cem_warm_start = bool(rl_cfg["training"].get("cem_warm_start", False))

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

    use_wandb = bool(rl_cfg["training"].get("use_wandb", False))

    # model_name = rl_cfg["training"].get("model_name", None)
    model_name = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dir = project_root / "models" / "rl" / "dqn-bts"
    model_dir.mkdir(parents=True, exist_ok=True)

    tag_for_saving = None

    # ---- load/init critics
    if continue_training:
        if model_name is not None:
            tag_for_saving = model_name
            tag_stem = f"ddpg_critic1_{model_name}"
            print(f"Loading double-critic models from tag: {model_name}")
            critics1, critics2, _ = load_doublecritics(model_dir, rl_cfg, tag_stem, device)
            
            print(f"Continuing training from tag: {model_name}")
        else:
            c1_h0, tag_stem = find_latest_doublecritic_checkpoint(model_dir, prefix=None)
            print("Continuing training from latest double-critic checkpoints:")
            print(f"  Critic1_0 : {c1_h0.name}")
            critics1, critics2, loaded_tag = load_doublecritics(model_dir, rl_cfg, tag_stem, device)
            tag_for_saving = loaded_tag
    else:
        critics1 = [Critic(state_dim, action_dim, hidden_dim).to(device) for _ in range(K)]
        critics2 = [Critic(state_dim, action_dim, hidden_dim).to(device) for _ in range(K)]

    opts1 = [torch.optim.Adam(c.parameters(), lr=critic_lr, weight_decay=1e-5) for c in critics1]
    opts2 = [torch.optim.Adam(c.parameters(), lr=critic_lr, weight_decay=1e-5) for c in critics2]

    # ---- buffers
    replay_buffer_big = ReplayBuffer(capacity=rl_cfg["training"]["replay_buffer_capacity"])
    replay_buffer_recent = ReplayBuffer(400)

    # ---- wandb
    if use_wandb:
        wandb.watch(critics1[0], log="gradients", log_freq=100)
        wandb.watch(critics2[0], log="gradients", log_freq=100)
        run_name = wandb.run.name.replace("-", "_")
    else:
        run_name = "local_run"

    if tag_for_saving is None:
        tag_for_saving = tmp_name if (env_type == "real" and tmp_name is not None) else run_name

    # ---- eval actor wrapper
    eval_actor = MeanPlannerActor(critics1, critics2, rl_cfg, device).to(device)

    # ---- real logging + load replay
    global_ep0 = 0
    episode_log_path = None
    episode_logger = None
    if env_type == "real":
        episode_log_path = project_root / "log" / "real_episodes" / "episode_logger_thompson.jsonl"
        episode_logger = EpisodeLoggerJsonl(episode_log_path)
        loaded_n = load_replay_from_jsonl(
            episode_log_path,
            replay_buffer_big,
            replay_buffer_recent,
            max_recent=1000,
            reward_cfg=rl_cfg.get("reward", None),
        )
        print(f"Loaded {loaded_n} episodes from {episode_log_path} into replay buffer.")
        if episode_log_path.exists():
            with episode_log_path.open("r") as f:
                global_ep0 = sum(1 for _ in f)
    

    ball_table = wandb.Table(columns=[
        "step",
        "x",
        "y",
        "reward",
        "in_hole",
        "planner",
    ])


    # Warm-start memory
    a_prev = None

    max_num_discs = rl_cfg["training"]["max_num_discs"]
    last_success_rate = 0.0
    last_last_success_rate = 0.0
    log_dict = {}

    for episode in range(episodes):
        # -------------------------------------------------
        # Periodic evaluation
        # -------------------------------------------------
        if env_type == "sim":
            eval_interval = int(rl_cfg["training"].get("eval_interval", 200))
            if (episode % eval_interval) == 0 and episode > 0:
                success_rate_eval, avg_reward_eval, avg_distance_eval = evaluation_policy_short(
                    eval_actor,
                    device,
                    mujoco_cfg,
                    rl_cfg,
                    num_episodes=int(rl_cfg["training"].get("eval_episodes", 10)),
                    max_num_discs=max_num_discs,
                    env_step=env_step,
                    env_type=env_type,
                )
                last_last_success_rate = last_success_rate
                last_success_rate = success_rate_eval
                
                # Increment num_discs
                if last_success_rate > 0.9 and last_last_success_rate > 0.9 and False:
                    max_num_discs = min(MAX_DISCS, max_num_discs + 1)
                    last_success_rate = 0.0
                    last_last_success_rate = 0.0
                    replay_buffer_recent.clear()
                    a_prev = None  # optional: reset CEM warm-start
                    print(f"[CURRICULUM] Increased max_num_discs -> {max_num_discs}")
                
                
                # print(
                #     f"[EVAL] Ep {episode}: success={success_rate_eval:.2f}, "
                #     f"avg_reward={avg_reward_eval:.3f}, avg_dist={avg_distance_eval:.3f}"
                # )
                if use_wandb:
                    wandb.log(
                        {
                            "success_rate": success_rate_eval,
                            "avg_reward_eval": avg_reward_eval,
                            "avg_distance_eval": avg_distance_eval,
                        },
                        step=episode,
                    )

        # -------------------------------------------------
        # Sample context
        # -------------------------------------------------
        if env_type == "sim":
            ball_start_obs, hole_pos_obs, disc_positions, x, y, hole_pos = sim_init_parameters(
                mujoco_cfg, max_num_discs
            )
        else:
            ball_start_obs, hole_pos_obs, disc_positions, chosen_hole = input_func(camera_index=camera_index_start, cap=cap)
            hole_pos = np.array(hole_pos_obs, dtype=float)

        # -------------------------------------------------
        # Encode + scale state (NO engineered features)
        # -------------------------------------------------
        state_vec = encode_state_with_discs(ball_start_obs, hole_pos_obs, disc_positions, 0)
        state_norm = scale_state_vec(state_vec)
        s = torch.tensor(state_norm, dtype=torch.float32, device=device)

        # -------------------------------------------------
        # Thompson head + CEM on min(Q1,Q2)
        # -------------------------------------------------
        head = int(np.random.randint(K))

        @torch.no_grad()
        def score_fn(s_batch, a_batch):
            q1 = critics1[head](s_batch, a_batch).squeeze(-1)
            q2 = critics2[head](s_batch, a_batch).squeeze(-1)
            return torch.minimum(q1, q2)

        init_mu = a_prev if (use_cem_warm_start and a_prev is not None) else None
        a_norm, cem_stats = cem_plan_action(
            s=s,
            score_fn=score_fn,
            action_dim=action_dim,
            device=device,
            cem_iters=cem_iters,
            cem_pop=cem_pop,
            cem_elite_frac=cem_elite_frac,
            init_std=cem_init_std,
            min_std=cem_min_std,
            init_mu=init_mu,
            return_stats=True,
        )
        a_prev = a_norm.detach()

        speed, angle_deg = get_input_parameters(a_norm.detach().cpu().numpy(), speed_low, speed_high, angle_low, angle_high)

        # sim actuator noise
        if env_type == "sim":
            # print(f"Before noise: speed={speed:.3f}, angle={angle_deg:.3f}")
            speed = np.clip(speed + np.random.normal(0, SPEED_NOISE_STD), speed_low, speed_high)
            angle_deg = np.clip(angle_deg + np.random.normal(0, ANGLE_NOISE_STD), angle_low, angle_high)
            # print(f"After noise: speed={speed:.3f}, angle={angle_deg:.3f}")

        # -------------------------------------------------
        # Step env
        # -------------------------------------------------
        if env_type == "sim":
            # disc_positions = [(2.0, -0.3), (2.1, 0.0), (2.0, 0.3), (2.4, -0.2), (2.4, 0.2)] # with 5 static discs
            # disc_positions = [(2.0, -0.3), (2.1, 0.0), (2.0, 0.3)] # with 3 static discs
            # disc_positions = [(2.1, 0.0)] # with 1 static disc
            disc_positions = [] # with 0 discs
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
                print(f"Episode {episode + 1}: failed twice. speed={speed}, angle={angle_deg}")
                if rl_cfg["training"].get("error_hard_stop", False) and env_type == "sim" and not mujoco_cfg["sim"]["render"]:
                    raise RuntimeError("Simulation failed twice — aborting training.")
                else:
                    continue

        ball_x, ball_y, in_hole, meta = result

        if env_type == "sim":
            meta = meta_from_trajectory_xy(meta, hole_pos_obs)

        is_out_of_bounds = False
        if env_type == "real" and isinstance(meta, dict):
            is_out_of_bounds = bool(meta.get("out_of_bounds", False))

        reward = float(compute_reward(
            ball_end_xy=np.array([ball_x, ball_y]),
            hole_xy=hole_pos_obs if env_type == "sim" else hole_pos,
            in_hole=in_hole,
            meta=meta,
            is_out_of_bounds=is_out_of_bounds,
            in_hole_reward=in_hole_reward,
            distance_scale=distance_scale,
            w_distance=w_distance,
            optimal_speed=optimal_speed,
            dist_at_hole_scale=dist_at_hole_scale,
            optimal_speed_scale=optimal_speed_scale,
        ))

        # ---- compute some extra metrics for logging
        with torch.no_grad():
            s1 = s.unsqueeze(0)
            a1 = a_norm.unsqueeze(0)
            q1s = torch.stack([c(s1, a1).squeeze() for c in critics1]).mean().item()
            q2s = torch.stack([c(s1, a1).squeeze() for c in critics2]).mean().item()
            q_min = min(q1s, q2s)

        if env_type == "real":
            log_payload = {
                "reward": float(reward),
                "in_hole": float(in_hole),
                "out_of_bounds": float(is_out_of_bounds) if env_type == "real" else 0.0,
                "speed": float(speed),
                "angle_deg": float(angle_deg),
                "chosen_hole": chosen_hole,
                "ball_start_x": float(ball_start_obs[0]),
                "ball_start_y": float(ball_start_obs[1]),
                "speed_at_hole": meta.get("speed_at_hole", None),
                "thompson_head": int(head),
                "q1_mean": q1s,
                "q2_mean": q2s,
                "q_min": q_min,
                "replay_big_size": len(replay_buffer_big.data),
                "replay_recent_size": len(replay_buffer_recent.data),

            }
        else:
            log_payload = {
                "reward": float(reward),
                "in_hole": float(in_hole),
                "speed": float(speed),
                "angle_deg": float(angle_deg),
                "thompson_head": int(head),
                "q1_mean": q1s,
                "q2_mean": q2s,
                "q_min": q_min,
                "replay_big_size": len(replay_buffer_big.data),
                "replay_recent_size": len(replay_buffer_recent.data),
            }
        if isinstance(meta, dict):
            if meta.get("dist_at_hole") is not None:
                log_payload["dist_at_hole"] = float(meta["dist_at_hole"])
            if meta.get("speed_at_hole") is not None:
                log_payload["speed_at_hole"] = float(meta["speed_at_hole"])

        log_payload.update({
            "cem_best_score": cem_stats["best_score"],
            "cem_std_mean": float(cem_stats["cem_std"].mean().item()),
            "cem_std_max": float(cem_stats["cem_std"].max().item()),
        })

        # -------------------------------------------------
        # Real logging
        # -------------------------------------------------
        if env_type == "real" and isinstance(meta, dict):
            used_for_training = bool(meta.get("used_for_training", True))
            if not used_for_training:
                print("Episode discarded by user; not adding to replay buffer.")
                continue_training_ = meta.get("continue_training", True)
                if not continue_training_:
                    print("Training aborted by user.")
                    break   # 👈 clean exit

                continue

            episode_logger.log(
                {
                    "reward": float(reward),
                    "in_hole": bool(in_hole),
                    "out_of_bounds": bool(meta.get("out_of_bounds", False)),
                    "ball_start_obs": ball_start_obs.tolist(),
                    "speed": float(speed),
                    "speed_at_hole": meta.get("speed_at_hole", None),
                    "angle_deg": float(angle_deg),
                    "thompson_head": int(head),
                    "q1_mean": q1s,
                    "q2_mean": q2s,
                    "q_min": q_min,
                    "cem_best_score": cem_stats["best_score"],
                    "cem_std_mean": float(cem_stats["cem_std"].mean().item()),
                    "cem_std_max": float(cem_stats["cem_std"].max().item()),
                    "replay_big_size": len(replay_buffer_big.data),
                    "replay_recent_size": len(replay_buffer_recent.data),
                    "episode": episode,
                    "used_for_training": used_for_training,                   
                    "hole_pos_obs": hole_pos_obs.tolist(),
                    "disc_positions": disc_positions,
                    "state_norm": state_norm.tolist(),
                    "action_norm": a_norm.detach().cpu().numpy().tolist(),                    
                    "ball_final_pos": [float(ball_x), float(ball_y)],                                      
                    "chosen_hole": chosen_hole,
                    "dist_at_hole": meta.get("dist_at_hole", None),
                    "exploring": True,
                    "planner": "ts_doublecritic_cem",
                }
            )

            if not meta.get("continue_training", True):
                print("Training aborted by user.")
                break

            if use_wandb:
                step = episode_logger.get_length() + 1
                wandb.log(log_payload, step=step)

        # -------------------------------------------------
        # Store
        # -------------------------------------------------
        replay_buffer_big.add(s.detach().cpu(), a_norm.detach().cpu(), float(reward))
        replay_buffer_recent.add(s.detach().cpu(), a_norm.detach().cpu(), float(reward))

        # -------------------------------------------------
        # Train
        # -------------------------------------------------
        if len(replay_buffer_big.data) >= batch_size:
            # print(f"Batch size: {batch_size}")
            if env_type == "real":
                print(f"Training on episode {episode + 1} with replay size {len(replay_buffer_big.data)}")

            for _ in range(grad_steps):
                states_b, actions_b, rewards_b = sample_mixed(
                    replay_buffer_recent,
                    replay_buffer_big,
                    recent_ratio=0.7,
                    batch_size=batch_size,
                    # ball_xy_idx=(0, 1),
                )
                states_b = states_b.to(device)
                actions_b = actions_b.to(device)
                rewards_b = rewards_b.to(device)

                for k in range(K):
                    mask = (torch.rand(rewards_b.shape[0], device=device) < float(bootstrap_p))
                    if int(mask.sum().item()) < 2:
                        continue

                    sb, ab, rb = states_b[mask], actions_b[mask], rewards_b[mask]

                    q1 = critics1[k](sb, ab)
                    loss1 = F.mse_loss(q1, rb)
                    opts1[k].zero_grad()
                    loss1.backward()
                    torch.nn.utils.clip_grad_norm_(critics1[k].parameters(), 1.0)
                    opts1[k].step()

                    q2 = critics2[k](sb, ab)
                    loss2 = F.mse_loss(q2, rb)
                    opts2[k].zero_grad()
                    loss2.backward()
                    torch.nn.utils.clip_grad_norm_(critics2[k].parameters(), 1.0)
                    opts2[k].step()

            if env_type == "real":
                print(f"Completed gradient steps for episode {episode + 1}. Saving models...")
                save_doublecritics(model_dir, tmp_name, critics1, critics2)
        # -------------------------------------------------
        # Prints + wandb
        # -------------------------------------------------
        if rl_cfg["training"].get("do_prints", False):
            print("========================================")
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward:.4f}, TS head: {head}")

    if env_type == "real" and cap is not None:
        cap.release()

    # -------------------------------------------------
    # Final evaluation + save
    # -------------------------------------------------
    if env_type == "sim":
        final_success, final_avg_reward, final_avg_dist = evaluation_policy_short(
            eval_actor,
            device,
            mujoco_cfg,
            rl_cfg,
            num_episodes=100,
            max_num_discs=max_num_discs,
            env_step=env_step,
            env_type=env_type,
        )
        print(
            f"[FINAL EVAL] success={final_success:.2f}, "
            f"avg_reward={final_avg_reward:.3f}, avg_dist={final_avg_dist:.3f}"
        )
        if use_wandb:
            wandb.log(
                {
                    "final_success_rate": final_success,
                    "final_avg_reward": final_avg_reward,
                    "final_avg_distance": final_avg_dist,
                }
            )
    
    if env_type == "sim":
        print(f"Saving final double-critic models to tag: {tag_for_saving}")
        save_doublecritics(model_dir, tag_for_saving, critics1, critics2)

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
    
    if env_type == "sim" and rl_cfg["training"].get("use_wandb", False):
        project_name = rl_cfg["training"].get("project_name", "rl_golf_wandb")
        wandb.init(
            project=project_name,
            group="dqn-bts",
            config={
                "rl_config": rl_cfg,
                "mujoco_config": mujoco_cfg,
            },
        )

    if wandb.run is not None:
        cfg = wandb.config

        rl_cfg["reward"]["distance_scale"]     = float(cfg.get("distance_scale", rl_cfg["reward"]["distance_scale"]))
        rl_cfg["reward"]["in_hole_reward"]     = float(cfg.get("in_hole_reward", rl_cfg["reward"]["in_hole_reward"]))
        rl_cfg["reward"]["w_distance"]         = float(cfg.get("w_distance", rl_cfg["reward"]["w_distance"]))
        rl_cfg["reward"]["optimal_speed"]      = float(cfg.get("optimal_speed", rl_cfg["reward"]["optimal_speed"]))
        rl_cfg["reward"]["dist_at_hole_scale"] = float(cfg.get("dist_at_hole_scale", rl_cfg["reward"]["dist_at_hole_scale"]))
        rl_cfg["reward"]["optimal_speed_scale"]= float(cfg.get("optimal_speed_scale", rl_cfg["reward"]["optimal_speed_scale"]))
        rl_cfg["training"]["cem_pop"]         = int(cfg.get("cem_pop", rl_cfg["training"]["cem_pop"]))
        rl_cfg["training"]["cem_iters"]       = int(cfg.get("cem_iters", rl_cfg["training"]["cem_iters"]))
        rl_cfg["training"]["critic_lr"]       = float(cfg.get("critic_lr", rl_cfg["training"]["critic_lr"]))
        rl_cfg["training"]["bootstrap_p"]     = float(cfg.get("bootstrap_p", rl_cfg["training"]["bootstrap_p"]))

    print("RL Config:", rl_cfg)
    training(
        rl_cfg,
        mujoco_cfg,
        project_root,
        continue_training=rl_cfg["training"].get("continue_training", False),
        env_step=env_step,
        env_type=env_type,
        tmp_name=tmp_name if env_type == "real" else None,
    )

    if env_type == "sim" and tmp_xml_path is not None:
        try:
            os.remove(tmp_xml_path)
        except FileNotFoundError:
            pass
