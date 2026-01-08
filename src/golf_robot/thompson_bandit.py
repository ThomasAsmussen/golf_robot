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


# =========================================================
# Algorithm: Critic-only Bootstrapped Thompson + CEM planning
# =========================================================
# - No actor networks (removes instability)
# - K bootstrapped critics approximate posterior over Q(s,a)=E[r|s,a]
# - Each episode: sample head k, use CEM to maximize Q_k(s,a) over action
# - Train critics by supervised regression Q(s,a)->r


# ---------------------------------------------------------
# Small helper: Ensemble wrapper to choose actions deterministically for eval
# ---------------------------------------------------------
class EnsembleCriticMeanPlanner:
    """
    Deterministic planner for evaluation:
      - score(a) = mean_k Q_k(s,a)
      - choose best action via CEM (or random shooting)
    """
    def __init__(self, critics, device):
        self.critics = critics
        self.device = device

    @torch.no_grad()
    def score_actions(self, s_batch, a_batch):
        # s_batch: [N, state_dim], a_batch: [N, action_dim]
        qs = []
        for c in self.critics:
            qs.append(c(s_batch, a_batch))  # [N,1]
        q = torch.stack(qs, dim=0).mean(dim=0)  # [N,1]
        return q.squeeze(-1)  # [N]

    def act(self, s, cfg):
        cem_pop_val = 256
        cem_iters_val = 2
        cem_elite_frac_val = 0.2
        cem_init_std_val = 0.4
        cem_min_std_val = 0.1
        return cem_plan_action(
            s=s,
            score_fn=self.score_actions,
            action_dim=cfg["model"]["action_dim"],
            device=self.device,
            cem_iters=cfg["training"].get("cem_iters_eval", cfg["training"].get("cem_iters", cem_iters_val)),
            cem_pop=cfg["training"].get("cem_pop_eval", cfg["training"].get("cem_pop", cem_pop_val)),
            cem_elite_frac=cfg["training"].get("cem_elite_frac", cem_elite_frac_val),
            action_low=-1.0,
            action_high=1.0,
            init_std=cfg["training"].get("cem_init_std", cem_init_std_val),
            min_std=cfg["training"].get("cem_min_std", cem_min_std_val),
        )


# ---------------------------------------------------------
# CEM planner
# ---------------------------------------------------------
def cem_plan_action(
    s,
    score_fn,
    action_dim,
    device,
    cem_iters=3,
    cem_pop=512,
    cem_elite_frac=0.1,
    action_low=-1.0,
    action_high=1.0,
    init_std=0.7,
    min_std=0.05,
):
    """
    Cross-Entropy Method in normalized action space [-1,1]^action_dim.

    Inputs:
      s: torch tensor [state_dim] on device
      score_fn: callable(s_batch, a_batch) -> scores [N] (higher is better)
    Returns:
      best_action: torch tensor [action_dim] on device
    """
    elite_n = max(1, int(round(cem_pop * cem_elite_frac)))

    mu = torch.zeros(action_dim, device=device)
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

    return best_a


# ---------------------------------------------------------
# Helper: Find newest checkpoint set (head0 critic as reference)
# ---------------------------------------------------------
def find_latest_bootcritic_checkpoint(model_dir: Path, prefix: str | None = None):
    """
    Find newest checkpoint set by looking for head0 critic file:
      ddpg_critic_<something>_h0.pth
    Returns: (critic_h0_path, tag_stem)
    where tag_stem is the stem with "_h0" stripped.
    """
    critic_files = list(model_dir.glob("ddpg_critic_*_h0.pth"))
    if prefix is not None:
        critic_files = [f for f in critic_files if prefix in f.stem]
    if not critic_files:
        raise FileNotFoundError("No matching bootstrapped critic (head0) checkpoints found.")

    critic_h0 = max(critic_files, key=lambda f: f.stat().st_mtime)
    tag_stem = critic_h0.stem.replace("_h0", "")
    return critic_h0, tag_stem


def load_bootcritics(model_dir: Path, rl_cfg, tag_stem: str, device):
    """
    Load all bootstrapped critics:
      expects files:
        ddpg_critic_<tag>_h{k}.pth
    where <tag> is tag_stem without "ddpg_critic_".
    """
    K = rl_cfg["training"].get("num_heads", 8)
    state_dim = rl_cfg["model"]["state_dim"]
    action_dim = rl_cfg["model"]["action_dim"]
    hidden_dim = rl_cfg["model"]["hidden_dim"]

    if not tag_stem.startswith("ddpg_critic_"):
        raise ValueError(f"Unexpected tag_stem format: {tag_stem}")

    tag = tag_stem.replace("ddpg_critic_", "", 1)

    critics = []
    for k in range(K):
        critic_path = model_dir / f"ddpg_critic_{tag}_h{k}.pth"
        if not critic_path.exists():
            raise FileNotFoundError(f"Missing critic head file for k={k}: {critic_path}")

        critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        critic.load_state_dict(torch.load(critic_path, map_location=device))
        critic.train()
        critics.append(critic)

    return critics, tag


def save_bootcritics(model_dir: Path, tag: str, critics):
    """
    Save critics to:
      ddpg_critic_<tag>_h{k}.pth
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    for k, c in enumerate(critics):
        torch.save(c.state_dict(), model_dir / f"ddpg_critic_{tag}_h{k}.pth")


# ---------------------------------------------------------
# Training loop (Bootstrapped Thompson + CEM planning)
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
    Good, stable algorithm for your setting:
      - Critic-only bootstrapped Thompson sampling
      - Action chosen by planning in action-space (CEM) against sampled critic head
      - Critics trained by supervised regression Q(s,a)->r

    This converges much more reliably than actor-critic for one-step tasks.
    """
    if env_step is None:
        raise ValueError("training() requires env_step (sim or real environment function).")

    episodes   = rl_cfg["training"]["episodes"]
    batch_size = rl_cfg["training"]["batch_size"]
    critic_lr  = rl_cfg["training"]["critic_lr"]
    grad_steps = rl_cfg["training"]["grad_steps"]

    # Bootstrapped Thompson controls
    K = rl_cfg["training"].get("num_heads", 8)
    bootstrap_p = rl_cfg["training"].get("bootstrap_p", 0.8)  # per-head mask prob

    # CEM controls
    cem_pop = rl_cfg["training"].get("cem_pop", 512)
    cem_iters = rl_cfg["training"].get("cem_iters", 3)
    cem_elite_frac = rl_cfg["training"].get("cem_elite_frac", 0.1)
    cem_init_std = rl_cfg["training"].get("cem_init_std", 0.7)
    cem_min_std = rl_cfg["training"].get("cem_min_std", 0.05)

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

    if env_type == "sim":
        use_wandb = rl_cfg["training"]["use_wandb"]
    else:
        use_wandb = False

    model_name = rl_cfg["training"].get("model_name", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dir = project_root / "models" / "rl" / "bandit"
    model_dir.mkdir(parents=True, exist_ok=True)

    tag_for_saving = None

    # -------------------------
    # Initialize critics (K heads)
    # -------------------------
    if continue_training:
        if model_name is not None:
            # Load by explicit tag
            tag_for_saving = model_name
            tag_stem = f"ddpg_critic_{model_name}"
            critics, _ = load_bootcritics(model_dir, rl_cfg, tag_stem, device)
            print(f"Continuing training from tag: {model_name}")
        else:
            # Load newest by head0 critic timestamp
            critic_h0, tag_stem = find_latest_bootcritic_checkpoint(model_dir, prefix=None)
            print("Continuing training from latest bootstrapped critics:")
            print(f"  Critic0 : {critic_h0.name}")
            critics, loaded_tag = load_bootcritics(model_dir, rl_cfg, tag_stem, device)
            tag_for_saving = loaded_tag
    else:
        critics = [Critic(state_dim, action_dim, hidden_dim).to(device) for _ in range(K)]

    # Use Adam (much more stable than SGD)
    critic_optimizers = [torch.optim.Adam(c.parameters(), lr=critic_lr) for c in critics]

    # Replay buffers
    replay_buffer_big = ReplayBuffer(capacity=rl_cfg["training"]["replay_buffer_capacity"])
    replay_buffer_recent = ReplayBuffer(1000)

    # wandb
    if use_wandb:
        wandb.watch(critics[0], log="gradients", log_freq=100)
        run_name = wandb.run.name.replace("-", "_")
    else:
        run_name = "local_run"

    # Choose save tag
    if tag_for_saving is None:
        if env_type == "real" and tmp_name is not None:
            tag_for_saving = tmp_name
        else:
            tag_for_saving = run_name

    # Evaluation planner: mean critic
    eval_planner = EnsembleCriticMeanPlanner(critics, device)

    # Real-robot episode logging + replay load
    episode_logger = None
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

    # Stats for prints
    log_dict = {}
    last_success_rate = 0.0
    last_last_success_rate = 0.0
    max_num_discs = 2

    for episode in range(episodes):
        # -------------------------------------------------
        # Periodic eval (real)
        # -------------------------------------------------
        if env_type == "real" and episode_logger.get_length() % 10000 == 0:
            # Use the mean-critic planner: we wrap it as a tiny "actor" with forward()
            class _EvalActor(nn.Module):
                def __init__(self, planner, cfg):
                    super().__init__()
                    self.planner = planner
                    self.cfg = cfg
                def forward(self, state):
                    # state is [B, state_dim], but evaluation_policy_short feeds [1, state_dim]
                    s = state.squeeze(0)
                    a = self.planner.act(s, self.cfg)
                    return a.unsqueeze(0)

            eval_actor = _EvalActor(eval_planner, rl_cfg).to(device)

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
        # Sample a context
        # -------------------------------------------------
        if env_type == "sim":
            ball_start_obs, hole_pos_obs, disc_positions, x, y, hole_pos = sim_init_parameters(
                mujoco_cfg, max_num_discs
            )
        else:
            ball_start_obs, hole_pos_obs, disc_positions, chosen_hole = input_func(
                camera_index=camera_index_start
            )
            hole_pos = np.array(hole_pos_obs, dtype=float)

        # -------------------------------------------------
        # Encode + scale state (match your original training)
        # -------------------------------------------------
        state_vec = encode_state_with_discs(ball_start_obs, hole_pos_obs, disc_positions, 5)
        state_norm = scale_state_vec(state_vec)
        s = torch.tensor(state_norm, dtype=torch.float32, device=device)

        # -------------------------------------------------
        # Thompson step: sample one critic head, plan action by CEM
        # -------------------------------------------------
        head = int(np.random.randint(K))
        lcb_lambda_val = 0.5 # bigger means more conservative
        lcb_lambda = rl_cfg["training"].get("lcb_lambda", lcb_lambda_val) 

        @torch.no_grad()
        def score_fn(s_batch, a_batch):
            qs = []
            for c in critics:
                qs.append(c(s_batch, a_batch))  # [N,1]
            q = torch.stack(qs, dim=0).squeeze(-1)  # [K,N]
            mu = q.mean(dim=0)                      # [N]
            std = q.std(dim=0, unbiased=False)      # [N]
            return mu - float(lcb_lambda) * std     # LCB score

        # def score_fn(s_batch, a_batch):
        #     # Use sampled head only
        #     q = critics[head](s_batch, a_batch)  # [N,1]
        #     return q.squeeze(-1)  # [N]

        a_norm = cem_plan_action(
            s=s,
            score_fn=score_fn,
            action_dim=action_dim,
            device=device,
            cem_iters=cem_iters,
            cem_pop=cem_pop,
            cem_elite_frac=cem_elite_frac,
            action_low=-1.0,
            action_high=1.0,
            init_std=cem_init_std,
            min_std=cem_min_std,
        )

        # Convert to physical action
        speed, angle_deg = get_input_parameters(
            a_norm.detach().cpu().numpy(), speed_low, speed_high, angle_low, angle_high
        )

        # Environment / actuator noise (sim)
        if env_type == "sim":
            speed_noise      = np.random.normal(0, SPEED_NOISE_STD)
            angle_deg_noise  = np.random.normal(0, ANGLE_NOISE_STD)
            speed     = np.clip(speed + speed_noise,     speed_low, speed_high)
            angle_deg = np.clip(angle_deg + angle_deg_noise, angle_low, angle_high)

        # -------------------------------------------------
        # Step environment
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
                    f"Episode {episode + 1}: failed twice. "
                    f"speed={speed}, angle={angle_deg}"
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

        # Reward
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
                rewards.append(float(reward_try))
            reward = float(max(rewards))
        else:
            reward = float(compute_reward(
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
            ))

        # -------------------------------------------------
        # Real logging
        # -------------------------------------------------
        if env_type == "real" and isinstance(meta, dict):
            used_for_training = bool(meta.get("used_for_training", True))
            out_of_bounds = bool(meta.get("out_of_bounds", False))

            if not used_for_training:
                print("Episode discarded by user; not adding to replay buffer.")
                continue

            logging_dict = {
                "episode": episode,
                "time": time.time(),
                "used_for_training": used_for_training,
                "ball_start_obs": ball_start_obs.tolist(),
                "hole_pos_obs": hole_pos_obs.tolist(),
                "disc_positions": disc_positions,
                "state_norm": state_norm.tolist(),
                "action_norm": a_norm.detach().cpu().numpy().tolist(),
                "speed": float(speed),
                "angle_deg": float(angle_deg),
                "ball_final_pos": [float(ball_x), float(ball_y)],
                "in_hole": bool(in_hole),
                "out_of_bounds": out_of_bounds,
                "reward": float(reward),
                "chosen_hole": chosen_hole,
                "dist_at_hole": meta.get("dist_at_hole", None),
                "speed_at_hole": meta.get("speed_at_hole", None),
                "exploring": True,
                "thompson_head": head,
                "planner": "boot_ts_cem",
            }
            episode_logger.log(logging_dict)

        # -------------------------------------------------
        # Store transition(s)
        # -------------------------------------------------
        s_train = s.detach().cpu()
        a_train = a_norm.detach().cpu()

        if using_all_holes:
            for r in rewards:
                replay_buffer_big.add(s_train, a_train, float(r))
                replay_buffer_recent.add(s_train, a_train, float(r))
        else:
            replay_buffer_big.add(s_train, a_train, float(reward))
            replay_buffer_recent.add(s_train, a_train, float(reward))

        # -------------------------------------------------
        # Train critics (supervised regression) with bootstrap masks
        # -------------------------------------------------
        if len(replay_buffer_big.data) >= batch_size:
            if env_type == "real":
                print("Updating critics...")

            for _ in range(grad_steps):
                states_b, actions_b, rewards_b = sample_mixed(
                    replay_buffer_recent,
                    replay_buffer_big,
                    recent_ratio=0.7,
                    batch_size=batch_size,
                )

                states_b  = states_b.to(device)
                actions_b = actions_b.to(device)
                rewards_b = rewards_b.to(device)  # [B,1]

                for k in range(K):
                    mask = (torch.rand(rewards_b.shape[0], device=device) < float(bootstrap_p))
                    if int(mask.sum().item()) < 2:
                        continue

                    sb = states_b[mask]
                    ab = actions_b[mask]
                    rb = rewards_b[mask]

                    q_pred = critics[k](sb, ab)
                    loss = F.mse_loss(q_pred, rb)

                    critic_optimizers[k].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(critics[k].parameters(), 1.0)
                    critic_optimizers[k].step()

            # Save periodically for real robot
            if env_type == "real" and tmp_name is not None:
                save_bootcritics(model_dir, tag_for_saving, critics)

        if env_type == "real" and isinstance(meta, dict):
            if not meta.get("continue_training", True):
                print("Training aborted by user.")
                break

        # -------------------------------------------------
        # Prints + wandb
        # -------------------------------------------------
        if rl_cfg["training"]["do_prints"]:
            print("========================================")
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward:.4f}, TS head: {head}")

        if use_wandb:
            log_dict["reward"] = float(reward)
            log_dict["thompson_head"] = int(head)
            wandb.log(log_dict, step=episode)

        # -------------------------------------------------
        # Periodic eval (sim)
        # -------------------------------------------------
        if (episode) % rl_cfg["training"]["eval_interval"] == 24 and env_type == "sim":

            # Wrap the mean planner as an "actor" for existing eval function
            class _EvalActor(nn.Module):
                def __init__(self, planner, cfg):
                    super().__init__()
                    self.planner = planner
                    self.cfg = cfg
                def forward(self, state):
                    s1 = state.squeeze(0)
                    a = self.planner.act(s1, self.cfg)
                    return a.unsqueeze(0)

            eval_actor = _EvalActor(eval_planner, rl_cfg).to(device)

            success_rate_eval, avg_reward_eval, avg_distance_to_hole_eval = evaluation_policy_short(
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
                wandb.log(
                    {
                        "success_rate": success_rate_eval,
                        "avg_reward": avg_reward_eval,
                        "avg_distance_to_hole": avg_distance_to_hole_eval,
                        "K_heads": int(K),
                        "bootstrap_p": float(bootstrap_p),
                        "cem_pop": int(cem_pop),
                        "cem_iters": int(cem_iters),
                    },
                    step=episode,
                )

    # -------------------------------------------------
    # Final evaluation + saving
    # -------------------------------------------------
    if env_type == "sim":
        class _EvalActor(nn.Module):
            def __init__(self, planner, cfg):
                super().__init__()
                self.planner = planner
                self.cfg = cfg
            def forward(self, state):
                s1 = state.squeeze(0)
                a = self.planner.act(s1, self.cfg)
                return a.unsqueeze(0)

        eval_actor = _EvalActor(eval_planner, rl_cfg).to(device)

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

    save_bootcritics(model_dir, tag_for_saving, critics)

    if use_wandb and env_type == "sim":
        wandb.log(
            {
                "final_avg_reward": final_avg_reward,
                "final_success_rate": final_success_rate,
                "final_avg_distance_to_hole": final_avg_distance_to_hole,
            }
        )

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

    if rl_cfg["training"]["use_wandb"]:
        wandb.init(
            project="rl_golf_boot_ts_cem",
            config={
                "rl_config": rl_cfg,
                "mujoco_config": mujoco_cfg,
            },
        )

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
