import os
from pathlib import Path
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
import copy
    

HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, ".."))
XML_PATH = os.path.join(REPO, "simulation", "run_sim.py")


def squash_to_range(x, lo, hi):
    """Squash a value in [-1, 1] to [lo, hi]."""
    return lo + 0.5 * (x + 1.0) * (hi - lo)


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
    

class Critic(nn.Module):
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
    
    
class ReplayBuffer:
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
    

def soft_update(target, source, rho):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(rho * param.data + (1.0 - rho) * target_param.data)


def compute_reward(ball_end_pos, hole_pos, in_hole):
    if in_hole:
        return 1
    
    dist = np.linalg.norm(ball_end_pos - hole_pos)
    # if dist > 0.5:
    #     return 0.0
    # else:
    #     return 1 - (dist / 0.5)
    k = 3.0
    reward = np.exp(-k * dist)
    return reward
    

def final_state_from_csv(csv_path):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    final_x = data[-1, 1]
    final_y = data[-1, 2]
    in_hole = int(data[-1, 4])
    return final_x, final_y, in_hole


def training(rl_cfg, mujoco_cfg, project_root, continue_training=False):
    episodes    = rl_cfg["training"]["episodes"]
    batch_size  = rl_cfg["training"]["batch_size"]
    actor_lr    = rl_cfg["training"]["actor_lr"]
    critic_lr   = rl_cfg["training"]["critic_lr"]
    noise_std   = rl_cfg["training"]["noise_std"]
    grad_steps  = rl_cfg["training"]["grad_steps"]

    state_dim   = rl_cfg["model"]["state_dim"]
    action_dim  = rl_cfg["model"]["action_dim"]
    hidden_dim  = rl_cfg["model"]["hidden_dim"]
    speed_low   = rl_cfg["model"]["speed_low"]
    speed_high  = rl_cfg["model"]["speed_high"]
    angle_low   = rl_cfg["model"]["angle_low"]
    angle_high  = rl_cfg["model"]["angle_high"]

    use_wandb   = rl_cfg["training"]["use_wandb"]

    gamma = 0.99

    csv_path = project_root / mujoco_cfg["sim"]["csv_path"]

    if continue_training:
        actor, device = load_actor(
            model_path=project_root / "models" / "rl" / "ddpg" / "ddpg_actor.pth",
            rl_cfg=rl_cfg,
        )
        critic, _ = load_critic(
            model_path=project_root / "models" / "rl" / "ddpg" / "ddpg_critic.pth",
            rl_cfg=rl_cfg,
        )

        target_actor, _ = load_actor(
            model_path=project_root / "models" / "rl" / "ddpg" / "ddpg_target_actor.pth",
            rl_cfg=rl_cfg,
        )

        critic, _ = load_critic(
            model_path=project_root / "models" / "rl" / "ddpg" / "ddpg_target_critic.pth",
            rl_cfg=rl_cfg,
        )


    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        target_actor = copy.deepcopy(actor).to(device)
        target_critic = copy.deepcopy(critic).to(device)

    print(f"Using device: {device}")

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    replay_buffer = ReplayBuffer(capacity=rl_cfg["training"]["replay_buffer_capacity"])

    if use_wandb:
        wandb.watch(actor, log="gradients", log_freq=100)
        wandb.watch(critic, log="gradients", log_freq=100)
        run_name = wandb.run.name.replace("-", "_")
    else:
        run_name = "local_run"

    log_dir = project_root / "log" / "ddpg"
    model_dir = project_root / "models" / "rl" / "ddpg"

    actor.train()
    critic.train()
    log_dict = {}

    last_avg_reward = None

    for episode in range(episodes):
        ball_start = np.array([0.0, 0.0])
        x, y = random_hole_in_donut(0.5, 1.5)
        hole_pos = np.array([x, y])
        s = torch.tensor(np.concatenate([ball_start, hole_pos]), dtype=torch.float32).to(device)

        critic_loss_value = None
        actor_loss_value = None

        with torch.no_grad():
            a_norm = actor(s.unsqueeze(0)).squeeze(0)
            noise = torch.normal(
                mean=torch.zeros_like(a_norm),
                std=noise_std * torch.ones_like(a_norm),
            )

        a_noisy = torch.clamp(a_norm + noise, -1.0, 1.0)

        speed, angle_deg = get_sim_input(a_noisy.cpu().numpy(), speed_low, speed_high)

        result = run_sim(angle_deg, speed, [x, y], mujoco_cfg)

        if result is None:
            print(f"Episode {episode + 1}: Simulation failed, trying again.")
            result = run_sim(angle_deg, speed, [x, y], mujoco_cfg)
            if result is None:
                print(f"Episode {episode + 1}: Simulation failed again. Bad action: speed={speed}, angle={angle_deg}")
                print(f"  Hole Position: x={x:.4f}, y={y:.4f}")
                if rl_cfg["training"]["error_hard_stop"] and not mujoco_cfg["sim"]["render"]:
                    raise RuntimeError("Simulation failed twice — aborting training.")
                else:
                    continue
            
                
        else:
            ball_x, ball_y, in_hole = result
        # ball_x, ball_y, in_hole = final_state_from_csv(csv_path)

        reward = compute_reward(
            ball_end_pos=np.array([ball_x, ball_y]),
            hole_pos=hole_pos,
            in_hole=in_hole,
        )

        s_train = s.detach().cpu()
        a_train = a_noisy.detach().cpu()

        replay_buffer.add(s_train, a_train, reward)

        if len(replay_buffer.data) >= batch_size:
            for _ in range(grad_steps):
                states_b, actions_b, rewards_b = replay_buffer.sample(batch_size)

                states_b = states_b.to(device)
                actions_b = actions_b.to(device)
                rewards_b = rewards_b.to(device)

                with torch.no_grad():
                    target_actions = target_actor(states_b)
                    target_q = target_critic(states_b, target_actions)

                td_target = rewards_b + gamma * target_q

                q_pred = critic(states_b, actions_b)
                critic_loss = F.mse_loss(q_pred, td_target)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                a_for_actor = actor(states_b)
                actor_loss = -critic(states_b, a_for_actor).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

            soft_update(target_actor, actor, rl_cfg["training"]["target_update_rho"])
            soft_update(target_critic, critic, rl_cfg["training"]["target_update_rho"])
            
            critic_loss_value = critic_loss.item()
            actor_loss_value = actor_loss.item()

        if (episode + 1) % 1 == 0:
            print("========================================")
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward:.4f}")
            print(f"  Hole Position: x={x:.4f}, y={y:.4f}")
            print(f"  Speed: {speed:.4f}, Angle: {angle_deg:.4f}")

        if (episode) % rl_cfg["training"]["eval_interval"] == 0:
            success_rate_eval, avg_reward_eval, avg_distance_to_hole_eval = evaluation_policy_short(
                actor,
                device,
                mujoco_cfg,
                project_root,
                rl_cfg,
                num_episodes=rl_cfg["training"]["eval_episodes"],
            )

            if last_avg_reward is not None:
                improvement = avg_reward_eval - last_avg_reward

                # Only react if change is significant (e.g. 0.02)
                if avg_reward_eval > 0.5 and improvement > 0.02:
                    noise_std *= 0.95  # slightly less exploration
                elif avg_reward_eval < 0.5 and improvement < -0.02:
                    noise_std *= 1.05  # slightly more exploration

                # Clip noise to reasonable bounds
                noise_std = float(np.clip(noise_std, 0.05, 0.6))

            # Update for next eval
            last_avg_reward = avg_reward_eval

            print(f"[EVAL] Success Rate: {success_rate_eval:.2f}, Avg Reward: {avg_reward_eval:.3f}")

            if use_wandb:
                log_dict["success_rate"] = success_rate_eval
                log_dict["avg_reward"] = avg_reward_eval
                log_dict["avg_distance_to_hole"] = avg_distance_to_hole_eval
                # wandb.log(log_dict, step=episode)

        if use_wandb:
            distance_to_hole = np.linalg.norm(np.array([ball_x, ball_y]) - hole_pos)
            # log_dict = {"reward": reward, "distance_to_hole": distance_to_hole}
            log_dict["reward"] = reward
            log_dict["distance_to_hole"] = distance_to_hole
            if critic_loss_value is not None:
                log_dict["critic_loss"] = critic_loss_value
            if actor_loss_value is not None:
                log_dict["actor_loss"] = actor_loss_value

            wandb.log(log_dict, step=episode)

    print("Sweep complete")


def load_actor(model_path, rl_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim   = rl_cfg["model"]["state_dim"]
    action_dim  = rl_cfg["model"]["action_dim"]
    hidden_dim  = rl_cfg["model"]["hidden_dim"]

    actor = Actor(state_dim, action_dim, hidden_dim).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()
    return actor, device


def load_critic(model_path, rl_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim   = rl_cfg["model"]["state_dim"]
    action_dim  = rl_cfg["model"]["action_dim"]
    hidden_dim  = rl_cfg["model"]["hidden_dim"]

    critic = Critic(state_dim, action_dim, hidden_dim).to(device)
    critic.load_state_dict(torch.load(model_path, map_location=device))
    critic.eval()
    return critic, device


def evaluate_policy_random(model_path, rl_cfg, mujoco_cfg, project_root, num_episodes, r_min=0.5, r_max=1.5):
    actor, device = load_actor(model_path, rl_cfg)

    speed_low   = rl_cfg["model"]["speed_low"]
    speed_high  = rl_cfg["model"]["speed_high"]
    angle_low     = rl_cfg["model"]["angle_low"]
    angle_high    = rl_cfg["model"]["angle_high"]

    csv_path = project_root / mujoco_cfg["sim"]["csv_path"]

    ball_start = np.array([0.0, 0.0])

    succeses = 0
    rewards = []

    for episode in range(num_episodes):
        x, y = random_hole_in_donut(r_min, r_max)
        hole_pos = np.array([x, y])

        state = torch.tensor(np.concatenate([ball_start, hole_pos]), dtype=torch.float32).to(device)

        with torch.no_grad():
            a_norm = actor(state.unsqueeze(0)).squeeze(0)

        speed, angle_deg = get_sim_input(a_norm.cpu().numpy(), speed_low, speed_high)

        ball_x, ball_y, in_hole = run_sim(angle_deg, speed, [x, y], mujoco_cfg)

        # ball_x, ball_y, in_hole = final_state_from_csv(csv_path)

        reward = compute_reward(ball_end_pos=np.array([ball_x, ball_y]), hole_pos=hole_pos, in_hole=in_hole)

        rewards.append(reward)

        succeses += int(in_hole == 1)

        print(f"Episode {episode + 1}/{num_episodes}, Reward: {reward:.4f}, In Hole: {in_hole}, Hole Pos: ({x:.4f}, {y:.4f}), Speed: {speed:.4f}, Angle: {angle_deg:.4f}")

    succes_rate = succeses / num_episodes
    avg_reward = np.mean(rewards)

    print("========================================")
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {succes_rate:.2f}, Average Reward: {avg_reward:.4f}")


def get_sim_input(a_norm, speed_low, speed_high):
    speed_norm, cos_theta, sin_theta = a_norm
    norm = np.hypot(sin_theta, cos_theta)
    if norm < 1e-6:
        cos_theta = 1.0
        sin_theta = 0.0
    else:
        cos_theta /= norm
        sin_theta /= norm
    
    angle_rad = np.arctan2(sin_theta, cos_theta)
    angle_deg = np.degrees(angle_rad)

    speed = squash_to_range(speed_norm, speed_low, speed_high)
    return speed, angle_deg


def evaluate_policy_grid(model_path, rl_cfg, mujoco_cfg, project_root, r_min=0.5, r_max=1.5, shots_per_hole=1):
    
    actor, device = load_actor(model_path, rl_cfg)

    speed_low   = rl_cfg["model"]["speed_low"]
    speed_high  = rl_cfg["model"]["speed_high"]
    angle_low     = rl_cfg["model"]["angle_low"]
    angle_high    = rl_cfg["model"]["angle_high"]

    x_min = -1.5
    x_max = 1.5
    y_min = -1.5
    y_max = 1.5
    num_x = 20
    num_y = 20
    
    csv_path = project_root / mujoco_cfg["sim"]["csv_path"]
    ball_start = np.array([0.0, 0.0])

    x_coords = np.linspace(x_min, x_max, num_x)
    y_coords = np.linspace(y_min, y_max, num_y)

    success_grid = np.full((num_x, num_y), np.nan)
    reward_grid = np.full((num_x, num_y), np.nan)

    for iy, y in enumerate(y_coords):
        for ix, x in enumerate(x_coords):
            r = np.sqrt(x**2 + y**2)

            if r < r_min or r > r_max:
                continue

            hole_pos = np.array([x, y])

            succeses = 0
            rewards = []

            for _ in range(shots_per_hole):
                state = torch.tensor(np.concatenate([ball_start, hole_pos]), dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    a_norm = actor(state).squeeze(0).cpu().numpy()

                speed, angle_deg = get_sim_input(a_norm, speed_low, speed_high)

                ball_x, ball_y, in_hole = run_sim(angle_deg, speed, [x, y], mujoco_cfg)

                # ball_x, ball_y, in_hole = final_state_from_csv(csv_path)

                reward = compute_reward(ball_end_pos=np.array([ball_x, ball_y]), hole_pos=hole_pos, in_hole=in_hole)
                
                print(f"Hole ({x:.2f}, {y:.2f}), Speed: {speed:.2f}, Angle: {angle_deg:.2f} => Ball ({ball_x:.2f}, {ball_y:.2f}), In Hole: {in_hole}, Reward: {reward:.4f}")
                rewards.append(reward)
                succeses += int(in_hole == 1)

                # print(ix+iy)
        
            success_rate = succeses / shots_per_hole
            avg_reward = np.mean(rewards)

            success_grid[ix, iy] = success_rate
            reward_grid[ix, iy] = avg_reward

    print("\n=== GRID POLICY EVALUATION ===")
    print(f"Grid size       : {num_x} x {num_y}")
    print(f"Shots per hole  : {shots_per_hole}")
    print("Finished grid evaluation.")
    print("================================\n")

    return x_coords, y_coords, success_grid, reward_grid    

            
def plot_reward_heatmap(x_coords, y_coords, reward_grid, log_to_wandb=False, project_path=None):
    fig, ax = plt.subplots()

    # Create grid of (x, y) positions
    X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")

    # Flatten for scatter; transpose to match your old imshow orientation
    sc = ax.scatter(
        X.ravel(),
        Y.ravel(),
        c=reward_grid.T.ravel(),  # .T to match the previous imshow(reward_grid.T)
        vmin=0,
        vmax=1,
        cmap="viridis",
        marker="o",       # circles (default)
        s=120,            # size of the circles, tweak as you like
        edgecolors="none" # avoid edgecolor warnings
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Success Rate")

    ax.set_xlabel("Hole X Position (m)")
    ax.set_ylabel("Hole Y Position (m)")
    ax.set_title("Policy Success Rate Heatmap")
    ax.set_aspect("equal")

    if log_to_wandb:
        wandb.log({"reward_heatmap": wandb.Image(fig)})
        plt.close(fig)
    else:
        # Save before show so it definitely writes to disk
        if project_path is not None:
            (project_path / "log/ddpg").mkdir(parents=True, exist_ok=True)
            fig.savefig(project_path / "log/ddpg/reward_heatmap.png", bbox_inches="tight")
        plt.show()


def evaluation_policy_short(actor, device, mujoco_cfg, project_root, rl_cfg, num_episodes):
    csv_path  = project_root / mujoco_cfg["sim"]["csv_path"]
    speed_low = rl_cfg["model"]["speed_low"]
    speed_high = rl_cfg["model"]["speed_high"]

    ball_start = np.array([0.0, 0.0])

    succeses = 0
    rewards  = []

    actor.eval()
    distances_to_hole = []
    with torch.no_grad():
        for _ in range(num_episodes):
            x, y = random_hole_in_donut(0.5, 1.5)
            hole_pos = np.array([x, y])

            state = torch.tensor(np.concatenate([ball_start, hole_pos]), dtype=torch.float32, device=device).unsqueeze(0)

            a_norm = actor(state).squeeze(0).cpu().numpy()

            speed, angle_deg = get_sim_input(a_norm, speed_low, speed_high)

            ball_x, ball_y, in_hole = run_sim(angle_deg, speed, [x, y], mujoco_cfg)

            # ball_x, ball_y, in_hole = final_state_from_csv(csv_path)

            reward = compute_reward(ball_end_pos=np.array([ball_x, ball_y]), hole_pos=hole_pos, in_hole=in_hole)
            rewards.append(reward)
            succeses += int(in_hole == 1)
            distance_to_hole = np.linalg.norm(np.array([ball_x, ball_y]) - hole_pos)
            distances_to_hole.append(distance_to_hole)

    avg_distance_to_hole = np.mean(distances_to_hole)

    actor.train()
    return succeses / num_episodes, np.mean(rewards), avg_distance_to_hole


def random_hole_in_donut(r_min, r_max):
    theta = np.random.uniform(0, 2 * np.pi)
    r = np.sqrt(np.random.uniform(r_min**2, r_max**2))
    return r * np.cos(theta), r * np.sin(theta)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    sim_dir = here.parent / "simulation"
    sys.path.append(str(sim_dir))

    from run_sim_rl import run_sim

    project_root = here.parents[2]
    mujoco_config_path = project_root / "configs" / "mujoco_config.yaml"
    rl_config_path = project_root / "configs" / "rl_config.yaml"

    with open(mujoco_config_path, "r") as f:
        mujoco_cfg = yaml.safe_load(f)

    with open(rl_config_path, "r") as f:
        rl_cfg = yaml.safe_load(f)

    if rl_cfg["training"]["use_wandb"]:
        # Define which hyperparams sweeps can override
        sweep_config = {
            "actor_lr":   rl_cfg["training"]["actor_lr"],
            "critic_lr":  rl_cfg["training"]["critic_lr"],
            "noise_std":  rl_cfg["training"]["noise_std"],
            "hidden_dim": rl_cfg["model"]["hidden_dim"],
        }

        wandb.init(
            project="rl_golf",
            config={
                **sweep_config,
                "rl_config": rl_cfg,
                "mujoco_config": mujoco_cfg,
            },
        )

        # Read back from wandb.config so sweeps can override
        cfg = wandb.config
        rl_cfg["training"]["actor_lr"]   = cfg.actor_lr
        rl_cfg["training"]["critic_lr"]  = cfg.critic_lr
        rl_cfg["training"]["noise_std"]  = cfg.noise_std
        rl_cfg["model"]["hidden_dim"]    = cfg.hidden_dim

    training(rl_cfg, mujoco_cfg, project_root, continue_training=rl_cfg["training"]["continue_training"])

    training(rl_cfg, mujoco_cfg, project_root, continue_training=rl_cfg["training"]["continue_training"])


    # evaluate_policy_random(
    #     model_path=project_root / "models" / "rl" / "ddpg" / "ddpg_actor.pth",
    #     rl_cfg=rl_cfg,
    #     mujoco_cfg=mujoco_cfg,
    #     project_root=project_root,
    #     num_episodes=100,
    # )
    run_name = wandb.run.name.replace("-", "_") if rl_cfg["training"]["use_wandb"] else "local_run"
    
    # x_coords, y_coords, success_grid, reward_grid = evaluate_policy_grid(
    #     model_path=project_root / "models" / "rl" / "ddpg" / f"ddpg_actor_{run_name}.pth",
    #     rl_cfg=rl_cfg,
    #     mujoco_cfg=mujoco_cfg,
    #     project_root=project_root,
    #     r_min=0.5,
    #     r_max=1.5,
    #     shots_per_hole=1,
    # )
    
    # plot_reward_heatmap(x_coords, y_coords, reward_grid, log_to_wandb=rl_cfg["training"]["use_wandb"], project_path=here)
    # run_sim(170, 1, [x, y], mujoco_cfg)