# infer_sac_action.py
from pathlib import Path
import yaml
import numpy as np
import torch

from rl_common import (
    SACActor,
    encode_state_with_discs,
    scale_state_vec,
    augment_state_features,
    scale_aug_features,
    get_input_parameters,
)

def _load_sac_actor(model_path: Path, rl_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim  = rl_cfg["model"]["state_dim"]
    action_dim = rl_cfg["model"]["action_dim"]
    hidden_dim = rl_cfg["model"]["hidden_dim"]

    actor = SACActor(state_dim, action_dim, hidden_dim).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()
    return actor, device

def find_latest_sac_actor(model_dir: Path):
    actor_files = list(model_dir.glob("sac_actor_*.pth"))
    if not actor_files:
        raise FileNotFoundError(f"No sac_actor_*.pth found in {model_dir}")
    return max(actor_files, key=lambda f: f.stat().st_mtime)

def build_state_norm(ball_start_obs, hole_pos_obs, disc_positions, rl_cfg):
    # must match training
    MAX_DISCS_FEATS = 5
    state_vec = encode_state_with_discs(
        ball_start_obs, hole_pos_obs, disc_positions, max_num_discs=MAX_DISCS_FEATS
    )
    state_norm = scale_state_vec(state_vec)  # 19 dims in your comment

    # optional augmented features (when rl_cfg["model"]["state_dim"] == 37)
    if rl_cfg["model"]["state_dim"] == 37:
        aug = augment_state_features(state_vec, max_num_discs=MAX_DISCS_FEATS, dist_clip=5.0)
        aug_norm = scale_aug_features(aug, max_num_discs=MAX_DISCS_FEATS, dist_clip=5.0)
        state_norm = np.concatenate([state_norm, aug_norm], axis=0)

    return state_norm

@torch.no_grad()
def actor_forward_once(actor, device, state_norm, rl_cfg):
    s = torch.tensor(state_norm, dtype=torch.float32, device=device).unsqueeze(0)  # [1, state_dim]

    # This matches your training usage:
    # a_noisy, _, _ = actor.sample(...)
    a_norm, logp, mean = actor.sample(s)  # shapes typically: [1,2], [1,1], [1,2] (depends on your impl)
    a_norm = a_norm.squeeze(0).cpu().numpy()

    mean = mean.squeeze(0).cpu().numpy()
    
    speed_low  = rl_cfg["model"]["speed_low"]
    speed_high = rl_cfg["model"]["speed_high"]
    angle_low  = rl_cfg["model"]["angle_low"]
    angle_high = rl_cfg["model"]["angle_high"]

    speed, angle_deg = get_input_parameters(mean, speed_low, speed_high, angle_low, angle_high)
    return a_norm, speed, angle_deg

def main():
    # adjust these paths if you run from a different working directory
    project_root = Path(__file__).resolve().parents[2]  # change if needed
    rl_config_path = project_root / "configs" / "rl_config.yaml"
    model_dir = project_root / "models" / "rl" / "sac"

    with open(rl_config_path, "r") as f:
        rl_cfg = yaml.safe_load(f)

    actor_path = find_latest_sac_actor(model_dir)
    actor, device = _load_sac_actor(actor_path, rl_cfg)

    ball_start_obs = np.array([0.0, 0.0], dtype=np.float32)
    hole_pos_obs   = np.array([3.675, -0.275], dtype=np.float32)


    disc_positions = []  # IMPORTANT: exactly as training expects

    state_norm = build_state_norm(ball_start_obs, hole_pos_obs, disc_positions, rl_cfg)
    a_norm, speed, angle_deg = actor_forward_once(actor, device, state_norm, rl_cfg)

    print(f"Loaded actor: {actor_path.name}")
    # print(f"state_dim = {rl_cfg['model']['state_dim']}, action_dim = {rl_cfg['model']['action_dim']}")
    # print(f"a_norm (in [-1,1]^2): {a_norm}")
    print(f"ball_start_obs: {ball_start_obs}")
    print(f"hole_pos_obs: {hole_pos_obs}")
    print(f"speed: {speed:.4f}")
    print(f"angle_deg: {angle_deg:.4f}")

    hole_pos_obs   = np.array([3.85, 0.0], dtype=np.float32)

    state_norm = build_state_norm(ball_start_obs, hole_pos_obs, disc_positions, rl_cfg)
    a_norm, speed, angle_deg = actor_forward_once(actor, device, state_norm, rl_cfg)

    print(f"Loaded actor: {actor_path.name}")
    # print(f"state_dim = {rl_cfg['model']['state_dim']}, action_dim = {rl_cfg['model']['action_dim']}")
    # print(f"a_norm (in [-1,1]^2): {a_norm}")
    print(f"ball_start_obs: {ball_start_obs}")
    print(f"hole_pos_obs: {hole_pos_obs}")
    print(f"speed: {speed:.4f}")
    print(f"angle_deg: {angle_deg:.4f}")

    hole_pos_obs   = np.array([3.85, 0.33], dtype=np.float32)
    state_norm = build_state_norm(ball_start_obs, hole_pos_obs, disc_positions, rl_cfg)
    a_norm, speed, angle_deg = actor_forward_once(actor, device, state_norm, rl_cfg)
    print(f"Loaded actor: {actor_path.name}")
    # print(f"state_dim = {rl_cfg['model']['state_dim']}, action_dim = {rl_cfg['model']['action_dim']}")
    # print(f"a_norm (in [-1,1]^2): {a_norm}")
    print(f"ball_start_obs: {ball_start_obs}")
    print(f"hole_pos_obs: {hole_pos_obs}")
    print(f"speed: {speed:.4f}")
    print(f"angle_deg: {angle_deg:.4f}")

if __name__ == "__main__":
    main()
