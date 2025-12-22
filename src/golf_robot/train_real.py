import os
from pathlib import Path
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid
from reinforcement_learning.contextual_bandit import load_actor, load_critic, compute_reward, ReplayBuffer 

here    = Path(__file__).resolve().parent
project_root        = here.parents[1]
rl_config_path      = project_root / "configs" / "rl_config.yaml"


# Load in parameters from rl_config.yaml
with open(rl_config_path, "r") as f:
    rl_cfg = yaml.safe_load(f)

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

model_name = rl_cfg["training"].get("model_name", None)


# Load pretrained model

