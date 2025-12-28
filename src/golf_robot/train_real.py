import os
from pathlib import Path
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid
import subprocess
import re
import random
import uuid
from contextual_bandit2 import training
from vision.ball2hole_distance import get_ball_final_position
from vision.ball_start_position import get_ball_start_position
from planning.generate_trajectory_csv import generate_trajectory_csv

OPERATING_SYSTEM = "linux"  # "windows" or "linux"

def confirm_continue():
    # Confirm
    key = input("Press y to continue: ").lower()
    if key == "y":
        print("Confirmed")
    else:
        print("Aborted by user")
        sys.exit(0)


def real_init_parameters(camera_index):
    # Ball
    bx, by = get_ball_start_position(debug=True, debug_raw=False, use_cam=True, camera_index=camera_index, operating_system=OPERATING_SYSTEM)
    ball_start_position = np.array([bx, by])  # in meters
    print(ball_start_position)
    
    # Holes
    # chosen_hole = random.choice([1,2,3])
    chosen_hole = 3  # for testing purposes
    here = Path(__file__).resolve().parent
    config_dir = here.parents[1] / "configs"
    with open(config_dir / "hole_config.yaml", "r") as f:
        hole_positions = yaml.safe_load(f)
    # hole_number = 1  # choose hole number
    hx = hole_positions[chosen_hole]["x"]
    hy = hole_positions[chosen_hole]["y"]
    hole_position = np.array([hx, hy]) # choose first hole for now
    print("hole_position: ", hole_position)
    
    # Discs:
    disc_positions = [] # not used for now in real training
    
    # Confirm
    confirm_continue()
    
    return ball_start_position, hole_position, disc_positions, chosen_hole
    

def run_real(impact_velocity, swing_angle, ball_start_position, planner = "quintic", check_rtt=False, chosen_hole=None):
    print(f"Impact velocity: {impact_velocity} m/s, swing angle: {swing_angle} deg, ball start pos: {ball_start_position} m")
    if planner == "quintic":
        generate_trajectory_csv(impact_velocity, swing_angle, ball_start_position[0], ball_start_position[1])
    if planner == "linear":
        sys.exit("Linear planner not implemented for real robot yet.")
    
    if OPERATING_SYSTEM == "windows":
        if check_rtt:
            out = subprocess.check_output(
                ["wsl", "ping", "-c", "3", "192.38.66.227"],
                text=True
            )

            rtts = [
                float(m.group(1))
                for m in re.finditer(r'time=([\d.]+)\s*ms', out)
            ]

            print("RTTs (ms):", rtts)
            print("avg:", sum(rtts)/len(rtts))
    elif OPERATING_SYSTEM == "linux":
        if check_rtt:
            out = subprocess.check_output(
                ["ping", "-c", "3", "192.38.66.227"], text=True
            )

            rtts = [
                float(m.group(1))
                for m in re.finditer(r'time=([\d.]+)\s*ms', out)
            ]

            print("RTTs (ms):", rtts)
            print("avg:", sum(rtts)/len(rtts))
    
    confirm_continue()
    
    here = Path(__file__).resolve().parent
    traj_exe = here / "communication/traj_streamer"
    print(traj_exe)
    win_path = Path(__file__).resolve().parent / "communication" / "traj_streamer"

    if OPERATING_SYSTEM == "windows":
        wsl_path = subprocess.check_output(
            ["wsl", "wslpath", "-a", str(win_path)],
            text=True
        ).strip()

        result = subprocess.run(
            ["wsl", wsl_path],
            check=True, capture_output=True, text=True
        )
    elif OPERATING_SYSTEM == "linux":
        try:
            result = subprocess.run(
                [str(traj_exe)],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            print("traj_streamer failed with return code:", e.returncode)
            print("---- stdout ----")
            print(e.stdout)
            print("---- stderr ----")
            print(e.stderr)
            raise
    print("Trajectory streamer output:")
    print(result.stdout)
    
    
    # Measure 
    key = input(f"Is ball in hole {chosen_hole}? (Press y) - Is ball out of bounds (Press o)").lower()
    if key == "y":
        print("Ball in hole confirmed")
        in_hole = True
        out_of_bounds = False
    elif key == "o":
        print("Ball out of bounds confirmed")
        in_hole = False
        out_of_bounds = True
    else:
        in_hole = False
        out_of_bounds = False
    
    if not out_of_bounds:
        ball_final_position = get_ball_final_position(camera_index=2, chosen_hole=1, use_cam=True, debug=True, operating_system=OPERATING_SYSTEM)
    
    if out_of_bounds:
        ball_final_position = np.array([0, 0])

    key = input("Use shot for training? ").lower()
    if key != "y":
        print("Shot discarded by user")
        sys.exit(0)

    return ball_final_position[0], ball_final_position[1], in_hole, out_of_bounds
    

#ball_start_position, hole_position, disc_positions = real_init_parameters(camera_index=0)
#run_real(impact_velocity=1.0, swing_angle=0.0, ball_start_position=ball_start_position, planner="quintic", check_rtt=True)

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
    
tmp_name     = f"golf_world_tmp_{os.getpid()}_{uuid.uuid4().hex}.xml"
    
training(rl_cfg=rl_cfg, mujoco_cfg=mujoco_cfg, project_root=project_root, continue_training=rl_cfg["training"]["continue_training"], input_func=real_init_parameters, env_step=run_real, env_type="real", tmp_name=tmp_name)


