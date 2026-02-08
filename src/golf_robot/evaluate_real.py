import os
from pathlib import Path
import sys
import yaml
import numpy as np
import uuid
import subprocess
import re
import random
import threading
import time
import socket
import glob
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
import cv2
#from contextual_bandit2 import training
# from SAC_bandit import training
from vision.ball2hole_distance import get_ball_final_position
from vision.ball_start_position import get_ball_start_position
from planning.generate_trajectory_csv import generate_trajectory_csv
from planning_linear.linear_planner import send_swing
#from vision.record_camera import record_from_camera
from vision.ball_at_hole_state import process_video
from gui.gui import *
from rl_common import *
from hand_designed_agent import hand_tuned_policy
import torch

OPERATING_SYSTEM = "linux"  # "windows" or "linux"
CAMERA_INDEX_START = 4  # starting camera index for real robot
CAMERA_INDEX_END   = 4  # ending camera index for real robot
actor_name = "ucb"  # "hand_tuned_policy", "SAC_bandit", "thompson_bandit", "contextual_bandit2", "ucb"
planner = "quintic"  # "quintic" or "linear"

# CAMERA_END = r'@device_pnp_\\?\usb#vid_046d&pid_08e5&mi_00#7&23aa88cc&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global'
#CAMERA_END = r'@device_pnp_\\?\usb#vid_046d&pid_08e5&mi_00#8&2e31d80&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global'
#CAMERA_END = r'@device_pnp_\\?\usb#vid_046d&pid_08e5&mi_00#8&2e31d80&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global'
END_POS = [-2.47, -2.38, -1.55, 1.66, 0.49, -0.26]
LOG_SHOTS = True

def real_init_parameters(camera_index, chosen_hole=None):
    # Ball
    bx, by, dbg = get_ball_start_position(debug=True, return_debug_image=True, debug_raw=False, use_cam=True, camera_index=camera_index, operating_system=OPERATING_SYSTEM)
    ball_start_position = np.array([bx, by])  # in meters
    print(f"ball_start_position: {ball_start_position}")
    
    if dbg is not None:
        prompter.show_image(dbg, title="Ball start detection")
    
    # Holes
    if chosen_hole is None:
        # chosen_hole = random.choice([1,2,3])
        chosen_hole = 2
    # chosen_hole = 1  # for testing purposes
    here = Path(__file__).resolve().parent
    config_dir = here.parents[1] / "configs"
    with open(config_dir / "hole_config.yaml", "r") as f:
        hole_positions = yaml.safe_load(f)
    # hole_number = 1  # choose hole number
    hx = hole_positions[chosen_hole]["x"]
    hy = hole_positions[chosen_hole]["y"]
    hole_position = np.array([hx, hy]) # choose first hole for now
    print(f"Chosen hole {chosen_hole} at position:", hole_position)
    
    # Discs:
    disc_positions = [] # not used for now in real training
    
    # Confirm
    prompter.show_hole(chosen_hole) # Show big hole number
    prompter.confirm_or_exit("Trajectory ready. Continue to execute?")
    prompter.clear_hole() # clear headline
    
    return ball_start_position, hole_position, disc_positions, chosen_hole
    

def run_real(impact_velocity, swing_angle, ball_start_position, planner = "quintic", check_rtt=False, chosen_hole=None):
    print(f"Impact velocity: {impact_velocity} m/s, swing angle: {swing_angle} deg, ball start pos: {ball_start_position} m")
    
    if planner == "quintic":
        results = generate_trajectory_csv(impact_velocity, swing_angle, ball_start_position[0], ball_start_position[1])
        if results is None:
            print("Trajectory not feasible, aborting.")
            sys.exit(0)
            
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
        
        
        prompter.confirm_or_exit("Ready to execute trajectory. Continue?")

        here = Path(__file__).resolve().parent
        traj_exe = here / "communication" / "traj_streamer"

        if OPERATING_SYSTEM == "linux":
            traj_cmd = [str(traj_exe)]

        elif OPERATING_SYSTEM == "windows":
            win_path = here / "communication" / "traj_streamer"
            wsl_path = subprocess.check_output(
                ["wsl", "wslpath", "-a", str(win_path)],
                text=True
            ).strip()
            traj_cmd = ["wsl", wsl_path]

        else:
            raise RuntimeError(f"Unsupported OS: {OPERATING_SYSTEM}")
        
        def run_traj():
            return subprocess.run(
                traj_cmd,
                check=False,
                capture_output=False,
                text=True,
            )

        result = prompter.run_with_spinner("Shooting", run_traj)

        print("Trajectory streamer output:")
        print(result.stdout)


    if planner == "linear":
        HOST = "192.38.66.227"   # UR10
        #PORT_logger = 30003
        PORT_cmd = 30002
        #time_sleep = 10.0

        # 1) Set up your logger on its own socket/connection
        # logger = UR10Logger(HOST, port=PORT_logger, log_folder="log")
        # logger.connect()
        # logger.start_logging()

        # 2) Open a separate socket for sending the program
        traj_cmd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        traj_cmd.connect((HOST, PORT_cmd))

        print("Sending swing...")
        x_ball_origo=-0.62823
        ball_radius=0.021335
        offset = 0.25 # 0.33 max
        z_buffer = 0.01
        x_start=x_ball_origo+ball_radius-offset   + ball_start_position[0]
        x_end=x_ball_origo+ball_radius+offset   + ball_start_position[0]
        y_ball_origo=-0.57480 #-0.546
        y_ball = y_ball_origo + ball_start_position[1]
        z_ball=0.15512+z_buffer #-0.006

        swing_meta = send_swing(traj_cmd, x_start=x_start, x_end=x_end,
            y_ball=y_ball, z_ball=z_ball,#0.01+Z_PALLET, #-0.040 old
            path_angle_deg=swing_angle, attack_angle_deg=0.0,
            vel=impact_velocity, acc=5.5)

        # swing_meta = prompter.run_with_spinner("Shooting", 
        #     send_swing(traj_cmd, x_start=x_start, x_end=x_end,
        #     y_ball=y_ball, z_ball=z_ball,#0.01+Z_PALLET, #-0.040 old
        #     path_angle_deg=swing_angle, attack_angle_deg=0.0,
        #     vel=impact_velocity, acc=5.0))
        

        # 3) Let the swing run and the logger collect a bit extra
        # time.sleep(time_sleep)   # 8.0 adjust to cover your full motion

        # 4) Clean up
        try:
            traj_cmd.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        traj_cmd.close()
        
        

    if LOG_SHOTS:

        time.sleep(2.0)
        # Stop vision recording
        print("Saving recording from replay buffer")
        obs_replay_save()

        print("Recording thread joined. Done.")

        data_dir = "data/OBS_saved_replay_buffer"
        prefix ="Replay"
        pattern = os.path.join(data_dir, f"{prefix}*")
        # print(pattern)
        files = glob.glob(pattern)
        # print(files)
        video_path = max(files, key=os.path.getmtime)
        
        print("Recorded video path:", video_path)

    ur_movej(
        robot_ip="192.38.66.227",
        q=END_POS,
    )
    print("Return to home position command sent.")

    if LOG_SHOTS:
        # Measure 
        # Deafaults
        dist_at_hole = None
        speed_at_hole = None
        ball_final_position = np.array([0.0, 0.0])  # optional: safe default
        continue_evaluation = False
        on_green = False
        
        state = prompter.ask_hole_oob(chosen_hole)

        wrong_hole = False

        if state == "in_hole":
            print("Ball in hole confirmed")
            in_hole = True
            out_of_bounds = False
            dist_at_hole, speed_at_hole = None, None
            here = Path(__file__).resolve().parent
            config_dir = here.parents[1] / "configs"
            with open(config_dir / "hole_config.yaml", "r") as f:
                hole_positions = yaml.safe_load(f)
            # hole_number = 1  # choose hole number
            hx = hole_positions[chosen_hole]["x"]
            hy = hole_positions[chosen_hole]["y"]
            ball_final_position = np.array([hx, hy]) # choose first hole for now

        elif state == "oob":
            print("Ball out of bounds confirmed")
            in_hole = False
            out_of_bounds = True

        elif state == "wrong_hole":
            chosen_hole = prompter.ask_which_hole()
            print(f"User indicated ball went to wrong hole, new hole is {chosen_hole}")
            in_hole = True
            out_of_bounds = False
            wrong_hole = chosen_hole

        else:
            in_hole = False
            out_of_bounds = False
            
        if not out_of_bounds and not in_hole:
            on_green = prompter.ask_yes_no("Is ball on green?")
            if on_green:
                obs_screenshot()
                ball_final_position = get_ball_final_position(camera_index=CAMERA_INDEX_END, chosen_hole=chosen_hole, use_cam=False, debug=True, operating_system=OPERATING_SYSTEM)
            
            else: 
                compute_all_holes = False
                traj_3_worked = False
                if compute_all_holes:
                    dist_at_hole = []
                    speed_at_hole = []
                    
                    for i in range(1,4):
                        dist_at_hole_tmp, speed_at_hole_tmp, xs, ys, hole_xo, hole_yo, bx, by = process_video(
                            video_path, chosen_hole=i, real_time_show=False, compute_all_holes=False
                        )
                        dist_at_hole.append(dist_at_hole_tmp)
                        speed_at_hole.append(speed_at_hole_tmp)
                    print("Distance at hole list:", dist_at_hole)
                    print("Speed at hole list:", speed_at_hole)
                    if dist_at_hole[2] is not None:
                        traj_3_worked = True

                else:
                    dist_at_hole, speed_at_hole, xs, ys, hole_xo, hole_yo, bx, by = process_video(
                        video_path, chosen_hole=chosen_hole, real_time_show=False, compute_all_holes=False
                    )
                if dist_at_hole is not None and traj_3_worked:
                    prompter.show_trajectory_plot(xs, ys, hole_xo, hole_yo, bx, by)

                ball_final_position = np.array([0, 0]) # dummy value

        if out_of_bounds:
            ball_final_position = np.array([0, 0]) # dummy value
            dist_at_hole = None
            speed_at_hole = None
   
        used_for_training = prompter.ask_yes_no("Log shot for evaluation?")
        if not used_for_training:
            print("Shot discarded by user")


        continue_evaluation = prompter.ask_yes_no("Continue evaluation?")
        if not continue_evaluation:
            print("Training aborted by user")
        
        meta = {
            "dist_at_hole": dist_at_hole,
            "speed_at_hole": speed_at_hole,
            "used_for_training": used_for_training,
            "continue_evaluation": continue_evaluation,
            "out_of_bounds": out_of_bounds,
            "wrong_hole": wrong_hole,
            "on_green": on_green,
        }
    
    else:
        continue_evaluation = prompter.ask_yes_no("Continue evaluation?")
        if not continue_evaluation:
            print("Evaluation aborted by user")
        ball_final_position = np.array([0.0, 0.0])  # safe default
        in_hole = False
        meta = {
            "dist_at_hole": None,
            "speed_at_hole": None,
            "used_for_training": False,
            "continue_evaluation": continue_evaluation,
            "out_of_bounds": False,
            "wrong_hole": False,
            "on_green": False,
        }

    print("meta:", meta)
    return ball_final_position[0], ball_final_position[1], in_hole, meta


def load_correct_actor(actor_name, rl_cfg):
    # actor, device = load_actor(rl_cfg["training"]["model_name"], rl_cfg)
    project_root = Path(__file__).parents[2]
    if actor_name == "hand_tuned_policy":
        print("Using hand-tuned policy")
        return hand_tuned_policy, None

    elif actor_name == "SAC_bandit":
        from SAC_bandit import load_actor
        print("Using SAC bandit policy")
        
    elif actor_name == "thompson_bandit":
        from thompson_bandit import find_latest_doublecritic_checkpoint, load_doublecritics, MeanPlannerActor
        device = rl_cfg["training"].get("device", "cpu")
        model_dir = project_root / "models" / "rl" / "dqn-bts"
        print("Model dir:", model_dir)

        c1_h0, tag_stem = find_latest_doublecritic_checkpoint(model_dir, prefix=None)
        print("Using Thompson bandit policy")
        print("Continuing training from latest double-critic checkpoints:")
        print(f"  Critic1_0 : {c1_h0.name}")
        critics1, critics2, loaded_tag = load_doublecritics(model_dir, rl_cfg, tag_stem, device)
        return MeanPlannerActor(critics1, critics2, rl_cfg, device).to(device), None


    elif actor_name == "ucb":
        # IMPORTANT: import from the file that actually defines MeanPlannerActor + checkpoint helpers
        from ucb_bandit import find_latest_critic_checkpoint, load_critics, MeanPlannerActor

        print("Using UCB bandit policy (MeanPlannerActor)")

        # Prefer torch.device, not string
        dev_str = rl_cfg["training"].get("device", "cpu")
        device = torch.device(dev_str)

        model_dir = project_root / "models" / "rl" / "ucb"
        print("Model dir:", model_dir)

        # find_latest_critic_checkpoint returns (path_to_head0, tag_stem)
        _, tag_stem = find_latest_critic_checkpoint(model_dir, prefix=None)

        # load_critics returns (critics_list, tag)
        critics, _ = load_critics(model_dir, rl_cfg, tag_stem, device)

        actor = MeanPlannerActor(critics=critics, rl_cfg=rl_cfg, device=device).to(device)
        actor.eval()
        return actor, device
    
    
    elif actor_name == "contextual_bandit2":
        from contextual_bandit2 import load_actor
        print("Using Contextual bandit v2 policy")

    return load_actor(actor_name, rl_cfg)

def main():
    global prompter

    prompter = HumanPrompter()

    here = Path(__file__).resolve().parent

    if LOG_SHOTS:
        obs_replay_start()
        project_root = here.parents[1]
        episode_log_path = project_root / "log" / "real_episodes_eval" / "episode_logger_eval_ucb.jsonl"
        print("Episode log path:", episode_log_path)
        episode_logger = EpisodeLoggerJsonl(episode_log_path)

    else:
        episode_logger = None

    project_root       = here.parents[1]
    mujoco_config_path = project_root / "configs" / "mujoco_config.yaml"
    rl_config_path     = project_root / "configs" / "rl_config.yaml"

    with open(mujoco_config_path, "r") as f:
        mujoco_cfg = yaml.safe_load(f)

    with open(rl_config_path, "r") as f:
        rl_cfg = yaml.safe_load(f)

    actor, device = load_correct_actor(actor_name, rl_cfg)
    device = "cpu"
    # actor = hand_tuned_policy
    
    try:
        if actor_name == "hand_tuned_policy":
            evaluation_policy_hand_tuned(actor, mujoco_cfg, rl_cfg, num_episodes=30, max_num_discs=0, env_step=run_real, env_type="real", input_func=real_init_parameters, planner=planner, camera_index=CAMERA_INDEX_START)
        else:
            evaluation_policy_short(actor, device, mujoco_cfg, rl_cfg, num_episodes=16, max_num_discs=0, env_step=run_real, env_type="real", input_func=real_init_parameters, big_episode_logger=episode_logger)
    finally:
        # Always close the Tk window (even on exceptions / sys.exit)
        if LOG_SHOTS:
            try:
                prompter.close()
                obs_replay_stop()
            except Exception:
                pass

if __name__ == "__main__":
    main()