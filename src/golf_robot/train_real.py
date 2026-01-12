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
import glob
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
import cv2
#from contextual_bandit2 import training
from SAC_bandit import training
from vision.ball2hole_distance import get_ball_final_position
from vision.ball_start_position import get_ball_start_position
from planning.generate_trajectory_csv import generate_trajectory_csv
#from vision.record_camera import record_from_camera
from vision.ball_at_hole_state import process_video
from gui.gui import *
#from vision.ffmpeg_record import start_ffmpeg_record_windows, stop_ffmpeg_record, get_video_frame_count, trim_last_seconds_reencode


OPERATING_SYSTEM = "linux"  # "windows" or "linux"
CAMERA_INDEX_START = 4  # starting camera index for real robot
CAMERA_INDEX_END   = 2  # ending camera index for real robot
# CAMERA_END = r'@device_pnp_\\?\usb#vid_046d&pid_08e5&mi_00#7&23aa88cc&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global'
#CAMERA_END = r'@device_pnp_\\?\usb#vid_046d&pid_08e5&mi_00#8&2e31d80&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global'
#CAMERA_END = r'@device_pnp_\\?\usb#vid_046d&pid_08e5&mi_00#8&2e31d80&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global'
END_POS = [-2.47, -2.38, -1.55, 1.66, 0.49, -0.26]


def real_init_parameters(camera_index, chosen_hole=None):
    # Ball
    bx, by, dbg = get_ball_start_position(debug=True, return_debug_image=True, debug_raw=False, use_cam=True, camera_index=camera_index, operating_system=OPERATING_SYSTEM)
    ball_start_position = np.array([bx, by])  # in meters
    print(f"ball_start_position: {ball_start_position}")
    
    if dbg is not None:
        prompter.show_image(dbg, title="Ball start detection")
    
    # Holes
    if chosen_hole is None:
        chosen_hole = random.choice([1,2,3])
        # chosen_hole = 1
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
    

    time.sleep(2.0)
    # Stop vision recording
    print("Saving recording from replay buffer")
    obs_replay_save()
    # stop_event.set()
    # recording_thread.join()
    #stop_ffmpeg_record(rec)
    
    
    print("Recording thread joined. Done.")

    data_dir = "data/OBS_saved_replay_buffer"
    prefix ="Replay"
    pattern = os.path.join(data_dir, f"{prefix}*")
    # print(pattern)
    files = glob.glob(pattern)
    # print(files)
    video_path = max(files, key=os.path.getmtime)
    
    print("Recorded video path:", video_path)
    # frames_captured = get_video_frame_count(video_path)
    # expected_frames = int(15 * 30)
    # print(frames_captured)

    
    # print(f"Captured {frames_captured}/{expected_frames} frames")
    # print(f"Drop ratio: {(1 - frames_captured/expected_frames)*100:.1f}%")
    
    # Start thread for return to home position
    ur_movej(
        robot_ip="192.38.66.227",
        q=END_POS,
    )
    print("Return to home position command sent.")

    # Measure 
    # Deafaults
    dist_at_hole = None
    speed_at_hole = None
    ball_final_position = np.array([0.0, 0.0])  # optional: safe default
    continue_training = False
    on_green = False
    
    wrong_hole = None
    state = prompter.ask_hole_oob(chosen_hole)


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
        #   ball_final_position = get_ball_final_position(camera_index=CAMERA_INDEX_END, chosen_hole=chosen_hole, use_cam=True, debug=True, operating_system=OPERATING_SYSTEM)
            ball_final_position = get_ball_final_position(camera_index=CAMERA_INDEX_END, chosen_hole=chosen_hole, use_cam=False, debug=True, operating_system=OPERATING_SYSTEM)
        else: 
            # data_dir = "data"
            # prefix ="trajectory_recording"
            # pattern = os.path.join(data_dir, f"{prefix}_*_last15s.avi")
            # files = glob.glob(pattern)
            # video_path = max(files, key=os.path.getmtime)
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
            #dist_at_hole, speed_at_hole = process_video(
            #    video_path, chosen_hole=chosen_hole,
            #    real_time_show=False,   # turn off video if running batch
            #)
            ball_final_position = np.array([0, 0]) # dummy value

    
    if out_of_bounds:
        ball_final_position = np.array([0, 0]) # dummy value
        dist_at_hole = None
        speed_at_hole = None

    
    used_for_training = prompter.ask_yes_no("Use shot for training?")
    if not used_for_training:
        print("Shot discarded by user")


    continue_training = prompter.ask_yes_no("Continue training?")
    if not continue_training:
        print("Training aborted by user")
    
    meta = {
        "dist_at_hole": dist_at_hole,
        "speed_at_hole": speed_at_hole,
        "used_for_training": used_for_training,
        "continue_training": continue_training,
        "out_of_bounds": out_of_bounds,
        "wrong_hole": wrong_hole,
        "on_green": on_green,
    }
    print("meta:", meta)
    return ball_final_position[0], ball_final_position[1], in_hole, meta
    

#ball_start_position, hole_position, disc_positions = real_init_parameters(camera_index=0)
#run_real(impact_velocity=1.0, swing_angle=0.0, ball_start_position=ball_start_position, planner="quintic", check_rtt=True)

def main():
    global prompter

    # Create UI once (Tk must live in main thread)
    prompter = HumanPrompter(title="Golf Human Input")

    here = Path(__file__).resolve().parent
    sim_dir = here / "simulation"
    sys.path.append(str(sim_dir))

    obs_replay_start()

    project_root       = here.parents[1]
    mujoco_config_path = project_root / "configs" / "mujoco_config.yaml"
    rl_config_path     = project_root / "configs" / "rl_config.yaml"

    with open(mujoco_config_path, "r") as f:
        mujoco_cfg = yaml.safe_load(f)

    with open(rl_config_path, "r") as f:
        rl_cfg = yaml.safe_load(f)

    tmp_name = f"golf_world_tmp_{os.getpid()}_{uuid.uuid4().hex}"

    try:
        training(
            rl_cfg=rl_cfg,
            mujoco_cfg=mujoco_cfg,
            project_root=project_root,
            continue_training=rl_cfg["training"]["continue_training"],
            input_func=real_init_parameters,
            env_step=run_real,
            env_type="real",
            tmp_name=tmp_name,
            camera_index_start=CAMERA_INDEX_START,
        )
    finally:
        # Always close the Tk window (even on exceptions / sys.exit)
        try:
            prompter.close()
            obs_replay_stop()
        except Exception:
            pass


if __name__ == "__main__":
    # Test
    #global prompter
    #prompter = HumanPrompter(title="Golf Human Input")
    #chosen_hole = 3
    #prompter.show_hole(chosen_hole) # Show big hole number
    #prompter.confirm_or_exit("Trajectory ready. Continue to execute?")
    #prompter.clear_hole() # clear headline
    #prompter.close()
    main()
