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
from contextual_bandit2 import training
from vision.ball2hole_distance import get_ball_final_position
from vision.ball_start_position import get_ball_start_position
from planning.generate_trajectory_csv import generate_trajectory_csv
from vision.record_camera import record_from_camera
from vision.ball_at_hole_state import process_video

OPERATING_SYSTEM = "linux"  # "windows" or "linux"
CAMERA_INDEX_START = 2  # starting camera index for real robot
CAMERA_INDEX_END   = 4  # ending camera index for real robot




class HumanPrompter:
    """
    Tiny Tkinter UI for human confirmations.
    - Call methods like confirm_continue(), ask_hole_oob(), ask_yes_no()
    - Methods BLOCK until a button is pressed, but do not require terminal focus.
    """

    def __init__(self, title="Golf Human Input"):
        self._root = tk.Tk()
        self._root.title(title)
        self._root.attributes("-topmost", True)  # keep on top (optional)
        self._root.geometry("460x220")   # width x height in pixels
        self._root.resizable(False, False)


        self._frame = tk.Frame(self._root, padx=12, pady=12)
        self._frame.pack(fill="both", expand=True)

        self._label = tk.Label(self._frame, text="", justify="left", wraplength=420)
        self._label.pack(fill="x", pady=(0, 10))

        self._btn_row = tk.Frame(self._frame)
        self._btn_row.pack(fill="both", expand=True)   # allow the row to expand vertically too
        self._btn_row.grid_columnconfigure(0, weight=1)  # center container

        self._status = tk.Label(self._frame, text="", justify="center")
        self._status.pack(fill="x", pady=(8, 0))

        self._spinner_job = None
        self._spinner_base = ""
        self._spinner_i = 0


        # state for current question
        self._event = threading.Event()
        self._result = None

        # handle window close
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # start with an idle UI
        self._set_buttons([])

        # make sure it appears
        self._root.update()

    def _on_close(self):
        # Treat closing the window as "abort"
        self._result = None
        self._event.set()
        self._root.destroy()

    def _set_message(self, text: str):
        self._label.config(text=text)

    def _set_buttons(self, buttons):
        # buttons: list of (text, value)
        for w in self._btn_row.winfo_children():
            w.destroy()

        # A centered inner frame that holds the buttons
        inner = tk.Frame(self._btn_row)
        inner.grid(row=0, column=0, sticky="nsew")  # centered by parent column weight

        # Make buttons evenly sized
        btn_font = ("TkDefaultFont", 12)   # bigger text
        btn_padx = 12
        btn_pady = 10
        min_width_chars = 14              # width in "characters" for tk.Button
        min_height = 2                    # height in text rows

        # Configure columns so buttons distribute nicely
        for i in range(len(buttons)):
            inner.grid_columnconfigure(i, weight=1)

        for i, (txt, val) in enumerate(buttons):
            b = tk.Button(
                inner,
                text=txt,
                font=btn_font,
                width=min_width_chars,
                height=min_height,
                command=lambda v=val: self._choose(v),
            )
            b.grid(row=0, column=i, padx=btn_padx, pady=btn_pady, sticky="ew")

    def _choose(self, value):
        self._result = value
        self._event.set()

    def _wait(self):
        # keep UI responsive while blocking caller
        self._event.clear()
        while not self._event.is_set():
            try:
                self._root.update()
            except tk.TclError:
                # window destroyed
                break

        return self._result

    def _tick_spinner(self):
        dots = "." * (self._spinner_i % 4)  # "", ".", "..", "..."
        self._status.config(text=f"{self._spinner_base}{dots}")
        self._spinner_i += 1
        self._spinner_job = self._root.after(300, self._tick_spinner)

        
    # ---------- Public API ----------
    def busy_start(self, text="Working..."):
        """Show an animated 'spinner' text and disable buttons."""
        self._spinner_base = text
        self._spinner_i = 0

        # disable any existing buttons
        for child in self._btn_row.winfo_children():
            for b in child.winfo_children():
                if isinstance(b, tk.Button):
                    b.config(state="disabled")

        # start animation
        self._status.config(text=text)
        self._tick_spinner()
        
    def busy_stop(self):
        """Stop spinner and clear status."""
        if self._spinner_job is not None:
            try:
                self._root.after_cancel(self._spinner_job)
            except tk.TclError:
                pass
            self._spinner_job = None
        self._status.config(text="")

    def run_with_spinner(self, text, fn, *args, **kwargs):
        """
        Run a blocking function in a worker thread while showing spinner.
        Returns fn's return value, or re-raises its exception.
        """
        result = {"value": None, "err": None}
        done = threading.Event()

        def worker():
            try:
                result["value"] = fn(*args, **kwargs)
            except Exception as e:
                result["err"] = e
            finally:
                done.set()

        self.busy_start(text)
        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while not done.is_set():
            try:
                self._root.update()
            except tk.TclError:
                break
            time.sleep(0.01)

        self.busy_stop()

        if result["err"] is not None:
            raise result["err"]
        return result["value"]


    def confirm_or_exit(self, text="Press Continue to proceed."):
        ok = self.confirm_continue(text)
        if not ok:
            print("Aborted by user")
            self.close()
            sys.exit(0)
        print("Confirmed")

    def confirm_continue(self, text="Continue?"):
        self._set_message(text)
        self._set_buttons([("Continue", True), ("Abort", False)])
        res = self._wait()
        return bool(res)

    def ask_hole_oob(self, chosen_hole: int):
        self._set_message(f"Hole {chosen_hole}: what happened?")
        self._set_buttons([
            ("✓ In hole", "in_hole"),
            ("✘ Out of bounds", "oob"),
            ("/ Neither", "neither"),
        ])
        return self._wait()

    def ask_yes_no(self, question: str, default=False):
        self._set_message(question)
        self._set_buttons([("Yes", True), ("No", False)])
        res = self._wait()
        if res is None:
            return default
        return bool(res)

    def close(self):
        try:
            self._root.destroy()
        except tk.TclError:
            pass

def real_init_parameters(camera_index):
    # Ball
    bx, by = get_ball_start_position(debug=True, debug_raw=False, use_cam=True, camera_index=camera_index, operating_system=OPERATING_SYSTEM)
    ball_start_position = np.array([bx, by])  # in meters
    print(ball_start_position)
    
    # Holes
    chosen_hole = random.choice([1,2,3])
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
    prompter.confirm_or_exit("Trajectory ready. Continue to execute?")
    
    return ball_start_position, hole_position, disc_positions, chosen_hole
    

def run_real(impact_velocity, swing_angle, ball_start_position, planner = "quintic", check_rtt=False, chosen_hole=None):
    print(f"Impact velocity: {impact_velocity} m/s, swing angle: {swing_angle} deg, ball start pos: {ball_start_position} m")
   
    # Start vision recording
    stop_event = threading.Event()

    recording_thread = threading.Thread(
        target=record_from_camera,
        kwargs={
            "camera_index": CAMERA_INDEX_END,
            "stop_event": stop_event,       # <-- this is what makes it stop after training
            "keep_last_seconds": 15.0,       # <-- rolling buffer size
            "fps": 30,
            "frame_width": 1920,
            "frame_height": 1080,
            "output_folder": "data",
            "filename_prefix": "trajectory_recording",
            "show_preview": False,
            "operating_system": OPERATING_SYSTEM,
        },
        daemon=True,
    )
    recording_thread.start()

    time.sleep(2.0)  # optional warm-up
        
    
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
            check=True,
            capture_output=True,
            text=True,
        )

    result = prompter.run_with_spinner("Shooting", run_traj)

    print("Trajectory streamer output:")
    print(result.stdout)

    
    # Stop vision recording
    print("Stopping camera recording...")
    stop_event.set()
    recording_thread.join()
    print("Recording thread joined. Done.")
    
    # Measure 
    # Deafaults
    dist_at_hole = None
    speed_at_hole = None
    ball_final_position = np.array([0.0, 0.0])  # optional: safe default
    continue_training = False
    
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
    else:
        in_hole = False
        out_of_bounds = False
        
    if not out_of_bounds and not in_hole:
        on_green = prompter.ask_yes_no("Is ball on green?")
        if on_green:
           ball_final_position = get_ball_final_position(camera_index=CAMERA_INDEX_END, chosen_hole=chosen_hole, use_cam=True, debug=True, operating_system=OPERATING_SYSTEM)
        else: 
            data_dir = "data"
            prefix ="trajectory_recording"
            pattern = os.path.join(data_dir, f"{prefix}_*_last15s.avi")
            files = glob.glob(pattern)
            video_path = max(files, key=os.path.getmtime)
            
            dist_at_hole, speed_at_hole = process_video(
                video_path, chosen_hole=chosen_hole,
                real_time_show=False,   # turn off GUI if running batch
            )
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
    }

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
        except Exception:
            pass


if __name__ == "__main__":
    main()
