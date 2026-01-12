import sys
from pathlib import Path
import subprocess
import glob
import threading
import os
import time
import tkinter as tk
import socket
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


# ------ OBS ------ #
OBS_ENV = {
    **os.environ,
    "OBSWS_HOST": "127.0.0.1",
    "OBSWS_PORT": "4455",
    #"OBSWS_PASSWORD": "jeff123",
    #"OBSWS_LOG_LEVEL": "DEBUG",
}

def obs_replay_start():
    subprocess.run(["obsws-cli", "replaybuffer", "start"], check=True, env=OBS_ENV)

def obs_replay_save():
    # Saves last N seconds (as configured in OBS: 15s)
    # subprocess.run(["obsws-cli", "replaybuffer", "save"], check=True, env=OBS_ENV)
    # time.sleep(3)  # finalization cushion

    subprocess.run(
        ["obsws-cli", "hotkey", "trigger", "ReplayBuffer.Save"],
        check=True,
        env=OBS_ENV,
    )
    time.sleep(3)

def obs_screenshot():
    print("Taking OBS screenshot")
    here = Path(__file__).resolve().parents[2]
    
    ball_path = f"{glob.glob(os.path.join('data', 'OBS_ball_on_green'))}"
    ball_path = os.path.join(here, "data", "OBS_ball_on_green", "ball_on_green.png")
    print(ball_path)
    subprocess.run(
        ["obsws-cli", "screenshot", "save", "camera_end", ball_path],
        check=True,
        env=OBS_ENV,
    )
    time.sleep(3)

def obs_replay_stop():
    subprocess.run(["obsws-cli", "replaybuffer", "stop"], check=True, env=OBS_ENV)
# ----- OBS ----- #


# For return to home position
def ur_movej(
    robot_ip,
    q,
    a=1.2,
    v=0.25,
):
    """
    Send a blocking movej command via URScript over port 30002.
    """
    assert len(q) == 6

    script = (
        "def py_move():\n"
        f"  movej([{','.join(f'{x:.6f}' for x in q)}], a={a}, v={v})\n"
        "end\n"
        "py_move()\n"
    )

    with socket.create_connection((robot_ip, 30002), timeout=5) as s:
        s.sendall(script.encode("utf-8"))


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
        self._root.geometry("860x640")   # width x height in pixels
        self._root.resizable(False, False)

        self._frame = tk.Frame(self._root, padx=12, pady=12)
        self._frame.pack(fill="both", expand=True)
        
        self._headline = tk.Label(
            self._frame,
            text="",
            font=("TkDefaultFont", 28, "bold"),
            justify="center"
        )
        self._headline.pack(fill="x", pady=(0, 8))

        self._label = tk.Label(self._frame, text="", justify="left", wraplength=420)
        self._label.pack(fill="x", pady=(0, 10))

        self._content = tk.Frame(self._frame)
        self._content.pack(fill="both", expand=True)
        self._content.grid_columnconfigure(0, weight=1)
        #self._btn_row = tk.Frame(self._frame)
        #self._btn_row.pack(fill="both", expand=True)   # allow the row to expand vertically too
        #self._btn_row.grid_columnconfigure(0, weight=1)  # center container

        self._status = tk.Label(self._frame, text="", justify="center")
        self._status.pack(fill="x", pady=(8, 0))

        self._spinner_job = None
        self._spinner_base = ""
        self._spinner_i = 0

        # If you want a big center label while busy:
        self._busy_label = None

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
        if self._busy_label is not None:
            self._busy_label.config(text=f"{self._spinner_base}{dots}")
        self._spinner_i += 1
        self._spinner_job = self._root.after(300, self._tick_spinner)

    def _clear_content(self):
        for w in self._content.winfo_children():
            w.destroy()
    
    def _set_buttons(self, buttons):
        self._clear_content()

        inner = tk.Frame(self._content)
        inner.grid(row=0, column=0, sticky="nsew")

        btn_font = ("TkDefaultFont", 12)
        btn_padx = 12
        btn_pady = 10
        min_width_chars = 14
        min_height = 2

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

    
    # ---------- Public API ----------
    def show_image(self, bgr_img, title=""):
        """
        Show a BGR OpenCV image inside the main GUI, with Continue button at the top.
        Blocks until Continue is pressed.
        """
        self._clear_content()

        # 2-row grid: row0 buttons, row1 image
        self._content.grid_rowconfigure(0, weight=0)
        self._content.grid_rowconfigure(1, weight=1)
        self._content.grid_columnconfigure(0, weight=1)

        # --- Top controls (same place as other buttons) ---
        top = tk.Frame(self._content)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top.grid_columnconfigure(0, weight=1)

        if title:
            lbl = tk.Label(top, text=title, font=("TkDefaultFont", 14, "bold"))
            lbl.pack(side="left")

        btn = tk.Button(
            top,
            text="Continue",
            font=("TkDefaultFont", 12),
            width=14,
            height=2,
            command=lambda: self._choose(True),
        )
        btn.pack(side="right")

        # --- Image area ---
        img_frame = tk.Frame(self._content)
        img_frame.grid(row=1, column=0, sticky="nsew")
        img_frame.grid_rowconfigure(0, weight=1)
        img_frame.grid_columnconfigure(0, weight=1)

        # Convert BGR -> RGB -> PIL
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # Fit image to available area (keep aspect)
        self._root.update_idletasks()
        max_w = img_frame.winfo_width() or 820
        max_h = img_frame.winfo_height() or 560

        pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil)

        img_label = tk.Label(img_frame, image=tk_img)
        img_label.image = tk_img  # IMPORTANT: keep reference
        img_label.grid(row=0, column=0, sticky="nsew")

        self._wait()

    
    # Tkinter plot for trajectory, used in GUI mode
    def show_trajectory_plot(self, xs, ys, hole_xo, hole_yo, bx, by):
        self._clear_content()

        # Make _content a 2-row grid: top = buttons, bottom = plot (expand)
        self._content.grid_rowconfigure(0, weight=0)  # buttons row
        self._content.grid_rowconfigure(1, weight=1)  # plot row expands
        self._content.grid_columnconfigure(0, weight=1)

        # --- Top bar (same place as other buttons) ---
        top = tk.Frame(self._content)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top.grid_columnconfigure(0, weight=1)

        btn = tk.Button(
            top,
            text="Continue",
            font=("TkDefaultFont", 12),
            command=lambda: self._choose(True),
            width=14,
            height=2,
        )
        btn.pack()  # centered in the top bar

        # In show_trajectory_plot, before creating plot_frame:
        self._set_message("Trajectory result")  # optional
        self._set_buttons([("Continue", True)])
        # then create plot_frame in row=1 instead of row=0 (so _set_buttons uses row 0)

        # --- Plot area (fills rest) ---
        plot_frame = tk.Frame(self._content)
        plot_frame.grid(row=1, column=0, sticky="nsew")

        fig = Figure(figsize=(6.8, 5.0), dpi=100)
        ax = fig.add_subplot(111)

        ax.plot(xs, ys, marker="o")
        ax.scatter([hole_xo], [hole_yo], s=150, c="black", marker="o")
        ax.scatter([bx], [by], s=150, c="orange", marker="o")

        ax.set_aspect("equal", "box")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Ball trajectory (Kalman filtered)")
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self._wait()

    
    def show_hole(self, hole_number: int):
        self._headline.config(text=f"HOLE {hole_number}")
        
    def clear_hole(self):
        self._headline.config(text="")
    
    def busy_start(self, text="Working..."):
        """Show an animated 'spinner' text inside the main content area."""
        self._spinner_base = text
        self._spinner_i = 0

        # Replace whatever is in the content area (buttons/plot) with a big label
        self._clear_content()
        self._busy_label = tk.Label(
            self._content,
            text=text,
            font=("TkDefaultFont", 18, "bold"),
            justify="center",
        )
        self._busy_label.pack(expand=True)

        self._tick_spinner()

    def busy_stop(self):
        """Stop spinner and remove busy UI."""
        if self._spinner_job is not None:
            try:
                self._root.after_cancel(self._spinner_job)
            except tk.TclError:
                pass
            self._spinner_job = None

        # Clear busy label (don’t recreate buttons here; next prompt will)
        self._clear_content()
        self._busy_label = None

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
            ("Wrong hole", "wrong_hole"),
        ])
        return self._wait()

    def ask_which_hole(self):
        self._set_message("Which hole was targeted?")
        self._set_buttons([
            ("Hole 1", 1),
            ("Hole 2", 2),
            ("Hole 3", 3),
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