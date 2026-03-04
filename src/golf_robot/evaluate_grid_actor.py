import os
import sys
import csv
import yaml
import numpy as np
import subprocess
import re
import socket
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from vision.ball_start_position import get_ball_start_position
from planning.generate_trajectory_csv import generate_trajectory_csv
from planning_linear.linear_planner import send_swing
from gui.gui import *
from rl_common import *


# Set exactly one of these to True
PRINT_MODE = False
EXECUTE_MODE = True

OPERATING_SYSTEM = "linux"
CAMERA_INDEX_START = 2
planner = "quintic"  # "quintic" or "linear"
END_POS = [-2.47, -2.38, -1.55, 1.66, 0.49, -0.26]
LOG_SHOTS = False

assert PRINT_MODE ^ EXECUTE_MODE, "Set exactly one of PRINT_MODE / EXECUTE_MODE to True"


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]  

LOOKUP_CSV_CANDIDATES = [
    PROJECT_ROOT / "log" / "lookups" / "state_action_lookup.csv"
]

LOOKUP_CSV_PATH = None
for p in LOOKUP_CSV_CANDIDATES:
    if p.exists():
        LOOKUP_CSV_PATH = p
        break
if LOOKUP_CSV_PATH is None:
    raise FileNotFoundError(f"Could not find lookup CSV. Tried: {LOOKUP_CSV_CANDIDATES}")

HOLE_CONFIG_PATH = PROJECT_ROOT / "configs" / "hole_config.yaml"
MUJOCO_CONFIG_PATH = PROJECT_ROOT / "configs" / "mujoco_config.yaml"
RL_CONFIG_PATH = PROJECT_ROOT / "configs" / "rl_config.yaml"


def ban_self_from_cores(cores):
    allowed = set(os.sched_getaffinity(0))
    ban = set(cores)
    new_allowed = allowed - ban
    if new_allowed != allowed:
        os.sched_setaffinity(0, new_allowed)
        print(f"[Affinity] Python allowed cores now: {sorted(new_allowed)}")


# =========================================================
# Lookup-based deterministic interpolating actor (speed, angle_deg)
# =========================================================

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


@dataclass
class HoleLookupGrid:
    hole_xy: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    points: Dict[Tuple[int, int], np.ndarray]  # (qx,qy)->[2]
    # cache for robust nearest fallback
    keys_list: List[Tuple[float, float]]
    keys_arr: np.ndarray          # [M,2]
    vals_arr: np.ndarray          # [M,2]

    @property
    def bounds(self):
        return float(self.gx[0]), float(self.gx[-1]), float(self.gy[0]), float(self.gy[-1])


import csv
from pathlib import Path
from typing import List

import csv
from pathlib import Path
from typing import List

Q = 1000  # 3 decimals

def q3(x: float) -> int:
    return int(round(float(x) * Q))

def uq3(k: int) -> float:
    return k / Q

def load_lookup_csv(path: Path) -> List[HoleLookupGrid]:
    """
    Expects columns:
      hole_x,hole_y,ball_x,ball_y,speed,angle_deg

    Uses EXACT points from the CSV (no cartesian product completion).
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = ["hole_x", "hole_y", "ball_x", "ball_y", "speed", "angle_deg"]
        for k in required:
            if k not in (r.fieldnames or []):
                raise ValueError(f"Lookup CSV missing '{k}'. Found: {r.fieldnames}")

        for d in r:
            hx = float(d["hole_x"]); hy = float(d["hole_y"])
            bx = float(d["ball_x"]); by = float(d["ball_y"])
            sp = float(d["speed"]);  ang = float(d["angle_deg"])
            rows.append((hx, hy, bx, by, sp, ang))

    if not rows:
        raise RuntimeError(f"Lookup CSV is empty: {path}")

    holes = sorted({(hx, hy) for (hx, hy, *_rest) in rows})

    grids: List[HoleLookupGrid] = []
    for (hx, hy) in holes:
        sub = [(bx, by, sp, ang) for (hhx, hhy, bx, by, sp, ang) in rows if (hhx, hhy) == (hx, hy)]
        if not sub:
            continue

        gx = np.array(sorted({float(bx) for (bx, _, _, _) in sub}), dtype=np.float32)
        gy = np.array(sorted({float(by) for (_, by, _, _) in sub}), dtype=np.float32)

        points = {}
        for (bx, by, sp, ang) in sub:
            points[(q3(bx), q3(by))] = np.array([float(sp), float(ang)], dtype=np.float32)

        keys_list = list(points.keys())
        keys_arr = np.array(keys_list, dtype=np.float64)   # keep precision
        vals_arr = np.stack([points[k] for k in keys_list], axis=0).astype(np.float32)

        grids.append(HoleLookupGrid(
            hole_xy=np.array([hx, hy], dtype=np.float32),
            gx=gx, gy=gy,
            points=points,
            keys_list=keys_list,
            keys_arr=keys_arr,
            vals_arr=vals_arr,
        ))

    print(f"[LookupActor] Loaded {len(grids)} hole lookups from {path}")
    for g in grids:
        print(f"  hole={g.hole_xy.tolist()}  |gx|={g.gx.size} |gy|={g.gy.size} |pts|={len(g.points)}")
    return grids


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)

class LookupInterpolatingActor:
    def __init__(self, grids: List[HoleLookupGrid]):
        self.grids = grids
        self.holes = np.stack([g.hole_xy for g in grids], axis=0)  # [H,2]

    def _hole_index_nearest(self, hole_xy: np.ndarray) -> int:
        hole_xy = np.asarray(hole_xy, dtype=np.float32).reshape(-1)[:2]
        d2 = np.sum((self.holes - hole_xy[None, :]) ** 2, axis=1)
        return int(np.argmin(d2))

    @staticmethod
    def _nearest_point_action(hg: HoleLookupGrid, x: float, y: float) -> np.ndarray:
        qx, qy = q3(x), q3(y)
        keys = np.array(list(hg.points.keys()), dtype=np.int32)  # [M,2]
        d2 = np.sum((keys - np.array([qx, qy], dtype=np.int32)[None, :]) ** 2, axis=1)
        idx = int(np.argmin(d2))
        k = (int(keys[idx, 0]), int(keys[idx, 1]))
        return hg.points[k]

    @staticmethod
    def _bilinear_if_cell_complete(hg: HoleLookupGrid, x: float, y: float) -> np.ndarray:
        gx, gy = hg.gx, hg.gy
        x0, x1, y0, y1 = hg.bounds
        x = _clamp(float(x), x0, x1)
        y = _clamp(float(y), y0, y1)

        # find bracketing indices
        i = int(np.searchsorted(gx, x, side="right") - 1)
        j = int(np.searchsorted(gy, y, side="right") - 1)
        i = max(0, min(i, len(gx) - 2))
        j = max(0, min(j, len(gy) - 2))

        xL, xR = float(gx[i]), float(gx[i + 1])
        yB, yT = float(gy[j]), float(gy[j + 1])

        p = hg.points
        k00 = (q3(xL), q3(yB))
        k10 = (q3(xR), q3(yB))
        k01 = (q3(xL), q3(yT))
        k11 = (q3(xR), q3(yT))

        if (k00 not in p) or (k10 not in p) or (k01 not in p) or (k11 not in p):
            return LookupInterpolatingActor._nearest_point_action(hg, x, y)

        a00 = p[k00]
        a10 = p[k10]
        a01 = p[k01]
        a11 = p[k11]

        tx = 0.0 if xR == xL else (x - xL) / (xR - xL)
        ty = 0.0 if yT == yB else (y - yB) / (yT - yB)

        a0 = (1.0 - tx) * a00 + tx * a10
        a1 = (1.0 - tx) * a01 + tx * a11
        a  = (1.0 - ty) * a0  + ty * a1
        return a.astype(np.float32)

    def __call__(self, state):
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        bx, by = float(s[0]), float(s[1])
        hx, hy = float(s[2]), float(s[3])

        hi = self._hole_index_nearest(np.array([hx, hy], dtype=np.float32))
        hg = self.grids[hi]

        out = self._bilinear_if_cell_complete(hg, bx, by)  # [2] (speed, angle_deg)
        return float(out[0]), float(out[1])


# =========================================================
# Real init parameters
# =========================================================
def real_init_parameters(camera_index, chosen_hole=None):
    bx, by, dbg = get_ball_start_position(
        debug=True, return_debug_image=True, debug_raw=False,
        use_cam=True, camera_index=camera_index,
        operating_system=OPERATING_SYSTEM
    )
    ball_start_position = np.array([bx, by])
    print(f"ball_start_position: {ball_start_position}")

    if dbg is not None:
        prompter.show_image(dbg, title="Ball start detection")

    if chosen_hole is None:
        chosen_hole = 2

    with open(HOLE_CONFIG_PATH, "r") as f:
        hole_positions = yaml.safe_load(f)

    hx = hole_positions[chosen_hole]["x"]
    hy = hole_positions[chosen_hole]["y"]
    hole_position = np.array([hx, hy])
    print(f"Chosen hole {chosen_hole} at position:", hole_position)

    disc_positions = []

    prompter.show_hole(chosen_hole)
    prompter.confirm_or_exit("Trajectory ready. Continue to execute?")
    prompter.clear_hole()

    return ball_start_position, hole_position, disc_positions, chosen_hole


# =========================================================
# Real execution 
# =========================================================
def run_real(impact_velocity, swing_angle, ball_start_position, planner="quintic", check_rtt=False, chosen_hole=None):
    print(f"Impact velocity: {impact_velocity} m/s, swing angle: {swing_angle} deg, ball start pos: {ball_start_position} m")

    if planner == "quintic":
        results = generate_trajectory_csv(impact_velocity, swing_angle, ball_start_position[0], ball_start_position[1])
        if results is None:
            print("Trajectory not feasible, aborting.")
            sys.exit(0)

        if OPERATING_SYSTEM == "windows":
            if check_rtt:
                out = subprocess.check_output(["wsl", "ping", "-c", "3", "192.38.66.227"], text=True)
                rtts = [float(m.group(1)) for m in re.finditer(r'time=([\d.]+)\s*ms', out)]
                print("RTTs (ms):", rtts)
                print("avg:", sum(rtts) / len(rtts))

        prompter.confirm_or_exit("Ready to execute trajectory. Continue?")

        traj_exe = HERE / "communication" / "traj_streamer"

        def run_traj():
            with open(os.devnull, "w") as dn:
                p = subprocess.Popen([str(traj_exe)], stdout=dn, stderr=dn)

                os.sched_setaffinity(p.pid, {10, 11})
                print("[Affinity] traj_streamer allowed cores:", sorted(os.sched_getaffinity(p.pid)))

                try:
                    _ = subprocess.run(
                        ["sudo", "-n", "/usr/bin/chrt", "-f", "-p", "80", str(p.pid)],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
                    )
                except Exception:
                    pass

                rc = p.wait()
                return subprocess.CompletedProcess(args=[str(traj_exe)], returncode=rc)

        _ = prompter.run_blocking("Shooting…", run_traj)

    if planner == "linear":
        HOST = "192.38.66.227"
        PORT_cmd = 30002

        traj_cmd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        traj_cmd.connect((HOST, PORT_cmd))

        print("Sending swing...")
        x_ball_origo = -0.62823
        ball_radius  = 0.021335
        offset       = 0.25
        z_buffer     = 0.01
        x_start = x_ball_origo + ball_radius - offset + ball_start_position[0]
        x_end   = x_ball_origo + ball_radius + offset + ball_start_position[0]
        y_ball_origo = -0.57480
        y_ball = y_ball_origo + ball_start_position[1]
        z_ball = 0.15512 + z_buffer

        _ = send_swing(
            traj_cmd,
            x_start=x_start, x_end=x_end,
            y_ball=y_ball, z_ball=z_ball,
            path_angle_deg=swing_angle, attack_angle_deg=0.0,
            vel=impact_velocity, acc=5.5
        )

        try:
            traj_cmd.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        traj_cmd.close()

    ur_movej(robot_ip="192.38.66.227", q=END_POS)
    print("Return to home position command sent.")

    # ---- logging bits (kept as-is) ----
    if LOG_SHOTS:
        dist_at_hole = 0
        speed_at_hole = 0
        ball_final_position = np.array([0.0, 0.0])
        continue_evaluation = False
        on_green = False

        state = prompter.ask_hole_oob(chosen_hole)
        wrong_hole = False

        if state == "in_hole":
            print("Ball in hole confirmed")
            in_hole = True
            out_of_bounds = False
            dist_at_hole, speed_at_hole = None, None

            with open(HOLE_CONFIG_PATH, "r") as f:
                hole_positions = yaml.safe_load(f)
            hx = hole_positions[chosen_hole]["x"]
            hy = hole_positions[chosen_hole]["y"]
            ball_final_position = np.array([hx, hy])

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
            on_green = False

        if not out_of_bounds and not in_hole:
            ball_final_position = np.array([0, 0])  # dummy

        if out_of_bounds:
            ball_final_position = np.array([0, 0])
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
        ball_final_position = np.array([0.0, 0.0])
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


# =========================================================
# Main
# =========================================================
def main():
    grids = load_lookup_csv(LOOKUP_CSV_PATH)
    actor = LookupInterpolatingActor(grids)

    if PRINT_MODE:
        # HARD-CODED QUERY EXAMPLE (only for debugging)
        CHOSEN_HOLE_ID = 1

        # Your query in ball-start coordinates (meters)
        BALL_X = -0.16
        BALL_Y = -0.10

        with open(HOLE_CONFIG_PATH, "r") as f:
            hole_positions = yaml.safe_load(f)
        hx = float(hole_positions[CHOSEN_HOLE_ID]["x"])
        hy = float(hole_positions[CHOSEN_HOLE_ID]["y"])

        # state format: [bx, by, hx, hy, ...]
        s = np.array([BALL_X, BALL_Y, hx, hy], dtype=np.float32)
        speed, angle = actor(s)
        print(f"[PRINT_MODE] hole_id={CHOSEN_HOLE_ID} hole_xy=({hx:.3f},{hy:.3f}) ball=({BALL_X:.3f},{BALL_Y:.3f})")
        print(f"[PRINT_MODE] -> speed={speed:.6f}  angle_deg={angle:.6f}")
        return

    if EXECUTE_MODE:
        ban_self_from_cores([10, 11])
        global prompter
        prompter = HumanPrompter()

        with open(MUJOCO_CONFIG_PATH, "r") as f:
            mujoco_cfg = yaml.safe_load(f)
        with open(RL_CONFIG_PATH, "r") as f:
            rl_cfg = yaml.safe_load(f)

        try:
            evaluation_policy_hand_tuned(
                actor,
                mujoco_cfg,
                rl_cfg,
                num_episodes=30,
                max_num_discs=0,
                env_step=run_real,
                env_type="real",
                input_func=real_init_parameters,
                planner=planner,
                camera_index=CAMERA_INDEX_START,
            )
        finally:
            if LOG_SHOTS:
                try:
                    prompter.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()