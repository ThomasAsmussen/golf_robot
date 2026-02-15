import os
from pathlib import Path
import sys
import yaml
import numpy as np
import subprocess
import re
import socket
import json

from vision.ball_start_position import get_ball_start_position
from planning.generate_trajectory_csv import generate_trajectory_csv
from planning_linear.linear_planner import send_swing
from gui.gui import *
from rl_common import *
import torch


# =========================================================
# Config
# =========================================================
OPERATING_SYSTEM = "linux"  # "windows" or "linux"
CAMERA_INDEX_START = 2
CAMERA_INDEX_END   = 2

planner = "quintic"  # "quintic" or "linear"
END_POS = [-2.47, -2.38, -1.55, 1.66, 0.49, -0.26]
LOG_SHOTS = True

# KNN settings
K_NEIGHBORS = 5

# JSONL to load (you said you attached it)
# - In this ChatGPT environment it's at /mnt/data/episode_logger_thompson.jsonl
# - In your repo, you can set it to project_root/log/... if preferred
JSONL_PATH_OVERRIDE = "/mnt/data/episode_logger_thompson.jsonl"

# If your JSONL key names differ, edit these:
JSON_KEYS = {
    "hole_xy": "hole_pos_obs",
    "ball_xy": "ball_start_obs",
    "action":  "action_norm",
    "in_hole": "in_hole",
    "oob":     "out_of_bounds",
}

# Stitching / thresholds
REQUIRE_IN_HOLE = True
REQUIRE_NOT_OOB = True


def ban_self_from_cores(cores):
    allowed = set(os.sched_getaffinity(0))
    ban = set(cores)
    new_allowed = allowed - ban
    if new_allowed != allowed:
        os.sched_setaffinity(0, new_allowed)
        print(f"[Affinity] Python allowed cores now: {sorted(new_allowed)}")

import json
import numpy as np

GRID_NX = 5
GRID_NY = 3

GRID_X_DEFAULT = None  # e.g. np.array([-0.10, -0.05, 0.00, 0.05, 0.10])
GRID_Y_DEFAULT = None  # e.g. np.array([-0.06,  0.00, 0.06])

# KNN for filling each grid point
K_NEIGHBORS = 5

JSON_KEYS = {
    "hole_xy": "hole_pos_obs",
    "ball_xy": "ball_start_obs",
    "action":  "action_norm",
    "in_hole": "in_hole",
    "oob":     "out_of_bounds",
}

REQUIRE_IN_HOLE = True
REQUIRE_NOT_OOB = True


def _as_xy(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    if a.size < 2:
        raise ValueError(f"Expected xy, got {a}")
    return a[:2]


class HoleGridDB:
    """
    For each hole (by coordinate), build a 5x3 action grid:
      - The 15 grid points are hand-tuned (GRID_X/Y) or auto-spanned from data.
      - Each gridpoint action is approximated by averaging K nearest SUCCESSFUL shots.
      - At runtime: bilinear interpolation between surrounding grid points.
    """

    def __init__(self, jsonl_path: str, k: int = 5,
                 grid_x_default=None, grid_y_default=None,
                 nx: int = 5, ny: int = 3):
        self.jsonl_path = str(jsonl_path)
        self.k = int(k)
        self.nx = int(nx)
        self.ny = int(ny)

        self.grid_x_default = None if grid_x_default is None else np.asarray(grid_x_default, dtype=np.float32)
        self.grid_y_default = None if grid_y_default is None else np.asarray(grid_y_default, dtype=np.float32)

        # Per-hole storage
        self.holes = None               # [H,2]
        self.ball_by_hole = []          # list of [N,2]
        self.act_by_hole  = []          # list of [N,2]
        self.grid_x = []                # list of [nx]
        self.grid_y = []                # list of [ny]
        self.grid_act = []              # list of [ny,nx,2]  (row=y, col=x)

        self._load_success_data()
        self._build_grids()

    def _load_success_data(self):
        rows = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if JSON_KEYS["hole_xy"] not in r or JSON_KEYS["ball_xy"] not in r or JSON_KEYS["action"] not in r:
                    continue

                if REQUIRE_IN_HOLE:
                    if float(r.get(JSON_KEYS["in_hole"], 0.0)) < 0.5:
                        continue
                if REQUIRE_NOT_OOB:
                    if float(r.get(JSON_KEYS["oob"], 0.0)) > 0.0:
                        continue

                hole_xy = _as_xy(r[JSON_KEYS["hole_xy"]])
                ball_xy = _as_xy(r[JSON_KEYS["ball_xy"]])
                act = np.asarray(r[JSON_KEYS["action"]], dtype=np.float32).reshape(-1)
                if act.size < 2:
                    continue
                act = act[:2]

                rows.append((hole_xy, ball_xy, act))

        if not rows:
            raise RuntimeError(f"[GridBaseline] No successful shots found in {self.jsonl_path}")

        holes_all = np.stack([h for (h, _, _) in rows], axis=0)
        holes_uniq = np.unique(np.round(holes_all, 6), axis=0).astype(np.float32)
        self.holes = holes_uniq

        holes_rounded = np.round(holes_all, 6)
        for h in self.holes:
            mask = np.linalg.norm(holes_rounded - h[None, :], axis=1) <= 1e-12
            balls = np.stack([rows[i][1] for i in range(len(rows)) if mask[i]], axis=0)
            acts  = np.stack([rows[i][2] for i in range(len(rows)) if mask[i]], axis=0)
            self.ball_by_hole.append(balls)
            self.act_by_hole.append(acts)

        print(f"[GridBaseline] Loaded {sum(b.shape[0] for b in self.ball_by_hole)} successful shots across {len(self.holes)} holes:")
        for i, h in enumerate(self.holes):
            print(f"  hole {h.tolist()} : n={self.ball_by_hole[i].shape[0]}")

    def _hole_index_nearest(self, hole_xy: np.ndarray) -> int:
        hole_xy = _as_xy(hole_xy)
        d2 = np.sum((self.holes - hole_xy[None, :]) ** 2, axis=1)
        return int(np.argmin(d2))

    def _auto_grid_from_data(self, balls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Auto-spans min/max of successful shots for that hole (reasonable default),
        # but you can replace with your hand-tuned GRID_X_DEFAULT/GRID_Y_DEFAULT.
        bx_min, by_min = balls.min(axis=0)
        bx_max, by_max = balls.max(axis=0)
        gx = np.linspace(bx_min, bx_max, self.nx, dtype=np.float32)
        gy = np.linspace(by_min, by_max, self.ny, dtype=np.float32)
        return gx, gy

    def _knn_mean_action_at(self, balls: np.ndarray, acts: np.ndarray, query_xy: np.ndarray) -> np.ndarray:
        # KNN in ball-start space
        d2 = np.sum((balls - query_xy[None, :]) ** 2, axis=1)
        k = min(self.k, d2.shape[0])
        idx = np.argpartition(d2, kth=k - 1)[:k]
        return acts[idx].mean(axis=0).astype(np.float32)

    def _build_grids(self):
        self.grid_x.clear()
        self.grid_y.clear()
        self.grid_act.clear()

        for hi in range(len(self.holes)):
            balls = self.ball_by_hole[hi]
            acts  = self.act_by_hole[hi]

            if self.grid_x_default is not None and self.grid_y_default is not None:
                gx = self.grid_x_default.astype(np.float32)
                gy = self.grid_y_default.astype(np.float32)
                assert gx.size == self.nx and gy.size == self.ny, "GRID_X/Y sizes must match nx/ny"
            else:
                gx, gy = self._auto_grid_from_data(balls)

            # Fill 15 grid actions via KNN averaging of successful shots
            grid = np.zeros((self.ny, self.nx, 2), dtype=np.float32)
            for j, y in enumerate(gy):
                for i, x in enumerate(gx):
                    q = np.array([x, y], dtype=np.float32)
                    grid[j, i, :] = self._knn_mean_action_at(balls, acts, q)

            self.grid_x.append(gx)
            self.grid_y.append(gy)
            self.grid_act.append(grid)

            print(f"[GridBaseline] Built grid for hole {self.holes[hi].tolist()} "
                  f"(nx={self.nx}, ny={self.ny})")

    @staticmethod
    def _clamp(v, lo, hi):
        return float(np.clip(v, lo, hi))

    def _bilinear(self, gx: np.ndarray, gy: np.ndarray, grid: np.ndarray, bx: float, by: float) -> np.ndarray:
        """
        Bilinear interpolation on a rectangular grid.
        gx: [nx] sorted
        gy: [ny] sorted
        grid: [ny,nx,2]
        """
        bx = self._clamp(bx, gx[0], gx[-1])
        by = self._clamp(by, gy[0], gy[-1])

        # find x cell
        ix1 = int(np.searchsorted(gx, bx, side="right"))
        ix0 = max(ix1 - 1, 0)
        ix1 = min(ix1, gx.size - 1)

        # find y cell
        iy1 = int(np.searchsorted(gy, by, side="right"))
        iy0 = max(iy1 - 1, 0)
        iy1 = min(iy1, gy.size - 1)

        x0, x1 = float(gx[ix0]), float(gx[ix1])
        y0, y1 = float(gy[iy0]), float(gy[iy1])

        # avoid divide-by-zero (degenerate grid line)
        tx = 0.0 if x1 == x0 else (bx - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (by - y0) / (y1 - y0)

        a00 = grid[iy0, ix0]  # bottom-left
        a10 = grid[iy0, ix1]  # bottom-right
        a01 = grid[iy1, ix0]  # top-left
        a11 = grid[iy1, ix1]  # top-right

        a0 = (1.0 - tx) * a00 + tx * a10
        a1 = (1.0 - tx) * a01 + tx * a11
        a  = (1.0 - ty) * a0  + ty * a1
        return a.astype(np.float32)

    def action_norm_from_state(self, ball_xy: np.ndarray, hole_xy: np.ndarray) -> np.ndarray:
        ball_xy = _as_xy(ball_xy)
        hole_xy = _as_xy(hole_xy)

        hi = self._hole_index_nearest(hole_xy)
        gx = self.grid_x[hi]
        gy = self.grid_y[hi]
        grid = self.grid_act[hi]

        return self._bilinear(gx, gy, grid, float(ball_xy[0]), float(ball_xy[1]))


# =========================================================
# The EXACT "structure" you asked to keep:
# make_grid_policy(...) -> policy(state) -> (speed, angle_deg)
# Works with evaluation_policy_hand_tuned.
# =========================================================
def make_grid_policy(jsonl_path: str, rl_cfg: dict,
                     k: int = K_NEIGHBORS,
                     grid_x_default=GRID_X_DEFAULT,
                     grid_y_default=GRID_Y_DEFAULT,
                     nx: int = GRID_NX,
                     ny: int = GRID_NY):

    db = HoleGridDB(
        jsonl_path=jsonl_path,
        k=k,
        grid_x_default=grid_x_default,
        grid_y_default=grid_y_default,
        nx=nx,
        ny=ny,
    )

    speed_low  = rl_cfg["model"]["speed_low"]
    speed_high = rl_cfg["model"]["speed_high"]
    angle_low  = rl_cfg["model"]["angle_low"]
    angle_high = rl_cfg["model"]["angle_high"]

    def policy(state):
        """
        Compatible with evaluation_policy_hand_tuned.

        state is RAW output of encode_state_with_discs:
          [bx, by, hx, hy, ...]
        """
        s = np.asarray(state, dtype=float).reshape(-1)
        ball_xy = s[0:2]
        hole_xy = s[2:4]

        a_norm = db.action_norm_from_state(ball_xy, hole_xy)

        # Uses your existing rl_common scaling to physical commands
        speed, angle_deg = get_input_parameters(
            a_norm, speed_low, speed_high, angle_low, angle_high
        )
        return float(speed), float(angle_deg)

    return policy

# =========================================================
# Real init parameters (same as yours)
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

    here = Path(__file__).resolve().parent
    config_dir = here.parents[1] / "configs"
    with open(config_dir / "hole_config.yaml", "r") as f:
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
# Real execution (same as yours)
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

        here = Path(__file__).resolve().parent
        traj_exe = here / "communication" / "traj_streamer"

        def run_traj():
            with open(os.devnull, "w") as dn:
                p = subprocess.Popen([str(traj_exe)], stdout=dn, stderr=dn)

                os.sched_setaffinity(p.pid, {10, 11})
                print("[Affinity] traj_streamer allowed cores:", sorted(os.sched_getaffinity(p.pid)))

                # Optional RT (non-fatal)
                try:
                    rt = subprocess.run(
                        ["sudo", "-n", "/usr/bin/chrt", "-f", "-p", "80", str(p.pid)],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
                    )
                    if rt.stderr.strip():
                        print("[chrt stderr]:", rt.stderr.strip())
                except Exception as e:
                    # not fatal
                    pass

                rc = p.wait()
                return subprocess.CompletedProcess(args=[str(traj_exe)], returncode=rc)

        result = prompter.run_blocking("Shooting…", run_traj)
        print("Trajectory streamer output:")
        print(result.stdout)

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

            here = Path(__file__).resolve().parent
            config_dir = here.parents[1] / "configs"
            with open(config_dir / "hole_config.yaml", "r") as f:
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
# KNN policy implementation
# =========================================================
def _as_xy(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    if a.size != 2:
        raise ValueError(f"Expected 2D xy, got {a.shape} (size={a.size})")
    return a


class HoleCoordKNNDB:
    """
    Stores successful shots, grouped by hole coordinate (3 unique holes in your JSONL).
    For a query (ball_xy, hole_xy), match to nearest hole coordinate, then KNN in ball_xy.
    """

    def __init__(self, jsonl_path: str, k: int = 5):
        self.jsonl_path = jsonl_path
        self.k = int(k)

        self.holes = None          # [H,2]
        self.ball_by_hole = []     # list of [N,2]
        self.act_by_hole  = []     # list of [N,2]
        self._load()

    def _load(self):
        rows = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # required
                if JSON_KEYS["hole_xy"] not in r or JSON_KEYS["ball_xy"] not in r or JSON_KEYS["action"] not in r:
                    continue

                # success filtering
                if REQUIRE_IN_HOLE:
                    in_hole = float(r.get(JSON_KEYS["in_hole"], 0.0))
                    if in_hole < 0.5:
                        continue
                if REQUIRE_NOT_OOB:
                    oob = float(r.get(JSON_KEYS["oob"], 0.0))
                    if oob > 0.0:
                        continue

                hole_xy = _as_xy(r[JSON_KEYS["hole_xy"]])
                ball_xy = _as_xy(r[JSON_KEYS["ball_xy"]])

                act = np.asarray(r[JSON_KEYS["action"]], dtype=np.float32).reshape(-1)
                if act.size != 2:
                    continue

                rows.append((hole_xy, ball_xy, act))

        if not rows:
            raise RuntimeError(f"No usable successful shots found in: {self.jsonl_path}")

        holes_all = np.stack([h for (h, _, _) in rows], axis=0)
        holes_uniq = np.unique(np.round(holes_all, 6), axis=0).astype(np.float32)
        self.holes = holes_uniq

        self.ball_by_hole = []
        self.act_by_hole = []
        holes_rounded = np.round(holes_all, 6)

        for h in self.holes:
            mask = np.linalg.norm(holes_rounded - h[None, :], axis=1) <= 1e-12
            balls = np.stack([rows[i][1] for i in range(len(rows)) if mask[i]], axis=0)
            acts  = np.stack([rows[i][2] for i in range(len(rows)) if mask[i]], axis=0)
            self.ball_by_hole.append(balls)
            self.act_by_hole.append(acts)

        print(f"[KNN] Loaded {sum(b.shape[0] for b in self.ball_by_hole)} successful shots across {len(self.holes)} holes:")
        for i, h in enumerate(self.holes):
            print(f"  hole {h.tolist()} : n={self.ball_by_hole[i].shape[0]}")

    def _hole_index_nearest(self, hole_xy: np.ndarray) -> int:
        d2 = np.sum((self.holes - hole_xy[None, :]) ** 2, axis=1)
        return int(np.argmin(d2))

    def mean_action_norm(self, ball_xy, hole_xy) -> np.ndarray:
        ball_xy = _as_xy(ball_xy)
        hole_xy = _as_xy(hole_xy)

        hi = self._hole_index_nearest(hole_xy)
        balls = self.ball_by_hole[hi]
        acts  = self.act_by_hole[hi]

        d2 = np.sum((balls - ball_xy[None, :]) ** 2, axis=1)
        k = min(self.k, d2.shape[0])
        idx = np.argpartition(d2, kth=k - 1)[:k]
        return acts[idx].mean(axis=0).astype(np.float32)


def _denorm_action(action_norm: np.ndarray, rl_cfg: dict):
    """
    Convert normalized [-1,1]^2 action -> (impact_velocity, swing_angle).

    We try a few common config patterns; if none exist, we fall back to identity
    (which will be wrong if your run_real expects physical units).

    If you already have a helper in rl_common (e.g. denormalize_action),
    we use it automatically.
    """
    a = np.asarray(action_norm, dtype=np.float32).reshape(-1)
    if a.size != 2:
        raise ValueError("Expected 2D action")

    # 1) Use rl_common helper if present
    for fn_name in ["denormalize_action", "unnormalize_action", "action_unnormalize"]:
        fn = globals().get(fn_name, None)
        if callable(fn):
            out = fn(a, rl_cfg)
            # expected (vel, angle) or array-like length 2
            out = np.asarray(out, dtype=np.float32).reshape(-1)
            if out.size == 2:
                return float(out[0]), float(out[1])

    # 2) Try common bounds in config
    candidates = []
    # a) rl_cfg["env"]
    env = rl_cfg.get("env", {})
    if "action_low" in env and "action_high" in env:
        candidates.append((env["action_low"], env["action_high"]))
    # b) rl_cfg["training"]
    tr = rl_cfg.get("training", {})
    if "action_low" in tr and "action_high" in tr:
        candidates.append((tr["action_low"], tr["action_high"]))
    # c) rl_cfg top-level
    if "action_low" in rl_cfg and "action_high" in rl_cfg:
        candidates.append((rl_cfg["action_low"], rl_cfg["action_high"]))

    for lo, hi in candidates:
        lo = np.asarray(lo, dtype=np.float32).reshape(-1)
        hi = np.asarray(hi, dtype=np.float32).reshape(-1)
        if lo.size >= 2 and hi.size >= 2:
            lo = lo[:2]
            hi = hi[:2]
            a01 = (a + 1.0) * 0.5
            out = lo + a01 * (hi - lo)
            return float(out[0]), float(out[1])

    # 3) Fallback identity
    print("[WARN] Could not denormalize action_norm from rl_cfg; using identity.")
    return float(a[0]), float(a[1])


def _try_extract_ball_hole_from_state(state):
    """
    Robust extraction for evaluation_policy_hand_tuned(actor(state)).

    Supports:
      - dict-like:
          state["ball_start_position"] / state["hole_position"]
          state["ball_start_obs"]      / state["hole_pos_obs"]
          state["ball_xy"]            / state["hole_xy"]
      - numpy array / list / tuple:
          assumes first 4 entries are [bx, by, hx, hy]
      - torch tensor:
          same assumption: first 4 are [bx, by, hx, hy]
    """
    # 1) dict-like
    if isinstance(state, dict):
        for bk, hk in [
            ("ball_start_position", "hole_position"),
            ("ball_start_obs", "hole_pos_obs"),
            ("ball_xy", "hole_xy"),
        ]:
            if bk in state and hk in state:
                return np.asarray(state[bk], dtype=np.float32).reshape(-1)[:2], \
                       np.asarray(state[hk], dtype=np.float32).reshape(-1)[:2]

        # sometimes nested
        if "obs" in state and isinstance(state["obs"], dict):
            return _try_extract_ball_hole_from_state(state["obs"])

    # 2) torch tensor
    if torch is not None and torch.is_tensor(state):
        s = state.detach().cpu().numpy()
        s = np.asarray(s, dtype=np.float32).reshape(-1)
        if s.size >= 4:
            return s[0:2], s[2:4]

    # 3) array-like
    if isinstance(state, (list, tuple, np.ndarray)):
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if s.size >= 4:
            return s[0:2], s[2:4]

    return None, None


def make_knn_policy(jsonl_path: str, rl_cfg: dict, k: int = 5):
    db = HoleCoordKNNDB(jsonl_path=jsonl_path, k=k)

    speed_low  = rl_cfg["model"]["speed_low"]
    speed_high = rl_cfg["model"]["speed_high"]
    angle_low  = rl_cfg["model"]["angle_low"]
    angle_high = rl_cfg["model"]["angle_high"]

    def policy(state):
        """
        Compatible with evaluation_policy_hand_tuned.

        state is RAW output of encode_state_with_discs:
        [bx, by, hx, hy, ...]
        """

        state = np.asarray(state, dtype=float).reshape(-1)

        # ---- extract ball + hole directly ----
        ball_xy = state[0:2]
        hole_xy = state[2:4]

        # ---- get mean normalized action from KNN ----
        action_norm = db.mean_action_norm(ball_xy, hole_xy)

        # ---- map normalized -> physical using rl_common helper ----
        speed, angle_deg = get_input_parameters(
            action_norm,
            speed_low,
            speed_high,
            angle_low,
            angle_high,
        )

        return float(speed), float(angle_deg)

    return policy


# =========================================================
# Main (mirrors your script: uses evaluation_policy_hand_tuned)
# =========================================================
def main():
    ban_self_from_cores([10, 11])
    global prompter
    prompter = HumanPrompter()

    here = Path(__file__).resolve().parent

    if LOG_SHOTS:
        project_root = here.parents[1]
        episode_log_path = project_root / "log" / "real_episodes_eval" / "episode_logger_eval_knn.jsonl"
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

    # Pick JSONL path: override if it exists, else fall back to repo log path
    jsonl_path = JSONL_PATH_OVERRIDE
    if not Path(jsonl_path).exists():
        # fallback guess in your repo
        candidate = project_root / "log" / "real_episodes" / "episode_logger_thompson.jsonl"
        if candidate.exists():
            jsonl_path = str(candidate)
        else:
            raise FileNotFoundError(
                f"Could not find JSONL. Tried:\n  {JSONL_PATH_OVERRIDE}\n  {candidate}"
            )

    print("[KNN] Using JSONL:", jsonl_path)
    knn_policy = make_knn_policy(jsonl_path=jsonl_path, rl_cfg=rl_cfg, k=K_NEIGHBORS)

    # actor must be callable: actor(state)->(speed, angle_deg)
    actor = make_grid_policy(
        jsonl_path=project_root / "log" / "real_episodes" / "episode_logger_thompson.jsonl",  # or your repo path
        rl_cfg=rl_cfg,
        k=5,
        # If you want true “hand-tuned 15 grid points”, set GRID_X_DEFAULT / GRID_Y_DEFAULT above.
    )


    try:
        # EXACTLY like your script’s hand-tuned branch: call evaluation_policy_hand_tuned(...)
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