import os
import math
import argparse
import csv as _csv
from typing import List, Dict, Tuple
import contextlib
from collections import defaultdict

import numpy as np
import mujoco
import optuna
import optunahub
import glob
import re


# ------------------------------ Paths ---------------------------------
HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
XML_PATH_DEFAULT = os.path.join(REPO, "models", "mujoco", "golf_world_no_hole.xml")
DEFAULT_DB_PATH = os.path.join(HERE, "optuna_calib.db")


# --------------------------- Small utilities --------------------------
@contextlib.contextmanager
def pushd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def mj_id(model, obj, name):
    return mujoco.mj_name2id(model, obj, name)


def get_ids(model):
    return {
        "club_head_gid": mj_id(model, mujoco.mjtObj.mjOBJ_GEOM,  "club_head"),
        "ball_gid":      mj_id(model, mujoco.mjtObj.mjOBJ_GEOM,  "ball_geom"),
        "hinge_jid":     mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "club_hinge"),
        "act_id":        mj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "club_motor"),
        "ball_jid":      mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"),
        "base_jid":      mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "mount_free"),
        "flex_jid":      mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_flex_joint"),
    }


# ---------------------- Measurement loading (X only) ------------------
def load_csv_x(path: str) -> Dict[str, np.ndarray]:
    """
    Accept columns (case-insensitive):
      time|t,  x|ball_x,  [vx optional]
    If vx missing, compute it by finite difference on x.
    """
    with open(path, "r", newline="") as f:
        rows = list(_csv.reader(f))
    if not rows:
        raise ValueError(f"{path}: empty CSV")

    header = [h.strip().lower() for h in rows[0]]
    data_rows = rows[1:] if any(header) else rows

    def find(col_names):
        for n in col_names:
            if n in header:
                return header.index(n)
        return None

    ti = find(["time", "t", "time_s", "time (s)", "time[s]"])
    xi = find(["x", "ball_x", "x_m", "x (m)", "x[m]"])
    vxi = find(["vx", "vx_m_s", "vx (m/s)", "vx[m/s]"])
    
    if ti is None or xi is None:
        raise ValueError(f"{path}: must include time and x columns")

    arr = np.array([[float(c) for c in r] for r in data_rows if len(r) >= 2], dtype=float)
    t = arr[:, ti]
    x = arr[:, xi]

    if vxi is not None and vxi < arr.shape[1]:
        vx = arr[:, vxi]
    else:
        vx = np.zeros_like(x)
        if len(t) > 1:
            vx[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2] + 1e-12)
            vx[0]    = (x[1] - x[0])     / (t[1] - t[0] + 1e-12)
            vx[-1]   = (x[-1] - x[-2])   / (t[-1] - t[-2] + 1e-12)

    #t = t - t[0]  # start at zero
    return {"t": t, "x": x, "vx": vx}


def parse_csvvx_arg(s: str) -> Tuple[str, float]:
    if "|" not in s:
        raise ValueError(f'--csvvx entry must be "PATH|VX_DES": got {s}')
    path, vx_str = s.rsplit("|", 1)
    path = path.strip().strip('"')
    vx = float(vx_str)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return path, vx


def load_measurements_with_vx(csvvx_list: List[str]) -> List[Dict[str, np.ndarray]]:
    """Each item: {'t','x','vx','vx_des'}; vx_des is scalar for this file."""
    out = []
    for item in csvvx_list:
        path, vx_des = parse_csvvx_arg(item)
        m = load_csv_x(path)
        m["vx_des"] = float(vx_des)
        out.append(m)
    return out


_NAME_RE = re.compile(
    r"^test_(?P<vx>[0-9]+(?:[.,][0-9]+)?)-(?P<idx>\d+)_trajectory_time_shifted",
    re.IGNORECASE,
)

def infer_vx_from_filename(path: str) -> float:
    """
    Parse a speed from filenames like:
      test_0.6-1_trajectory.csv
      test_0.8-5_trajectory.csv
      test_1.0-3_trajectory.csv
    Returns vx_des as float (e.g. 0.6, 0.8, 1.0).
    """
    base = os.path.splitext(os.path.basename(path))[0]
    m = _NAME_RE.match(base)
    if not m:
        raise ValueError(f"Cannot infer vx_des from filename: {base}")
    vx_str = m.group("vx").replace(",", ".")
    return float(vx_str)


def load_measurements_from_dir(dir_path: str) -> List[Dict[str, np.ndarray]]:
    """
    Scan a directory for files named like test_<vx>-<id>_trajectory*.csv,
    load each with load_csv_x, and attach the inferred vx_des.
    """
    pattern = os.path.join(dir_path, "test_*_trajectory_time_shifted*")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r}")

    print("[info] Found files:")
    for f in files:
        print("   ", f)
    
    out: List[Dict[str, np.ndarray]] = []
    for path in files:
        vx_des = infer_vx_from_filename(path)
        m = load_csv_x(path)   # expects CSV with time + x columns
        m["vx_des"] = float(vx_des)
        out.append(m)
        print(f"[info] loaded {os.path.basename(path)} with vx_des={vx_des}")
    return out


# ------------------------- Sim helpers (controller) -------------------
def reset_ball_state(model, data, start_pos=(0.0, 0.0, 0.02135)):
    jid = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    qadr = model.jnt_qposadr[jid]; vadr = model.jnt_dofadr[jid]
    data.qpos[qadr:qadr+7] = np.array([*start_pos, 1.0, 0.0, 0.0, 0.0], float)
    data.qvel[vadr:vadr+6] = 0.0


def reset_club_pose(model, data, start_deg=40.0):
    jid = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "club_hinge")
    qadr = model.jnt_qposadr[jid]; dof = model.jnt_dofadr[jid]
    lo_rad, hi_rad = model.jnt_range[jid]
    q0 = float(np.clip(np.deg2rad(start_deg), lo_rad, hi_rad))
    data.qpos[qadr] = q0
    data.qvel[dof]  = 0.0
    mujoco.mj_forward(model, data)


def bake_aim_pose(model, data, yaw_rad=0.0):
    jid = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "mount_free")
    qadr = model.jnt_qposadr[jid]; vadr = model.jnt_dofadr[jid]
    half = 0.5 * yaw_rad
    quat = np.array([math.cos(half), 0.0, 0.0, math.sin(half)], float)
    data.qpos[qadr:qadr+7] = np.array([0.0, 0.0, 0.02135, *quat], float)
    data.qvel[vadr:vadr+6] = 0.0
    mujoco.mj_forward(model, data)

def head_ball_dx(data, ids):
    return float(data.geom_xpos[ids["club_head_gid"]][0] - data.geom_xpos[ids["ball_gid"]][0])

def command_impact_speed(model, data, ids, vx_des=1.0, R=0.36):
    """
    Open-loop command:
    - Compute hinge angular speed that would give linear speed vx_des
      at 0° (unflexed configuration).
    - Do NOT use live club head pose in the Jacobian, so the motor
      doesn't compensate for shaft flex.
    """

    # Direction to the impact plane (same logic as before)
    dx = head_ball_dx(data, ids)   # <-- remove `model` here
    dir_to_plane = -np.sign(dx if dx != 0.0 else 1e-12)

    # Desired linear speed (along x) at 0°
    v_des = -float(abs(vx_des)) * dir_to_plane

    # Mapping v = omega * R  =>  omega = v / R
    if abs(R) < 1e-8:
        return 0.0

    omega_cmd = v_des / float(R)
    
    return float(omega_cmd)



# -------------------------- In-place parameter set --------------------
def find_pair_index(model, g1, g2):
    for i in range(model.npair):
        a = int(model.pair_geom1[i]); b = int(model.pair_geom2[i])
        if (a == g1 and b == g2) or (a == g2 and b == g1):
            return i
    return -1


def apply_params_inplace_only_requested(model, ids, trial):
    """
    Vary:
      - contact pair(ball_geom, club_head): solref[tc, dr], solimp[a,b,c]
      - global default <geom friction="slide spin roll">
    but only in a reasonably small neighborhood around the XML values.
    """
    if True:
        # ---- Baseline values from your XML ----
        base_tc   = 0.01
        base_dr   = 0.13
        base_a    = 0.6
        base_b    = 1.05
        base_c    = 0.001
        base_slide = 0.40
        base_spin  = 0.0001
        #base_roll  = 0.000845

        # ---- Ball–club contact pair (local search) ----
        pid = find_pair_index(model, ids["ball_gid"], ids["club_head_gid"])
        if pid == -1:
            raise RuntimeError("Ball–club contact pair not found in compiled model.")

        # solref = [timeconst, damp_ratio]
        tc = trial.suggest_float(
            "pair.solref.tc",
            base_tc * 0.5,
            base_tc * 2.0,
            log=True,
        )
        dr = trial.suggest_float(
            "pair.solref.dr",
            base_dr * 0.5,
            base_dr * 2.0,
        )
        model.pair_solref[pid][0] = tc
        model.pair_solref[pid][1] = dr

        # solimp = [a, b, c]
        a = trial.suggest_float(
            "pair.solimp.a",
            base_a - 0.3,
            base_a + 0.3,
        )
        b = trial.suggest_float(
            "pair.solimp.b",
            base_b - 0.3,
            base_b + 0.3,
        )
        c = trial.suggest_float(
            "pair.solimp.c",
            base_c * 0.5,
            base_c * 2.0,
            log=True,
        )
        model.pair_solimp[pid][0] = a
        model.pair_solimp[pid][1] = b
        model.pair_solimp[pid][2] = c

        # ---- Global default <geom friction="slide spin roll"> (local search) ----
        slide = trial.suggest_float(
            "default.geom.friction.slide",
            base_slide * 0.5,
            base_slide * 2.0,
        )
        spin = trial.suggest_float(
            "default.geom.friction.spin",
            base_spin * 0.01,
            base_spin * 10.0,
            log=True,
        )
        #roll = trial.suggest_float(
        #    "default.geom.friction.roll",
        #    base_roll * 0.1,
        #    base_roll * 10.0,
        #    log=True,
        #)

        # Apply to all contact geoms
        for gid in range(model.ngeom):
            try:
                contype = int(model.geom_contype[gid])
                if contype == 0:
                    continue
            except Exception:
                pass
            model.geom_friction[gid][0] = slide
            model.geom_friction[gid][1] = spin
            #model.geom_friction[gid][2] = roll
        
    else:
        # ---- Ball–club contact pair (local search) ----
        pid = find_pair_index(model, ids["ball_gid"], ids["club_head_gid"])
        if pid == -1:
            raise RuntimeError("Ball–club contact pair not found in compiled model.")

        base_tc   = 0.0945187990260779
        base_dr   = 0.25991
        # solref = [timeconst, damp_ratio]
        tc = trial.suggest_float(
            "pair.solref.tc",
            base_tc * 0.2,
            base_tc * 5.0,
            log=True,
        )
        dr = trial.suggest_float(
            "pair.solref.dr",
            base_dr * 0.2,
            base_dr * 5.0,
        )
        model.pair_solref[pid][0] = tc
        model.pair_solref[pid][1] = dr

        base_slide = 0.40
        # ---- Global default <geom friction="slide spin roll"> (local search) ----
        slide = trial.suggest_float(
            "default.geom.friction.slide",
            base_slide * 0.5,
            base_slide * 2.0,
        )
        for gid in range(model.ngeom):
            try:
                contype = int(model.geom_contype[gid])
                if contype == 0:
                    continue
            except Exception:
                pass
            model.geom_friction[gid][0] = slide
            #model.geom_friction[gid][1] = spin
            #model.geom_friction[gid][2] = roll
    
    # ---- Actuator gain: kv for the club velocity motor ----
    #kv = trial.suggest_float(
    #    "actuator.club_motor.kv",
    #    base_kv * 0.0001,
    #    base_kv * 1.1,
    #    log=True,
    #)
    #act_id = ids["act_id"]
    #model.actuator_gainprm[act_id][0] = kv
    
    #hinge_jid = ids["hinge_jid"]
    
    # HINGE STUFF:
    #hinge_dof = model.jnt_dofadr[hinge_jid]

    #base_damp = 0.01
    #damp = trial.suggest_float(
    #    "joint.club_hinge.damping",
    #    base_damp * 0.001,
    #    base_damp * 100.0,
    #    log=True,
    #)
    #model.dof_damping[hinge_dof] = damp
    
    #hinge_qadr = model.jnt_qposadr[hinge_jid]
    #base_stiff = 5

    #stiff = trial.suggest_float(
    #    "joint.club_hinge.stiffness",
    #    base_stiff * 0.01,
    #    base_stiff * 100.0,
    #    log=True,
    #)

    #model.qpos_spring[hinge_qadr] = stiff
    
    #speed_scale = trial.suggest_float(
    #    "controller.speed_scale",
    #    0.5,     # lower bound (tune as needed)
    #    2.0,     # upper bound (tune as needed)
    #    log=True
    #)

    # Store in model for later (safe hack: use model.user_data or similar)
    #model.userdata = np.array([speed_scale], float)
    
    
    # ------------------------------------------------------------------
    # NEW: flex hinge dynamics as hyperparameters
    # ------------------------------------------------------------------
    #flex_jid = ids["flex_jid"]
    #flex_dof = model.jnt_dofadr[flex_jid]

    # Baseline from compiled model (fallbacks for safety)
    #base_flex_damp = 0.1

    # Damping (log-scale around baseline)
    #flex_damp = trial.suggest_float(
    #    "joint.shaft_flex.damping",
    #    base_flex_damp * 0.001,
    #    base_flex_damp * 1000.0,
    #    log=True,
    #)
    #model.dof_damping[flex_dof] = flex_damp

    # Stiffness (log-scale around baseline)
    # If baseline is zero, use a nominal range [1, 1000].

    #base_flex_stiff = 10.0
    #flex_stiff = trial.suggest_float(
    #    "joint.shaft_flex.stiffness",
    #    base_flex_stiff*0.01,
    #    base_flex_stiff*100.0,
    #    log=True,
    #)

    # Apply stiffness back to the joint (if available)
    #try:
    #    model.jnt_stiffness[flex_jid] = flex_stiff
    #except AttributeError:
        # If this field doesn't exist in your mujoco bindings, you may need a
        # different mechanism (e.g. qpos_spring, equality constraints, etc.)
    #    pass


# ------------------- Dense rollout + interpolation --------------------
def simulate_dense_trajectory(model, ids,
                              vx_des: float,
                              aim_yaw_deg: float,
                              start_angle_deg: float,
                              sim_timestep: float,
                              safe_substep: float,
                              T_end: float,
                              sample_dt: float):
    """
    One rollout on a uniform dense grid [0, T_end] with spacing sample_dt.
    Returns (t_dense, x_dense, vx_dense) all same length.

    mount_free is welded (like in run_sim_forever.py): its pose is reset and
    its velocity zeroed every step, so the base does not move.
    """
    data = mujoco.MjData(model)

    # Use sim_timestep directly, one physics step per call
    model.opt.timestep = float(sim_timestep)
    nstep = 1  # kept as a variable in case you want to experiment later

    # Initial conditions
    reset_ball_state(model, data)
    bake_aim_pose(model, data, yaw_rad=math.radians(aim_yaw_deg))
    reset_club_pose(model, data, start_deg=start_angle_deg)

    # Cache baked base pose (mount_free) to weld it
    base_jid = ids["base_jid"]
    qadr_base = model.jnt_qposadr[base_jid]
    vadr_base = model.jnt_dofadr[base_jid]
    base_pose = np.array(data.qpos[qadr_base:qadr_base+7], dtype=float)
    # velocities for a welded base are always zero; we just overwrite with zeros each step

    act_id = ids["act_id"]
    lo, hi = model.actuator_ctrlrange[act_id]
    max_vel = hi

    N = int(math.floor(T_end / sample_dt)) + 1
    t_dense = np.linspace(0.0, (N - 1) * sample_dt, N, dtype=float)
    x_dense = np.zeros(N, dtype=float)
    vx_dense = np.zeros(N, dtype=float)

    # initial sample
    x_dense[0] = float(data.geom_xpos[ids["ball_gid"]][0])

    next_sample_t = sample_dt
    i = 1
    prev_x = x_dense[0]
    prev_t = float(data.time)

    while i < N:
        # advance until >= next_sample_t
        while float(data.time) < next_sample_t:
            # Weld the base: keep mount_free fixed like in run_sim_forever.py
            data.qpos[qadr_base:qadr_base+7] = base_pose
            data.qvel[vadr_base:vadr_base+6] = 0.0

            qd_cmd = command_impact_speed(model, data, ids, vx_des=vx_des)
            qd_cmd = float(np.clip(qd_cmd, -max_vel, max_vel))
            qd_cmd = float(np.clip(qd_cmd, lo, hi))
            data.ctrl[act_id] = qd_cmd

            mujoco.mj_step(model, data, nstep=nstep)

        cur_t = float(data.time)
        cur_x = float(data.geom_xpos[ids["ball_gid"]][0])
        # simple hold is OK with small sample_dt; use linear interp for better accuracy
        if cur_t == prev_t:
            x_samp = cur_x
        else:
            alpha = (next_sample_t - prev_t) / (cur_t - prev_t)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            x_samp = (1.0 - alpha) * prev_x + alpha * cur_x
        x_dense[i] = x_samp
        prev_t, prev_x = cur_t, cur_x
        i += 1
        next_sample_t += sample_dt

    if N > 1:
        dt = np.diff(t_dense)
        vx_dense[1:] = np.diff(x_dense) / (dt + 1e-12)
        vx_dense[0] = vx_dense[1]

    return t_dense, x_dense, vx_dense


def sample_from_dense(t_dense: np.ndarray, x_dense: np.ndarray, vx_dense: np.ndarray, t_query: np.ndarray):
    """Interpolate x and vx from dense uniform trajectory at arbitrary times."""
    xq  = np.interp(t_query, t_dense, x_dense, left=x_dense[0], right=x_dense[-1])
    vxq = np.interp(t_query, t_dense, vx_dense, left=vx_dense[0], right=vx_dense[-1])
    return xq, vxq


# ---------------------- Alignment (on dense rollout) ------------------
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def best_time_shift_rmse_dense(t_dense, x_dense, vx_dense,
                               m, x_w, vx_w,
                               win, step, refine_step):
    # Coarse search
    best = (float("inf"), 0.0)
    dt = -win
    t_meas = m["t"]
    x_gt, vx_gt = m["x"], m["vx"]
    while dt <= win + 1e-12:
        tq = t_meas + dt
        xs, vxs = sample_from_dense(t_dense, x_dense, vx_dense, tq)
        cost = x_w * rmse(xs, x_gt) + vx_w * rmse(vxs, vx_gt)
        if cost < best[0]:
            best = (cost, dt)
        dt += step

    # Refine
    center = best[1]
    dt = center - 2*refine_step
    hi = center + 2*refine_step
    while dt <= hi + 1e-12:
        tq = t_meas + dt
        xs, vxs = sample_from_dense(t_dense, x_dense, vx_dense, tq)
        cost = x_w * rmse(xs, x_gt) + vx_w * rmse(vxs, vx_gt)
        if cost < best[0]:
            best = (cost, dt)
        dt += refine_step

    # Evaluate at best
    tq = t_meas + best[1]
    xs, vxs = sample_from_dense(t_dense, x_dense, vx_dense, tq)
    final_cost = x_w * rmse(xs, x_gt) + vx_w * rmse(vxs, vx_gt)
    return final_cost, best[1], xs, vxs


def best_time_shift_ncc_dense(t_dense, x_dense, vx_dense,
                              m, x_w, vx_w,
                              win, step, refine_step):
    # NCC on vx (z-scored)
    vx_gt = m["vx"]
    vx_gt_z = (vx_gt - np.mean(vx_gt)) / (np.std(vx_gt) + 1e-12)

    best_dt = 0.0
    best_ncc = -1e9
    t_meas = m["t"]

    dt = -win
    while dt <= win + 1e-12:
        tq = t_meas + dt
        _, vxs = sample_from_dense(t_dense, x_dense, vx_dense, tq)
        vxs_z = (vxs - np.mean(vxs)) / (np.std(vxs) + 1e-12)
        ncc = float(np.mean(vxs_z * vx_gt_z))  # same length by construction
        if ncc > best_ncc:
            best_ncc = ncc
            best_dt = dt
        dt += step

    # Refine around best_dt
    dt = best_dt - 2*refine_step
    hi = best_dt + 2*refine_step
    while dt <= hi + 1e-12:
        tq = t_meas + dt
        _, vxs = sample_from_dense(t_dense, x_dense, vx_dense, tq)
        vxs_z = (vxs - np.mean(vxs)) / (np.std(vxs) + 1e-12)
        ncc = float(np.mean(vxs_z * vx_gt_z))
        if ncc > best_ncc:
            best_ncc = ncc
            best_dt = dt
        dt += refine_step

    # Evaluate at best_dt (for output trajectories)
    tq = t_meas + best_dt
    xs, vxs = sample_from_dense(t_dense, x_dense, vx_dense, tq)

    # Use NCC itself as the "score" (higher is better).
    # We will tell Optuna to MAXIMIZE this when align_metric == "ncc".
    cost = best_ncc
    return cost, best_dt, xs, vxs



# --------------------------- Objective (X only) -----------------------
def make_objective(model, ids,
                   measurements: List[Dict[str, np.ndarray]],
                   aim_yaw_deg: float, start_angle_deg: float,
                   sim_timestep: float, safe_substep: float,
                   x_w: float, vx_w: float,
                   align_window: float, align_step: float, align_refine: float,
                   align_metric: str):
    # Group CSVs by desired vx to simulate once per group
    meas_by_vx = defaultdict(list)
    for m in measurements:
        meas_by_vx[float(m["vx_des"])].append(m)

    # How long to simulate per group: max t + window + small margin
    Tmax_by_vx = {vx: max(np.max(m["t"]) for m in lst) + align_window + 0.05
                  for vx, lst in meas_by_vx.items()}

    # Dense sample spacing (same as sim timestep is fine)
    dense_dt = sim_timestep

    def objective(trial: optuna.Trial) -> float:
        # Mutate ONLY the requested params
        apply_params_inplace_only_requested(model, ids, trial)

        power_scale = trial.suggest_float(
            "controller.power_scale",
            0.3,
            1.7,
        )

        v_ref = 0.8 # the speed where you want actual = command (at this speed, no scaling)

        total_cost = 0.0
        total_n = 0

        # For each unique desired velocity, simulate once
        for vx_val, lst in meas_by_vx.items():
            vx_cmd = v_ref * (vx_val / v_ref) ** power_scale
            t_dense, x_dense, vx_dense = simulate_dense_trajectory(
                model, ids,
                vx_des=vx_cmd,
                aim_yaw_deg=aim_yaw_deg,
                start_angle_deg=start_angle_deg,
                sim_timestep=sim_timestep,
                safe_substep=safe_substep,
                T_end=Tmax_by_vx[vx_val],
                sample_dt=dense_dt
            )

            # Evaluate all CSVs for this vx via alignment on the dense traj
            for m in lst:
                if align_metric == "ncc":
                    best_cost, best_dt, _, _ = best_time_shift_ncc_dense(
                        t_dense, x_dense, vx_dense,
                        m, x_w, vx_w,
                        align_window, align_step, align_refine
                    )
                else:
                    best_cost, best_dt, _, _ = best_time_shift_rmse_dense(
                        t_dense, x_dense, vx_dense,
                        m, x_w, vx_w,
                        align_window, align_step, align_refine
                    )
                total_cost += best_cost * len(m["t"])
                total_n    += len(m["t"])

        return total_cost / max(total_n, 1)

    return objective


# -------------------------------- CLI ---------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="Tune ONLY ball–club pair solref/solimp and default geom friction, fitting x & vx with fast alignment."
    )
    p.add_argument("--xml", type=str, default=XML_PATH_DEFAULT, help="Path to MJCF XML.")
    p.add_argument(
        "--csvvx",
        type=str,
        action="append",
        required=False,
        help='Repeat as "PATH|VX_DES". Example: --csvvx "C:\\path\\file.csv|0.8"',
    )
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--jobs", type=int, default=1, help="Keep 1 (shared model).")
    p.add_argument("--sampler", type=str, default="tpe",
                   choices=["tpe","sobol","random","cmaes"])
    p.add_argument("--sampler-seed", type=int, default=42)
    p.add_argument("--no-pruner", action="store_true")
    # controller/sim
    p.add_argument("--aim-yaw", type=float, default=0.0)
    p.add_argument("--club-start-angle", type=float, default=40.0)
    p.add_argument("--sim-timestep", type=float, default=0.002)   # matches your XML
    p.add_argument("--safe-substep", type=float, default=5e-4)    # kept for API, not really used now
    # weights
    p.add_argument("--x-weight", type=float, default=1.0)
    p.add_argument("--vx-weight", type=float, default=0.5)
    # alignment
    p.add_argument("--align-window", type=float, default=4.0, help="Max |Δt| shift (s).")
    p.add_argument("--align-step", type=float, default=0.01, help="Coarse Δt step (s).")
    p.add_argument("--align-refine", type=float, default=0.002, help="Refine Δt step (s).")
    p.add_argument("--align-metric", type=str, default="ncc", choices=["rmse","ncc"],
                   help="Pick dt by min RMSE or max NCC on vx.")
    # Optuna storage for dashboard
    p.add_argument("--storage", type=str,
                   default=f"sqlite:///{DEFAULT_DB_PATH}",
                   help='Optuna storage URL (e.g., sqlite:///C:/path/optuna.db)')
    p.add_argument("--study-name", type=str, default="golf-calib",
                   help="Optuna study name for dashboard/resume.")
    p.add_argument(
    "--auto-dir",
    type=str,
    default=None,
    help=(
        "Directory containing files named like "
        "test_<vx>-<id>_trajectory*.csv. The <vx> part "
        "is parsed as vx_des for each file."
    ),)
    return p


def make_sampler(name: str, seed: int):
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed, n_startup_trials=64, multivariate=True)
    if name == "sobol":
        return optuna.samplers.SobolSampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed, restart_strategy="ipop")
    raise ValueError("Unknown sampler")


def main():
    args = build_argparser().parse_args()

    # Load measurements with per-file vx_des
    # Load measurements (either from --auto-dir or explicit --csvvx)
    if args.auto_dir:
        measurements = load_measurements_from_dir(args.auto_dir)
    elif args.csvvx:
        measurements = load_measurements_with_vx(args.csvvx)
    else:
        raise ValueError("You must provide either --auto-dir or at least one --csvvx.")


    # Build model once from PATH so relative assets resolve
    xml_abs = os.path.abspath(args.xml)
    xml_dir = os.path.dirname(xml_abs)
    with pushd(xml_dir):
        model = mujoco.MjModel.from_xml_path(xml_abs)
    ids = get_ids(model)
    if ids["ball_gid"] == -1 or ids["club_head_gid"] == -1:
        raise RuntimeError("Could not find ball_geom or club_head by name in the model.")

    # Optuna
    sampler = make_sampler(args.sampler, args.sampler_seed)
    pruner = None if args.no_pruner else optuna.pruners.MedianPruner(n_startup_trials=10)
    study = optuna.create_study(
        direction="maximize" if args.align_metric == "ncc" else "minimize",
        sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(),
        pruner=pruner,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    objective = make_objective(
        model, ids, measurements,
        aim_yaw_deg=args.aim_yaw,
        start_angle_deg=args.club_start_angle,
        sim_timestep=args.sim_timestep,
        safe_substep=args.safe_substep,
        x_w=args.x_weight, vx_w=args.vx_weight,
        align_window=args.align_window,
        align_step=args.align_step,
        align_refine=args.align_refine,
        align_metric=args.align_metric
    )

    print(f"[info] starting calibration on X & VX …")
    print(f"[info] Optuna storage: {args.storage}")
    print(f"[info] Study name    : {args.study_name}")
    os.environ.setdefault("MJ_THREADS", "1")  # one thread per process
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True, n_jobs=args.jobs)

    # Report best params (paste these into the XML)
    print("\nBest trial value:", study.best_value)
    params = study.best_trial.params

    tc = params["pair.solref.tc"]; dr = params["pair.solref.dr"]
    a  = params["pair.solimp.a"];  b  = params["pair.solimp.b"];  c = params["pair.solimp.c"]
    slide = params["default.geom.friction.slide"]
    spin  = params["default.geom.friction.spin"]
    roll  = params["default.geom.friction.roll"]

    print("\n=== Paste these into your XML ===")
    print('  <contact>')
    print('    <pair geom1="ball_geom" geom2="club_head" '
          f'solref="{tc:.6g} {dr:.6g}" solimp="{a:.6g} {b:.6g} {c:.6g}"/>')
    print('  </contact>')
    print('\n  <default>')
    print('    <geom condim="6" solref="0.002 1" solimp="0.95 0.99 0.001" '
          f'friction="{slide:.6g} {spin:.6g} {roll:.6g}"/>')
    print('  </default>')

    print("\nAll params:")
    for k in sorted(params):
        print(f"  {k}: {params[k]}")

    print("\nTo open the Optuna Dashboard:")
    print(f'  optuna-dashboard "{args.storage}" --study-name "{args.study_name}"')
    print("If needed: pip install optuna-dashboard")


if __name__ == "__main__":
    main()
