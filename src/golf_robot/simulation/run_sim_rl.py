import os
import math
import uuid
import numpy as np
import mujoco
from mujoco import viewer
from contextlib import nullcontext
import csv
import yaml
from pathlib import Path
import argparse


#python run_sim.py --aim-yaw 0.0 --vx 1.0 --sim-timestep 0.001 --csv "C:\Users\marti\OneDrive - Danmarks Tekniske Universitet\DTU\Mini-golf robot Master\Physics Comparison using Detection\log_0\ball_log.csv"        
# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
XML_PATH = os.path.join(REPO, "models", "mujoco", "golf_world.xml")
XML_PATH_NEW = os.path.join(REPO, "models", "mujoco", "golf_world_new.xml")


def set_model_options_from_cfg(model, cfg):
    """Set MuJoCo model options from cfg dictionary."""
    ts = float(cfg["sim"].get("timestep", model.opt.timestep))
    model.opt.timestep = ts


def get_ids(model):
    """Get MuJoCo model IDs for key objects."""

    return {
        "club_head_gid": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,  "club_head"),
        "ball_gid":      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,  "ball_geom"),
        "hole_sid":      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,  "hole_center"),
        "hinge_jid":     mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "club_hinge"),
        "act_id":        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "club_motor"),
        "base_jid":      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mount_free"),
        "base_bid":      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "club_mount"),
        "ball_jid":      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"),
    }


def move_hole(model, hole_xy):
    hx, hy = hole_xy
    cup_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    assert cup_bid != -1, "cup body not found"
    cup_pos = model.body_pos[cup_bid].copy()
    cup_pos[0] = hx
    cup_pos[1] = hy
    model.body_pos[cup_bid] = cup_pos


def move_ground_cutout(model, hole_xy, x_min=-1.0, x_max=7.0, y_min=-2.0, y_max=2.0, hole_half=0.053):
    hx, hy = hole_xy

    g_plus_y    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "green_plus_y")
    g_neg_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "green_neg_y")
    g_plus_x   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "green_plus_x")
    g_neg_x  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "green_neg_x")

    if min(g_plus_y, g_neg_y, g_plus_x, g_neg_x) == -1:
        raise RuntimeError("One or more ground cutout geoms not found")

    plus_y_min_y    = hy + hole_half
    plus_y_max_y    = y_max
    plus_y_center_y = 0.5 * (plus_y_min_y + plus_y_max_y)
    plus_y_min_x    = x_min
    plus_y_max_x    = x_max
    plus_y_center_x = 0.5 * (plus_y_min_x + plus_y_max_x)
    plus_y_size_y   = 0.5 * (plus_y_max_y - plus_y_min_y)
    plus_y_size_x   = 0.5 * (plus_y_max_x - plus_y_min_x)

    model.geom_pos[g_plus_y][0]  = plus_y_center_x
    model.geom_pos[g_plus_y][1]  = plus_y_center_y
    model.geom_size[g_plus_y][0] = plus_y_size_x
    model.geom_size[g_plus_y][1] = plus_y_size_y

    neg_y_min_y    = y_min
    neg_y_max_y    = hy - hole_half
    neg_y_center_y = 0.5 * (neg_y_min_y + neg_y_max_y)
    neg_y_min_x    = x_min
    neg_y_max_x    = x_max
    neg_y_center_x = 0.5 * (neg_y_min_x + neg_y_max_x)
    neg_y_size_y   = 0.5 * (neg_y_max_y - neg_y_min_y)
    neg_y_size_x   = 0.5 * (neg_y_max_x - neg_y_min_x)

    model.geom_pos[g_neg_y][0]  = neg_y_center_x
    model.geom_pos[g_neg_y][1]  = neg_y_center_y
    model.geom_size[g_neg_y][0] = neg_y_size_x
    model.geom_size[g_neg_y][1] = neg_y_size_y

    plus_x_min_x    = hx + hole_half
    plus_x_max_x    = x_max
    plus_x_center_x = 0.5 * (plus_x_min_x + plus_x_max_x)
    plus_x_min_y    = neg_y_max_y
    plus_x_max_y    = plus_y_min_y
    plus_x_center_y = 0.5 * (plus_x_min_y + plus_x_max_y)
    plus_x_size_x   = 0.5 * (plus_x_max_x - plus_x_min_x)
    plus_x_size_y   = 0.5 * (plus_x_max_y - plus_x_min_y)

    model.geom_pos[g_plus_x][0]  = plus_x_center_x
    model.geom_pos[g_plus_x][1]  = plus_x_center_y
    model.geom_size[g_plus_x][0] = plus_x_size_x
    model.geom_size[g_plus_x][1] = plus_x_size_y

    neg_x_min_x    = x_min
    neg_x_max_x    = hx - hole_half
    neg_x_center_x = 0.5 * (neg_x_min_x + neg_x_max_x)
    neg_x_min_y    = neg_y_max_y
    neg_x_max_y    = plus_y_min_y
    neg_x_center_y = 0.5 * (neg_x_min_y + neg_x_max_y)
    neg_x_size_x   = 0.5 * (neg_x_max_x - neg_x_min_x)
    neg_x_size_y   = 0.5 * (neg_x_max_y - neg_x_min_y)

    model.geom_pos[g_neg_x][0]  = neg_x_center_x
    model.geom_pos[g_neg_x][1]  = neg_x_center_y
    model.geom_size[g_neg_x][0] = neg_x_size_x
    model.geom_size[g_neg_x][1] = neg_x_size_y


def move_discs(model, disc_xy_list):
    for i, (dx, dy) in enumerate(disc_xy_list):
        disc_name = f"disc{i}"
        disc_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, disc_name)
        x, y = disc_xy_list[i]
        model.body_pos[disc_bid] = np.array([x, y, 0.0], float)


def generate_disc_positions(max_num_discs, x_min, x_max, y_min, y_max, hole_xy):
    hole_x, hole_y = hole_xy
    num_discs = np.random.randint(0, max_num_discs + 1)
    # print(f"Generating {num_discs} disc positions.")
    disc_positions = []
    min_dist_from_objects = 0.25 

    x_lo = x_min + min_dist_from_objects
    x_hi = x_max - min_dist_from_objects
    y_lo = y_min + min_dist_from_objects
    y_hi = y_max - min_dist_from_objects

    max_tries_per_disc = 100

    for _ in range(num_discs):
        placed = False
        for _ in range(max_tries_per_disc):
            x = np.random.uniform(x_lo, x_hi)
            y = np.random.uniform(y_lo, y_hi)

            if np.hypot(x - hole_x, y - hole_y) < min_dist_from_objects:
                continue
            too_close = False

            for (dx, dy) in disc_positions:
                if np.hypot(x - dx, y - dy) < min_dist_from_objects:
                    too_close = True
                    break

            if too_close:
                continue

            disc_positions.append((x, y))
            placed = True
            break
        if not placed:
            raise RuntimeError("Could not place all discs without overlap.")
        
    return disc_positions

def quat_yaw_deg(q):
    """
    Given a quaternion [w,x,y,z], return the yaw (rotation about +Z) in degrees.

    Input:
        q: Quaternion as iterable of 4 floats [w,x,y,z]

    Output: 
        Yaw angle in degrees (float)
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))  # ZYX yaw
    return math.degrees(yaw)


def assert_velocity_actuator(model, ids):
    aid, jid = ids["act_id"], ids["hinge_jid"]
    if aid == -1 or jid == -1:
        raise ValueError("Missing actuator or hinge joint.")
    if model.actuator_trnid[aid][0] != jid:
        raise RuntimeError("'club_motor' must target 'club_hinge'.")
    lo, hi = model.actuator_ctrlrange[aid]
    if not np.isfinite([lo, hi]).all():
        raise RuntimeError("'club_motor' must have finite ctrlrange.")
    kv = float(model.actuator_gainprm[aid][0]) if model.actuator_gainprm[aid][0] != 0 else None
    aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    msg = f"[actuator] {aname} → '{jname}', range=({lo:.2f},{hi:.2f})"
    if kv:
        msg += f", kv={kv:g}"
    # print(msg)

# ---------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------
def reset_ball_state(model, data, cfg, ids):
    jid = ids["ball_jid"]
    if jid == -1:
        return
    qadr = model.jnt_qposadr[jid]
    vadr = model.jnt_dofadr[jid]
    pos = cfg.get("ball", {}).get("start_pos", [0.0, 0.0, 0.02135])
    quat = (1.0, 0.0, 0.0, 0.0)
    data.qpos[qadr:qadr+7] = np.array([*pos, *quat], float)
    data.qvel[vadr:vadr+6] = 0


def is_ball_in_hole(data, cfg, ids):
    gid = ids["ball_gid"]; sid = ids["hole_sid"]
    if gid == -1 or sid == -1:
        return False
    ball_p = np.array(data.geom_xpos[gid])
    hole_p = np.array(data.site_xpos[sid])
    r = float(cfg.get("hole", {}).get("radius", 0.055))
    return float(np.linalg.norm(ball_p - hole_p)) < r


def reset_club_pose(model, data, ids, start_deg):
    jid = ids["hinge_jid"]
    if jid == -1:
        return
    qadr = model.jnt_qposadr[jid]
    dof  = model.jnt_dofadr[jid]
    lo_rad, hi_rad = model.jnt_range[jid]
    q0 = float(np.clip(np.deg2rad(start_deg), lo_rad, hi_rad))
    data.qpos[qadr] = q0
    data.qvel[dof]  = 0
    mujoco.mj_forward(model, data)

def bake_aim_pose(model, data, base_free="mount_free", ball_xyz=(0,0,0.02135), yaw_rad=0.0):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, base_free)
    assert jid != -1
    qadr = model.jnt_qposadr[jid]; vadr = model.jnt_dofadr[jid]
    half = 0.5 * yaw_rad
    quat = np.array([math.cos(half), 0.0, 0.0, math.sin(half)], float)
    data.qpos[qadr:qadr+7] = np.array([*ball_xyz, *quat], float)
    data.qvel[vadr:vadr+6] = 0.0
    mujoco.mj_forward(model, data)
    return (qadr, vadr, np.array(ball_xyz, float), quat)

# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def head_ball_dx(data, ids):
    return float(data.geom_xpos[ids["club_head_gid"]][0] - data.geom_xpos[ids["ball_gid"]][0])

def sign_crossed(prev, now):
    return prev is not None and ((prev > 0 and now < 0) or (prev < 0 and now > 0) or now == 0.0)

def club_head_vx(model, data, ids):
    jid = ids["hinge_jid"]; gid = ids["club_head_gid"]
    if gid == -1 or jid == -1:
        return 0.0, 0.0
    r   = np.array(data.geom_xpos[gid]) - np.array(data.xanchor[jid])
    den = float(np.cross(np.array(data.xaxis[jid]), r)[0])
    qd  = float(data.qvel[model.jnt_dofadr[jid]])
    vx  = qd * den
    return vx, den

# ---------------------------------------------------------------------
# Command mapping
# ---------------------------------------------------------------------
def command_impact_speed(data, ids, aim_yaw_rad, v_des=1.0):
    """
    Map desired club head speed along the 2D aim direction to hinge velocity.
    aim_yaw_rad: aim yaw angle in radians (world-frame)
    v_des: desired linear speed along aim direction (m/s)
    """
    jid = ids["hinge_jid"]
    gid = ids["club_head_gid"]
    ball_gid = ids["ball_gid"]

    if gid == -1 or jid == -1 or ball_gid == -1:
        return 0.0

    # Vector from hinge anchor to club head
    r = np.array(data.geom_xpos[gid]) - np.array(data.xanchor[jid])

    # Hinge axis in world frame
    axis = np.array(data.xaxis[jid])

    # Linear velocity for qd = 1 rad/s: v = axis × r
    Jv = np.cross(axis, r)  # 3D

    # 2D aim direction in world frame
    d = np.array([math.cos(aim_yaw_rad), math.sin(aim_yaw_rad), 0.0], float)

    # Component of Jv along aim direction
    comp = float(np.dot(Jv, d))
    if abs(comp) < 1e-8:
        # Geometry degenerate, no way to create velocity along d
        return 0.0

    speed_des = abs(v_des)

    # magnitude of qdot needed
    qd_mag = speed_des / abs(comp)

    # We want v_parallel = sign(v_des)*speed_des
    # But v_parallel = qdot * comp  =>  sign(qdot) = sign(v_des) / sign(comp)
    sign_v = math.copysign(1.0, v_des if v_des != 0.0 else 1.0)
    sign_comp = math.copysign(1.0, comp)
    sign_qd = sign_v / sign_comp

    qd_cmd = sign_qd * qd_mag
    return qd_cmd


# ---------------------------------------------------------------------
# Viewer wrapper
# ---------------------------------------------------------------------
class NullViewer:
    def __init__(self, data, max_duration_sec=5.0):
        self._running = True
        self._end_time = float(data.time) + float(max_duration_sec)
        self._lock = nullcontext(); self._data = data
    def __enter__(self): return self
    def __exit__(self, *a): self._running = False
    def is_running(self): return self._running and (float(self._data.time) <= self._end_time)
    def sync(self):
        if float(self._data.time) > self._end_time:
            self._running = False
    def lock(self): return self._lock

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def run_loop(model, data, cfg,
             base_idx, base_pose,
             ids, nstep, max_vel, vx_des, aim_yaw_rad):
    """
    Run the main simulation loop with viewer or headless.
    inputs:
        model, data: MuJoCo model and data
        cfg: Configuration dictionary
        base_idx: Tuple of (qpos_addr, qvel_addr) for the base free joint
        base_pose: Tuple of (pos, quat) for the base joint to hold each step
        ids: Dictionary of MuJoCo IDs for key objects
        nstep: Number of substeps per mj_step
        max_vel: Maximum actuator target velocity (rad/s)
        vx_des: Desired club head X velocity at impact (m/s)
        start_immediately: If True, reset and swing immediately on start
    
    """
    trajectory = []
    act_id = ids['act_id']; hinge_jid = ids['hinge_jid']
    qadr = model.jnt_qposadr[hinge_jid]
    qadr_base, vadr_base = base_idx
    base_pos_baked, base_quat_baked = base_pose
    # base_jid = ids["base_jid"]  # for telemetry
    pending_actions = []; 
    dx_prev = None

    # if stop_sim_at_still is requested, track last ball movement:
    last_xy = None
    last_move_t = None
    started = False

    # ---- CSV setup ----------------------------------------------------
    csv_path = cfg["sim"].get("csv_path", None)
    csv_period = float(cfg["sim"]["csv_period"])
    csv_file = None
    csv_writer = None
    next_csv_t = 0.0
    if csv_path:
        try:
            dir_ = os.path.dirname(csv_path)
            if dir_:
                os.makedirs(dir_, exist_ok=True)
            csv_file = open(csv_path, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["time", "ball_x", "ball_y", "ball_z", "in_hole"])
        except Exception as e:
            print(f"[warn] Could not open CSV '{csv_path}': {e}")
            csv_file = None
            csv_writer = None
    # ------------------------------------------------------------------

    def key_callback(keycode):
        if keycode in (ord('r'), ord('R')):
            pending_actions.append("reset_and_swing")
        if keycode in (ord('q'), ord('Q'), 27):  # ESC
            pending_actions.append("breaking")

    viewer_ctx = (viewer.launch_passive(model, data, key_callback=key_callback)
                  if cfg["sim"].get("render", True)
                  else NullViewer(data, max_duration_sec=cfg["sim"].get("max_duration_sec", 5.0)))

    if cfg["sim"]["start_immediately"]:
        pending_actions.append("reset_and_swing")

    next_print_t = 0.0; 
    print_period = cfg["sim"].get("print_period", 0.1)
    do_print = cfg["sim"]["do_print"]

    # try:
    with viewer_ctx as v:
        while v.is_running():
            while pending_actions:
                act = pending_actions.pop(0)
                if act == "reset_and_swing":
                    with v.lock():
                        reset_ball_state(model, data, cfg, ids)
                        reset_club_pose(model, data, ids, cfg["club"].get("start_angle_deg", 40.0))
                        mujoco.mj_forward(model, data)
                    if do_print:
                        print(f"[{data.time:7.3f}s] RESET + SWING")

                if act == "breaking":
                    if do_print:
                        print(f"[{data.time:7.3f}s] BREAKING")
                    return

            data.qpos[qadr_base:qadr_base+3] = base_pos_baked
            data.qpos[qadr_base+3:qadr_base+7] = base_quat_baked
            data.qvel[vadr_base:vadr_base+6] = 0.0

            qd_cmd = command_impact_speed(data, ids, aim_yaw_rad=aim_yaw_rad, v_des=vx_des)
            lo, hi = model.actuator_ctrlrange[act_id]
            qd_cmd = float(np.clip(qd_cmd, -max_vel, max_vel))
            qd_cmd = float(np.clip(qd_cmd, lo, hi))
            data.ctrl[act_id] = qd_cmd
            mujoco.mj_step(model, data, nstep=nstep)

            dx = head_ball_dx(data, ids)
            if sign_crossed(dx_prev, dx):
                vx_now, _ = club_head_vx(model, data, ids)
                if do_print:
                    print(f"IMPACT @ t={data.time:7.3f}s | Vx_des={vx_des:+.3f} m/s | Vx_hit={vx_now:+.3f} m/s")
            dx_prev = dx

            # telemetry ---------------------------------------------------
            ball_p = data.geom_xpos[ids["ball_gid"]]
            t = float(data.time)

            trajectory.append( ( t, float(ball_p[0]), float(ball_p[1]) ) )

            if t >= next_print_t:
                vx_now, _ = club_head_vx(model, data, ids)
                q_deg = math.degrees(float(data.qpos[qadr]))
                base_bid = ids["base_bid"]
                quat = data.xquat[base_bid]
                yaw_deg = quat_yaw_deg(quat)
                if do_print:
                    print(
                        f"Time: {t:7.3f}s | "
                        f"q: {q_deg:7.2f}° | "
                        f"qd_cmd: {qd_cmd:7.3f} rad/s | "
                        f"Vx_des: {vx_des:+6.3f} | "
                        f"Vx_now: {vx_now:+6.3f} | "
                        f"Yaw: {yaw_deg:+6.2f}° | "
                        f"Ball: ({ball_p[0]:+.3f}, {ball_p[1]:+.3f}, {ball_p[2]:+.3f})"
                    )
                if is_ball_in_hole(data, cfg, ids):
                    if do_print:
                        print(f"[{data.time:6.3f}s] Ball within hole radius — physics should drop it.")

                next_print_t += print_period

            # Stop sim if ball has been still for 0.2s ----------------------
            if cfg["sim"]["stop_sim_at_still"]:
                xy = np.array(ball_p[:2])
                z = ball_p[2]
                if last_xy is not None:
                    dist = np.linalg.norm(xy - last_xy)

                    if dist > 1e-4:
                        # ball has moved
                        last_xy = xy.copy()
                        last_move_t = t

                        if not started:
                            started = True

                    elif started and ((t - last_move_t) >= 0.05):
                        if do_print:
                            print(f"[{t:7.3f}s] Ball has been still for 0.05s, ending simulation.")
                        
                        # print(f"Returning state: {ball_p[0]}, {ball_p[1]}, {is_ball_in_hole(data, cfg, ids)}")
                        return ball_p[0], ball_p[1], is_ball_in_hole(data, cfg, ids), np.array(trajectory)
                      
                else:
                    last_xy = xy.copy()
                    last_move_t = t

                if z < -0.2:
                    if do_print:
                        print(f"[{t:7.3f}s] Ball has fallen below z=-0.2m, ending simulation.")
                    
                    # print(f"Returning state: {ball_p[0]}, {ball_p[1]}, {is_ball_in_hole(data, cfg, ids)}")
                    return ball_p[0], ball_p[1], is_ball_in_hole(data, cfg, ids), np.array(trajectory)
                   
                
                if is_ball_in_hole(data, cfg, ids):
                    if do_print:
                        print(f"[{t:7.3f}s] Ball is in the hole, ending simulation.")
                    csv_writer.writerow([f"{t:.6f}", f"{ball_p[0]:.9f}", f"{ball_p[1]:.9f}", f"{ball_p[2]:.9f}", int(True)])
                    
                    return ball_p[0], ball_p[1], True, np.array(trajectory)
               
                


            # ---- CSV logging at requested rate -------------------------
            if csv_writer is not None and t >= next_csv_t:
                bp = data.geom_xpos[ids["ball_gid"]]
                csv_writer.writerow([f"{t:.6f}", f"{bp[0]:.9f}", f"{bp[1]:.9f}", f"{bp[2]:.9f}", int(is_ball_in_hole(data, cfg, ids))])
                # optional flush for safety during long runs:
                # csv_file.flush()
                next_csv_t += csv_period
            # ------------------------------------------------------------
            

            v.sync()
    print("Viewer closed, ending simulation.")
    print(f"Returning state: {ball_p[0]}, {ball_p[1]}, {is_ball_in_hole(data, cfg, ids)}")
    print(f"Failing simulation return without ball stillness detection for: vx_des={vx_des}, aim_yaw_rad={aim_yaw_rad}")
    print(f"Mujoco cfg: cfg={cfg}")

def debug_print_green_spans(model):
    names = ["green_top", "green_bottom", "green_left", "green_right"]
    for name in names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        pos = model.geom_pos[gid]
        size = model.geom_size[gid]
        x_min, x_max = pos[0] - size[0], pos[0] + size[0]
        y_min, y_max = pos[1] - size[1], pos[1] + size[1]
        print(
            f"{name}: x[{x_min:+.3f}, {x_max:+.3f}], "
            f"y[{y_min:+.3f}, {y_max:+.3f}]"
        )
# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def run_sim(aim_yaw_deg, vx_des, hole_pos_xy, cfg, disc_positions=None):
    """
    Run the golf robot MuJoCo simulation with given parameters.
    inputs:
        aim_yaw_deg: Aiming yaw in degrees
        vx_des: Desired club head X velocity at impact (m/s)
        cfg: Configuration dictionary
        start: If True, reset and swing immediately on start
    """
    # print(disc_positions)
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    move_hole(model, hole_xy=hole_pos_xy)
    move_ground_cutout(model, hole_xy=hole_pos_xy)
    move_discs(model, disc_positions)

    new_xml_path = str(cfg["sim"].get("xml_path", XML_PATH_NEW))
    # print(f"Saving modified XML to: {new_xml_path}")
    mujoco.mj_saveLastXML(new_xml_path, model)
    model = mujoco.MjModel.from_xml_path(new_xml_path)

    real_start_pos = cfg["ball"]["start_pos"]
    real_ball_xy = real_start_pos[:2]
    obs_start_pos = cfg["ball"]["obs_start_pos"]
    obs_ball_xy = obs_start_pos[:2]
    cfg["disc_positions"] = disc_positions
    set_model_options_from_cfg(model, cfg)
    data = mujoco.MjData(model)
    cfg_nsub = int(cfg["sim"].get("nsubsteps", 1))
    safe_substep = float(cfg["sim"].get("safe_substep", 5e-05))
    nstep = max(cfg_nsub, int(math.ceil(model.opt.timestep / safe_substep)))

    ids = get_ids(model)
    assert_velocity_actuator(model, ids)
    reset_ball_state(model, data, cfg, ids)

    # move hole and ground cutout to desired position

    # debug_print_green_spans(model)

    mujoco.mj_forward(model, data)

    base_jid = ids["base_jid"]
    qadr_base0 = model.jnt_qposadr[base_jid]
    base_z0 = 0.37865 - 0.36

    base_pos_xyz = np.array([obs_ball_xy[0], obs_ball_xy[1], base_z0], float)


    qadr_base, vadr_base, base_pos_baked, base_quat_baked = bake_aim_pose(
        model, data, ball_xyz=base_pos_xyz, yaw_rad=math.radians(aim_yaw_deg)
    )
    reset_club_pose(model, data, ids, cfg["club"].get("start_angle_deg", 40.0))
    max_vel = math.radians(float(cfg["club"].get("max_vel_deg_s", 1000.0)))

    return run_loop(model, data, cfg,
             (qadr_base, vadr_base),
             (base_pos_baked, base_quat_baked),
             ids, nstep, max_vel, float(vx_des), math.radians(aim_yaw_deg))

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    
    here = Path(__file__).resolve().parent
    # print(here)
    project_root = here.parents[2]
    config_path = project_root / "configs" / "mujoco_config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    p = argparse.ArgumentParser()
    # Simulation behavior (defaults from cfg)
    p.add_argument("--render", action="store_true", help="Force render viewer on")
    p.add_argument("--no-render", action="store_true", help="Force headless mode")


    # High-level controls (not stored in cfg by design)
    p.add_argument("--aim-yaw", type=float, default=0.0, help="Aiming yaw in degrees")
    p.add_argument("--vx", type=float, default=1.0, help="Desired head X velocity (m/s)")

    p.add_argument("--no-start", action="store_true", help="Disable auto reset+swing")
    p.add_argument("--csv", type=str, default=None,
                help="Path to CSV to log ball position (time,x,y,z)")
    p.add_argument("--csv-period", type=float, default=0.01,
                help="How often to sample the ball pose in seconds")

    args = p.parse_args()

    # Determine render setting (tri-state so you can keep cfg default yet allow overrides)
    render = None
    if args.render and not args.no_render:
        render = True
    if args.no_render and not args.render:
        render = False

    if render is not None:
        cfg["sim"]["render"] = bool(render)

    cfg["sim"]["csv_period"] = args.csv_period
    aim_yaw = 10.258417129516602
    vx_des =  2.3321681022644043
    hole_pos_xy = [5, 0]
    # disc_positions = generate_disc_positions(5, -3.0, 3.0, -2.0, 2.0, hole_pos_xy)
    disc_positions = [(1.4956361419452078, -0.29096378977334414), (-0.20827146563258125, -0.5665742751926417), (-1.1686851660689785, -0.7053216711723784), (-2.1658353815296345, -0.6954762744879543)]
    cfg["ball"]["start_pos"] = [np.float64(-0.10126499661175004), np.float64(-0.48780616912307617), 0.02135]
    cfg["ball"]["obs_start_pos"] = [np.float64(-0.1013574481871513), np.float64(-0.4878724069881774), 0.02135]
    run_sim(
        aim_yaw_deg=aim_yaw,
        vx_des=vx_des,
        hole_pos_xy=hole_pos_xy,
        cfg=cfg,
        disc_positions=disc_positions
    )

if __name__ == "__main__":
    main()
