import os
import math
import copy
import numpy as np
import mujoco
from mujoco import viewer
from contextlib import nullcontext
import csv

#python run_sim.py --aim-yaw 0.0 --vx 1.0 --sim-timestep 0.001 --csv "C:\Users\marti\OneDrive - Danmarks Tekniske Universitet\DTU\Mini-golf robot Master\Physics Comparison using Detection\log_0\ball_log.csv"        
# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
XML_PATH = os.path.join(REPO, "models", "mujoco", "golf_world.xml")

# ---------------------------------------------------------------------
# Default configuration (single source of truth for defaults)
# ---------------------------------------------------------------------
DEFAULT_CFG = {
    "sim": {
        "timestep": 0.0002,
        "nsubsteps": 1,
        "safe_substep": 5e-05,
        "duration_sec": 30.0,
        "render": True,            # default behavior unless CLI flips it
    },
    "club": {
        "max_vel_deg_s": 500.0,
        "start_angle_deg": 40.0,   # moved from global into cfg
        "swing": {
            "vx": 2.0,
            "post_margin": 0.05,
        },
    },
    "hole": {"radius": 0.055},
    "ball": {
        "radius": 0.02135,
        "start_pos": [0.0, 0.0, 0.02135],
    },
}

# ---------------------------------------------------------------------
# Model options from cfg
# ---------------------------------------------------------------------
def set_model_options_from_cfg(model, cfg):
    try:
        ts = float(cfg["sim"].get("timestep", model.opt.timestep))
        model.opt.timestep = ts
    except Exception:
        pass

# ---------------------------------------------------------------------
# ID cache / checks
# ---------------------------------------------------------------------
def get_ids(model):
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

def quat_yaw_deg(q):
    """Return yaw (rotation about +Z) in degrees for MuJoCo quat [w,x,y,z]."""
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
    print(msg)

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

def is_ball_in_hole(model, data, cfg, ids):
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
def command_impact_speed(model, data, ids, vx_des=1.0):
    dx = head_ball_dx(data, ids)
    dir_to_plane = -np.sign(dx if dx != 0.0 else 1e-12)
    v_des = float(abs(vx_des)) * dir_to_plane
    jid = ids["hinge_jid"]; gid = ids["club_head_gid"]
    r   = np.array(data.geom_xpos[gid]) - np.array(data.xanchor[jid])
    den = float(np.cross(np.array(data.xaxis[jid]), r)[0])
    if abs(den) < 1e-8:
        return 0.0
    return float(v_des / den)

# ---------------------------------------------------------------------
# Viewer wrapper
# ---------------------------------------------------------------------
class NullViewer:
    def __init__(self, model, data, duration_sec=5.0):
        self._running = True
        self._end_time = float(data.time) + float(duration_sec)
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
             ids, nstep, max_vel, vx_des,
             *, start_immediately=False):
    act_id = ids['act_id']; hinge_jid = ids['hinge_jid']
    qadr = model.jnt_qposadr[hinge_jid]
    qadr_base, vadr_base = base_idx
    base_pos_baked, base_quat_baked = base_pose
    base_jid = ids["base_jid"]  # for telemetry
    pending_actions = []; dx_prev = None

    # ---- CSV setup ----------------------------------------------------
    csv_path = cfg["sim"].get("csv_path", None)
    csv_period = float(cfg["sim"].get("csv_period", 0.01))
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
            csv_writer.writerow(["time", "ball_x", "ball_y", "ball_z"])
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
                  else NullViewer(model, data, duration_sec=cfg["sim"].get("duration_sec", 5.0)))

    if start_immediately:
        pending_actions.append("reset_and_swing")

    next_print_t = 0.0; PRINT_PERIOD = 0.01

    try:
        with viewer_ctx as v:
            while v.is_running():
                while pending_actions:
                    act = pending_actions.pop(0)
                    if act == "reset_and_swing":
                        with v.lock():
                            reset_ball_state(model, data, cfg, ids)
                            reset_club_pose(model, data, ids, cfg["club"].get("start_angle_deg", 40.0))
                            mujoco.mj_forward(model, data)
                        print(f"[{data.time:7.3f}s] RESET + SWING")
                    if act == "breaking":
                        print(f"[{data.time:7.3f}s] BREAKING")
                        return

                data.qpos[qadr_base:qadr_base+3] = base_pos_baked
                data.qpos[qadr_base+3:qadr_base+7] = base_quat_baked
                data.qvel[vadr_base:vadr_base+6] = 0.0

                qd_cmd = command_impact_speed(model, data, ids, vx_des=vx_des)
                lo, hi = model.actuator_ctrlrange[act_id]
                qd_cmd = float(np.clip(qd_cmd, -max_vel, max_vel))
                qd_cmd = float(np.clip(qd_cmd, lo, hi))
                data.ctrl[act_id] = qd_cmd
                mujoco.mj_step(model, data, nstep=nstep)

                dx = head_ball_dx(data, ids)
                if sign_crossed(dx_prev, dx):
                    vx_now, _ = club_head_vx(model, data, ids)
                    print(f"IMPACT @ t={data.time:7.3f}s | Vx_des={vx_des:+.3f} m/s | Vx_hit={vx_now:+.3f} m/s")
                dx_prev = dx

                # telemetry ---------------------------------------------------
                t = float(data.time)
                if t >= next_print_t:
                    vx_now, _ = club_head_vx(model, data, ids)
                    q_deg = math.degrees(float(data.qpos[qadr]))
                    ball_p = data.geom_xpos[ids["ball_gid"]]
                    base_bid = ids["base_bid"]
                    quat = data.xquat[base_bid]
                    yaw_deg = quat_yaw_deg(quat)
                    print(
                        f"Time: {t:7.3f}s | "
                        f"q: {q_deg:7.2f}° | "
                        f"qd_cmd: {qd_cmd:7.3f} rad/s | "
                        f"Vx_des: {vx_des:+6.3f} | "
                        f"Vx_now: {vx_now:+6.3f} | "
                        f"Yaw: {yaw_deg:+6.2f}° | "
                        f"Ball: ({ball_p[0]:+.3f}, {ball_p[1]:+.3f}, {ball_p[2]:+.3f})"
                    )
                    next_print_t += PRINT_PERIOD

                # ---- CSV logging at requested rate -------------------------
                if csv_writer is not None and t >= next_csv_t:
                    bp = data.geom_xpos[ids["ball_gid"]]
                    csv_writer.writerow([f"{t:.6f}", f"{bp[0]:.9f}", f"{bp[1]:.9f}", f"{bp[2]:.9f}"])
                    # optional flush for safety during long runs:
                    # csv_file.flush()
                    next_csv_t += csv_period
                # ------------------------------------------------------------
                
                if is_ball_in_hole(model, data, cfg, ids):
                    print(f"[{data.time:6.3f}s] Ball within hole radius — physics should drop it.")
                v.sync()
    finally:
        if csv_file is not None:
            try:
                csv_file.flush()
                csv_file.close()
                print(f"[info] CSV saved to {csv_path}")
            except Exception as e:
                print(f"[warn] Could not close CSV '{csv_path}': {e}")

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def run_sim(aim_yaw_deg, vx_des, cfg, *, start=True):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    set_model_options_from_cfg(model, cfg)
    data = mujoco.MjData(model)
    cfg_nsub = int(cfg["sim"].get("nsubsteps", 1))
    safe_substep = float(cfg["sim"].get("safe_substep", 5e-05))
    nstep = max(cfg_nsub, int(math.ceil(model.opt.timestep / safe_substep)))

    ids = get_ids(model)
    assert_velocity_actuator(model, ids)
    reset_ball_state(model, data, cfg, ids)
    mujoco.mj_forward(model, data)

    qadr_base, vadr_base, base_pos_baked, base_quat_baked = bake_aim_pose(
        model, data, yaw_rad=math.radians(aim_yaw_deg)
    )
    reset_club_pose(model, data, ids, cfg["club"].get("start_angle_deg", 40.0))
    max_vel = math.radians(float(cfg["club"].get("max_vel_deg_s", 1000.0)))

    run_loop(model, data, cfg,
             (qadr_base, vadr_base),
             (base_pos_baked, base_quat_baked),
             ids, nstep, max_vel, float(vx_des),
             start_immediately=start)

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    import argparse

    cfg_defaults = DEFAULT_CFG  # alias for brevity

    p = argparse.ArgumentParser()
    # Simulation behavior (defaults from cfg)
    p.add_argument("--sim-timestep", type=float,
                   default=cfg_defaults["sim"]["timestep"],
                   help="MuJoCo model.opt.timestep")
    p.add_argument("--sim-nsubsteps", type=int,
                   default=cfg_defaults["sim"]["nsubsteps"],
                   help="Integrator substeps per mj_step")
    p.add_argument("--sim-safe-substep", type=float,
                   default=cfg_defaults["sim"]["safe_substep"],
                   help="Max step size when deriving nstep")
    p.add_argument("--sim-duration", type=float,
                   default=cfg_defaults["sim"]["duration_sec"],
                   help="Viewer duration in headless mode (s)")
    p.add_argument("--render", action="store_true", help="Force render viewer on")
    p.add_argument("--no-render", action="store_true", help="Force headless mode")

    # Club/ball/hole params (defaults from cfg)
    p.add_argument("--club-max-vel-deg-s", type=float,
                   default=cfg_defaults["club"]["max_vel_deg_s"],
                   help="Max actuator target velocity in deg/s")
    p.add_argument("--club-start-angle", type=float,
                   default=cfg_defaults["club"]["start_angle_deg"],
                   help="Initial club hinge angle in degrees")
    p.add_argument("--hole-radius", type=float,
                   default=cfg_defaults["hole"]["radius"],
                   help="Hole radius (m)")
    p.add_argument("--ball-radius", type=float,
                   default=cfg_defaults["ball"]["radius"],
                   help="Ball radius (m) (not directly used)")
    p.add_argument("--ball-start-pos", type=float, nargs=3,
                   default=list(cfg_defaults["ball"]["start_pos"]),
                   metavar=("X","Y","Z"),
                   help="Ball free-joint start position (m)")

    # High-level controls (not stored in cfg by design)
    p.add_argument("--aim-yaw", type=float, default=0.0, help="Aiming yaw in degrees")
    p.add_argument("--vx", type=float, default=1.0, help="Desired head X velocity (m/s)")
    p.add_argument("--duration", type=float, default=None,
                   help="(Deprecated) Overrides --sim-duration if provided")
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

    # Build cfg from args (start from deepcopy of defaults to avoid accidental mutation)
    cfg = copy.deepcopy(cfg_defaults)
    cfg["sim"]["timestep"] = args.sim_timestep
    cfg["sim"]["nsubsteps"] = args.sim_nsubsteps
    cfg["sim"]["safe_substep"] = args.sim_safe_substep
    cfg["sim"]["duration_sec"] = args.sim_duration if args.duration is None else float(args.duration)
    # keep default render unless user explicitly overrode
    if render is not None:
        cfg["sim"]["render"] = bool(render)

    cfg["club"]["max_vel_deg_s"] = args.club_max_vel_deg_s
    cfg["club"]["start_angle_deg"] = args.club_start_angle

    cfg["hole"]["radius"] = args.hole_radius
    cfg["ball"]["radius"] = args.ball_radius
    cfg["ball"]["start_pos"] = list(args.ball_start_pos)
    cfg["sim"]["csv_path"] = args.csv
    cfg["sim"]["csv_period"] = args.csv_period

    run_sim(
        aim_yaw_deg=args.aim_yaw,
        vx_des=args.vx,
        cfg=cfg,
        start=(not args.no_start),
    )

if __name__ == "__main__":
    main()
