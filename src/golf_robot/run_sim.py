import os
import time
import yaml
import math
import numpy as np
import mujoco
from mujoco import viewer

# Path helpers
HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
YAML_PATH = os.path.join(REPO, "configs", "mujoco.yaml")
XML_PATH  = os.path.join(REPO, "models", "mujoco", "golf_world.xml")

START_ANGLE_DEG = -20.0

def load_params():
    with open(YAML_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def set_model_options_from_cfg(model, cfg):
    # set global timestep
    if "sim" in cfg and "timestep" in cfg["sim"]:
        model.opt.timestep = float(cfg["sim"]["timestep"])
    # friction: green geom
    green_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "green")
    if green_id != -1:
        fr = cfg["green"]["friction"]
        model.geom_friction[green_id][:] = fr
        # set plane elevation
        z = cfg["green"].get("elevation", 0.0)
        model.geom_pos[green_id][2] = z
    # club tuning
    jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "club_hinge")
    if jnt != -1:
        j = jnt
        # damping/armature from YAML
        model.dof_damping[ model.jnt_dofadr[j] ] = cfg["club"]["damping"]
        model.dof_armature[ model.jnt_dofadr[j] ] = cfg["club"]["armature"]
    # motor gear
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "club_motor")
    if act_id != -1:
        model.actuator_gear[act_id][0] = cfg["club"]["motor_gear"]

    # hole visuals (site size, body position)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hole_center")
    hole_cfg = cfg["hole"]
    if site_id != -1:
        model.site_pos[site_id][:] = hole_cfg["position"]
        model.site_size[site_id][0] = hole_cfg["radius"]

    # move the hole body too (so visuals line up)
    hole_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hole")
    if hole_body != -1:
        model.body_pos[hole_body][:] = hole_cfg["position"]

def reset_ball_state(model, data, cfg):
    # Set ball position & zero velocities
    qpos_addr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]
    x, y, z = cfg["start"]["ball_pos"]
    qw, qx, qy, qz = cfg["start"]["ball_quat"]
    data.qpos[qpos_addr:qpos_addr+7] = np.array([x, y, z, qw, qx, qy, qz])
    data.qvel[qpos_addr:qpos_addr+6] = 0.0

def ball_position(model, data):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    # geom_xpos is populated after forward()
    return np.array(data.geom_xpos[gid])

def is_ball_in_hole(model, data, cfg):
    hole_pos = np.array(cfg["hole"]["position"])
    hole_r   = float(cfg["hole"]["radius"])
    rim_h    = float(cfg["hole"]["rim_height"])
    bpos     = ball_position(model, data)
    # Horizontal distance to hole center
    dxy = np.linalg.norm(bpos[:2] - hole_pos[:2])
    # Check radial & height thresholds
    return (dxy <= hole_r) and (bpos[2] <= hole_pos[2] + rim_h)

def drop_ball_to_cup(model, data, cfg):
    """Teleport ball into the cup bottom (visual approximation)."""
    hole_pos = np.array(cfg["hole"]["position"])
    depth    = float(cfg["hole"]["cup_depth"])
    r        = float(cfg["ball"]["radius"])
    # place ball at cup bottom, slightly above bottom
    new_pos = hole_pos + np.array([0, 0, -depth + r + 0.002])
    qpos_addr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]
    data.qpos[qpos_addr:qpos_addr+3] = new_pos
    data.qvel[qpos_addr:qpos_addr+6] = 0.0

def main():
    cfg = load_params()
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    set_model_options_from_cfg(model, cfg)
    data  = mujoco.MjData(model)

    # Reset the ball once
    reset_ball_state(model, data, cfg)
    mujoco.mj_forward(model, data)

    # --- set initial CLUB angle here (once, from code) ---
    j_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "club_hinge")
    if j_id == -1:
        raise ValueError("Joint 'club_hinge' not found in the model.")
    qadr = model.jnt_qposadr[j_id]          # qpos index of the hinge angle
    dof  = model.jnt_dofadr[j_id]           # qvel index of the hinge rate
    data.qpos[qadr] = np.deg2rad(START_ANGLE_DEG)
    data.qvel[dof]  = 0.0
    mujoco.mj_forward(model, data)

    # Keyboard-triggered swing (simple impulse)
    swing_active = False
    swing_end_t  = 0.0

    def key_callback(keycode):
        nonlocal swing_active, swing_end_t
        if keycode in (ord('r'), ord('R')):
            reset_ball_state(model, data, cfg)
            mujoco.mj_forward(model, data)
        elif keycode in (ord('s'), ord('S')):
            swing_active = True
            swing_end_t = data.time + float(cfg["club"]["swing_duration"])

    # Viewer loop
    if cfg["sim"]["render"]:
        with viewer.launch_passive(model, data) as v:
            # v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            # v.cam.lookat[:]  = [0.0, 0.0, 0.0]
            # v.cam.distance   = 4.035878063137433
            # v.cam.azimuth    = 0.5
            # v.cam.elevation  = -18.625
            # cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
            # if cam_id != -1:
            #     v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            #     v.cam.fixedcamid = cam_id
            if hasattr(v, "user_key_callback"):
                v.user_key_callback = key_callback
            else:
                v.user_keyboard = key_callback

            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0

            while v.is_running():
                # print("Camera lookat:", v.cam.lookat)     # 3D point the camera looks at
                # print("Camera distance:", v.cam.distance) # distance from lookat
                # print("Camera azimuth:", v.cam.azimuth)   # horizontal angle (deg)
                # print("Camera elevation:", v.cam.elevation) 
                if swing_active:
                    data.ctrl[0] = float(cfg["club"]["swing_impulse"])
                    if data.time >= swing_end_t:
                        swing_active = False
                        data.ctrl[0] = 0.0
                else:
                    data.ctrl[0] = 0.0

                mujoco.mj_step(model, data, nstep=cfg["sim"]["nsubsteps"])

                if is_ball_in_hole(model, data, cfg):
                    print(f"[{data.time:6.3f}s] Ball sunk!")
                    drop_ball_to_cup(model, data, cfg)
                    mujoco.mj_forward(model, data)

                v.sync()
    else:
        steps = int(cfg["sim"]["duration_sec"] / model.opt.timestep)
        for _ in range(steps):
            mujoco.mj_step(model, data, nstep=cfg["sim"]["nsubsteps"])
            if is_ball_in_hole(model, data, cfg):
                print(f"[{data.time:6.3f}s] Ball sunk!")
                drop_ball_to_cup(model, data, cfg)
                mujoco.mj_forward(model, data)



if __name__ == "__main__":
    main()
