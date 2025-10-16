import os
import math
import numpy as np
import mujoco
from mujoco import viewer

# Paths (same style as your run_sim.py)
HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
XML_PATH  = os.path.join(REPO, "models", "mujoco", "golf_world.xml")

# Target rollout: v0=1 m/s, s≈2 m  =>  Crr = v0^2/(2 g s)
G = 9.81
V0 = 1.0
S_TARGET = 2.0
CRR = (V0**2) / (2.0 * G * S_TARGET)   # ≈ 0.025478

# Use a balanced split so (ball_roll * green_roll) ≈ CRR
ROLL_SPLIT = math.sqrt(CRR)             # ≈ 0.1596
BALL_ROLL = ROLL_SPLIT
GREEN_ROLL = ROLL_SPLIT

SPEED_STOP = 0.02        # m/s threshold to consider "stopped"
NSUBSTEPS = 1

def move_club_out_of_the_way(model):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "club_base")
    if cid != -1:
        model.body_pos[cid][:] = np.array([1000.0, 1000.0, 1000.0])

def ensure_ball_condim6(model):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    if gid != -1:
        model.geom_condim[gid] = 6

def set_rolling_friction(model, ball_roll, green_roll):
    ball_gid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    green_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "green")
    if ball_gid != -1:
        model.geom_friction[ball_gid][2] = float(ball_roll)
    if green_gid != -1:
        model.geom_friction[green_gid][2] = float(green_roll)

def set_ball_state(model, data, pos, quat=(1,0,0,0), linvel=(0,0,0), angvel=(0,0,0)):
    jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    qadr = model.jnt_qposadr[jid]
    vadr = model.jnt_dofadr[jid]
    data.qpos[qadr:qadr+7] = np.array([*pos, *quat], dtype=float)
    data.qvel[vadr:vadr+6] = np.array([*linvel, *angvel], dtype=float)

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # Make sure rolling friction is actually modeled
    ensure_ball_condim6(model)

    # Set rolling friction so product ≈ Crr
    set_rolling_friction(model, BALL_ROLL, GREEN_ROLL)

    # Keep contacts a bit firmer so friction transmits (optional)
    model.opt.impratio = 5.0

    # Hide the club
    move_club_out_of_the_way(model)

    # Ball radius from geom size, start at origin on plane
    ball_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    r = float(model.geom_size[ball_gid][0])
    set_ball_state(model, data, pos=(0.0, 0.0, r), linvel=(V0, 0.0, 0.0))
    mujoco.mj_forward(model, data)

    # Keep initial XY for distance measurement
    bj  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    bq  = model.jnt_qposadr[bj]
    bv  = model.jnt_dofadr[bj]
    x0, y0 = data.qpos[bq:bq+2].copy()

    with viewer.launch_passive(model, data) as v:
        # Nice camera aimed at the rollout
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.lookat[:] = [1.2, 0.0, 0.0]
        v.cam.distance  = 3.5
        v.cam.azimuth   = 0.0
        v.cam.elevation = -15.0

        stopped = False
        while v.is_running():
            mujoco.mj_step(model, data, nstep=NSUBSTEPS)

            speed = np.linalg.norm(data.qvel[bv:bv+3])
            if not stopped and speed < SPEED_STOP:
                stopped = True
                x, y = data.qpos[bq:bq+2]
                s = float(np.linalg.norm([x - x0, y - y0]))
                eff_roll = model.geom_friction[ball_gid][2] * model.geom_friction[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "green")][2]
                print("\n=== Rollout result ===")
                print(f"v0: {V0:.3f} m/s   stop speed < {SPEED_STOP:.3f} m/s")
                print(f"Distance rolled: {s:.3f} m (target ~ {S_TARGET:.3f} m)")
                print(f"Effective rolling (product): {eff_roll:.6f} (target Crr ~ {CRR:.6f})")
                print("======================\n")

            v.sync()

if __name__ == "__main__":
    main()
