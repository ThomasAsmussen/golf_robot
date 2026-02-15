import os
import math
import argparse
import csv
from collections import defaultdict
from typing import List, Dict

import numpy as np
import mujoco
import optuna
import optunahub
import contextlib


# ------------------------------ Paths ---------------------------------
HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
XML_PATH_DEFAULT = os.path.join(REPO, "models", "mujoco", "golf_world_no_hole.xml")
DEFAULT_DB_PATH = os.path.join(HERE, "optuna_calib.db")


# --------------------------- Utilities --------------------------------
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
    }


# ---------------------- Measurement loading ----------------------------
def load_finalpos_csv(path: str) -> List[Dict[str, float]]:
    """
    CSV columns (case-insensitive):
      impact_velocity, distance
    """
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))

    header = [h.strip().lower() for h in rows[0]]
    data = rows[1:]

    vxi = header.index("impact_velocity")
    xi  = header.index("distance")

    out = []
    for r in data:
        out.append({
            "vx_des": float(r[vxi]),
            "x_final": float(r[xi]),
        })
    return out


# ------------------------- Reset helpers -------------------------------
def reset_ball_state(model, data):
    jid = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    qadr = model.jnt_qposadr[jid]
    vadr = model.jnt_dofadr[jid]
    data.qpos[qadr:qadr+7] = np.array([0, 0, 0.02135, 1, 0, 0, 0])
    data.qvel[vadr:vadr+6] = 0.0


def reset_club_pose(model, data, start_deg=40.0):
    jid = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "club_hinge")
    qadr = model.jnt_qposadr[jid]
    dof  = model.jnt_dofadr[jid]
    data.qpos[qadr] = np.deg2rad(start_deg)
    data.qvel[dof] = 0.0
    mujoco.mj_forward(model, data)


def bake_aim_pose(model, data, yaw_deg=0.0):
    jid = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "mount_free")
    qadr = model.jnt_qposadr[jid]
    vadr = model.jnt_dofadr[jid]
    half = 0.5 * math.radians(yaw_deg)
    quat = [math.cos(half), 0, 0, math.sin(half)]
    data.qpos[qadr:qadr+7] = [0, 0, 0.02135, *quat]
    data.qvel[vadr:vadr+6] = 0.0
    mujoco.mj_forward(model, data)


# -------------------------- Controller ---------------------------------
def command_impact_speed(model, data, ids, vx_des, R=0.36):
    dx = (
        data.geom_xpos[ids["club_head_gid"]][0]
        - data.geom_xpos[ids["ball_gid"]][0]
    )
    direction = -np.sign(dx if dx != 0 else 1e-6)
    v = -abs(vx_des) * direction
    return v / R


# -------------------------- Simulation ---------------------------------
def simulate_final_x(model, ids, vx_cmd, T=10.0, dt=0.002):
    data = mujoco.MjData(model)
    model.opt.timestep = dt

    reset_ball_state(model, data)
    bake_aim_pose(model, data)
    reset_club_pose(model, data)

    base_jid = ids["base_jid"]
    qadr = model.jnt_qposadr[base_jid]
    vadr = model.jnt_dofadr[base_jid]
    base_pose = np.copy(data.qpos[qadr:qadr+7])

    act_id = ids["act_id"]
    lo, hi = model.actuator_ctrlrange[act_id]

    steps = int(T / dt)
    for _ in range(steps):
        data.qpos[qadr:qadr+7] = base_pose
        data.qvel[vadr:vadr+6] = 0.0

        cmd = np.clip(command_impact_speed(model, data, ids, vx_cmd), lo, hi)
        data.ctrl[act_id] = cmd
        mujoco.mj_step(model, data)

    x_final = float(data.geom_xpos[ids["ball_gid"]][0])

    ball_jid = ids["ball_jid"]
    vadr_ball = model.jnt_dofadr[ball_jid]
    vxyz = data.qvel[vadr_ball:vadr_ball+3]
    speed_final = float(np.linalg.norm(vxyz))

    return x_final, speed_final


# -------------------- Parameters to tune --------------------------------
def find_pair_index(model, g1, g2):
    for i in range(model.npair):
        if {model.pair_geom1[i], model.pair_geom2[i]} == {g1, g2}:
            return i
    raise RuntimeError("Ball–club contact pair not found")


def apply_params(model, ids, trial):
    pid = find_pair_index(model, ids["ball_gid"], ids["club_head_gid"])

    model.pair_solref[pid][0] = trial.suggest_float("solref_tc", 0.005, 0.1, log=True)
    model.pair_solref[pid][1] = trial.suggest_float("solref_dr", 0.05, 0.5)

    model.pair_solimp[pid][0] = trial.suggest_float("solimp_a", 0.7, 1.2)
    model.pair_solimp[pid][1] = trial.suggest_float("solimp_b", 0.8, 1.2)
    model.pair_solimp[pid][2] = trial.suggest_float("solimp_c", 1e-4, 1e-2, log=True)

    #slide = trial.suggest_float("friction_slide", 0.2, 0.8)
    #spin  = trial.suggest_float("friction_spin", 1e-5, 1e-3, log=True)

    #for g in range(model.ngeom):
    #    if model.geom_contype[g]:
    #        model.geom_friction[g][0] = slide
    #        model.geom_friction[g][1] = spin


# --------------------------- Objective ---------------------------------
def make_objective(model, ids, measurements):
    grouped = defaultdict(list)
    for m in measurements:
        grouped[m["vx_des"]].append(m["x_final"])

    def objective(trial):
        apply_params(model, ids, trial)

        se = 0.0
        n = 0

        print("\n[Trial", trial.number, "]")

        for vx, xs in grouped.items():
            x_sim, v_end = simulate_final_x(model, ids, vx)
            x_meas_mean = float(np.mean(xs))

            print(
                f"  vx={vx:.2f} | "
                f"x_sim={x_sim:.3f} m | "
                f"x_meas_mean={x_meas_mean:.3f} m | "
                f"v_end={v_end:.4e} m/s"
            )

            for x_meas in xs:
                err = x_sim - x_meas
                se += err * err
                n += 1

        return se / n


    return objective


# ------------------------------ Main ----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=XML_PATH_DEFAULT)
    ap.add_argument("--csv", required=True, help="CSV with impact_velocity,distance")
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--storage", default=f"sqlite:///{DEFAULT_DB_PATH}")
    ap.add_argument("--study-name", default="golf-final-x")
    args = ap.parse_args()

    measurements = load_finalpos_csv(args.csv)

    with pushd(os.path.dirname(args.xml)):
        model = mujoco.MjModel.from_xml_path(os.path.abspath(args.xml))

    ids = get_ids(model)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.CmaEsSampler(),  # <-- FORCE CMA-ES #sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(),
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    study.optimize(
        make_objective(model, ids, measurements),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    print("\nBest MSE:", study.best_value)
    print("Best parameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
