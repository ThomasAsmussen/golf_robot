import os, math, numpy as np, mujoco

# Paths (same as your repo)
HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
XML_PATH = os.path.join(REPO, "models", "mujoco", "golf_world.xml")

G = 9.81
H_DROP = 0.40     # meters
H_REB  = 0.15     # meters (your real test)
E_TARGET = math.sqrt(H_REB / H_DROP)   # ~0.612

# Hinge hold gains to emulate a rigidly supported club
KP_HOLD = 800.0
KD_HOLD = 20.0
Q_DES   = 0.0     # hold club_hinge at 0 rad (face-up)

MAX_TIME = 2.0    # seconds to simulate per trial
NSUB = 1          # substeps per step

def id_of(model, objtype, name):  # tiny helper
    return mujoco.mj_name2id(model, objtype, name)

def hold_club_torque(model, data):
    """Return motor torque to hold the hinge at Q_DES with PD."""
    j = id_of(model, mujoco.mjtObj.mjOBJ_JOINT, "club_hinge")
    if j == -1:
        return 0.0
    dof = model.jnt_dofadr[j]
    q   = float(data.qpos[model.jnt_qposadr[j]])
    dq  = float(data.qvel[dof])
    return KP_HOLD*(Q_DES - q) - KD_HOLD*dq

def set_ball_solparams(model, tconst, zeta=1.0, imp=None, width=None, mid=None):
    """Set ball_geom solref[timeconst, zeta] and optionally solimp."""
    gid = id_of(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    if gid == -1: raise RuntimeError("ball_geom not found")
    model.geom_solref[gid][0] = float(tconst)
    model.geom_solref[gid][1] = float(zeta)
    if imp is not None:   model.geom_solimp[gid][0] = float(imp)
    if width is not None: model.geom_solimp[gid][1] = float(width)
    if mid is not None:   model.geom_solimp[gid][2] = float(mid)

def get_head_top_z(model, data):
    """Compute the world z of the club-head top face when hinge is at Q_DES."""
    head_gid = id_of(model, mujoco.mjtObj.mjOBJ_GEOM, "club_head")
    if head_gid == -1: raise RuntimeError("club_head geom not found")
    # After mj_forward, geom_xpos is center; size[2] is half-thickness along local z for a box
    zc = float(data.geom_xpos[head_gid][2])
    hz = float(model.geom_size[head_gid][2])
    # With hinge at 0, head local z aligns with world z: top face at zc + hz
    return zc + hz

def ball_radius(model):
    gid = id_of(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    return float(model.geom_size[gid][0])

def set_ball_com_at(model, data, x, y, z, vx=0.0, vy=0.0, vz=0.0):
    bj   = id_of(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    qadr = model.jnt_qposadr[bj]
    vadr = model.jnt_dofadr[bj]
    # qpos = [x y z qw qx qy qz], here keep quaternion = [1,0,0,0]
    data.qpos[qadr:qadr+7] = np.array([x, y, z, 1.0, 0.0, 0.0, 0.0])
    data.qvel[vadr:vadr+6] = np.array([vx, vy, vz, 0.0, 0.0, 0.0])

def drop_and_measure_e(model, data):
    """Run one drop, return measured e = sqrt(h_rebound / H_DROP)."""
    # 1) set hinge exactly at Q_DES and zero velocities
    j = id_of(model, mujoco.mjtObj.mjOBJ_JOINT, "club_hinge")
    if j == -1: raise RuntimeError("club_hinge not found")
    qadr = model.jnt_qposadr[j]
    dof  = model.jnt_dofadr[j]
    data.qpos[qadr] = Q_DES
    data.qvel[dof]  = 0.0

    # 2) assemble and forward
    mujoco.mj_forward(model, data)

    # 3) compute head top z and place ball center H_DROP above it (plus radius)
    top_z = get_head_top_z(model, data)
    r     = ball_radius(model)
    # put ball centered above head in x/y
    head = id_of(model, mujoco.mjtObj.mjOBJ_GEOM, "club_head")
    hx, hy = data.geom_xpos[head][0], data.geom_xpos[head][1]
    set_ball_com_at(model, data, hx, hy, top_z + r + H_DROP, 0, 0, 0)
    mujoco.mj_forward(model, data)

    # 4) simulate with hinge hold; track max post-bounce height
    first_contact = False
    post_contact  = False
    z_peak = -1e9
    t_end  = data.time + MAX_TIME

    # actuator index 0 is your club_motor; we drive it as a PD position hold
    while data.time < t_end:
        # apply hold torque every step
        data.ctrl[0] = hold_club_torque(model, data)
        mujoco.mj_step(model, data, nstep=NSUB)

        # detect contact with club_head
        contact_now = (data.ncon > 0)
        if contact_now and not first_contact:
            first_contact = True
        if first_contact and not contact_now:
            post_contact = True  # we have bounced and left contact

        if post_contact:
            # track peak COM height after bounce
            bj = id_of(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
            zc = float(data.qpos[model.jnt_qposadr[bj] + 2])
            if zc > z_peak:
                z_peak = zc
            # stop once vertical velocity is downward again and we’ve seen a peak
            vz = float(data.qvel[model.jnt_dofadr[bj] + 2])
            if zc < z_peak - 1e-4 and vz < 0.0:
                break

    if z_peak < 0:
        raise RuntimeError("No bounce detected; check geometry alignment and hold gains.")
    # rebound height relative to contact plane:
    h_reb = max(0.0, z_peak - (top_z + r))
    e_meas = math.sqrt(max(0.0, h_reb / H_DROP))
    return e_meas, h_reb

def tune_solref_timeconst(model, data, e_target=E_TARGET, iters=10, tmin=1e-4, tmax=0.03):
    """
    Binary search on ball geom solref time-constant to match e_target.
    Keeps zeta=1.0 and current solimp.
    """
    best = None
    for k in range(iters):
        t = 0.5*(tmin + tmax)
        set_ball_solparams(model, tconst=t, zeta=1.0)
        mujoco.mj_forward(model, data)
        e_meas, h = drop_and_measure_e(model, data)

        # record
        best = (t, e_meas, h)

        # adjust bounds: smaller t => stiffer/shorter contact => generally higher e
        if e_meas < e_target:
            # need more bounce
            tmax = t
        else:
            # too bouncy
            tmin = t

        # print progress
        print(f"[{k+1:02d}] tconst={t:.6f}  e_meas={e_meas:.4f}  (target={e_target:.4f})  h_reb={h:.4f} m")

    return best

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # Make contacts a bit firmer overall (optional)
    model.opt.impratio = 5.0

    # Initial guess for ball solparams (start reasonably stiff)
    set_ball_solparams(model, tconst=0.005, zeta=1.0, imp=0.95, width=0.95, mid=0.001)

    print(f"Target COR from 0.40->0.15 m: e_target = {E_TARGET:.4f}\nTuning ball geom solref[0] (time-constant)...\n")

    t_opt, e_meas, h = tune_solref_timeconst(model, data, e_target=E_TARGET, iters=12)

    print("\n=== Result ===")
    print(f"Optimal ball solref: timeconst={t_opt:.6f}, zeta=1.0")
    print(f"Measured e: {e_meas:.4f}  (target {E_TARGET:.4f}), rebound h={h:.4f} m")
    print("\nPut this on your ball geom in XML, e.g.:")
    print(f'  solref="{t_opt:.6g} 1"  solimp="0.95 0.95 0.001"')
    print("================\n")

if __name__ == "__main__":
    main()
