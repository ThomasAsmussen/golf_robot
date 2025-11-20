from kinematics import numeric_jacobian
import numpy as np

# -------------------------
# Inputs
# -------------------------
# q_hit = np.array([np.deg2rad(-127.23), np.deg2rad(-153.93), np.deg2rad(-100.71), np.deg2rad(74.59), np.deg2rad(35.18), np.deg2rad(-150.02)])
q_hit   = np.array([np.deg2rad(-127.23), np.deg2rad(-153.93), np.deg2rad(-100.71), np.deg2rad(74.59), np.deg2rad(35.18), np.deg2rad(28.72)])
v_dir = np.array([1.0, 0.0, 0.0])          # desired TCP direction
v_dir = v_dir / np.linalg.norm(v_dir)       # unit vector
joint_vel_max = np.array([3.0] * 6)         # rad/s per joint
eps = 1e-12

# -------------------------
# Compute max feasible TCP speed in v_dir at q_hit
# -------------------------
J = numeric_jacobian(q_hit)
J_lin = J[:3, :]

# Joint velocity that produces 1 m/s in v_dir (least-squares)
dq_dir = np.linalg.pinv(J_lin) @ v_dir  # shape (6,)

# Find which joints meaningfully contribute (avoid divide-by-zero)
idx = np.where(np.abs(dq_dir) > eps)[0]
if idx.size == 0:
    # Direction essentially unreachable from linear Jacobian at this pose
    v_mag_max = 0.0
    dq_at_limit = np.zeros(6)
    limiting_joint = None
else:
    # Scale so the first joint to hit its limit defines the maximum
    per_joint_scale = joint_vel_max[idx] / np.abs(dq_dir[idx])  # same shape
    scaling = np.min(per_joint_scale)
    limiting_joint = idx[np.argmin(per_joint_scale)]

    dq_at_limit = dq_dir * scaling
    v_tcp_max = J_lin @ dq_at_limit
    v_mag_max = float(np.linalg.norm(v_tcp_max))

# -------------------------
# Report
# -------------------------
print("\n[Maximum Achievable TCP Speed in Given Direction]")
print("Direction (unit):", v_dir)
print("Max linear speed (m/s):", round(v_mag_max, 6))
print("Joint velocities at that limit (rad/s):", np.round(dq_at_limit, 6))
print("Joint limits (rad/s):", joint_vel_max)
print("Limiting joint index:", limiting_joint)
