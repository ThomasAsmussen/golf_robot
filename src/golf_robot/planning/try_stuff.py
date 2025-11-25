from kinematics import fk_ur10, pick_ik_solution, move_point_xyz
import numpy as np
Z_PALLET = 0.155  # from config.py
home=np.array([-2.1694, -2.6840, -1.4999, 1.0404, 0.5964, -2.7892])
# Waypoints for movel:
# q_start = np.array([np.deg2rad(-144.36), np.deg2rad(-167.85), np.deg2rad(-58.16),  np.deg2rad(45.97), np.deg2rad(52.32), np.deg2rad(28.72)])
# q_hit   = np.array([np.deg2rad(-127.23), np.deg2rad(-153.93), np.deg2rad(-100.71), np.deg2rad(74.59), np.deg2rad(35.18), np.deg2rad(28.72)])
# q_exit  = np.array([np.deg2rad(-94.89),  np.deg2rad(-152.06), np.deg2rad(-123.42), np.deg2rad(94.85), np.deg2rad(2.85),  np.deg2rad(28.72)])

q_hit = move_point_xyz(0.0, 0.0, Z_PALLET, home, home)[0]

print("Home: ")
print(home)
print("Picked IK solution for q_hit:")
print(q_hit)
# Good waypoints for speedj:
q_start = np.array([np.deg2rad(-144.36), np.deg2rad(-162.82), np.deg2rad(-51.80),  np.deg2rad(66.92), np.deg2rad(42.60), np.deg2rad(6.51)])
q_hit   = np.array([np.deg2rad(-127.23), np.deg2rad(-153.93), np.deg2rad(-100.71), np.deg2rad(74.59), np.deg2rad(35.18), np.deg2rad(28.72)])
q_exit  = np.array([np.deg2rad(-87.65),  np.deg2rad(-135.05), np.deg2rad(-108.29), np.deg2rad(85.72), np.deg2rad(4.39),  np.deg2rad(-12.23)])

q0_start = np.array([-2.35718127, -2.84501045, -0.88769778,  1.27293301,  0.61129035, -0.03746998])
q0_hit  = np.array([-2.18480298, -2.68658532, -1.75772109,  1.30184109,  0.61400683,  0.50125856])  # reference impact joint config (+X direction)
q0_exit = np.array([-0.74857402, -2.44563742, -1.97906305,  0.87441754, -1.14358243,  0.7267085])
print(q_hit)
print(q0_hit)
# q_exit   = np.array([np.deg2rad(-127.23), np.deg2rad(-153.93), np.deg2rad(-100.71), np.deg2rad(74.59), np.deg2rad(35.18), np.deg2rad(28.72)])

def move_point_xyz(x_change, y_change, z_change, q_init):
    T_init = fk_ur10(q_init)[-1]
    x_init, y_init, z_init = T_init[:3,3]
    T_new = T_init.copy()
    T_new[0,3] = x_init + x_change
    T_new[1,3] = y_init + y_change
    T_new[2,3] = z_init + z_change
    q_new, info = pick_ik_solution(T_new, q_init)
    return q_new, info

q_start, _ = move_point_xyz(-0.6, 0.0, 0.25, q0_hit)
print("Picked IK solution for q_start:")
print(q_start)

q_end, _ = move_point_xyz(0.6, 0.0, 0.25, q0_hit)
print("Picked IK solution for q_end:")
print(q_end)

x,y,z = fk_ur10(q_start)[9][:3,3]
print(f"xyz for joint 9 at q_start:")
print(f"x: {x}")
print(f"y: {y}")
print(f"z: {z}")

x_hit,y_hit,z_hit = fk_ur10(q0_hit)[9][:3,3]
print(f"xyz for joint 9 at q_hit:")
print(f"x: {x_hit}")
print(f"y: {y_hit}")
print(f"z: {z_hit}")

x,y,z = fk_ur10(q_end)[9][:3,3]
print(f"xyz for joint 9 at q_exit:")
print(f"x: {x}")
print(f"y: {y}")
print(f"z: {z}")

print([np.deg2rad(-127.23), np.deg2rad(-153.93), np.deg2rad(-100.71), np.deg2rad(74.59), np.deg2rad(35.18), np.deg2rad(28.72)])