import numpy as np
from ur10_logger import UR10Logger

# --- small rotation helpers ---
def _rvec_to_R(rvec):
    rvec = np.asarray(rvec, dtype=float)
    th = np.linalg.norm(rvec)
    if th < 1e-12:
        return np.eye(3)
    k = rvec / th
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)

def _axis_angle_from_R(R):
    tr = np.clip(np.trace(R), -1.0, 3.0)
    ang = np.arccos((tr - 1.0) / 2.0)
    if np.isclose(ang, 0.0):
        return (0.0, 0.0, 0.0)
    if np.isclose(ang, np.pi):
        rx = np.sqrt(max(0, (R[0,0] + 1)/2)); ry = np.sqrt(max(0, (R[1,1] + 1)/2)); rz = np.sqrt(max(0, (R[2,2] + 1)/2))
        rx = np.copysign(rx, R[2,1] - R[1,2]); ry = np.copysign(ry, R[0,2] - R[2,0]); rz = np.copysign(rz, R[1,0] - R[0,1])
        axis = np.array([rx, ry, rz]); axis /= (np.linalg.norm(axis) + 1e-12)
        r = axis * ang; return (float(r[0]), float(r[1]), float(r[2]))
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2*np.sin(ang))
    axis /= (np.linalg.norm(axis) + 1e-12)
    r = axis * ang; return (float(r[0]), float(r[1]), float(r[2]))

def _pose_p(xyz, R, face_roll=0.0):
    Rb = R.copy()
    if abs(face_roll) > 0:
        c, s = np.cos(face_roll), np.sin(face_roll)
        Rroll = np.array([[1, 0, 0],
                          [0, c, -s],
                          [0, s,  c]])
        Rb = Rb @ Rroll  # local X roll (open/close face)
    rx, ry, rz = _axis_angle_from_R(Rb)
    return f"p[{xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}, {rx:.6f}, {ry:.6f}, {rz:.6f}]"

def _frame_from_direction(dir_x, up_hint=np.array([0.0, 0.0, 1.0])):
    """Return R with columns [x,y,z], x aligned with dir_x, z as upright as possible."""
    x = np.asarray(dir_x, dtype=float)
    x = x / (np.linalg.norm(x) + 1e-12)
    z = up_hint - np.dot(up_hint, x)*x
    if np.linalg.norm(z) < 1e-8:   # degenerate (dir parallel to up)
        up_hint = np.array([0.0, 1.0, 0.0])
        z = up_hint - np.dot(up_hint, x)*x
    z = z / (np.linalg.norm(z) + 1e-12)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)
    z = np.cross(x, y)
    return np.column_stack([x, y, z])

# --- swing sender ---
def send_swing(
    sock,
    *,
    # planes & ball
    x_start=-0.997,
    x_end=-0.297,
    y_ball=-0.546, #-0.546
    z_ball=-0.006, #-0.006
    # angles
    path_angle_deg=0.0,    # yaw in XY (in-to-out/out-to-in). + rotates toward +Y as X increases
    attack_angle_deg=0.0,  # pitch in XZ (down/up strike). + goes downward in -Z as X increases
    base_rvec=(0.8161, -1.4130, -0.8167),  # your 0° club pose (UR axis-angle)
    face_roll_deg=0.0,     # roll about club X (open/close face)
    # motion
    vel=0.3, acc=6.0,
    home=(-2.095, -2.618, -1.745, 1.22, 0.5236, -2.618),
    settle_s=0.5
):
    """
    Program: movej(home) -> movel(start) -> movel(end) -> sleep -> movej(home)

    Geometry:
      x_start, x_end are fixed planes.
      x_ball = (x_start + x_end)/2.
      y_start/y_end mirror around y_ball with slope dy/dx = tan(path_angle).
      z_start/z_end mirror around z_ball with slope dz/dx = -tan(attack_angle).

    Orientation:
      R = Rz(path_angle) * Ry(attack_angle) * R0
      (left-multiply so we rotate your 0° pose in world coordinates)
    """
    # Midpoint
    x_ball = 0.5 * (x_start + x_end)

    # Slopes from angles
    th_yaw   = np.deg2rad(path_angle_deg)
    th_pitch = np.deg2rad(attack_angle_deg)
    slope_y = np.tan(th_yaw)         # dy/dx
    slope_z = -np.tan(th_pitch)      # dz/dx (minus => +angle = descending)

    # Mirror Y and Z around the ball
    y_start = y_ball + slope_y * (x_start - x_ball)
    y_end   = y_ball + slope_y * (x_end   - x_ball)
    z_start = z_ball + slope_z * (x_start - x_ball)
    z_end   = z_ball + slope_z * (x_end   - x_ball)

    # Direction of travel (start -> end), i.e. the line through the ball
    d = np.array([x_end - x_start, y_end - y_start, z_end - z_start])
    Rdir = _frame_from_direction(d)        # X points along the swing line

    # Your 0° “mounting” orientation as a fixed offset
    R0 = _rvec_to_R(base_rvec)             # from your axis-angle
    Rtool = Rdir @ R0                      # club now points along the path

    # Optional: roll the face about the strike axis (local X)
    face_roll = np.deg2rad(face_roll_deg)
    if abs(face_roll) > 0:
        c, s = np.cos(face_roll), np.sin(face_roll)
        Rroll = np.array([[1, 0, 0],
                        [0, c, -s],
                        [0, s,  c]])
        Rtool = Rtool @ Rroll

    # Build poses with this orientation
    p_start = _pose_p((x_start, y_start, z_start), Rtool)
    p_end   = _pose_p((x_end,   y_end,   z_end),   Rtool)
    p_mid   = _pose_p((x_ball,  y_ball,  z_ball),  Rtool)

    # URScript
    prog = []
    prog.append("def swing():")
    print(f"Home: {home}")
    print(f"Start: {p_start}")
    print(f"Mid:   {p_mid}")
    print(f"End:   {p_end}")
    prog.append(f"  movej({list(home)}, 0.2, 0.2)")
    prog.append(f"  movel({p_start}, a=0.2, v=0.2)")
    prog.append(f"  movel({p_end},   a={acc:.3f}, v={vel:.3f})")
    if settle_s and settle_s > 0:
        prog.append(f"  sleep({float(settle_s):.3f})")
    
    prog.append(f"  movej({list(home)}, 0.2, 0.2)")
    # prog.append(f"  movel({p_mid}, a=0.2, v=0.2)")
    prog.append("end")
    prog.append("swing()")
    sock.sendall(("\n".join(prog) + "\n").encode("utf-8"))

    # Optional meta to annotate your own plots
    return {
        "x_start": x_start, "x_end": x_end,
        "x_ball": x_ball, "y_ball": y_ball, "z_ball": z_ball,
        "y_start": y_start, "y_end": y_end,
        "z_start": z_start, "z_end": z_end,
        "path_angle_deg": path_angle_deg,
        "attack_angle_deg": attack_angle_deg,
        "vel": vel, "acc": acc
    }

import time
from ur10_logger import UR10Logger
import socket

# ---------- example usage ----------
if __name__ == "__main__":
    # HOST = "192.168.56.101" #SIM
    HOST = "192.38.66.227"   # UR10
    PORT_logger = 30003
    PORT_cmd = 30002

    # 1) Set up your logger on its own socket/connection
    logger = UR10Logger(HOST, port=PORT_logger, log_folder="log")
    logger.connect()
    logger.start_logging()

    # 2) Open a separate socket for sending the program
    cmd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cmd.connect((HOST, PORT_cmd))

    print("Sending swing...")
    swing_meta = send_swing(cmd, x_start=-0.997, x_end=-0.297,
           y_ball=-0.456, z_ball=-0.040,
           path_angle_deg=-2, attack_angle_deg=0.0,
           vel=1.2, acc=7.5)

    # 3) Let the swing run and the logger collect a bit extra
    time.sleep(20.0)   # adjust to cover your full motion

    # 4) Clean up
    try:
        cmd.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    cmd.close()

    logger.stop_logging()
    logger.close()

    report = logger.remove_outliers()
    print(report)

    # 5) Save whatever streams you want (no plotting here)
    csv_path = logger.save_csv(which=("tcp","dtcp","q","dq"), suffix="swing")
    print(f"Saved: {csv_path}")

    # If you want plots, call your own helpers elsewhere, e.g.:
    logger.plot("q",   pi_axis=True,  save=True, show=True)
    logger.plot("dq",  pi_axis=False, save=True, show=False)
    # Pose time-series from FK:
    logger.plot_tcp("tcp",  show=True,  ztool=0.0)                 # x,y,z,rx,ry,rz
    # Twist time-series from FK-differences:
    logger.plot_tcp("dtcp", show=True,  ztool=0.0, smoothing=5)    # vx..wz
    logger.plot_tcp_xy(save=True, show=True, fk=True)  # XY path with equal axes (square)
    logger.plot_tcp_xyz(save=True, show=True, fk=True)        # 3D path (equal axis)
    # and you can use swing_meta to annotate the ball/planes in your own plot code.