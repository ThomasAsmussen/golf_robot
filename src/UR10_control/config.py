import math
import numpy as np

# Network / I/O
HOST_ROBOT   = "192.168.56.101"
PORT_SCRIPT  = 30002       # URScript secondary
PORT_SERVOJ  = 30003       # stream
LOCAL_PORT   = 30020       # reverse-socket back to PC
STATE_FILE   = "last_q.npy"

# Timing
DT = 0.008                 # 125 Hz

# UR10 (CB2) DH (matches your MATLAB)
A = np.array([0.0, -0.612, -0.5723, 0.0, 0.0, 0.0], dtype=float)
D = np.array([0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922], dtype=float)
ZTOOL = 0.0

# Limits
Q_MIN   = np.array([-2*math.pi]*6)
Q_MAX   = np.array([ 2*math.pi]*6)
DQ_MAX  = np.deg2rad([120,120,180,180,180,180])
DDQ_MAX = np.deg2rad([300,300,400,400,400,400])

# Keep-outs (world/base frame, meters)
KEEP_OUT_AABBS = []  # [((xmin,ymin,zmin),(xmax,ymax,zmax)), ...]

# World <- Base (default identity: world == base)
T_WB = np.eye(4)

# Paramters at impact
IMPACT = {
    "enable": True,         # set False to ignore velocity at impact
    "index": 1,             # which waypoint is the "impact" pose (0-based in main_move waypoints)
    "speed_mps": 1,       # desired linear TCP speed at impact (m/s)
    # One of the two directions below must be provided (the other should be None):
    "direction_base": np.array([1.0, 0.0, 0.0]),  # unit-ish dir in BASE frame (will be normalized)
    "direction_tool": None,                       # OR dir in TOOL frame (e.g., [1,0,0] = tool x-axis)
}