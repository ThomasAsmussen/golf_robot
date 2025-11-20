import math
import numpy as np

# Network / I/O
HOST_ROBOT   = "127.0.0.1"
PORT_SCRIPT  = 30002       # URScript secondary
PORT_SERVOJ  = 30003       # stream
LOCAL_PORT   = 30020       # reverse-socket back to PC
STATE_FILE   = "last_q.npy"

# Timing
DT = 0.008                 # 125 Hz

# UR10 (CB2) DH (matches your MATLAB)
A = np.array([0.0, -0.612, -0.5723, 0.0, 0.0, 0.0], dtype=float)
D = np.array([0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922], dtype=float)
Z_TOOL = 0.0
Z_CLUB = 0.10
X_CLUB_SHORT = 0.040
Y_CLUB_SHORT = 0.755
X_CLUB_LONG = -0.065
Y_CLUB_LONG = 0.788
CLUB_MID = np.array([0.5*(X_CLUB_LONG+X_CLUB_SHORT),
                     0.5*(Y_CLUB_LONG+Y_CLUB_SHORT),
                     Z_CLUB], dtype=float)

# Limits
Q_MIN   = np.array([-2*math.pi, -math.pi, -2*math.pi, -2*math.pi, -2*math.pi, -2*math.pi])
Q_MAX   = np.array([ 2*math.pi,  0,        2*math.pi,  2*math.pi,  2*math.pi,  2*math.pi])
DQ_MAX  = np.deg2rad([120, 120, 180, 180, 180, 180])
DDQ_MAX  = np.deg2rad([180, 180, 270, 270, 270, 270])
# DDQ_MAX = np.deg2rad([240, 240, 360, 360, 360, 360])

# ============================================================================
# UNIFIED VELOCITY CONFIGURATION
# ============================================================================
# Configure both motion planner and controller velocity limits in ONE place
# 
# VMAX_JOINT: Maximum joint velocity limits (rad/s) - SINGLE SOURCE OF TRUTH
#   - This value controls BOTH the planner (dq_max) and controller (vmax) limits
#   - To increase/decrease robot speed, ONLY adjust this parameter
#   - Example values:
#       * Conservative: [π/4, π/4, π/3, π/3, π/3, π/3] (~45°/s, ~60°/s)
#       * Default:      [π/3, π/3, π/2, π/2, π/2, π/2] (~60°/s, ~90°/s)
#       * Aggressive:   [π/2, π/2, π,   π,   π,   π]   (~90°/s, ~180°/s)
#
# PLANNER_SAFETY_FACTOR: Safety margin for motion planner (0 < factor ≤ 1)
#   - The planner uses: dq_max = PLANNER_SAFETY_FACTOR * VMAX_JOINT
#   - This reserves headroom for the PD controller to track without saturating
#   - Default 0.8 means planner uses 80% of controller limit (20% headroom)
#   - Increase (e.g., 0.9) for more aggressive planning, decrease (e.g., 0.7) for more tracking margin
#
# To adjust robot speed:
#   1. Modify VMAX_JOINT to scale both planner and controller proportionally
#   2. If you see scale_factor warnings > 1.0, multiply VMAX_JOINT by that factor
#   3. Optionally adjust PLANNER_SAFETY_FACTOR if you want different planner/controller ratio
# ============================================================================

VMAX_JOINT = np.array([np.pi/2, np.pi/2, np.pi, np.pi, np.pi, np.pi])
PLANNER_SAFETY_FACTOR = 0.8

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