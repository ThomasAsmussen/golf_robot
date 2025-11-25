import math
import numpy as np

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
Z_PALLET = 0.155
CLUB_MID = np.array([0.5*(X_CLUB_LONG+X_CLUB_SHORT),
                     0.5*(Y_CLUB_LONG+Y_CLUB_SHORT),
                     Z_CLUB], dtype=float)

# Limits
Q_MIN   = np.array([-2*math.pi, -math.pi, -2*math.pi, -2*math.pi, -2*math.pi, -2*math.pi])
Q_MAX   = np.array([ 2*math.pi,  0,        2*math.pi,  2*math.pi,  2*math.pi,  2*math.pi])
DQ_MAX  = np.deg2rad([120, 120, 180, 180, 180, 180])
DDQ_MAX  = np.deg2rad([180, 180, 270, 270, 270, 270])