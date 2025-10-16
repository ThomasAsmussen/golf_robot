import numpy as np
from config import Q_MIN, Q_MAX, KEEP_OUT_AABBS
from kinematics import fk_ur10

def within_aabb(p, aabb):
    (xmin,ymin,zmin),(xmax,ymax,zmax)=aabb
    return (xmin<=p[0]<=xmax) and (ymin<=p[1]<=ymax) and (zmin<=p[2]<=zmax)

def sample_link_points(frames):
    pts = [frames[i][:3,3] for i in range(1,7)]
    pts.append(frames[6][:3,3])  # TCP
    return pts

def first_safety_violation(Q):
    for k,q in enumerate(Q):
        if not (np.all(q>=Q_MIN) and np.all(q<=Q_MAX)):
            return False, "joint_limits", k, {"q": q}
        frames = fk_ur10(q)
        for P in sample_link_points(frames):
            for box in KEEP_OUT_AABBS:
                if within_aabb(P, box):
                    return False, "keepout", k, {"point": P, "box": box}
    return True, None, None, {}
