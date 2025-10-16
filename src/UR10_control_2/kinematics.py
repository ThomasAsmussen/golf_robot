import numpy as np
from config import A, D, ZTOOL, Q_MIN, Q_MAX
from utils import rotm_from_rpy, T_from_xyz_rpy, unwrap_to_seed

def fk_ur10(q):
    c,s = np.cos, np.sin
    q1,q2,q3,q4,q5,q6 = q.tolist()
    T12 = np.array([[ c(q1), 0,  s(q1), 0],
                    [ s(q1), 0, -c(q1), 0],
                    [    0 , 1,     0 , D[0]],[0,0,0,1]])
    T23 = np.array([[ c(q2), -s(q2), 0, A[1]*c(q2)],
                    [ s(q2),  c(q2), 0, A[1]*s(q2)],
                    [    0 ,     0 , 1, 0],[0,0,0,1]])
    T34 = np.array([[ c(q3), -s(q3), 0, A[2]*c(q3)],
                    [ s(q3),  c(q3), 0, A[2]*s(q3)],
                    [    0 ,     0 , 1, 0],[0,0,0,1]])
    T45 = np.array([[ c(q4), 0,  s(q4), 0],
                    [ s(q4), 0, -c(q4), 0],
                    [   0  , 1,    0  , D[3]],[0,0,0,1]])
    T56 = np.array([[ c(q5), 0, -s(q5), 0],
                    [ s(q5), 0,  c(q5), 0],
                    [   0  ,-1,    0  , D[4]],[0,0,0,1]])
    T67 = np.array([[ c(q6), -s(q6), 0, 0],
                    [ s(q6),  c(q6), 0, 0],
                    [   0  ,    0 , 1, D[5]+ZTOOL],[0,0,0,1]])
    T01 = np.eye(4); T02 = T01 @ T12; T03 = T02 @ T23
    T04 = T03 @ T34; T05 = T04 @ T45; T06 = T05 @ T56; T07 = T06 @ T67
    return [T01,T02,T03,T04,T05,T06,T07]

def _norm(v): return float(np.linalg.norm(v))
def _saturate(x, lo=-1.0, hi=1.0): return float(np.clip(x, lo, hi))

def ur10_inv_kin(T):
    T_adj = np.array(T, float)
    T_adj[:3,3] = T[:3,3] - T[:3,2]*ZTOOL
    a = A.copy(); d = D.copy()

    a1 = -T_adj[1,3] + T_adj[1,2]*d[5]
    b1 =  T_adj[0,3] - T_adj[0,2]*d[5]
    temp = a1*a1 + b1*b1 - d[3]*d[3]
    if temp < -1e-12:
        return np.empty((0,6)), np.empty((0,), int)

    k = np.arctan2(np.sqrt(max(temp,0.0)), d[3])
    v1s = np.array([np.arctan2(b1,a1)+k, np.arctan2(b1,a1)-k])

    sols=[]; errs=[]
    for v1 in v1s:
        n1 = np.array([np.sin(v1), -np.cos(v1), 0.0])
        n5 = np.cross(n1, T_adj[:3,2]); n5 = n5/_norm(n5) if _norm(n5)>1e-6 else np.array([0,0,1.0])
        for flip in (1.0,-1.0):
            n5f = n5*flip
            p3 = T_adj[:3,3] - d[5]*T_adj[:3,2] + n5f*d[4]
            p1 = d[3]*n1 + np.array([0,0,d[0]])
            l  = p3 - p1; ll = _norm(l)
            err = int(ll > abs(a[1])+abs(a[2])+1e-12)

            cos_v3 = _saturate((a[1]**2 + a[2]**2 - ll**2)/(2*abs(a[1])*abs(a[2])))
            v3_abs = np.pi - np.arccos(cos_v3)

            denom = ll if ll>1e-12 else 1e-12
            lhat  = l/denom
            proj  = np.array([-np.cos(v1), -np.sin(v1), 0.0])
            phi   = np.arctan2(lhat[2], proj @ lhat)
            cos_t = _saturate((a[1]**2 - a[2]**2 + ll**2)/(2*abs(a[1])*ll))

            for elbow in (1,-1):
                v3 =  v3_abs if elbow==1 else -v3_abs
                v2 = -np.arccos(cos_t)-phi if elbow==1 else  np.arccos(cos_t)-phi
                v4 = np.arctan2(-n5f[2], n5f @ proj) - v3 - v2 + np.pi/2.0
                v5 = np.arctan2(np.dot(np.cross(-n5f, n1), T_adj[:3,2]), float(np.dot(n1, T_adj[:3,2])))
                v6 = np.arctan2(float(n5f @ T_adj[:3,0]), float(n5f @ T_adj[:3,1]))
                sols.append(np.array([v1,v2,v3,v4,v5,v6])); errs.append(err)
    if not sols: return np.empty((0,6)), np.empty((0,), int)
    return np.vstack(sols), np.array(errs, int)

def pick_ik_solution(T, q_seed=None, q_min=Q_MIN, q_max=Q_MAX):
    sols, errs = ur10_inv_kin(T)
    if sols.shape[0]==0: return None, None
    cands=[]; e2=[]
    for q,e in zip(sols,errs):
        q2 = unwrap_to_seed(q, q_seed)
        if np.all(q2>=q_min) and np.all(q2<=q_max):
            cands.append(q2); e2.append(e)
    if not cands: return None, None
    if q_seed is None: return cands[0], e2[0]
    i = int(np.argmin([np.linalg.norm(q - q_seed) for q in cands]))
    return cands[i], e2[i]

def numeric_jacobian(q, eps=1e-6):
    T0 = fk_ur10(q)[-1]
    p0, R0 = T0[:3,3], T0[:3,:3]
    J = np.zeros((6,6))
    for i in range(6):
        q_ = q.copy(); q_[i]+=eps
        Ti = fk_ur10(q_)[-1]
        pi, Ri = Ti[:3,3], Ti[:3,:3]
        v = (pi - p0)/eps
        dR = Ri @ R0.T
        w = np.array([dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]])/(2*eps)
        J[:,i] = np.hstack([v,w])
    return J
