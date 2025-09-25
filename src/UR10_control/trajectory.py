import numpy as np
from .config import DQ_MAX, DDQ_MAX, DT
from .utils import unwrap_to_seed_all
from .safety import first_safety_violation
from .kinematics import numeric_jacobian


def quintic_coeffs(q0,dq0,ddq0,qf,dqf,ddqf,T):
    T2,T3,T4,T5=T*T,T**3,T**4,T**5
    a0=q0; a1=dq0; a2=0.5*ddq0
    M=np.array([[T3,T4,T5],[3*T2,4*T3,5*T4],[6*T,12*T2,20*T3]])
    b=np.array([qf-(a0+a1*T+a2*T2), dqf-(a1+2*a2*T), ddqf-(2*a2)])
    a3,a4,a5=np.linalg.solve(M,b)
    return np.array([a0,a1,a2,a3,a4,a5])

def eval_quintic(c,t):
    a0,a1,a2,a3,a4,a5=c
    q=a0+a1*t+a2*t**2+a3*t**3+a4*t**4+a5*t**5
    dq=a1+2*a2*t+3*a3*t**2+4*a4*t**3+5*a5*t**4
    ddq=2*a2+6*a3*t+12*a4*t**2+20*a5*t**3
    return q,dq,ddq

def auto_time_and_discretize(q0,dq0,ddq0,qf,dqf,ddqf,
                             dq_max=DQ_MAX, ddq_max=DDQ_MAX,
                             dt=DT, T_guess=1.0, T_max=8.0,
                             T_min=None, T_scale=1.0):
    T=T_guess
    for _ in range(30):
        coeffs=np.vstack([quintic_coeffs(q0[i],dq0[i],ddq0[i],qf[i],dqf[i],ddqf[i],T) for i in range(6)])
        ts=np.arange(0,T+1e-9,dt)
        Q=np.zeros((len(ts),6));DQ=np.zeros_like(Q);DDQ=np.zeros_like(Q)
        for k,t in enumerate(ts):
            for i in range(6):
                Q[k,i],DQ[k,i],DDQ[k,i]=eval_quintic(coeffs[i],t)
        if np.all(np.abs(DQ)<=dq_max+1e-9) and np.all(np.abs(DDQ)<=ddq_max+1e-9):
            break
        T*=1.2
        if T>T_max: break
    if T_min is not None: T=max(T,float(T_min))
    T*=float(T_scale)
    coeffs=np.vstack([quintic_coeffs(q0[i],dq0[i],ddq0[i],qf[i],dqf[i],ddqf[i],T) for i in range(6)])
    ts=np.arange(0,T+1e-9,dt)
    Q=np.zeros((len(ts),6));DQ=np.zeros_like(Q);DDQ=np.zeros_like(Q)
    for k,t in enumerate(ts):
        for i in range(6):
            Q[k,i],DQ[k,i],DDQ[k,i]=eval_quintic(coeffs[i],t)
    return ts,Q,DQ,DDQ,coeffs,T

def joint_vel_from_tcp(q, v_lin=None, w_ang=None, dq_max=DQ_MAX, damping=1e-4):
    """Map desired TCP velocity to a feasible joint velocity via damped least squares."""
    J = numeric_jacobian(q)  # 6x6 (linear; angular) columns per joint
    if v_lin is not None and w_ang is None:
        Jsel = J[:3, :]
        target = np.asarray(v_lin, float).reshape(3)
    else:
        if v_lin is None: v_lin = np.zeros(3)
        if w_ang is None: w_ang = np.zeros(3)
        Jsel = J
        target = np.hstack([v_lin, w_ang]).astype(float)

    H = Jsel.T @ Jsel + damping*np.eye(6)
    dq = np.linalg.solve(H, Jsel.T @ target)

    # Uniformly scale if any joint exceeds limits
    scale = np.max(np.abs(dq) / (dq_max + 1e-9))
    if scale > 1.0:
        dq /= scale
    return dq

def plan_piecewise_quintic(waypoints, dt=DT, impact=None, T_min_map=None, T_scale_map=None):
    """
    waypoints: [{'q': np.array(6), 'hold_vel': bool}, ...]
    impact: {'index': k, 'v_lin': np.array(3), 'w_ang': optional}
            Enforces joint end-velocity at waypoint k (end of segment k-1)
            and uses the same as the start-velocity of segment k (continuity).
    """
    T_min_map   = T_min_map or {}
    T_scale_map = T_scale_map or {}

    segments=[]
    prev_end_dq = np.zeros(6)
    for seg in range(len(waypoints)-1):
        q0 = waypoints[seg]['q']
        qf = unwrap_to_seed_all(waypoints[seg+1]['q'], q0)

        # boundary conditions (defaults)
        dq0  = prev_end_dq if seg > 0 else np.zeros(6)
        ddq0 = np.zeros(6)
        dqf  = np.zeros(6)
        ddqf = np.zeros(6)

        # if this segment ENDS at the impact waypoint, impose TCP velocity there
        if impact and (seg + 1) == impact['index']:
            v_lin = impact.get('v_lin', None)
            w_ang = impact.get('w_ang', None)
            dqf   = joint_vel_from_tcp(qf, v_lin=v_lin, w_ang=w_ang)
            next_start_dq = dqf.copy()   # continuity into next segment
        else:
            next_start_dq = np.zeros(6)

        # time-scaling and sampling
        ts,Q,DQ,DDQ,coeffs,T = auto_time_and_discretize(
            q0,dq0,ddq0, qf,dqf,ddqf,
            dt=dt,
            T_min=T_min_map.get(seg, None),
            T_scale=T_scale_map.get(seg, 1.0),
        )

        ok,reason,kbad,detail = first_safety_violation(Q)
        if not ok:
            raise RuntimeError(f"Safety violation on segment {seg} ({reason}) at sample {kbad}: {detail}")

        segments.append({'ts':ts,'Q':Q})
        prev_end_dq = next_start_dq

    Q_all = np.vstack([s['Q'] for s in segments])
    return segments, Q_all

# trajectory.py
def concat_with_holds(segments, dt, hold_after=None, hold_before=None):
    hold_after  = hold_after  or {}
    hold_before = hold_before or {}
    blocks = []
    for i, seg in enumerate(segments):
        if i in hold_before and hold_before[i] > 0.0:          # dwell BEFORE seg i
            n = max(1, int(hold_before[i] / dt))
            blocks.append(np.repeat(seg['Q'][0][None, :], n, axis=0))
        blocks.append(seg['Q'])
        if i in hold_after and hold_after[i] > 0.0:             # dwell AFTER seg i
            n = max(1, int(hold_after[i] / dt))
            blocks.append(np.repeat(seg['Q'][-1][None, :], n, axis=0))
    return np.vstack(blocks)
