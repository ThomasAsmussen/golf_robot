import numpy as np, time
from config import HOST_ROBOT, DT, STATE_FILE, IMPACT
from utils import T_from_xyz_rpy, normalize
from kinematics import pick_ik_solution, fk_ur10
from trajectory import plan_piecewise_quintic, concat_with_holds
from comms import read_actual_q_reverse_socket, stream_servoj, stream_speedj, read_q_stream_realtime, movej_and_wait, stream_servoj_with_feedback, start_joint_publisher, read_joint_stream, read_q_stream_primary, movej_blocking
from plotter import tcp_path_from_Q, plot_paths
from ur10_logger import UR10Logger

def load_last_q(path):
    try:
        q=np.load(path); 
        return q if q.shape==(6,) else None
    except Exception:
        return None
def save_last_q(path,q):
    try: np.save(path, np.asarray(q,float))
    except Exception: pass

def main():
    # 1) Start state (actual joints if possible)
    q_start = read_actual_q_reverse_socket(HOST_ROBOT, timeout_s=5.0)
    if q_start is None:
        try:
            q_start = load_last_q(STATE_FILE)
        except:
            np.array([0.0,-1.57,1.57,-1.57,1.57,0.0])
            print("Using fallback start:", q_start)
    else:
        print("Start from robot:", q_start)

    # 2) Task-space waypoints (your example with roll=-pi/2, tool-down)
    T_preimpact = T_from_xyz_rpy(-0.1, -0.5, 0.30, -np.pi/2, np.pi, 0.0)
    T_impact    = T_from_xyz_rpy( 0.0, -0.5, 0.30, -np.pi/2, np.pi, 0.0)
    T_exit      = T_from_xyz_rpy( 0.1, -0.5, 0.30, -np.pi/2, np.pi, 0.0)

    # 3) IK chain (unwrap handled inside)
    q_pre,_ = pick_ik_solution(T_preimpact, q_seed=q_start);  assert q_pre is not None, "IK preimpact fail"
    q_imp,_ = pick_ik_solution(T_impact,    q_seed=q_pre);    assert q_imp is not None, "IK impact fail"
    q_exi,_ = pick_ik_solution(T_exit,      q_seed=q_imp);    assert q_exi is not None, "IK exit fail"
    
    # 4) Build impact-velocity request from config
    impact = None
    if IMPACT.get("enable", False):
        speed = float(IMPACT.get("speed_mps", 0.0))
        dir_base = IMPACT.get("direction_base", None)
        dir_tool = IMPACT.get("direction_tool", None)

        if dir_base is not None:
            v_dir_base = normalize(dir_base)
        elif dir_tool is not None:
            T_imp = fk_ur10(q_imp)[-1]
            R_imp = T_imp[:3, :3]
            v_dir_base = normalize(R_imp @ np.asarray(dir_tool, float))
        else:
            raise ValueError("IMPACT config must set either 'direction_base' or 'direction_tool'.")

        v_lin = v_dir_base * speed
        impact = {"index": int(IMPACT.get("index", 1)), "v_lin": v_lin}


    waypoints = [
        {'q': q_pre, 'hold_vel': True},    # index 0
        {'q': q_imp, 'hold_vel': False},   # index 1  <-- impact waypoint
        {'q': q_exi, 'hold_vel': True},    # index 2
    ]

    # slow approach to impact, then normal exit (optional per-seg timing via T_min_map/T_scale_map)
    segments, Q_all = plan_piecewise_quintic(waypoints, dt=DT, impact=impact)
    

    q0 = segments[0]['Q'][0]                  # first sample of the trajectory
    movej_and_wait(q0, a=1.2, v=0.30, wait_s=5)  # or wait_s=5.0 for fixed sleep
    # tiny settle (optional)
    #time.sleep(0.2)

    # planned TCP path once (optional, if you plot)
    P_plan = tcp_path_from_Q(Q_all, fk_ur10)

    # how long the swing runs (sec)
    swing_time = DT * len(Q_all)
    read_time  = swing_time + 1.5  # margin

    # --- start UR10Logger on 30003 (records q, dq, tcp/dtcp if present) ---
    logger = UR10Logger(HOST_ROBOT, port=30013, log_folder="log")
    logger.connect(timeout=2.0)
    logger.start_logging(print_rows=False)
    try:
        # --- stream the trajectory over 30002 (controller executes URScript) ---
        stream_speedj(Q_all, host=HOST_ROBOT, port=30003, dt=DT, a=3.0, ema=0.2)
        #stream_servoj(Q_all, host=HOST_ROBOT, port=30002, dt=DT, lookahead=0.05, gain=1200)  # :contentReference[oaicite:1]{index=1}
    finally:
        # ensure the logger stops even if streaming raises
        logger.stop_logging()
        logger.close()


    # --- pull recorded data from logger ---
    q_actual_series = logger.as_array("q")     # Nx6, may be empty if connection failed
    t_series        = logger.time()

    if q_actual_series is not None and len(q_actual_series) > 1:
        q_last = q_actual_series[-1]
    else:
        q_last = Q_all[-1]

    save_last_q(STATE_FILE, q_last)
    print("Saved last joints to", STATE_FILE)

    path = logger.save_csv(which=("tcp","dtcp","q","dq"), suffix="run1")
    print("Saved CSV:", path)

    # Optional: plots into ./log
    logger.plot("q",  pi_axis=True,  save=True, show=False, title="Joint positions vs time")
    logger.plot("dq", pi_axis=False, save=True, show=False, title="Joint velocities vs time")

    # FK-based pose/twist (robust even if tcp/dtcp aren’t streamed)
    logger.plot_tcp("tcp",  show=False, ztool=0.0)
    logger.plot_tcp("dtcp", show=False, ztool=0.0, smoothing=5)
    logger.plot_tcp_xy(save=True, show=False, fk=True)
    logger.plot_tcp_xyz(save=True, show=False, fk=True)

if __name__ == "__main__":
    main()
