import re, socket, struct, time, numpy as np
from .config import HOST_ROBOT, PORT_SCRIPT, PORT_SERVOJ, LOCAL_PORT, STATE_FILE, DT
from .utils import spin_until

_FLOAT_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

def _local_ip_for_peer(peer_ip: str) -> str:
    s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((peer_ip, 9)); ip=s.getsockname()[0]
    finally:
        s.close()
    return ip

def _parse_q_from_bytes(b: bytes):
    txt=b.decode("utf-8","ignore"); nums=_FLOAT_RE.findall(txt)
    vals=[float(x) for x in nums[:6]]
    return np.array(vals, float) if len(vals)==6 else None

def read_actual_q_reverse_socket(robot_ip=HOST_ROBOT, local_port=LOCAL_PORT, timeout_s=5.0):
    local_ip=_local_ip_for_peer(robot_ip)
    srv=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((local_ip, local_port)); srv.listen(1); srv.settimeout(timeout_s)
    script=f"""def send_q():
  socket_open("{local_ip}", {local_port})
  q = get_actual_joint_positions()
  socket_send_string(to_str(q[0])+","+to_str(q[1])+","+to_str(q[2])+","+to_str(q[3])+","+to_str(q[4])+","+to_str(q[5])+"\\n")
  socket_close()
end
send_q()
"""
    try:
        cli=socket.socket(socket.AF_INET, socket.SOCK_STREAM); cli.settimeout(2.0)
        cli.connect((robot_ip, PORT_SCRIPT)); cli.sendall(script.encode("ascii")); cli.close()
    except Exception:
        srv.close(); return None
    q=None
    try:
        conn,_=srv.accept(); conn.settimeout(timeout_s)
        data=b""
        while True:
            chunk=conn.recv(1024)
            if not chunk: break
            data+=chunk
            if b"\n" in data: break
        conn.close(); q=_parse_q_from_bytes(data)
    except Exception:
        q=None
    finally:
        srv.close()
    return q

def movej(q_target, host=HOST_ROBOT, a=1.0, v=0.15):
    prog=f"movej([{','.join(f'{x:.6f}' for x in q_target)}], a={a:.3f}, v={v:.3f})\n"
    s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, PORT_SCRIPT)); s.sendall(prog.encode('ascii')); s.close()

def stream_servoj(Q, host=HOST_ROBOT, port=PORT_SERVOJ, dt=DT, lookahead=0.20, gain=1200,
                  prewarm_time=0.30, settle_time=0.50):
    s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.connect((host,port))
    try:
        # prewarm at first point
        q0=Q[0]; t_next=time.perf_counter()+0.05
        for _ in range(max(1,int(prewarm_time/dt))):
            cmd=f"servoj([{','.join(f'{x:.6f}' for x in q0)}], t={dt:.6f}, lookahead_time={lookahead:.3f}, gain={gain})\n"
            s.send(cmd.encode('ascii')); spin_until(t_next); t_next+=dt
        # stream
        for q in Q:
            cmd=f"servoj([{','.join(f'{x:.6f}' for x in q)}], t={dt:.6f}, lookahead_time={lookahead:.3f}, gain={gain})\n"
            s.send(cmd.encode('ascii')); spin_until(t_next); t_next+=dt
        # hold at end
        q_last=Q[-1]
        for _ in range(max(1,int(settle_time/dt))):
            cmd=f"servoj([{','.join(f'{x:.6f}' for x in q_last)}], t={dt:.6f}, lookahead_time={lookahead:.3f}, gain={gain})\n"
            s.send(cmd.encode('ascii')); spin_until(t_next); t_next+=dt
        s.send(b"stopj(1.0)\n")
    finally:
        s.close()

def stream_servoj_with_feedback(Q, host=HOST_ROBOT, fb_port=LOCAL_PORT, dt=DT,
                                    lookahead=0.20, gain=1200): # for CB2
    """
    CB2-safe feedback streamer:
      - PC sends SIX newline-separated floats per setpoint.
      - Robot reads one float at a time and calls servoj().
      - Robot sends back actual joints as CSV per cycle.
    Returns: np.ndarray [M,6] of actual joints.
    """
    local_ip = _local_ip_for_peer(host)

    # 1) Server for duplex connection
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((local_ip, fb_port))
    srv.listen(1)

    # 2) URScript (CB2-safe: read 1 float at a time)
    script = f"""def follow_stream_cb2():
  socket_open("{local_ip}", {fb_port})
  t  = {dt:.6f}
  la = {lookahead:.3f}
  g  = {gain:.0f}
  while True:
    f0 = socket_read_ascii_float(1, 2.0)
    if f0[0] == 9999.0:
      break
    f1 = socket_read_ascii_float(1, 2.0)
    f2 = socket_read_ascii_float(1, 2.0)
    f3 = socket_read_ascii_float(1, 2.0)
    f4 = socket_read_ascii_float(1, 2.0)
    f5 = socket_read_ascii_float(1, 2.0)
    tgt = [f0[0], f1[0], f2[0], f3[0], f4[0], f5[0]]
    servoj(tgt, t=t, lookahead_time=la, gain=g)
    q = get_actual_joint_positions()
    socket_send_string(to_str(q[0])+","+to_str(q[1])+","+to_str(q[2])+","+to_str(q[3])+","+to_str(q[4])+","+to_str(q[5])+"\\n")
  socket_close()
end
follow_stream_cb2()
"""

    # 3) Start program on robot
    try:
        cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cli.settimeout(2.0)
        cli.connect((host, PORT_SCRIPT))
        cli.sendall(script.encode("ascii"))
        cli.close()
    except Exception as e:
        srv.close()
        raise RuntimeError(f"Could not start controller stream program: {e}")

    # 4) Accept connection and stream targets while reading feedback
    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    q_actual = []
    try:
        for q in Q:
            # send SIX lines, each a single float
            for x in q:
                conn.sendall(f"{x:.6f}\n".encode("ascii"))
            # read one CSV line back with actual joints
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(1024)
                if not chunk:
                    break
                data += chunk
            ja = _parse_q_from_bytes(data)
            if ja is not None:
                q_actual.append(ja)

        # sentinel: six lines of 9999 to stop
        for _ in range(6):
            conn.sendall(b"9999\n")
    finally:
        conn.close()
        srv.close()

    return np.asarray(q_actual, dtype=float)


def start_joint_publisher(host=HOST_ROBOT, local_port=LOCAL_PORT, duration_s=4.0, sample_dt=0.03):
    """
    Start ONE URScript program that pushes actual joints to us at sample_dt
    for ~duration_s. Returns (srv, conn) sockets so caller can read lines.
    Close both when done.
    """
    local_ip = _local_ip_for_peer(host)

    # server to receive the stream
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((local_ip, local_port))
    srv.listen(1)

    script = f"""def joint_pub():
  socket_open("{local_ip}", {local_port})
  t0 = get_robot_time()
  while (get_robot_time() - t0) < {duration_s:.3f}:
    q = get_actual_joint_positions()
    socket_send_string(to_str(q[0])+","+to_str(q[1])+","+to_str(q[2])+","+to_str(q[3])+","+to_str(q[4])+","+to_str(q[5])+"\\n")
    sleep({sample_dt:.3f})
  socket_close()
end
joint_pub()
"""
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.settimeout(2.0)
    cli.connect((host, PORT_SCRIPT))
    cli.sendall(script.encode("ascii"))
    cli.close()

    conn, _ = srv.accept()
    conn.settimeout(2.0)
    return srv, conn

def read_joint_stream(conn, out_list):
    """Blocking reader: append np.array(6,) to out_list for each CSV line."""
    buf = b""
    try:
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                i = buf.find(b"\n")
                if i < 0:
                    break
                line = buf[:i+1]; buf = buf[i+1:]
                q = _parse_q_from_bytes(line)
                if q is not None:
                    out_list.append(q)
    finally:
        try: conn.close()
        except: pass
        

def _find_q_offset(pkt):
    """
    Heuristic: scan for 6 consecutive big-endian doubles with values in [-2π, 2π].
    Returns byte offset or -1 if not found.
    """
    lo, hi = -2*np.pi - 0.1, 2*np.pi + 0.1
    n = len(pkt)
    # Doubles must be 8-byte aligned; step by 8
    for off in range(0, n - 6*8 + 1, 8):
        try:
            vals = struct.unpack_from("!6d", pkt, off)
        except struct.error:
            break
        ok = all(lo <= v <= hi for v in vals)
        if ok:
            return off
    return -1

def read_q_stream_primary(host, duration_s=4.0, max_hz=125):
    """
    Connect to 30001 and read joint angles for up to duration_s.
    Returns np.ndarray [N,6] of q_actual in radians.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    s.connect((host, 30001))
    s.settimeout(0.2)

    t_end = time.time() + float(duration_s)
    q_list = []
    buf = b""
    q_off = None
    last_t = 0.0
    min_dt = 1.0 / float(max_hz)

    try:
        while time.time() < t_end:
            try:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
            except socket.timeout:
                pass

            # Packets on 30001 are framed: 4-byte big-endian length prefix
            while len(buf) >= 4:
                (plen,) = struct.unpack_from("!i", buf, 0)
                if plen <= 0 or plen > 20000:
                    # Desync; drop one byte
                    buf = buf[1:]
                    continue
                if len(buf) < 4 + plen:
                    break
                pkt = buf[4:4+plen]
                buf = buf[4+plen:]

                # pick offset once
                if q_off is None:
                    q_off = _find_q_offset(pkt)
                    if q_off < 0:
                        continue

                # rate-limit
                now = time.time()
                if now - last_t < min_dt:
                    continue
                last_t = now

                try:
                    q = struct.unpack_from("!6d", pkt, q_off)
                    q_list.append(q)
                except struct.error:
                    # if layout changed mid-run, reset detection
                    q_off = None
                    continue
    finally:
        s.close()

    return np.asarray(q_list, dtype=float)


def read_q_stream_realtime(host, duration_s=4.0, max_hz=125):
    """
    Passive reader for real-time client stream on port 30003.
    Returns np.ndarray [N,6] of joint positions (rad).
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    s.connect((host, 30003))
    s.settimeout(0.2)

    t_end = time.time() + float(duration_s)
    q_list = []
    buf = b""
    q_off = None
    last_t = 0.0
    min_dt = 1.0 / float(max_hz)

    try:
        while time.time() < t_end:
            try:
                chunk = s.recv(8192)
                if not chunk:
                    break
                buf += chunk
            except socket.timeout:
                pass

            # Frames are: 4-byte big-endian length, then payload
            while len(buf) >= 4:
                (plen,) = struct.unpack_from("!i", buf, 0)
                if plen <= 0 or plen > 20000:
                    buf = buf[1:]  # desync guard
                    continue
                if len(buf) < 4 + plen:
                    break
                pkt = buf[4:4+plen]
                buf = buf[4+plen:]

                if q_off is None:
                    q_off = _find_q_offset(pkt)
                    if q_off < 0:
                        continue  # try next packet

                now = time.time()
                if now - last_t < min_dt:
                    continue
                last_t = now

                try:
                    q = struct.unpack_from("!6d", pkt, q_off)
                    q_list.append(q)
                except struct.error:
                    q_off = None  # layout changed; rescan
                    continue
    finally:
        s.close()

    return np.asarray(q_list, dtype=float)



def movej_blocking(q_target, host=HOST_ROBOT, a=1.2, v=0.20,
                   tol=8e-4, max_wait_s=15.0, poll_period=0.05):
    """
    Send movej, then poll actual joints via reverse socket until ||q-q_target|| < tol.
    Returns True if reached, else False.
    """
    # fire movej
    prog = f"movej([{','.join(f'{x:.6f}' for x in q_target)}], a={a:.3f}, v={v:.3f})\n"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, PORT_SCRIPT)); s.sendall(prog.encode('ascii')); s.close()

    # poll until inside tolerance (rotate local ports to avoid TIME_WAIT issues)
    t0 = time.time()
    i = 0
    base_port = 30020
    consecutive = 0
    while time.time() - t0 < max_wait_s:
        port = base_port + (i % 8); i += 1
        q = read_actual_q_reverse_socket(host, local_port=port, timeout_s=0.8)
        if q is not None:
            if np.linalg.norm(q - q_target) < tol:
                consecutive += 1
                if consecutive >= 3:  # a few consistent reads
                    return True
            else:
                consecutive = 0
        time.sleep(poll_period)
    return False


def movej_and_wait(q_target, host=HOST_ROBOT, a=1.2, v=0.25,
                   wait_s=None, settle_s=0.2, q_start=None):
    """
    Fire movej() and then wait:
      - if wait_s is given, just sleep that long
      - else estimate time from joint distance using trapezoid/triangle profile
    """
    # send movej
    prog = f"movej([{','.join(f'{x:.6f}' for x in q_target)}], a={a:.3f}, v={v:.3f})\n"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, PORT_SCRIPT)); s.sendall(prog.encode('ascii')); s.close()

    # decide how long to wait
    if wait_s is None:
        # try to estimate from current joints (optional)
        if q_start is None:
            q_start = read_actual_q_reverse_socket(host, timeout_s=1.0)
        if q_start is None:
            wait_s = 5.0  # pure "sleep 5s" fallback
        else:
            dq = np.abs(np.asarray(q_target) - np.asarray(q_start))
            # trapezoid/triangle per joint with peak vel v and accel a
            t_list = []
            for di in dq:
                if di <= (v*v)/a:           # triangular
                    t = 2.0*np.sqrt(di/a)
                else:                       # trapezoidal
                    t = (di - (v*v)/a)/v + 2.0*(v/a)
                t_list.append(t)
            wait_s = max(t_list) + float(settle_s)
    time.sleep(float(wait_s))
    
    
    
def read_q_stream_realtime_locked(host, duration_s=4.0, q_hint=None,
                                  max_hz=125, max_jump_norm=0.6, relock=True,
                                  print_debug=False):
    """
    Listen on 30003 and extract q_actual by:
      • scanning each frame for all 6-double candidates (big-endian),
      • first lock = candidate closest to q_hint (if given) or first candidate,
      • continuity gate: reject jumps > max_jump_norm [rad],
      • optional re-lock if continuity breaks.

    Returns: np.ndarray [N,6] in radians.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    s.connect((host, 30003))
    s.settimeout(0.2)

    t_end   = time.time() + float(duration_s)
    q_list  = []
    buf     = b""
    last_q  = None if q_hint is None else np.asarray(q_hint, float)
    min_dt  = 1.0 / float(max_hz)
    last_t  = 0.0
    have_lock = False
    lock_off  = None

    def scan_candidates(pkt):
        """Yield (off, qvec) for 6-double big-endian blocks inside pkt."""
        n = len(pkt)
        for off in range(0, n - 6*8 + 1, 8):  # 8-byte aligned
            try:
                q = struct.unpack_from("!6d", pkt, off)
            except struct.error:
                break
            # coarse angle range guard
            if all(-2*np.pi - 0.2 <= v <= 2*np.pi + 0.2 for v in q):
                yield off, np.asarray(q, float)

    try:
        while time.time() < t_end:
            try:
                chunk = s.recv(8192)
                if not chunk:
                    break
                buf += chunk
            except socket.timeout:
                pass

            # Frames: 4-byte big-endian length prefix, then payload
            while len(buf) >= 4:
                (plen,) = struct.unpack_from("!i", buf, 0)
                if plen <= 0 or plen > 20000:
                    buf = buf[1:]  # resync
                    continue
                if len(buf) < 4 + plen:
                    break
                pkt = buf[4:4+plen]
                buf = buf[4+plen:]

                now = time.time()
                if now - last_t < min_dt:
                    continue

                if not have_lock:
                    # choose the candidate closest to the hint/last_q
                    best = None; best_d = 1e9; best_off = None
                    for off, q in scan_candidates(pkt):
                        d = 0.0 if last_q is None else float(np.linalg.norm(q - last_q))
                        if d < best_d:
                            best_d, best, best_off = d, q, off
                    if best is not None:
                        # sanity: if we had a hint, require “reasonably close”
                        if (last_q is None) or (best_d < 1.0):
                            lock_off = best_off
                            have_lock = True
                            if print_debug:
                                print(f"[30003] lock at byte {lock_off}, d={best_d:.3f}")
                            q_cur = best
                        else:
                            continue
                    else:
                        continue
                else:
                    # read from the locked offset
                    try:
                        q_cur = np.asarray(struct.unpack_from("!6d", pkt, lock_off), float)
                    except struct.error:
                        have_lock = False
                        lock_off  = None
                        continue
                    if last_q is not None:
                        if float(np.linalg.norm(q_cur - last_q)) > max_jump_norm:
                            if relock:
                                have_lock = False
                                lock_off  = None
                                if print_debug:
                                    print("[30003] continuity break → re-lock")
                                continue
                            else:
                                # reject outlier
                                continue

                q_list.append(q_cur)
                last_q = q_cur
                last_t = now
    finally:
        s.close()

    return np.asarray(q_list, dtype=float)