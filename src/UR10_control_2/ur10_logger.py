# ur10_logger.py
import os, time, socket, struct, threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

def _pi_formatter(val, pos):
    n = val / np.pi
    if np.isclose(n, 0): return r"$0$"
    if np.isclose(n, 1): return r"$\pi$"
    if np.isclose(n, -1): return r"$-\pi$"
    if float(n).is_integer(): return rf"${int(n)}\pi$"
    return rf"${n:.2f}\pi$"

class UR10Logger:
    """
    Async, schema-driven logger for UR10 CB2/CB3 real-time stream (30003).
    - Default fields: q, dq, tcp, dtcp
    - Extend with extra_fields={"name":[idx,...], ...}
    - plot(name="q"|"dq"|"tcp"|"dtcp", pi_axis=True/False)
    - save_csv(which=("q","dq","tcp","dtcp"))
    """
    def __init__(self, host, port=30003, log_folder="log", extra_fields=None):
        self.host, self.port = host, port
        self.log_folder = log_folder
        os.makedirs(self.log_folder, exist_ok=True)

        # ---- your requested defaults ----
        self.fields = {
            "q":   [30, 31, 32, 33, 34, 35],
            "dq":  [36, 37, 38, 39, 40, 41],
            "tcp": [54, 55, 56, 57, 58, 59], # ONLY WORKS ON SIM
            "dtcp":[60, 61, 62, 63, 64, 65], # ONLY WORKS ON SIM
        }
        if extra_fields:
            self.fields.update(extra_fields)

        # precompute byte offsets
        self.offsets = {k: [12 + 8*i for i in idxs] for k, idxs in self.fields.items()}

        # data buffers
        self.data = {"t": []}
        for k in self.fields:
            self.data[k] = []  # list of lists (N x width)

        # sockets/threading
        self.sock = None
        self.t0 = None
        self._run = threading.Event()
        self._thr = None
        self._lock = threading.Lock()  # protect self.data

    # ---- connection ----
    def connect(self, timeout=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if timeout is not None:
            self.sock.settimeout(timeout)
        self.sock.connect((self.host, self.port))
        self.sock.settimeout(0.05)  # short timeout so stop is responsive
        self.t0 = time.time()

    def close(self):
        self.stop_logging()
        if self.sock:
            try: self.sock.close()
            except: pass
            self.sock = None

    # ---- async logging ----
    def start_logging(self, recv_bytes=4096, print_rows=False):
        if not self.sock or self.t0 is None:
            raise RuntimeError("Not connected. Call connect() first.")
        if self._thr and self._thr.is_alive():
            return
        self._run.set()
        self._thr = threading.Thread(
            target=self._loop, args=(recv_bytes, print_rows), daemon=True
        )
        self._thr.start()

    def stop_logging(self):
        self._run.clear()
        if self._thr and self._thr.is_alive():
            self._thr.join(timeout=1.0)
        self._thr = None

    def _loop(self, recv_bytes, print_rows):
        while self._run.is_set():
            self._log_step(recv_bytes, print_rows)

    def _log_step(self, recv_bytes, print_rows):
        if not self.sock: return
        try:
            pkt = self.sock.recv(recv_bytes)
        except socket.timeout:
            return
        if not pkt: return

        parsed = {}
        for name, offs in self.offsets.items():
            vals = []
            for off in offs:
                chunk = pkt[off:off+8]
                if len(chunk) < 8:
                    vals = None
                    break
                vals.append(struct.unpack(">d", chunk)[0])
            if vals is not None:
                parsed[name] = vals

        if not parsed:
            return

        t_rel = time.time() - self.t0
        with self._lock:
            self.data["t"].append(t_rel)
            for name in self.fields:
                width = len(self.fields[name])
                self.data[name].append(parsed.get(name, [np.nan]*width))

        if print_rows:
            q = parsed.get("q")
            dq = parsed.get("dq")
            q_str = ", ".join(f"{x:.3f}" for x in q) if q else "-"
            dq_str = ", ".join(f"{x:.3f}" for x in dq) if dq else "-"
            print(f"t={t_rel:.3f} | q=[{q_str}] | dq=[{dq_str}]")

    # ---- accessors ----
    def time(self) -> np.ndarray:
        with self._lock:
            return np.asarray(self.data["t"], dtype=float)

    def as_array(self, name: str) -> np.ndarray:
        if name not in self.fields:
            raise KeyError(f"Unknown field '{name}'. Known: {list(self.fields)}")
        with self._lock:
            rows = self.data[name]
            w = len(self.fields[name])
            if not rows:
                return np.empty((0, w), dtype=float)
            return np.asarray(rows, dtype=float)

    # ---- save ----
    def save_csv(self, which=("q","dq"), suffix=None):
        t = self.time()
        if t.size == 0:
            raise RuntimeError("No data to save.")
        if isinstance(which, str):
            which = (which,)

        mats = [t.reshape(-1,1)]
        header = ["t"]
        for name in which:
            A = self.as_array(name)
            mats.append(A)
            if name == "q":
                header += [f"q{i+1}" for i in range(A.shape[1])]
            elif name == "dq":
                header += [f"dq{i+1}" for i in range(A.shape[1])]
            elif name == "tcp":
                header += ["x","y","z","rx","ry","rz"][:A.shape[1]]
            elif name == "dtcp":
                header += ["vx","vy","vz","wx","wy","wz"][:A.shape[1]]
            else:
                header += [f"{name}{i+1}" for i in range(A.shape[1])]

        M = np.concatenate(mats, axis=1)
        ts = time.strftime("%Y%m%d_%H%M%S")
        if suffix is None:
            suffix = "_".join(which)
        path = os.path.join(self.log_folder, f"{suffix}_{ts}.csv")
        np.savetxt(path, M, delimiter=",", header=",".join(header), comments="")
        return path

    # ---- plot ----
    def plot(self, name="q", pi_axis=False, save=True, show=False, title=None):
        t = self.time()
        A = self.as_array(name)
        if A.shape[0] == 0:
            print("No data to plot.")
            return None

        labels = {
            "q":   ["Base","Shoulder","Elbow","Wrist1","Wrist2","Wrist3"],
            "dq":  ["Base","Shoulder","Elbow","Wrist1","Wrist2","Wrist3"],
            "tcp": ["x","y","z","rx","ry","rz"],
            "dtcp":["vx","vy","vz","wx","wy","wz"],
        }.get(name, [f"{name}{i+1}" for i in range(A.shape[1])])

        plt.close("all")
        fig, ax = plt.subplots()
        for i in range(A.shape[1]):
            ax.plot(t, A[:, i], label=labels[i], linewidth=1)

        if name == "q":
            ax.set_ylabel("Joint position [rad]")
            if pi_axis:
                ax.yaxis.set_major_locator(MultipleLocator(np.pi/4))
                ax.yaxis.set_major_formatter(FuncFormatter(_pi_formatter))
        elif name == "dq":
            ax.set_ylabel("Joint velocity [rad/s]")
        elif name == "tcp":
            ax.set_ylabel("TCP pose [SI units]")
        elif name == "dtcp":
            ax.set_ylabel("TCP twist [SI units]")

        ax.set_xlabel("Time [s]")
        ax.set_title(title or f"{name} over time")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        fig.tight_layout()

        out = None
        if save:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out = os.path.join(self.log_folder, f"log_{name}_{ts}.png")
            fig.savefig(out, dpi=300)
        if show: plt.show()
        else: plt.close(fig)
        return out


    def _ufwdkin(self, q, ztool=0.0):
        """
        UR10 FK to tool tip. q: iterable length 6 (rad). Returns 4x4 T.
        DH-like chain matching your earlier code (meters).
        """
        q1, q2, q3, q4, q5, q6 = [float(x) for x in q]
        a = [0.0, -0.612, -0.5723, 0.0, 0.0, 0.0]
        d = [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922]

        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        c3, s3 = np.cos(q3), np.sin(q3)
        c4, s4 = np.cos(q4), np.sin(q4)
        c5, s5 = np.cos(q5), np.sin(q5)
        c6, s6 = np.cos(q6), np.sin(q6)

        T12 = np.array([[ c1,  0.0,  s1,  0.0],
                        [ s1,  0.0, -c1,  0.0],
                        [0.0,  1.0,  0.0,  d[0]],
                        [0.0,  0.0,  0.0,  1.0]])
        T23 = np.array([[ c2, -s2, 0.0, a[1]*c2],
                        [ s2,  c2, 0.0, a[1]*s2],
                        [0.0,  0.0, 1.0,      0.0],
                        [0.0,  0.0, 0.0,      1.0]])
        T34 = np.array([[ c3, -s3, 0.0, a[2]*c3],
                        [ s3,  c3, 0.0, a[2]*s3],
                        [0.0,  0.0, 1.0,      0.0],
                        [0.0,  0.0, 0.0,      1.0]])
        T45 = np.array([[ c4,  0.0,  s4,  0.0],
                        [ s4,  0.0, -c4,  0.0],
                        [0.0,  1.0,  0.0,  d[3]],
                        [0.0,  0.0,  0.0,  1.0]])
        T56 = np.array([[ c5,  0.0, -s5,  0.0],
                        [ s5,  0.0,  c5,  0.0],
                        [0.0, -1.0,  0.0,  d[4]],
                        [0.0,  0.0,  0.0,  1.0]])
        T67 = np.array([[ c6, -s6, 0.0, 0.0],
                        [ s6,  c6, 0.0, 0.0],
                        [0.0,  0.0, 1.0, d[5] + ztool],
                        [0.0,  0.0, 0.0, 1.0]])
        return (((T12 @ T23) @ T34) @ T45) @ T56 @ T67

    def _tcp_from_joints(self, q, ztool=0.0):
        T = self._ufwdkin(q, ztool=ztool)
        p = T[:3, 3]
        return float(p[0]), float(p[1]), float(p[2])

    def _rmat_to_axis_angle(self, R: np.ndarray):
        tr = float(np.clip(np.trace(R), -1.0, 3.0))
        ang = float(np.arccos((tr - 1.0) / 2.0))
        if np.isclose(ang, 0.0):
            return 0.0, 0.0, 0.0
        if np.isclose(ang, np.pi):
            rx = np.sqrt(max(0.0, (R[0,0] + 1)/2))
            ry = np.sqrt(max(0.0, (R[1,1] + 1)/2))
            rz = np.sqrt(max(0.0, (R[2,2] + 1)/2))
            rx = np.copysign(rx, R[2,1] - R[1,2])
            ry = np.copysign(ry, R[0,2] - R[2,0])
            rz = np.copysign(rz, R[1,0] - R[0,1])
            axis = np.array([rx, ry, rz], dtype=float)
            axis /= (np.linalg.norm(axis) + 1e-12)
            r = axis * ang
            return float(r[0]), float(r[1]), float(r[2])
        axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], dtype=float)
        axis /= (2.0*np.sin(ang) + 1e-12)
        axis /= (np.linalg.norm(axis) + 1e-12)
        r = axis * ang
        return float(r[0]), float(r[1]), float(r[2])

    def _fk_tcp6_series(self, ztool: float = 0.0):
        """(t, TCP6, Rlist) with TCP6=[x y z rx ry rz] via FK."""
        Q = self.as_array("q")
        t = self.time()
        if Q.shape[0] == 0:
            return t[:0], np.empty((0,6)), []
        TCP = np.empty((Q.shape[0], 6), dtype=float)
        Rlist = []
        for i, q in enumerate(Q):
            T = self._ufwdkin(q, ztool=ztool)
            p = T[:3, 3]
            rx, ry, rz = self._rmat_to_axis_angle(T[:3, :3])
            TCP[i, :3] = p
            TCP[i, 3:] = (rx, ry, rz)
            Rlist.append(T[:3, :3])
        return t, TCP, Rlist

    def _fk_dtcp6_series(self, ztool: float = 0.0, smoothing: int = 1):
        """(t_mid, V) with V=[vx vy vz wx wy wz] from FK TCP via central diffs + SO(3) log."""
        t, TCP, Rlist = self._fk_tcp6_series(ztool=ztool)
        if TCP.shape[0] < 3:
            return t[:0], np.empty((0,6))
        p = TCP[:, :3]

        # Linear velocity (central difference, midpoints)
        dt = t[2:] - t[:-2]
        dp = p[2:, :] - p[:-2, :]
        dt[dt <= 0] = np.nan
        v = dp / dt[:, None]
        t_mid_v = 0.5*(t[2:] + t[:-2])

        # Angular velocity on edges, then average to centers
        w_edges = []
        for i in range(len(Rlist) - 1):
            dR = Rlist[i].T @ Rlist[i+1]
            r = self._so3_log(dR)
            Δt = t[i+1] - t[i]
            w_edges.append(r / (Δt if Δt > 0 else np.nan))
        w_edges = np.asarray(w_edges)
        if w_edges.shape[0] < 2:
            return t[:0], np.empty((0,6))
        w = 0.5*(w_edges[1:] + w_edges[:-1])  # align to central grid
        t_mid = t_mid_v

        V = np.empty((t_mid.shape[0], 6), dtype=float)
        V[:, :3] = v
        V[:, 3:] = w

        if smoothing and smoothing > 1 and V.shape[0] > 0:
            k = int(smoothing); ker = np.ones(k)/k
            for j in range(6):
                V[:, j] = np.convolve(V[:, j], ker, mode="same")

        return self._finite_time_and_series(t_mid, V)


    def _so3_log(self, R: np.ndarray) -> np.ndarray:
        tr = float(np.clip(np.trace(R), -1.0, 3.0))
        th = float(np.arccos((tr - 1.0) / 2.0))
        if th < 1e-12:
            return np.zeros(3)
        w = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2.0*np.sin(th))
        w = w / (np.linalg.norm(w) + 1e-12)
        return w * th

    def _finite_time_and_series(self, t: np.ndarray, Y: np.ndarray):
        if t.size == 0 or Y.size == 0:
            return t[:0], Y[:0]
        m = np.isfinite(t) & np.isfinite(Y).all(axis=1)
        return t[m], Y[m]


    def _compute_tcp_xyz_from_q(self, ztool=0.0):
        """
        Vectorized-ish: compute Nx3 TCP positions from self.as_array('q').
        Returns (mask, X, Y, Z) where mask aligns with self.time().
        """
        Q = self.as_array("q")
        if Q.shape[0] == 0:
            return np.zeros((0,), dtype=bool), np.array([]), np.array([]), np.array([])
        X = np.empty(Q.shape[0]); Y = np.empty(Q.shape[0]); Z = np.empty(Q.shape[0])
        for i, q in enumerate(Q):
            try:
                x,y,z = self._tcp_from_joints(q, ztool=ztool)
            except Exception:
                x=y=z=np.nan
            X[i], Y[i], Z[i] = x, y, z
        mask = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
        return mask, X, Y, Z


    def plot_tcp_xy(self, save=True, show=False, title="TCP XY path (cartesian)", equal=True, fk=False, ztool=0.0):
        """
        Plot TCP XY using either streamed 'tcp' field or FK from joints.
        Set fk=True to force FK. If fk=False and 'tcp' is empty/NaN, auto-fallback to FK.
        """
        use_fk = fk
        if not use_fk:
            A = self.as_array("tcp")
            if A.shape[0] == 0 or not np.isfinite(A[:, :2]).any():
                use_fk = True

        if use_fk:
            mask, X, Y, _ = self._compute_tcp_xyz_from_q(ztool=ztool)
            x, y = X[mask], Y[mask]
        else:
            A = self.as_array("tcp")
            x, y = A[:, 0], A[:, 1]
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]

        if x.size == 0:
            print("TCP XY has no finite samples.")
            return None

        plt.close("all")
        fig, ax = plt.subplots()
        ax.plot(x, y, linewidth=1, label="TCP path")
        ax.scatter(x[0],  y[0],  marker="o", s=30, label="start")
        ax.scatter(x[-1], y[-1], marker="x", s=40, label="end")
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.set_title(title + (" (FK)" if use_fk else ""))
        ax.grid(True, linestyle="--", alpha=0.7); ax.legend()

        if equal:
            ax.set_aspect("equal", adjustable="box")
            xmin, xmax = float(np.min(x)), float(np.max(x))
            ymin, ymax = float(np.min(y)), float(np.max(y))
            cx, cy = (xmin+xmax)/2.0, (ymin+ymax)/2.0
            half = max(xmax-xmin, ymax-ymin)/2.0
            pad = 0.05 * (half if half > 0 else 1.0)
            half += pad
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
        else:
            ax.set_aspect("equal", adjustable="datalim")

        fig.tight_layout()
        out = None
        if save:
            ts = time.strftime("%Y%m%d_%H%M%S")
            suffix = "fk" if use_fk else "tcp"
            out = os.path.join(self.log_folder, f"log_tcp_xy_{suffix}_{ts}.png")
            fig.savefig(out, dpi=300)
        if show: plt.show()
        else: plt.close(fig)
        return out


    def plot_tcp_xyz(self, save=True, show=False, title="TCP 3D path", color_by_time=True, equal=True, fk=False, ztool=0.0):
        """
        Plot TCP XYZ using either streamed 'tcp' field or FK from joints.
        Set fk=True to force FK. If fk=False and 'tcp' is empty/NaN, auto-fallback to FK.
        """
        use_fk = fk
        if not use_fk:
            A = self.as_array("tcp")
            if A.shape[0] == 0 or not np.isfinite(A[:, :3]).any():
                use_fk = True

        if use_fk:
            t = self.time()
            mask, X, Y, Z = self._compute_tcp_xyz_from_q(ztool=ztool)
            x, y, z = X[mask], Y[mask], Z[mask]
            t = t[mask]
        else:
            A = self.as_array("tcp")
            x, y, z = A[:, 0], A[:, 1], A[:, 2]
            m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            x, y, z = x[m], y[m], z[m]
            t = self.time()[m]

        if x.size == 0:
            print("TCP XYZ has no finite samples.")
            return None

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        if color_by_time:
            sc = ax.scatter(x, y, z, c=t, s=6)
            cb = fig.colorbar(sc, ax=ax, pad=0.1)
            cb.set_label("Time [s]")
            ax.plot3D(x, y, z, linewidth=0.8, alpha=0.7)
        else:
            ax.plot3D(x, y, z, linewidth=1)

        ax.scatter(x[0], y[0], z[0], s=30, marker="o", label="start")
        ax.scatter(x[-1], y[-1], z[-1], s=40, marker="x", label="end")
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        ax.set_title(title + (" (FK)" if use_fk else "")); ax.legend(loc="upper left")

        if equal:
            xmin, xmax = float(np.min(x)), float(np.max(x))
            ymin, ymax = float(np.min(y)), float(np.max(y))
            zmin, zmax = float(np.min(z)), float(np.max(z))
            cx, cy, cz = (xmin+xmax)/2.0, (ymin+ymax)/2.0, (zmin+zmax)/2.0
            half = max(xmax-xmin, ymax-ymin, zmax-zmin)/2.0
            pad = 0.05 * (half if half > 0 else 1.0)
            half += pad
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_zlim(cz - half, cz + half)
            try:
                ax.set_box_aspect([1, 1, 1])
            except Exception:
                pass

        fig.tight_layout()
        out = None
        if save:
            ts = time.strftime("%Y%m%d_%H%M%S")
            suffix = "fk" if use_fk else "tcp"
            out = os.path.join(self.log_folder, f"log_tcp_xyz_{suffix}_{ts}.png")
            fig.savefig(out, dpi=300)
        if show: plt.show()
        else: plt.close(fig)
        return out
    
    
    def plot_tcp(self,
                kind: str = "tcp",
                save: bool = True,
                show: bool = False,
                title: str | None = None,
                ztool: float = 0.0,
                smoothing: int = 1,
                components: tuple[str, ...] | None = None):
        """
        CB2-robust, FK-based time series plotter.
        kind: "tcp" -> [x,y,z,rx,ry,rz],  "dtcp" -> [vx,vy,vz,wx,wy,wz]
        """
        if kind not in ("tcp", "dtcp"):
            raise ValueError("kind must be 'tcp' or 'dtcp'")

        if kind == "tcp":
            comp_idx = {"x":0,"y":1,"z":2,"rx":3,"ry":4,"rz":5}
            if components is None: components = ("x","y","z","rx","ry","rz")
            idx = [comp_idx[c] for c in components]
            t, TCP, _ = self._fk_tcp6_series(ztool=ztool)
            if TCP.shape[0] == 0: print("No TCP samples."); return None
            Y = TCP[:, idx]
            ylab = "TCP components [m / rad]"
            ttl  = title or "TCP pose over time (FK)"
        else:
            comp_idx = {"vx":0,"vy":1,"vz":2,"wx":3,"wy":4,"wz":5}
            if components is None: components = ("vx","vy","vz","wx","wy","wz")
            idx = [comp_idx[c] for c in components]
            t, V = self._fk_dtcp6_series(ztool=ztool, smoothing=smoothing)
            if V.shape[0] == 0: print("No TCP velocity samples."); return None
            Y = V[:, idx]
            ylab = "TCP twist [m/s, rad/s]"
            ttl  = title or (f"TCP twist over time (FK diff)"
                            + ("" if smoothing in (None,1) else f", sma={int(smoothing)}"))

        t, Y = self._finite_time_and_series(t, Y)
        if t.size == 0: print("No finite samples to plot."); return None

        plt.close("all")
        fig, ax = plt.subplots()
        for j, name in enumerate(components):
            ax.plot(t, Y[:, j], linewidth=1, label=name)
        ax.set_xlabel("Time [s]"); ax.set_ylabel(ylab)
        ax.set_title(ttl); ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(); fig.tight_layout()

        out = None
        if save:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out = os.path.join(self.log_folder, f"log_{kind}_over_time_fk_{ts}.png")
            fig.savefig(out, dpi=300)
        if show: plt.show()
        else: plt.close(fig)
        return out
