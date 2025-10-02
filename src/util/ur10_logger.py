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
            "tcp": [54, 55, 56, 57, 58, 59],
            "dtcp":[60, 61, 62, 63, 64, 65],
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

    def plot_tcp_xy(self, save=True, show=False, title="TCP XY path (cartesian)", equal=True):
        """
        Plot the TCP trajectory in the XY plane (meters). Ensures equal axis scale
        and (optionally) same axis extents so the plot is square in data units.
        """
        # Grab TCP data: [x, y, z, rx, ry, rz]
        A = self.as_array("tcp")
        if A.shape[0] == 0:
            print("No TCP data to plot.")
            return None

        x = A[:, 0]
        y = A[:, 1]
        # Drop NaNs if any
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size == 0:
            print("TCP XY has no finite samples.")
            return None

        import matplotlib.pyplot as plt
        plt.close("all")
        fig, ax = plt.subplots()

        # Draw the path
        ax.plot(x, y, linewidth=1, label="TCP path")

        # Mark start/end
        ax.scatter(x[0],  y[0],  marker="o", s=30, label="start")
        ax.scatter(x[-1], y[-1], marker="x", s=40, label="end")

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

        # Ensure equal scaling and equal extents (square)
        # equal=True -> make scales equal AND make axes the same length
        if equal:
            ax.set_aspect("equal", adjustable="box")
            xmin, xmax = float(np.min(x)), float(np.max(x))
            ymin, ymax = float(np.min(y)), float(np.max(y))
            # Make the view square in data units
            cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
            half_range = max(xmax - xmin, ymax - ymin) / 2.0
            pad = 0.05 * (half_range if half_range > 0 else 1.0)
            half = half_range + pad
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
        else:
            # Still keep equal *scale* if requested but no forced square window
            ax.set_aspect("equal", adjustable="datalim")

        fig.tight_layout()

        out = None
        if save:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out = os.path.join(self.log_folder, f"log_tcp_xy_{ts}.png")
            fig.savefig(out, dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return out
    
    def plot_tcp_xyz(self, save=True, show=False, title="TCP 3D path", color_by_time=True, equal=True):
        """Plot the TCP path in 3D (meters). Equal aspect for true geometry."""
        A = self.as_array("tcp")
        if A.shape[0] == 0:
            print("No TCP data to plot."); return None

        x, y, z = A[:, 0], A[:, 1], A[:, 2]
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[m], y[m], z[m]
        if x.size == 0:
            print("TCP XYZ has no finite samples."); return None
        t = self.time()[m]

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (ensures 3D projection)
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
        ax.set_title(title); ax.legend(loc="upper left")

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
                ax.set_box_aspect([1, 1, 1])  # mpl ≥3.3
            except Exception:
                pass

        fig.tight_layout()
        out = None
        if save:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out = os.path.join(self.log_folder, f"log_tcp_xyz_{ts}.png")
            fig.savefig(out, dpi=300)
        if show: plt.show()
        else: plt.close(fig)
        return out
