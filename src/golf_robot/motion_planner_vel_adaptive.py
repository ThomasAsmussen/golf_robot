import numpy as np
import matplotlib.pyplot as plt

class MotionPlanner3DStrictVel:
    """
    Piecewise quintic with EXACT waypoint positions & velocity vectors.
    Interior continuity: acceleration & jerk.
    Boundary accelerations = 0 by default.
    """

    def __init__(self, waypoints, t_total=9.0):
        """
        waypoints: [((x,y,z), (vx,vy,vz)), ...]  for m >= 2
        t_total: total duration in seconds (can be retimed/optimized later)
        """
        self.P = np.asarray([p for p, _ in waypoints], float)  # (m,3)
        self.V = np.asarray([v for _, v in waypoints], float)  # (m,3)
        self.m = len(waypoints)
        assert self.m >= 2, "Need at least 2 waypoints."

        # initial time allocation proportional to segment length
        seg_len = np.linalg.norm(self.P[1:] - self.P[:-1], axis=1)

        if seg_len.sum() < 1e-9:
            seg_len[:] = 1.0

        dt = t_total * seg_len / seg_len.sum()
        self.times = np.concatenate(([0.0], np.cumsum(dt)))
        self.seg_times = dt

        self.coeffs = self._solve_coeffs()
        self.trajectory_setpoints, self.time_setpoints = self._sample_dense()

    # ---------- math helpers ----------
    def _Am(self, t):
        # pos, vel, acc, jerk, snap rows for quintic at local time t
        return np.array([
            [t**5,   t**4,  t**3, t**2, t, 1],
            [5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0],
            [20*t**3,12*t**2,6*t,   2,   0, 0],
            [60*t**2,24*t,   6,     0,   0, 0],
            [120*t,  24,     0,     0,   0, 0],
        ], float)

    def _solve_coeffs(self):
        """
        Per segment i:
          pos(0)=P[i], vel(0)=V[i],  pos(Ti)=P[i+1], vel(Ti)=V[i+1]
        Interior joints:
          acc_i(Ti) = acc_{i+1}(0),  jerk_i(Ti) = jerk_{i+1}(0)
        Boundary:
          acc_0(0)=0,  acc_last(T_last)=0
        """
        m = self.m
        nseg = m - 1
        A0 = self._Am(0.0)
        coeffs = np.zeros((6*nseg, 3))

        for dim in range(3):
            A = np.zeros((6*nseg, 6*nseg))
            b = np.zeros(6*nseg)
            row = 0

            # per-segment boundary position/velocity
            for i in range(nseg):
                Ti = self.seg_times[i]
                Af = self._Am(Ti)
                sl = slice(i*6, (i+1)*6)

                A[row, sl] = A0[0]; b[row] = self.P[i, dim]; row += 1
                A[row, sl] = A0[1]; b[row] = self.V[i, dim]; row += 1
                A[row, sl] = Af[0]; b[row] = self.P[i+1, dim]; row += 1
                A[row, sl] = Af[1]; b[row] = self.V[i+1, dim]; row += 1

            # interior continuity: acc & jerk
            for i in range(nseg - 1):
                Ti = self.seg_times[i]
                Af = self._Am(Ti)
                sl_i = slice(i*6,   (i+1)*6)
                sl_j = slice((i+1)*6,(i+2)*6)

                A[row, sl_i] = Af[2]; A[row, sl_j] = -A0[2]; b[row] = 0.0; row += 1
                A[row, sl_i] = Af[3]; A[row, sl_j] = -A0[3]; b[row] = 0.0; row += 1

            # boundary accelerations = 0
            A[row, 0:6] = A0[2]; b[row] = 0.0; row += 1
            T_last = self.seg_times[-1]
            Af_last = self._Am(T_last)
            A[row, -6:] = Af_last[2]; b[row] = 0.0; row += 1

            assert row == 6*nseg
            coeffs[:, dim] = np.linalg.solve(A, b)

        return coeffs

    def _sample_dense(self, disc_steps=30):
        N = disc_steps * self.m
        t_query = np.linspace(self.times[0], self.times[-1], N)
        nseg = self.m - 1
        cx, cy, cz = self.coeffs[:,0], self.coeffs[:,1], self.coeffs[:,2]

        x = np.zeros(N); y = np.zeros(N); z = np.zeros(N)
        vx = np.zeros(N); vy = np.zeros(N); vz = np.zeros(N)
        ax = np.zeros(N); ay = np.zeros(N); az = np.zeros(N)
        jx = np.zeros(N); jy = np.zeros(N); jz = np.zeros(N)

        for k, t in enumerate(t_query):
            i = min(max(np.searchsorted(self.times, t) - 1, 0), nseg - 1)
            tl = t - self.times[i]
            Am = self._Am(tl)
            sl = slice(i*6, (i+1)*6)
            x[k]  = Am[0] @ cx[sl]; y[k]  = Am[0] @ cy[sl]; z[k]  = Am[0] @ cz[sl]
            vx[k] = Am[1] @ cx[sl]; vy[k] = Am[1] @ cy[sl]; vz[k] = Am[1] @ cz[sl]
            ax[k] = Am[2] @ cx[sl]; ay[k] = Am[2] @ cy[sl]; az[k] = Am[2] @ cz[sl]
            jx[k] = Am[3] @ cx[sl]; jy[k] = Am[3] @ cy[sl]; jz[k] = Am[3] @ cz[sl]

        yaw = np.arctan2(vy, vx, dtype=float, where=(vx*vx+vy*vy)>0)
        traj = np.column_stack([x, y, z, yaw])

        self._last_metrics = {
            "v_max": float(np.sqrt(vx*vx + vy*vy + vz*vz).max()),
            "a_max": float(np.sqrt(ax*ax + ay*ay + az*az).max()),
            "j_max": float(np.sqrt(jx*jx + jy*jy + jz*jz).max()),
        }
        return traj, t_query

    def rescale_total_time(self, new_total_time):
        """Scale all segment times, recompute coeffs and samples."""
        cur_total = self.times[-1] - self.times[0]
        alpha = new_total_time / max(cur_total, 1e-9)
        self.seg_times = self.seg_times * alpha
        self.times = np.concatenate(([0.0], np.cumsum(self.seg_times)))
        self.coeffs = self._solve_coeffs()
        self.trajectory_setpoints, self.time_setpoints = self._sample_dense()

    # ---------- NEW: adaptive time optimization ----------
    def optimize_total_time(self, vel_lim=None, acc_lim=None, jerk_lim=None,
                            t_lower=None, t_upper=None, tol=1e-3, max_expand=20):
        """
        Find the *minimum* total time that satisfies the given limits.
        Monotone bisection on t_total (expand upper bound until feasible).
        Returns: (t_opt, metrics_dict)
        """
        # hard feasibility check: waypoint speeds must be <= vel_lim
        if vel_lim is not None:
            wp_speeds = np.linalg.norm(self.V, axis=1)
            if wp_speeds.max() > vel_lim + 1e-9:
                raise ValueError(f"Waypoint velocity {wp_speeds.max():.3f} exceeds vel_lim={vel_lim:.3f}.")

        base_T = self.times[-1]
        lo = t_lower if t_lower is not None else max(1e-3, 0.1 * base_T)
        hi = t_upper if t_upper is not None else base_T

        def feasible(T):
            self.rescale_total_time(T)
            mets = self._last_metrics
            cond_v = (vel_lim  is None) or (mets["v_max"] <= vel_lim)
            cond_a = (acc_lim  is None) or (mets["a_max"] <= acc_lim)
            cond_j = (jerk_lim is None) or (mets["j_max"] <= jerk_lim)
            return cond_v and cond_a and cond_j

        # ensure we have a feasible upper bound
        if not feasible(hi):
            for _ in range(max_expand):
                hi *= 1.5
                if feasible(hi):
                    break
            else:
                raise RuntimeError("Could not find a feasible total time within expansion budget.")

        # bisection for minimal feasible time
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if feasible(mid):
                hi = mid
            else:
                lo = mid

        # set to optimal (hi) and return metrics
        self.rescale_total_time(hi)
        return hi, dict(self._last_metrics)

# ------------------------ demo ------------------------

def run_demo():
    # Positions & exact velocity vectors (you control these)
    start_pos = (0.0, 0.0, 0.4)
    mid_pos   = (1.0, 0.0, 0.8)
    end_pos   = (2.0, 0.0, 0.4)

    # Exact velocity vectors (m/s) at the waypoints:
    v_start = (0.0, 0.0, 0.0)       # start at rest
    v_mid   = (1.0, 1.0, 0.0)      # your explicit mid velocity vector
    v_end   = (0.0, 0.0, 0.0)       # end at rest

    waypoints = [(start_pos, v_start), (mid_pos, v_mid), (end_pos, v_end)]

    mp = MotionPlanner3DStrictVel(waypoints, t_total=6.0)

    # Set your dynamic limits; optimizer finds the smallest total time that meets them
    t_opt, mets = mp.optimize_total_time(vel_lim=2.0, acc_lim=6.0, jerk_lim=60.0, tol=1e-3)
    print(f"Optimized total time: {t_opt:.3f} s | metrics: {mets}")

    # Verify mid velocity is exact
    Am_end = mp._Am(mp.seg_times[0])
    c = mp.coeffs
    v_mid_ach = np.array([Am_end[1] @ c[0:6, d] for d in range(3)])
    print("Requested mid vel:", v_mid, " | Achieved mid vel:", v_mid_ach)

    # Plot
    traj, tq = mp.trajectory_setpoints, mp.time_setpoints
    x, y, z = traj[:,0], traj[:,1], traj[:,2]
    vx = np.gradient(x, tq); vy = np.gradient(y, tq); vz = np.gradient(z, tq)
    speed = np.sqrt(vx*vx + vy*vy + vz*vz)

    fig1 = plt.figure(figsize=(7,6))
    ax3d = fig1.add_subplot(111, projection='3d')
    ax3d.plot(x, y, z, linewidth=2, label="Trajectory")
    P = np.array([start_pos, mid_pos, end_pos], float)
    ax3d.scatter(P[:,0], P[:,1], P[:,2], s=60, label="Waypoints")
    ax3d.quiver(P[1,0], P[1,1], P[1,2], v_mid[0], v_mid[1], v_mid[2],
                length=1.0, normalize=False, linewidth=2, label="Mid vel")
    for name, p in zip(["Start","Mid","End"], P): ax3d.text(*p, name)
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.set_title("Quintic with exact waypoint velocities + optimized total time")
    ax3d.legend()
    ax3d.view_init(elev=25, azim=-60)

    fig2, ax = plt.subplots(4,1, figsize=(8,8), sharex=True)
    ax[0].plot(tq, x); ax[0].set_ylabel("x [m]")
    ax[1].plot(tq, y); ax[1].set_ylabel("y [m]")
    ax[2].plot(tq, z); ax[2].set_ylabel("z [m]")
    ax[3].plot(tq, speed); ax[3].set_ylabel("|v| [m/s]"); ax[3].set_xlabel("time [s]")
    fig2.suptitle("Time profiles")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_demo()
