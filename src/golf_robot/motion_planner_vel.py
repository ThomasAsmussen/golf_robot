import numpy as np
import matplotlib.pyplot as plt

class MotionPlanner3DStrictVel:
    """
    Piecewise quintic with EXACT waypoint position *and velocity vector* constraints.
    Continuity at interior waypoints: acceleration & jerk.
    Boundary accelerations set to zero (change if you want).
    """

    def __init__(self, waypoints, t_total=9.0):
        """
        waypoints: list of ((x,y,z), (vx,vy,vz)) for all waypoints (m >= 2)
                   Provide the exact velocity vectors you want at each waypoint.
        t_total:   total duration (s). Change this freely; waypoint velocities remain exact.
        """
        self.P = np.asarray([p for p, _ in waypoints], dtype=float)  # (m,3)
        self.V = np.asarray([v for _, v in waypoints], dtype=float)  # (m,3)
        self.m = len(waypoints)
        assert self.m >= 2, "Need at least 2 waypoints."

        # Time allocation proportional to segment length (you can change this)
        seg_len = np.linalg.norm(self.P[1:] - self.P[:-1], axis=1)
        if seg_len.sum() < 1e-9:
            seg_len[:] = 1.0
        dt = t_total * seg_len / seg_len.sum()
        self.times = np.concatenate(([0.0], np.cumsum(dt)))
        self.seg_times = dt

        # Solve & sample
        self.coeffs = self._solve_coeffs()
        self.trajectory_setpoints, self.time_setpoints = self._sample_dense()

    # -------- math helpers --------
    def _Am(self, t):
        """Rows: pos, vel, acc, jerk, snap for quintic at local time t."""
        return np.array([
            [t**5,   t**4,  t**3, t**2, t, 1],
            [5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0],
            [20*t**3,12*t**2,6*t,   2,   0, 0],
            [60*t**2,24*t,   6,     0,   0, 0],
            [120*t,  24,     0,     0,   0, 0],
        ], dtype=float)

    def _solve_coeffs(self):
        """
        Build A θ = b per axis with:
        For each segment i (between waypoint i and i+1):
          pos(0)=P[i],   vel(0)=V[i]
          pos(Ti)=P[i+1], vel(Ti)=V[i+1]
        For interior joints k=1..m-2:
          acc_i(Ti) == acc_{i+1}(0)
          jerk_i(Ti) == jerk_{i+1}(0)
        Boundary acc:
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

            # per-segment pos/vel at start & end
            for i in range(nseg):
                Ti = self.seg_times[i]
                Af = self._Am(Ti)
                sl = slice(i*6, (i+1)*6)

                # start pos = P[i]
                A[row, sl] = A0[0]; b[row] = self.P[i, dim]; row += 1
                # start vel = V[i]
                A[row, sl] = A0[1]; b[row] = self.V[i, dim]; row += 1
                # end pos = P[i+1]
                A[row, sl] = Af[0]; b[row] = self.P[i+1, dim]; row += 1
                # end vel = V[i+1]
                A[row, sl] = Af[1]; b[row] = self.V[i+1, dim]; row += 1

            # interior continuity: acc, jerk
            for i in range(nseg - 1):
                Ti = self.seg_times[i]
                Af = self._Am(Ti)
                sl_i = slice(i*6,   (i+1)*6)
                sl_j = slice((i+1)*6,(i+2)*6)

                # acc continuity
                A[row, sl_i] = Af[2]; A[row, sl_j] = -A0[2]; b[row] = 0.0; row += 1
                # jerk continuity
                A[row, sl_i] = Af[3]; A[row, sl_j] = -A0[3]; b[row] = 0.0; row += 1

            # boundary accelerations = 0
            # start
            A[row, 0:6] = A0[2]; b[row] = 0.0; row += 1
            # end
            T_last = self.seg_times[-1]
            Af_last = self._Am(T_last)
            A[row, -6:] = Af_last[2]; b[row] = 0.0; row += 1

            assert row == 6*nseg, "Row/unknowns mismatch; constraints not square."

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

        for k, t in enumerate(t_query):
            i = min(max(np.searchsorted(self.times, t) - 1, 0), nseg - 1)
            tl = t - self.times[i]
            Am = self._Am(tl)
            sl = slice(i*6, (i+1)*6)
            x[k]  = Am[0] @ cx[sl]; y[k]  = Am[0] @ cy[sl]; z[k]  = Am[0] @ cz[sl]
            vx[k] = Am[1] @ cx[sl]; vy[k] = Am[1] @ cy[sl]; vz[k] = Am[1] @ cz[sl]
            ax[k] = Am[2] @ cx[sl]; ay[k] = Am[2] @ cy[sl]; az[k] = Am[2] @ cz[sl]

        yaw = np.arctan2(vy, vx, dtype=float, where=(vx*vx+vy*vy)>0)
        traj = np.column_stack([x, y, z, yaw])

        # quick report
        speed = np.sqrt(vx*vx + vy*vy + vz*vz)
        acc   = np.sqrt(ax*ax + ay*ay + az*az)
        print(f"Max |v|={speed.max():.3f} m/s (mean {speed.mean():.3f}), Max |a|={acc.max():.3f} m/s²")

        return traj, t_query

    def rescale_total_time(self, new_total_time):
        """Scale all segment times to hit a new total time, then recompute coeffs and samples."""
        alpha = new_total_time / (self.times[-1] - self.times[0])
        self.seg_times = self.seg_times * alpha
        self.times = np.concatenate(([0.0], np.cumsum(self.seg_times)))
        self.coeffs = self._solve_coeffs()
        self.trajectory_setpoints, self.time_setpoints = self._sample_dense()


def run_demo():
    # --- positions and EXACT velocity vectors at each waypoint ---
    start_pos = (0.0, 0.0, 0.4)
    mid_pos   = (1.0, 0.0, 0.8)
    end_pos   = (2.0, 0.0, 0.4)

    # Exact velocity vectors (m/s) at the waypoints:
    v_start = (0.0, 0.0, 0.0)       # start at rest
    v_mid   = (5.0, 0.0, 0.0)      # your explicit mid velocity vector
    v_end   = (0.0, 0.0, 0.0)       # end at rest

    waypoints = [
        (start_pos, v_start),
        (mid_pos,   v_mid),
        (end_pos,   v_end),
    ]

    # Choose total time. Increase if you see aggressive accelerations/jerk.
    t_total = 9.0
    mp = MotionPlanner3DStrictVel(waypoints, t_total=t_total)

    traj = mp.trajectory_setpoints
    tq   = mp.time_setpoints
    x, y, z = traj[:,0], traj[:,1], traj[:,2]
    vx = np.gradient(x, tq); vy = np.gradient(y, tq); vz = np.gradient(z, tq)
    speed = np.sqrt(vx*vx + vy*vy + vz*vz)

    # Sanity check: achieved mid velocity equals requested mid velocity
    # (evaluate at the end of segment 0)
    Am_end = mp._Am(mp.seg_times[0])
    c = mp.coeffs
    v_mid_ach = np.array([
        Am_end[1] @ c[0:6, 0],
        Am_end[1] @ c[0:6, 1],
        Am_end[1] @ c[0:6, 2],
    ])
    print("Requested mid vel:", v_mid, "  Achieved:", v_mid_ach)

    # --- plots ---
    fig1 = plt.figure(figsize=(7,6))
    ax3d = fig1.add_subplot(111, projection='3d')
    ax3d.plot(x, y, z, linewidth=2, label="Trajectory")
    P = np.array([start_pos, mid_pos, end_pos], float)
    ax3d.scatter(P[:,0], P[:,1], P[:,2], s=60, label="Waypoints")
    # draw mid velocity arrow
    M = np.array(mid_pos)
    ax3d.quiver(M[0], M[1], M[2], v_mid[0], v_mid[1], v_mid[2],
                length=1.0, normalize=False, linewidth=2, label="Mid velocity")
    for name, p in zip(["Start","Mid","End"], P): ax3d.text(*p, name)
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.set_title("Quintic with exact waypoint velocity vectors")
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
