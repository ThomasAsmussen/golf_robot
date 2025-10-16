import numpy as np

def tcp_path_from_Q(Q, fk_fn):
    """Return Nx3 TCP positions from a sequence of joint vectors using fk_fn."""
    P = np.zeros((len(Q), 3))
    for i, q in enumerate(Q):
        T = fk_fn(q)[-1]
        P[i] = T[:3, 3]
    return P

def plot_paths(P_planned, P_actual=None, show_xy=True, title="TCP path (planned vs actual)"):
    import matplotlib.pyplot as plt
    from numpy.linalg import norm
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(P_planned[:,0], P_planned[:,1], P_planned[:,2], linewidth=2, label="Planned")
    if P_actual is not None and len(P_actual) > 1:
        ax.plot(P_actual[:,0], P_actual[:,1], P_actual[:,2], linestyle="--", label="Actual")
        # annotate drift at end
        drift = norm(P_actual[-1] - P_planned[-1])
        ax.text(*P_actual[-1], f" drift {drift*1000:.1f} mm")
    ax.scatter(*P_planned[0], marker="o", s=40, label="Start")
    ax.scatter(*P_planned[-1], marker="x", s=60, label="Planned end")
    if P_actual is not None and len(P_actual) > 0:
        ax.scatter(*P_actual[-1], marker="^", s=60, label="Actual end")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title(title); ax.legend(); ax.grid(True)

    if show_xy:
        plt.figure(figsize=(7,6))
        import matplotlib.pyplot as plt
        plt.plot(P_planned[:,0], P_planned[:,1], label="Planned")
        if P_actual is not None:
            plt.plot(P_actual[:,0], P_actual[:,1], "--", label="Actual")
        plt.scatter(P_planned[0,0], P_planned[0,1], c="k", s=25)
        plt.xlabel("X [m]"); plt.ylabel("Y [m]"); plt.gca().set_aspect("equal", "box")
        plt.title("Top view (XY)"); plt.grid(True); plt.legend()

    plt.show()
