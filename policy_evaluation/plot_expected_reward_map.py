import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =========================
# PLOT CONFIG (EDIT)
# =========================
DATA_DIR = "computed_maps"
OUT_PLOT_DIR = "plots_per_hole"

# # Bigger text (increase more if you want)
# plt.rcParams.update({
#     "font.size": 20,
#     "axes.titlesize": 22,
#     "axes.labelsize": 20,
#     "xtick.labelsize": 18,
#     "ytick.labelsize": 18,
# })
plt.rcParams.update({
    "font.size": 30,
    "axes.titlesize": 28,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
})


# Keep colors "normal": autoscale each plot unless you set FIXED_LIMITS=True
FIXED_LIMITS = False

# If fixed limits are desired (for comparability), set these:
VALUE_LIMS = None          # e.g. (-0.2, 1.5)
SPEED_LIMS = None          # e.g. (1.0, 2.0)
ANGLE_LIMS = None          # e.g. (-20.0, 20.0)
# =========================

def global_limits(npz_files, key, qlo=0.0, qhi=1.0):
    mins, maxs = [], []
    for p in npz_files:
        d = np.load(p)
        arr = d[key]
        # quantile-based limits optional; set qlo=0, qhi=1 for true min/max
        lo = np.quantile(arr, qlo)
        hi = np.quantile(arr, qhi)
        mins.append(lo)
        maxs.append(hi)
    return float(min(mins)), float(max(maxs))

def plot_map(data, xlin, ylin, title, cbar_label, out_path, vmin=None, vmax=None, cmap=None):
    fig, ax = plt.subplots(figsize=(9.0, 6.2), dpi=260)

    im = ax.imshow(
        data,
        origin="lower",
        extent=[float(xlin.min()), float(xlin.max()), float(ylin.min()), float(ylin.max())],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )

    # ax.set_xlabel("ball start x (m)", labelpad=10)
    # ax.set_ylabel("ball start y (m)", labelpad=2)
    ax.set_xlabel(r"$x_{b,0}$ [m]", labelpad=6)

    ax.set_ylabel(r"$y_{b,0}$ [m]", rotation=0)
    ax.yaxis.set_label_coords(-0.05, 1.02)

    ax.set_title(title, pad=14)
    
    # Set ticks at 0.05m intervals, aligned with the grid lines
    ax.set_xticks([-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15])
    ax.set_yticks([-0.05, 0.00, 0.05, 0.10])

    # reserve top space for the colorbar strip
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.70)

    # horizontal colorbar above the plot
    cax = fig.add_axes([0.13, 0.86, 0.80, 0.035])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(cbar_label, labelpad=14)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.tick_params(labelsize=18, pad=2)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_PLOT_DIR, exist_ok=True)
    data_dir = Path(DATA_DIR)

    npz_files = sorted(data_dir.glob("planned_maps_hole*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {DATA_DIR}. Run compute_maps.py first.")

    # same heatbar across ALL holes
    VALUE_LIMS = global_limits(npz_files, "V_map", qlo=0.0, qhi=1.0)
    SPEED_LIMS = global_limits(npz_files, "speed_map", qlo=0.0, qhi=1.0)

    FIXED_LIMITS = True
    
    for npz_path in npz_files:
        hole_name = npz_path.stem.replace("planned_maps_", "")
        d = np.load(npz_path)

        xlin = d["xlin"]
        ylin = d["ylin"]
        V_map = d["V_map"]
        speed_map = d["speed_map"]
        angle_map = d["angle_map"]

        if FIXED_LIMITS:
            vV = VALUE_LIMS
            vS = SPEED_LIMS
            vA = ANGLE_LIMS
        else:
            vV = vS = vA = None

        plot_map(
            V_map, xlin, ylin,
            title=f"{hole_name}: planned value",
            cbar_label="V(s)",
            out_path=os.path.join(OUT_PLOT_DIR, f"planned_maps_{hole_name}_V.png"),
            vmin=None if vV is None else vV[0],
            vmax=None if vV is None else vV[1],
            cmap="viridis",
        )

        plot_map(
            speed_map, xlin, ylin,
            title=f"{hole_name}: planned speed",
            cbar_label=r"$v$ [m/s]",
            out_path=os.path.join(OUT_PLOT_DIR, f"planned_maps_{hole_name}_speed.png"),
            vmin=None if vS is None else vS[0],
            vmax=None if vS is None else vS[1],
            cmap="viridis",
        )

        plot_map(
            angle_map, xlin, ylin,
            title=f"{hole_name}: planned angle",
            cbar_label = r"$\theta$ [$^\circ$]",
            out_path=os.path.join(OUT_PLOT_DIR, f"planned_maps_{hole_name}_angle.png"),
            vmin=None if vA is None else vA[0],
            vmax=None if vA is None else vA[1],
            cmap="bwr",
        )

        print("Plotted:", hole_name)

    print("\nDONE. PNGs saved in:", OUT_PLOT_DIR)


if __name__ == "__main__":
    main()