import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# ------------------------------------------------------------------
# 1) CONFIG
# ------------------------------------------------------------------

RESULTS_DIR = os.path.join("data", "tuning_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def savefig(name: str):
    """Save current figure in multiple formats."""
    for ext in ("png", "pdf", "svg"):
        plt.savefig(os.path.join(RESULTS_DIR, f"{name}.{ext}"),
                    dpi=300, bbox_inches="tight")
# Path to simulation CSV (assumed single file)
SIM_DIR = os.path.join("data", "tuning_sim")
SIM_CSV = os.path.join(SIM_DIR, "sim_vxscale_tester.csv")

# Folder where all physical CSVs are stored
PHYS_DIR = os.path.join("data", "tuning_videos_processed")


# ------------------------------------------------------------------
# 2) LOAD SIMULATION DATA ONCE
# ------------------------------------------------------------------

sim = pd.read_csv(SIM_CSV)

# Adjust these column names if your sim CSV differs
time_sim = sim["time"].to_numpy()
time_sim = time_sim
x_sim = sim["ball_x"].to_numpy()
y_sim = sim["ball_z"].to_numpy()

# Sim velocities
vx_sim = np.gradient(x_sim, time_sim)
vy_sim = np.gradient(y_sim, time_sim)


# ------------------------------------------------------------------
# 3) LOOP OVER ALL PHYSICAL CSV FILES
# ------------------------------------------------------------------

# Get all .csv files in PHYS_DIR
phys_files = [f for f in os.listdir(PHYS_DIR) if f.lower().endswith("time_shifted.csv")]
phys_files.sort()

if not phys_files:
    print(f"No CSV files found in {PHYS_DIR}")
else:
    print(f"Found {len(phys_files)} physical CSV files in {PHYS_DIR}:")
    for f in phys_files:
        print("  -", f)

for phys_file in phys_files:
    phys_path = os.path.join(PHYS_DIR, phys_file)
    print(f"\nProcessing physical file: {phys_path}")

    phys = pd.read_csv(phys_path)

    # ------------------------------------------------------------------
    # 3a) Extract physical trajectory columns
    #     These names come from your tracking script:
    #     ["time_s", "x_m", "y_m", "vx_m_s", "vy_m_s", "speed_m_s"]
    # ------------------------------------------------------------------
    time_phys = phys["time_s"].to_numpy()
    x_phys = phys["x_m"].to_numpy()
    y_phys = phys["y_m"].to_numpy()

    # Normalize time and apply optional offset
    time_phys = time_phys

    # You *could* reuse vx_m_s / vy_m_s from the file, but we recompute
    # to be consistent with how we treated sim:
    vx_phys = np.gradient(x_phys, time_phys)
    vy_phys = np.gradient(y_phys, time_phys)

    # Smooth velocities (ensure window is odd and <= len(data))
    def smooth_vel(v, t, window=55, poly=3):
        n = len(v)
        if n < 5:
            # too short, just return as-is
            return v
        # window must be odd and <= n
        w = min(window, n if n % 2 == 1 else n - 1)
        if w < 5:
            w = 5 if 5 < n else (n if n % 2 == 1 else n - 1)
        if w <= poly:
            poly = max(1, w - 2)
        return savgol_filter(v, window_length=w, polyorder=poly)

    #vx_phys_smooth = smooth_vel(vx_phys, time_phys)
    #vy_phys_smooth = smooth_vel(vy_phys, time_phys)
    vx_phys_smooth = vx_phys
    vy_phys_smooth = vy_phys

    # Short name for figures
    base_name = os.path.splitext(phys_file)[0]

    # ------------------------------------------------------------------
    # 3b) POSITION PLOT: Sim vs Physical
    # ------------------------------------------------------------------
    plt.figure(figsize=(5, 5))
    plt.plot(time_sim, x_sim, label='Sim X')
    plt.plot(time_phys, x_phys, label='Physical X', linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('X Position [m]')
    plt.legend()
    plt.title(f'X Position vs Time ({base_name})')

    plt.tight_layout()
    savefig(f"position_comparison_{base_name}")


    # ------------------------------------------------------------------
    # 3c) VELOCITY PLOT: Sim vs Physical
    # ------------------------------------------------------------------
    plt.figure(figsize=(5, 5))
    plt.plot(time_sim, vx_sim, label='Sim Vx')
    plt.plot(time_phys, vx_phys_smooth, label='Physical Vx (smoothed)', linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity X [m/s]')
    plt.legend()
    plt.title(f'X Velocity vs Time ({base_name})')

    plt.tight_layout()
    savefig(f"velocity_comparison_{base_name}")


plt.show()
plt.close()

print("\nAll comparisons done.")
