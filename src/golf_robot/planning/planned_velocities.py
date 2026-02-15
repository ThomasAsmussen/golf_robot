#!/usr/bin/env python3
"""
plot_cmd_vs_planned_velocities.py

Plots planned joint velocities (from the planner CSV) vs commanded joint velocities
(from the streamer command log CSV produced by your C++ code).

Expected CSV formats:
  Planned (trajectory_sim.csv):   t,q0..q5,dq0..dq5
  Commanded (commanded_velocity.csv): t,dq_cmd0..dq_cmd5   (or t,dq0..dq5)

Usage:
  python plot_cmd_vs_planned_velocities.py

Edit PLANNED_CSV / CMD_CSV / OUT_DIR below if needed.
"""

from pathlib import Path
import csv
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import glob

# ---------------------- Config ----------------------
SHOW_PLOTS = True  # set False to save PNGs only (headless)
PLANNED_CSV = "log/trajectory_sim.csv"
CMD_CSV_GLOB = "log/commanded_velocity.csv"  # or "log/commanded_velocity_*.csv" if you timestamp it
OUT_DIR = "log"
TITLE_PREFIX = "Planned vs Commanded Joint Velocities"

# If your commanded file has different column names, add them here:
CMD_COL_CANDIDATES = (
    [f"dq_cmd{i}" for i in range(6)] +
    [f"dq{i}" for i in range(6)] +
    [f"cmd_dq{i}" for i in range(6)]
)

# ---------------------- IO helpers ----------------------
def _try_set_backend(show: bool) -> None:
    if show:
        for bk in ("TkAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
            try:
                matplotlib.use(bk)
                return
            except Exception:
                pass
    else:
        matplotlib.use("Agg")

_try_set_backend(SHOW_PLOTS)

def find_latest_file(glob_pattern: str):
    files = glob.glob(glob_pattern)
    if not files:
        return None
    files.sort()
    return files[-1]

def load_planned_csv(path: str):
    """Loads: t,q0..q5,dq0..dq5 -> t (N,), dQ (N,6)"""
    t_list, dQ_list = [], []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        hdr = next(reader, None)
        for row in reader:
            if not row or all((c.strip() == "" for c in row)):
                continue
            row = [c.strip() for c in row]
            if len(row) < 13:
                continue
            try:
                t = float(row[0])
                dq = [float(x) for x in row[7:13]]
            except ValueError:
                continue
            t_list.append(t)
            dQ_list.append(dq)
    if not t_list:
        raise RuntimeError(f"No valid rows parsed from planned CSV: {path}")
    t = np.asarray(t_list, float)
    dQ = np.asarray(dQ_list, float).reshape(-1, 6)
    return t, dQ

def load_cmd_csv(path: str):
    """
    Loads commanded velocity CSV.
    Supports both:
      - header with dq_cmd0..dq_cmd5 (preferred)
      - header with dq0..dq5
      - no header (assumes t + 6 cols)
    Returns t_cmd (M,), dQ_cmd (M,6)
    """
    t_list, dQ_list = [], []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            raise RuntimeError(f"Empty commanded CSV: {path}")

        # Detect header vs data
        def looks_like_header(r):
            s = ",".join([c.strip().lower() for c in r])
            return ("t" in s) or ("dq" in s) or ("time" in s)

        has_header = looks_like_header(first)

        if has_header:
            header = [c.strip() for c in first]
            col_idx = {name: i for i, name in enumerate(header)}

            # Find dq column indices in the header
            dq_idxs = []
            for name in CMD_COL_CANDIDATES:
                if name in col_idx:
                    dq_idxs.append(col_idx[name])
            # If we found fewer than 6 by name, try to infer contiguous after 't'
            if len(dq_idxs) < 6:
                # assume: t is col0, dq columns are 1..6
                if len(header) >= 7:
                    dq_idxs = list(range(1, 7))

            # time column
            t_idx = col_idx.get("t", 0)

            for row in reader:
                if not row:
                    continue
                row = [c.strip() for c in row]
                if len(row) <= max([t_idx] + dq_idxs):
                    continue
                try:
                    t = float(row[t_idx])
                    dq = [float(row[i]) for i in dq_idxs[:6]]
                except ValueError:
                    continue
                t_list.append(t)
                dQ_list.append(dq)
        else:
            # No header; first row is data
            rows = [first] + list(reader)
            for row in rows:
                if not row:
                    continue
                row = [c.strip() for c in row]
                if len(row) < 7:
                    continue
                try:
                    t = float(row[0])
                    dq = [float(x) for x in row[1:7]]
                except ValueError:
                    continue
                t_list.append(t)
                dQ_list.append(dq)

    if not t_list:
        raise RuntimeError(f"No valid rows parsed from commanded CSV: {path}")
    t = np.asarray(t_list, float)
    dQ = np.asarray(dQ_list, float).reshape(-1, 6)
    return t, dQ

# ---------------------- Plotting ----------------------
def plot_cmd_vs_planned(t_plan, dQ_plan, t_cmd, dQ_cmd, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # Optional: restrict to overlap window for cleaner plots
    t0 = max(t_plan.min(), t_cmd.min())
    t1 = min(t_plan.max(), t_cmd.max())

    plan_mask = (t_plan >= t0) & (t_plan <= t1)
    cmd_mask = (t_cmd >= t0) & (t_cmd <= t1)

    t_plan2 = t_plan[plan_mask]
    dQ_plan2 = dQ_plan[plan_mask]
    t_cmd2 = t_cmd[cmd_mask]
    dQ_cmd2 = dQ_cmd[cmd_mask]

    # Plot per joint
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=False)
    axes = axes.flatten()

    for j in range(6):
        ax = axes[j]
        ax.plot(t_plan2, dQ_plan2[:, j], "-", linewidth=2, label="Planned dq")
        ax.plot(t_cmd2, dQ_cmd2[:, j], ".", linewidth=0.8, label="Commanded dq")
        ax.set_title(f"Joint {j}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (rad/s)")
        ax.grid(True)
        ax.legend()

    fig.suptitle(TITLE_PREFIX)
    plt.tight_layout()

    out_png = os.path.join(out_dir, f"planned_vs_commanded_velocities_{ts}.png")

    if SHOW_PLOTS:
        try:
            plt.show()
        finally:
            plt.close(fig)
        return None
    else:
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
        return out_png

# ---------------------- Main ----------------------
def main():
    planned_path = Path(PLANNED_CSV)
    if not planned_path.exists():
        raise FileNotFoundError(f"Planned CSV not found: {planned_path}")

    cmd_path = find_latest_file(CMD_CSV_GLOB)
    if cmd_path is None:
        raise FileNotFoundError(f"No commanded CSV found with pattern: {CMD_CSV_GLOB}")

    print(f"[INFO] Planned CSV:   {planned_path}")
    print(f"[INFO] Commanded CSV: {cmd_path}")

    t_plan, dQ_plan = load_planned_csv(str(planned_path))
    t_cmd, dQ_cmd = load_cmd_csv(cmd_path)

    print(f"[INFO] Planned dQ shape:   {dQ_plan.shape}")
    print(f"[INFO] Commanded dQ shape: {dQ_cmd.shape}")

    out = plot_cmd_vs_planned(t_plan, dQ_plan, t_cmd, dQ_cmd, OUT_DIR)
    if out:
        print(f"[DONE] Saved: {out}")
    else:
        print("[DONE] Displayed plot interactively.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
