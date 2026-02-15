#!/usr/bin/env python3
"""
plot_stitched_success_rate_raw_mean.py

- Loads runs with final_success_rate >= THRESH
- Stitches final_success_rate at step=1000
- Computes RAW mean and std across runs per step (no rolling)
- Saves stitched data + aggregate stats to CSV
- Plots mean ± std
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm


# -------------------------
# Config
# -------------------------
ENTITY  = None
PROJECT = "golf_robot_thompson_v_zero_final_training"

FINAL_METRIC = "final_success_rate"
TRAIN_METRIC = "success_rate"
STEP_KEY     = "_step"

THRESH = 0.82
STITCH_STEP = 1000

SAVE_STITCHED_MATRIX = "stitched_success_matrix.csv"
SAVE_AGG_STATS       = "stitched_success_aggregate.csv"
SAVE_RUN_SUMMARY     = "stitched_run_summary.csv"


# -------------------------
# Helpers
# -------------------------
def fetch_training_history(run):
    try:
        hist = run.history(keys=[STEP_KEY, TRAIN_METRIC], pandas=True)
    except Exception:
        return None

    if hist is None or hist.empty or TRAIN_METRIC not in hist.columns:
        return None

    hist = hist[[STEP_KEY, TRAIN_METRIC]].dropna()
    if hist.empty:
        return None

    s = (
        pd.Series(hist[TRAIN_METRIC].values,
                  index=hist[STEP_KEY].astype(int))
        .groupby(level=0)
        .last()
        .sort_index()
    )
    return s


# -------------------------
# Main
# -------------------------
def main():
    api = wandb.Api()
    proj_path = f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT
    runs = api.runs(proj_path)

    loaded = []

    for run in tqdm(runs, desc="Loading runs"):
        final_val = run.summary.get(FINAL_METRIC, None)
        if final_val is None:
            continue

        try:
            final_val = float(final_val)
        except Exception:
            continue

        if final_val < THRESH:
            continue

        train_s = fetch_training_history(run)
        if train_s is None or train_s.empty:
            continue

        stitched = train_s.copy()
        stitched.loc[STITCH_STEP] = final_val
        stitched = stitched.sort_index()

        loaded.append({
            "name": run.name,
            "id": run.id,
            "final_success_rate": final_val,
            "n_train_points": len(train_s),
            "series": stitched
        })

    print(f"\nLoaded {len(loaded)} runs with {FINAL_METRIC} >= {THRESH}\n")

    if not loaded:
        print("No runs loaded.")
        return

    # -------------------------
    # Build stitched matrix
    # -------------------------
    all_steps = sorted(set().union(*[set(d["series"].index) for d in loaded]))
    df = pd.DataFrame(index=all_steps)

    for d in loaded:
        df[d["name"]] = d["series"].reindex(df.index)

    # Save full stitched matrix
    df_out = df.copy()
    df_out.insert(0, "step", df_out.index)
    df_out.to_csv(SAVE_STITCHED_MATRIX, index=False)
    print(f"Saved stitched matrix to: {SAVE_STITCHED_MATRIX}")

    # -------------------------
    # Raw aggregate stats (over runs)
    # -------------------------
    mean_curve = df.mean(axis=1, skipna=True)
    std_curve  = df.std(axis=1, skipna=True)
    count_curve = df.count(axis=1)

    agg = pd.DataFrame({
        "step": df.index,
        "mean_success": mean_curve.values,
        "std_success": std_curve.values,
        "n_runs_at_step": count_curve.values
    })

    agg.to_csv(SAVE_AGG_STATS, index=False)
    print(f"Saved aggregate stats to: {SAVE_AGG_STATS}")

    # -------------------------
    # Save per-run summary
    # -------------------------
    run_summary = pd.DataFrame([{
        "name": d["name"],
        "id": d["id"],
        "final_success_rate": d["final_success_rate"],
        "n_train_points": d["n_train_points"]
    } for d in loaded])

    run_summary.to_csv(SAVE_RUN_SUMMARY, index=False)
    print(f"Saved run summary to: {SAVE_RUN_SUMMARY}")

    # -------------------------
    # Plot raw mean ± std
    # -------------------------
    x = df.index.to_numpy()
    y_mean = mean_curve.to_numpy(dtype=float)
    y_std  = std_curve.to_numpy(dtype=float)

    plt.figure(figsize=(10, 5))

    # Optional: show individual runs lightly
    for col in df.columns:
        plt.plot(x, df[col].to_numpy(dtype=float), alpha=0.12)

    plt.plot(x, y_mean, linewidth=2.5, label="Mean across runs")
    plt.fill_between(x, y_mean - y_std, y_mean + y_std,
                     alpha=0.20, label="Std across runs (±1σ)")

    plt.ylim(-0.02, 1.02)
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print final sanity numbers
    print("\nSanity check:")
    print(f"Mean at 1000: {mean_curve.loc[STITCH_STEP]:.4f}")
    print(f"Std  at 1000: {std_curve.loc[STITCH_STEP]:.4f}")


if __name__ == "__main__":
    main()