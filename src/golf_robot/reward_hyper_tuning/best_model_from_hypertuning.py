#!/usr/bin/env python3
# compute_algo_stats.py
#
# Computes robust per-(algorithm, difficulty) statistics for final_success_rate:
#  - n runs
#  - max (best run)
#  - mean
#  - median
#  - std
#  - q25 / q75 / IQR
#  - mean of top-k runs (k=5 by default)
#  - pairwise win-probabilities P(algo A > algo B) per difficulty
#
# Assumes you have either:
#   wandb_all_runs.parquet  (preferred)
# or
#   wandb_all_runs.csv
#
# Target metric column expected:
#   metric.final_success_rate  (from our loader)
# If yours is sum.final_success_rate, change METRIC_COL below.

import os
import numpy as np
import pandas as pd

# -------------------------
# HARDCODED SETTINGS
# -------------------------
PARQUET_PATH = "data/wandb/wandb_all_runs.parquet"
CSV_PATH     = "data/wandb/wandb_all_runs.csv"

METRIC_COL = "metric.final_success_rate"   # change to "sum.final_success_rate" if needed
ALGO_COL = "algorithm"
DIFF_COL = "difficulty"

TOPK = 5
OUT_STATS = "algo_difficulty_stats.csv"
OUT_WINS  = "pairwise_win_probs_by_difficulty.csv"


def load_df() -> pd.DataFrame:
    if os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH)
        print(f"Loaded {PARQUET_PATH}: {df.shape}")
        return df
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded {CSV_PATH}: {df.shape}")
        return df
    raise FileNotFoundError(f"Could not find {PARQUET_PATH} or {CSV_PATH} in current directory.")


def topk_mean(x: pd.Series, k: int) -> float:
    x = x.dropna().to_numpy()
    if x.size == 0:
        return np.nan
    k = min(k, x.size)
    # sort descending and take top k
    return float(np.sort(x)[-k:].mean())


def compute_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure numeric metric
    df = df.copy()
    df[METRIC_COL] = pd.to_numeric(df[METRIC_COL], errors="coerce")
    df[DIFF_COL] = pd.to_numeric(df[DIFF_COL], errors="coerce")

    # Drop rows missing essentials
    df = df.dropna(subset=[ALGO_COL, DIFF_COL, METRIC_COL])

    g = df.groupby([ALGO_COL, DIFF_COL])[METRIC_COL]

    stats = g.agg(
        n="count",
        mean="mean",
        std="std",
        median="median",
        max="max",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        min="min",
    ).reset_index()

    stats["iqr"] = stats["q75"] - stats["q25"]

    # Top-k mean
    topk = g.apply(lambda s: topk_mean(s, TOPK)).reset_index(name=f"top{TOPK}_mean")
    stats = stats.merge(topk, on=[ALGO_COL, DIFF_COL], how="left")

    # Helpful rankings per difficulty
    stats["rank_by_max"] = stats.groupby(DIFF_COL)["max"].rank(ascending=False, method="min").astype(int)
    stats["rank_by_median"] = stats.groupby(DIFF_COL)["median"].rank(ascending=False, method="min").astype(int)
    stats["rank_by_topk"] = stats.groupby(DIFF_COL)[f"top{TOPK}_mean"].rank(ascending=False, method="min").astype(int)

    # Sort nicely
    stats = stats.sort_values([DIFF_COL, "rank_by_median", "rank_by_max"]).reset_index(drop=True)

    return stats, df


def compute_pairwise_win_probs(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    For each difficulty and each ordered pair (A,B), compute:
      P( score_A > score_B )
    estimated by all pairwise comparisons between runs from A and runs from B.
    """
    out_rows = []
    algos = sorted(df_clean[ALGO_COL].unique())
    diffs = sorted(df_clean[DIFF_COL].unique())

    for d in diffs:
        sub = df_clean[df_clean[DIFF_COL] == d]
        scores = {a: sub[sub[ALGO_COL] == a][METRIC_COL].dropna().to_numpy() for a in algos}

        for a in algos:
            for b in algos:
                if a == b:
                    continue
                xa = scores[a]
                xb = scores[b]
                if xa.size == 0 or xb.size == 0:
                    p = np.nan
                    na = xa.size
                    nb = xb.size
                else:
                    # Compute P(xa > xb) using vectorized comparison
                    # This is O(na*nb) but with ~50 runs it's fine.
                    p = float((xa[:, None] > xb[None, :]).mean())
                    na = xa.size
                    nb = xb.size

                out_rows.append({
                    "difficulty": d,
                    "algo_A": a,
                    "algo_B": b,
                    "P(A > B)": p,
                    "n_A": na,
                    "n_B": nb,
                })

    wins = pd.DataFrame(out_rows)
    # Add a symmetric "net advantage" convenience: P(A>B) - P(B>A) per pair
    # (optional; can be computed later)
    return wins


def main():
    df = load_df()

    # Quick check for the metric column
    if METRIC_COL not in df.columns:
        # Try common alternative
        alt = "sum.final_success_rate"
        if alt in df.columns:
            raise KeyError(
                f"Metric column '{METRIC_COL}' not found, but '{alt}' exists.\n"
                f"Set METRIC_COL = '{alt}' at top of script."
            )
        else:
            candidates = [c for c in df.columns if "final_success_rate" in c]
            raise KeyError(
                f"Metric column '{METRIC_COL}' not found. Columns containing 'final_success_rate': {candidates}"
            )

    stats, df_clean = compute_group_stats(df)
    print("\n=== Per (algorithm, difficulty) stats ===")
    print(stats.to_string(index=False))

    # Save stats
    stats.to_csv(OUT_STATS, index=False)
    print(f"\nWrote: {OUT_STATS}")

    # Pairwise win probabilities
    wins = compute_pairwise_win_probs(df_clean)
    print("\n=== Pairwise win probabilities (head) ===")
    print(wins.head(20).to_string(index=False))

    wins.to_csv(OUT_WINS, index=False)
    print(f"\nWrote: {OUT_WINS}")

    # Convenience: show "who is best" per difficulty under different criteria
    for crit in ["max", "median", f"top{TOPK}_mean"]:
        best = (
            stats.sort_values([DIFF_COL, crit], ascending=[True, False])
                 .groupby(DIFF_COL)
                 .head(1)[[DIFF_COL, ALGO_COL, "n", crit]]
        )
        print(f"\nBest by {crit}:")
        print(best.to_string(index=False))


if __name__ == "__main__":
    main()