#!/usr/bin/env python3
# analyze_reward_shaping.py

import os
import re
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


# -------------------------
# HARD-CODED SETTINGS
# -------------------------
DATA_PARQUET = "data/wandb/wandb_all_runs.parquet"
DATA_CSV     = "data/wandb/wandb_all_runs.csv"

TARGET_METRIC = "metric.final_success_rate"  # from the loader script
# If your column ended up as "sum.final_success_rate" instead, change TARGET_METRIC accordingly.

# If you KNOW the reward shaping keys, hardcode them here (recommended).
# Otherwise, leave as None and it will try to auto-detect from cfg.*
REWARD_KEYS_HARDCODED = None
# Example:
REWARD_KEYS_HARDCODED = [
    "cfg.w_distance",
    "cfg.in_hole_reward",
    "cfg.distance_scale",
    "cfg.optimal_speed",
    "cfg.optimal_speed_scale",
    "cfg.dist_at_hole_scale",
]


def load_df():
    if os.path.exists(DATA_PARQUET):
        df = pd.read_parquet(DATA_PARQUET)
        print(f"Loaded {DATA_PARQUET}: {df.shape}")
        return df
    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
        print(f"Loaded {DATA_CSV}: {df.shape}")
        return df
    raise FileNotFoundError(f"Could not find {DATA_PARQUET} or {DATA_CSV} in cwd.")


def auto_detect_reward_keys(df):
    """
    Heuristic: reward shaping tends to contain words like:
    reward, hole, distance, speed, scale, bonus, penalty, shaping, w_
    and should be numeric.
    """
    candidates = []
    pat = re.compile(r"(reward|hole|distance|speed|scale|bonus|penalty|shap|w_)", re.IGNORECASE)

    for c in df.columns:
        if not c.startswith("cfg."):
            continue
        if not pat.search(c):
            continue
        # numeric-ish?
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() < 0.2:
            continue
        candidates.append(c)

    # Prefer smaller set: drop clearly algo-specific things if they slipped in
    drop_pat = re.compile(r"(actor|critic|lr|batch|hidden|tau|rho|gamma|cem|bootstrap|noise)", re.IGNORECASE)
    candidates = [c for c in candidates if not drop_pat.search(c)]

    return sorted(candidates)


def prepare_matrix(df, reward_cols, target_col):
    # Ensure target exists and numeric
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Available examples: {df.columns[:20].tolist()}")

    y = pd.to_numeric(df[target_col], errors="coerce")

    # Convert reward cols to numeric
    X = df[reward_cols].copy()
    for c in reward_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Keep rows with enough data
    keep = y.notna()
    for c in reward_cols:
        keep &= X[c].notna()

    X = X.loc[keep]
    y = y.loc[keep]

    # Add context variables for deconfounding (algo + difficulty)
    # These are *not* reward shaping, but help isolate shaping effects.
    # Algorithm is categorical -> one-hot.
    if "algorithm" in df.columns:
        algo = df.loc[keep, "algorithm"].astype(str)
        X = pd.concat([X, pd.get_dummies(algo, prefix="algo", drop_first=False)], axis=1)

    if "difficulty" in df.columns:
        X["difficulty"] = pd.to_numeric(df.loc[keep, "difficulty"], errors="coerce").astype(float)

    return X, y, keep


def global_importance(X, y, reward_cols):
    model = RandomForestRegressor(
        n_estimators=400,
        random_state=0,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X, y)

    imp = permutation_importance(
        model, X, y,
        n_repeats=20,
        random_state=0,
        n_jobs=-1
    )

    imp_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": imp.importances_mean,
        "importance_std": imp.importances_std,
    }).sort_values("importance_mean", ascending=False)

    # Keep only reward shaping rows for the "reward shaping conclusion"
    imp_reward = imp_df[imp_df["feature"].isin(reward_cols)].copy()
    imp_reward = imp_reward.sort_values("importance_mean", ascending=False)

    return imp_reward, imp_df


def slice_importance(df, reward_cols, target_col, min_rows=50):
    """
    Compute permutation importance per (algorithm, difficulty) slice to estimate robustness.
    Returns per-slice importances and a robustness table.
    """
    results = []
    if "algorithm" not in df.columns or "difficulty" not in df.columns:
        print("Skipping slice importance: missing algorithm/difficulty columns.")
        return None, None

    for algo in sorted(df["algorithm"].dropna().unique()):
        for diff in sorted(df["difficulty"].dropna().unique()):
            sub = df[(df["algorithm"] == algo) & (df["difficulty"] == diff)].copy()
            if len(sub) < min_rows:
                continue

            # Only use reward cols for slice model (keep it simple per slice)
            Xs = sub[reward_cols].copy()
            ys = pd.to_numeric(sub[target_col], errors="coerce")
            for c in reward_cols:
                Xs[c] = pd.to_numeric(Xs[c], errors="coerce")
            keep = ys.notna()
            for c in reward_cols:
                keep &= Xs[c].notna()
            Xs = Xs.loc[keep]
            ys = ys.loc[keep]
            if len(Xs) < min_rows:
                continue

            model = RandomForestRegressor(
                n_estimators=300,
                random_state=0,
                n_jobs=-1,
                min_samples_leaf=2,
            )
            model.fit(Xs, ys)
            imp = permutation_importance(model, Xs, ys, n_repeats=10, random_state=0, n_jobs=-1)

            for i, c in enumerate(reward_cols):
                results.append({
                    "algorithm": algo,
                    "difficulty": diff,
                    "reward_param": c,
                    "importance": float(imp.importances_mean[i]),
                })

    per_slice = pd.DataFrame(results)
    if per_slice.empty:
        return per_slice, None

    robust = (
        per_slice.groupby("reward_param")["importance"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    robust["robust_score"] = robust["mean"] - 0.5 * robust["std"]
    robust = robust.sort_values("robust_score", ascending=False)

    return per_slice, robust


def recommend_values(df, reward_cols, target_col, bins=8):
    """
    Very practical: for each reward parameter, bin its values and compute mean success.
    This helps you pick a single global value (or range) that performs well broadly.
    """
    out = []
    y = pd.to_numeric(df[target_col], errors="coerce")

    for c in reward_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() < 200:
            continue

        xs = x[ok]
        ys = y[ok]

        # If only few unique values, just group directly
        if xs.nunique() <= 20:
            g = pd.DataFrame({"x": xs, "y": ys}).groupby("x")["y"].agg(["mean", "count"]).reset_index()
            g = g.sort_values("mean", ascending=False).head(5)
            best = g.iloc[0]
            out.append({
                "reward_param": c,
                "best_value_or_bin": str(best["x"]),
                "mean_success": float(best["mean"]),
                "support": int(best["count"]),
            })
        else:
            # Quantile binning
            try:
                b = pd.qcut(xs, q=bins, duplicates="drop")
            except ValueError:
                continue
            g = pd.DataFrame({"bin": b, "y": ys}).groupby("bin")["y"].agg(["mean", "count"]).reset_index()
            g = g.sort_values("mean", ascending=False).head(3)
            best = g.iloc[0]
            out.append({
                "reward_param": c,
                "best_value_or_bin": str(best["bin"]),
                "mean_success": float(best["mean"]),
                "support": int(best["count"]),
            })

    rec = pd.DataFrame(out).sort_values("mean_success", ascending=False)
    return rec


def main():
    df = load_df()

    # Choose reward shaping columns
    if REWARD_KEYS_HARDCODED is not None:
        reward_cols = [c for c in REWARD_KEYS_HARDCODED if c in df.columns]
        if len(reward_cols) != len(REWARD_KEYS_HARDCODED):
            missing = sorted(set(REWARD_KEYS_HARDCODED) - set(df.columns))
            raise KeyError(f"Missing hardcoded reward columns: {missing}")
    else:
        reward_cols = auto_detect_reward_keys(df)

    if not reward_cols:
        raise RuntimeError(
            "Could not detect reward shaping columns. "
            "Set REWARD_KEYS_HARDCODED to the exact cfg.* names."
        )

    print("\nReward shaping columns:")
    for c in reward_cols:
        print("  ", c)

    # Global importance with context controls
    X, y, keep = prepare_matrix(df, reward_cols, TARGET_METRIC)

    print(f"\nUsing {len(y)} runs after NaN filtering.")
    imp_reward, imp_all = global_importance(X, y, reward_cols)

    print("\n=== GLOBAL reward shaping importance (permutation) ===")
    print(imp_reward.to_string(index=False))

    # Robustness across slices
    per_slice, robust = slice_importance(df, reward_cols, TARGET_METRIC, min_rows=50)
    if robust is not None:
        print("\n=== Robustness across (algorithm, difficulty) slices ===")
        print(robust.to_string(index=False))

    # Recommend values/ranges
    rec = recommend_values(df, reward_cols, TARGET_METRIC, bins=8)
    if not rec.empty:
        print("\n=== High-performing global values / bins (quick guidance) ===")
        print(rec.to_string(index=False))

    # Save outputs
    imp_reward.to_csv("reward_global_importance.csv", index=False)
    if robust is not None:
        robust.to_csv("reward_robustness.csv", index=False)
    if per_slice is not None and not per_slice.empty:
        per_slice.to_csv("reward_importance_per_slice.csv", index=False)
    rec.to_csv("reward_value_recommendations.csv", index=False)

    print("\nWrote:")
    print("  reward_global_importance.csv")
    print("  reward_robustness.csv")
    print("  reward_importance_per_slice.csv (if computed)")
    print("  reward_value_recommendations.csv")


if __name__ == "__main__":
    main()