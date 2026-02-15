#!/usr/bin/env python3
# analyze_ucb_final_tuning.py
#
# Same idea as analyze_reward_shaping.py, but for the NEW loader output:
#   - input columns: param.<name> and metric.<metric_name>
#   - projects: golf_robot_ucb_v_*_final_tuning
#   - confounders: algorithm + version (instead of algorithm + difficulty)

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


# -------------------------
# HARD-CODED SETTINGS
# -------------------------
# Output from the loader script I gave you earlier
DATA_PARQUET = "data/wandb/wandb_ucb_final_tuning_runs.parquet"
DATA_CSV     = "data/wandb/wandb_ucb_final_tuning_runs.csv"

# Same sweep yaml you used when loading (used to get metric + parameter names)
SWEEP_YAML = "wandb_sweep_ucb.yaml"   # or "/mnt/data/wandb_sweep_ucb.yaml"

ONLY_FINISHED = True  # loader already filtered, but keep as safety

# If you KNOW the hyperparameter keys, hardcode them here.
# Otherwise, it will read them from SWEEP_YAML and look for param.<key> columns.
HYPER_KEYS_HARDCODED: Optional[List[str]] = None
# Example:
# HYPER_KEYS_HARDCODED = [
#   "bootstrap_p", "cem_pop", "cem_iters", "cem_init_std", "cem_min_std"
# ]


# -------------------------
# Helpers
# -------------------------
def load_df() -> pd.DataFrame:
    if os.path.exists(DATA_PARQUET):
        df = pd.read_parquet(DATA_PARQUET)
        print(f"Loaded {DATA_PARQUET}: {df.shape}")
        return df
    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
        print(f"Loaded {DATA_CSV}: {df.shape}")
        return df
    raise FileNotFoundError(f"Could not find {DATA_PARQUET} or {DATA_CSV} in cwd.")


def load_sweep_metric_and_params(sweep_yaml: str) -> Tuple[str, List[str]]:
    project_root = Path(__file__).parents[3]
    p = project_root / "configs" / "wandb_sweep_ucb.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"Could not find sweep yaml at '{p}'. Run from repo root or set SWEEP_YAML."
        )
    cfg = yaml.safe_load(p.read_text())
    metric_name = (cfg.get("metric") or {}).get("name", "final_success_rate")
    param_keys = list((cfg.get("parameters") or {}).keys())
    if not param_keys:
        raise RuntimeError(f"No parameters found in '{p}' under 'parameters:'")
    return metric_name, param_keys


def choose_hyper_cols(df: pd.DataFrame, param_keys_from_yaml: List[str]) -> List[str]:
    if HYPER_KEYS_HARDCODED is not None:
        cols = [f"param.{k}" for k in HYPER_KEYS_HARDCODED]
    else:
        cols = [f"param.{k}" for k in param_keys_from_yaml]

    existing = [c for c in cols if c in df.columns]
    missing = sorted(set(cols) - set(existing))
    if missing:
        print("\nWarning: missing some expected param columns (will be ignored):")
        for c in missing:
            print("  ", c)

    # As a fallback, if yaml-based columns are missing for some reason, auto-detect param.*
    if not existing:
        existing = sorted([c for c in df.columns if c.startswith("param.")])
    return existing


def prepare_matrix(df: pd.DataFrame, hyper_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found. "
            f"Available examples: {df.columns[:30].tolist()}"
        )

    # target
    y = pd.to_numeric(df[target_col], errors="coerce")

    # X: hyperparameters
    X = df[hyper_cols].copy()
    for c in hyper_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # keep rows with complete data
    keep = y.notna()
    for c in hyper_cols:
        keep &= X[c].notna()

    X = X.loc[keep].copy()
    y = y.loc[keep].copy()

    # Deconfounding controls: algorithm + version
    if "algorithm" in df.columns:
        algo = df.loc[keep, "algorithm"].astype(str)
        X = pd.concat([X, pd.get_dummies(algo, prefix="algo", drop_first=False)], axis=1)

    if "version" in df.columns:
        X["version"] = pd.to_numeric(df.loc[keep, "version"], errors="coerce").astype(float)

    return X, y


def global_importance(X: pd.DataFrame, y: pd.Series, hyper_cols: List[str]) -> pd.DataFrame:
    model = RandomForestRegressor(
        n_estimators=500,
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

    # keep only the hyperparameters for the conclusion
    imp_h = imp_df[imp_df["feature"].isin(hyper_cols)].copy()
    imp_h = imp_h.sort_values("importance_mean", ascending=False)
    return imp_h


def slice_importance(df: pd.DataFrame, hyper_cols: List[str], target_col: str, min_rows: int = 50):
    """
    Per-version importance (optionally per algorithm too if multiple algos appear).
    """
    if "version" not in df.columns:
        print("Skipping slice importance: missing 'version' column.")
        return None, None

    results = []

    # If algorithm exists and has >1 unique value, slice by (algorithm, version)
    has_algo = "algorithm" in df.columns and df["algorithm"].nunique(dropna=True) > 1

    if has_algo:
        for algo in sorted(df["algorithm"].dropna().unique()):
            for ver in sorted(df["version"].dropna().unique()):
                sub = df[(df["algorithm"] == algo) & (df["version"] == ver)].copy()
                if len(sub) < min_rows:
                    continue

                Xs = sub[hyper_cols].copy()
                ys = pd.to_numeric(sub[target_col], errors="coerce")
                for c in hyper_cols:
                    Xs[c] = pd.to_numeric(Xs[c], errors="coerce")

                keep = ys.notna()
                for c in hyper_cols:
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

                for i, c in enumerate(hyper_cols):
                    results.append({
                        "algorithm": algo,
                        "version": ver,
                        "hyper_param": c,
                        "importance": float(imp.importances_mean[i]),
                    })
    else:
        for ver in sorted(df["version"].dropna().unique()):
            sub = df[df["version"] == ver].copy()
            if len(sub) < min_rows:
                continue

            Xs = sub[hyper_cols].copy()
            ys = pd.to_numeric(sub[target_col], errors="coerce")
            for c in hyper_cols:
                Xs[c] = pd.to_numeric(Xs[c], errors="coerce")

            keep = ys.notna()
            for c in hyper_cols:
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

            for i, c in enumerate(hyper_cols):
                results.append({
                    "version": ver,
                    "hyper_param": c,
                    "importance": float(imp.importances_mean[i]),
                })

    per_slice = pd.DataFrame(results)
    if per_slice.empty:
        return per_slice, None

    robust = (
        per_slice.groupby("hyper_param")["importance"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    robust["robust_score"] = robust["mean"] - 0.5 * robust["std"]
    robust = robust.sort_values("robust_score", ascending=False)

    return per_slice, robust


def recommend_values(df: pd.DataFrame, hyper_cols: List[str], target_col: str, bins: int = 8) -> pd.DataFrame:
    """
    For each hyperparameter, bin values and report bins with best mean success.
    """
    out = []
    y = pd.to_numeric(df[target_col], errors="coerce")

    for c in hyper_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() < 70:
            continue

        xs = x[ok]
        ys = y[ok]

        # few unique -> group directly
        if xs.nunique() <= 25:
            g = pd.DataFrame({"x": xs, "y": ys}).groupby("x")["y"].agg(["mean", "count"]).reset_index()
            g = g.sort_values(["mean", "count"], ascending=[False, False]).head(6)
            best = g.iloc[0]
            out.append({
                "hyper_param": c,
                "best_value_or_bin": str(best["x"]),
                "mean_success": float(best["mean"]),
                "support": int(best["count"]),
            })
        else:
            try:
                b = pd.qcut(xs, q=bins, duplicates="drop")
            except ValueError:
                continue
            g = pd.DataFrame({"bin": b, "y": ys}).groupby("bin")["y"].agg(["mean", "count"]).reset_index()
            g = g.sort_values(["mean", "count"], ascending=[False, False]).head(3)
            best = g.iloc[0]
            out.append({
                "hyper_param": c,
                "best_value_or_bin": str(best["bin"]),
                "mean_success": float(best["mean"]),
                "support": int(best["count"]),
            })

    rec = pd.DataFrame(out).sort_values("mean_success", ascending=False)
    return rec


def main():
    df = load_df()

    # metric + hyperparameter keys from sweep yaml
    metric_name, param_keys = load_sweep_metric_and_params(SWEEP_YAML)
    target_col = f"metric.{metric_name}"

    # Optional: enforce only finished (in case file contains others)
    if ONLY_FINISHED and "state" in df.columns:
        df = df[df["state"] == "finished"].copy()

    # Choose hyperparameter columns
    hyper_cols = choose_hyper_cols(df, param_keys)
    if not hyper_cols:
        raise RuntimeError("No hyperparameter columns found (expected 'param.*').")

    print("\nHyperparameter columns:")
    for c in hyper_cols:
        print("  ", c)

    # Global importance (with deconfounding controls)
    X, y = prepare_matrix(df, hyper_cols, target_col)

    print(f"\nUsing {len(y)} runs after NaN filtering.")
    imp_h = global_importance(X, y, hyper_cols)

    print("\n=== GLOBAL hyperparameter importance (permutation) ===")
    print(imp_h.to_string(index=False))

    # Robustness across slices (by version, optionally algorithm)
    per_slice, robust = slice_importance(df, hyper_cols, target_col, min_rows=50)
    if robust is not None:
        print("\n=== Robustness across slices ===")
        print(robust.to_string(index=False))

    # Recommend values / ranges
    rec = recommend_values(df, hyper_cols, target_col, bins=8)
    if not rec.empty:
        print("\n=== High-performing global values / bins (quick guidance) ===")
        print(rec.to_string(index=False))

    # Save outputs
    imp_h.to_csv("ucb_hyper_global_importance.csv", index=False)
    if robust is not None:
        robust.to_csv("ucb_hyper_robustness.csv", index=False)
    if per_slice is not None and not per_slice.empty:
        per_slice.to_csv("ucb_hyper_importance_per_slice.csv", index=False)
    rec.to_csv("ucb_hyper_value_recommendations.csv", index=False)

    print("\nWrote:")
    print("  ucb_hyper_global_importance.csv")
    print("  ucb_hyper_robustness.csv")
    print("  ucb_hyper_importance_per_slice.csv (if computed)")
    print("  ucb_hyper_value_recommendations.csv")


if __name__ == "__main__":
    main()