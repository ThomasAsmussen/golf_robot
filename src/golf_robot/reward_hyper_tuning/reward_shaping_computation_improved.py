#!/usr/bin/env python3
# reward_shaping_computation_improved.py
#
# Analyzes reward shaping sweeps exported from W&B (parquet/csv).
# Produces:
# - Holdout validation metrics (R^2, RMSE) for the success-rate surrogate
# - Global permutation importance (with context controls: algorithm + difficulty)
# - Bootstrap CIs for permutation importance
# - Slice-level importance across (algorithm, difficulty) to estimate robustness
# - Quantile-binning based guidance for good global reward ranges (with simple CI)
# - Publication-friendly plots (matplotlib, no explicit colors)
# - PDP-style sensitivity curve for v_opt (cfg.optimal_speed)

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# -------------------------
# HARD-CODED SETTINGS
# -------------------------
DATA_PARQUET = "data/wandb/wandb_all_runs.parquet"
DATA_CSV = "data/wandb/wandb_all_runs.csv"

TARGET_METRIC = "metric.final_success_rate"  # from your loader script
# If your column ended up as "sum.final_success_rate" instead, change TARGET_METRIC accordingly.

# If you KNOW the reward shaping keys, hardcode them here (recommended).
# Otherwise, set to None and it will try to auto-detect from cfg.*
REWARD_KEYS_HARDCODED = [
    "cfg.w_distance",
    "cfg.in_hole_reward",
    "cfg.distance_scale",
    "cfg.optimal_speed",
    "cfg.optimal_speed_scale",
    "cfg.dist_at_hole_scale",
]

# Model eval
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Permutation importance
PERM_REPEATS_GLOBAL = 20
PERM_REPEATS_SLICE = 10

# Bootstrap CI for permutation importance
N_BOOT_CI = 60

# Robustness score definition: robust = mean - ROBUST_SIGMA_WEIGHT * std
ROBUST_SIGMA_WEIGHT = 0.5

# Plotting
PLOT_DIR = Path("reward_analysis_plots")
PLOT_DIR.mkdir(exist_ok=True, parents=True)
PLOT_TOP_K = 12  # plot top-k reward parameters for readability

# PDP / sensitivity curve for v_opt
PDP_FEATURE = "cfg.optimal_speed"  # v_opt
PDP_GRID_POINTS = 60
PDP_Q_LO = 0.01       # grid min as quantile of observed feature values
PDP_Q_HI = 0.99       # grid max as quantile of observed feature values
PDP_BAND_Q_LO = 0.05  # prediction band lower quantile
PDP_BAND_Q_HI = 0.95  # prediction band upper quantile


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


def auto_detect_reward_keys(df: pd.DataFrame) -> list[str]:
    """Heuristic detection of reward-shaping cfg.* columns."""
    candidates: list[str] = []
    pat = re.compile(r"(reward|hole|distance|speed|scale|bonus|penalty|shap|w_)", re.IGNORECASE)

    for c in df.columns:
        if not c.startswith("cfg."):
            continue
        if not pat.search(c):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() < 0.2:
            continue
        candidates.append(c)

    # Drop obviously algo-specific hyperparams if they slipped in
    drop_pat = re.compile(r"(actor|critic|lr|batch|hidden|tau|rho|gamma|cem|bootstrap|noise)", re.IGNORECASE)
    candidates = [c for c in candidates if not drop_pat.search(c)]

    return sorted(candidates)


def prepare_matrix(df: pd.DataFrame, reward_cols: list[str], target_col: str):
    """Prepare X/y and include contextual controls (algorithm + difficulty)."""
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found. "
            f"Example columns: {df.columns[:30].tolist()}"
        )

    y = pd.to_numeric(df[target_col], errors="coerce")

    X = df[reward_cols].copy()
    for c in reward_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    keep = y.notna()
    for c in reward_cols:
        keep &= X[c].notna()

    X = X.loc[keep]
    y = y.loc[keep]

    # Context variables to control for baseline differences
    if "algorithm" in df.columns:
        algo = df.loc[keep, "algorithm"].astype(str)
        X = pd.concat([X, pd.get_dummies(algo, prefix="algo", drop_first=False)], axis=1)

    if "difficulty" in df.columns:
        X["difficulty"] = pd.to_numeric(df.loc[keep, "difficulty"], errors="coerce").astype(float)

    return X, y, keep


def make_rf_model(*, n_estimators: int = 400) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=0,
        n_jobs=-1,
        min_samples_leaf=2,
        oob_score=True,
        bootstrap=True,
    )


def evaluate_model(model: RandomForestRegressor, X: pd.DataFrame, y: pd.Series):
    """Holdout validation to justify surrogate-model quality."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    m = clone(model)
    m.fit(Xtr, ytr)

    pred = m.predict(Xte)
    r2 = r2_score(yte, pred)
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))

    print("\n=== RF Validation (holdout) ===")
    print(f"R^2  : {r2:.4f}")
    print(f"RMSE : {rmse:.4f}")

    return r2, rmse


def global_importance(X: pd.DataFrame, y: pd.Series, reward_cols: list[str]):
    """Global permutation importance using full-data fit (after holdout eval)."""
    model = make_rf_model(n_estimators=400)

    # 1) Holdout metrics (separate fit)
    _r2, _rmse = evaluate_model(model, X, y)

    # 2) Fit on all data (for importance + OOB score)
    model.fit(X, y)
    if hasattr(model, "oob_score_"):
        print(f"OOB score (fit on all data): {model.oob_score_:.4f}")

    imp = permutation_importance(
        model,
        X,
        y,
        n_repeats=PERM_REPEATS_GLOBAL,
        random_state=0,
        n_jobs=-1,
    )

    imp_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": imp.importances_mean,
            "importance_std": imp.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    imp_reward = imp_df[imp_df["feature"].isin(reward_cols)].copy()
    imp_reward = imp_reward.sort_values("importance_mean", ascending=False)

    return model, imp_reward, imp_df


def bootstrap_importance_ci(
    model_template: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_boot: int = N_BOOT_CI,
    perm_repeats: int = 10,
) -> pd.DataFrame:
    """Bootstrap CIs for permutation importance.

    We resample rows with replacement, refit, and recompute permutation importances.
    Returns mean + 5/95 percentiles for each feature.
    """
    rng = np.random.default_rng(0)

    all_imps = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        Xb = X.iloc[idx]
        yb = y.iloc[idx]

        m = clone(model_template)
        m.fit(Xb, yb)

        imp = permutation_importance(
            m,
            Xb,
            yb,
            n_repeats=perm_repeats,
            random_state=0,
            n_jobs=-1,
        )
        all_imps.append(imp.importances_mean)

    all_imps = np.asarray(all_imps)  # [B, F]

    rows = []
    for j, f in enumerate(X.columns):
        vals = all_imps[:, j]
        rows.append(
            {
                "feature": f,
                "importance_mean": float(np.mean(vals)),
                "ci_low": float(np.percentile(vals, 5)),
                "ci_high": float(np.percentile(vals, 95)),
            }
        )

    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def plot_perm_importance_bar(df_imp: pd.DataFrame, *, out_path: Path, title: str, top_k: int = PLOT_TOP_K):
    """Bar plot for permutation importance. Expects columns: feature, importance_mean, importance_std."""
    d = df_imp.copy().head(top_k)
    d = d.iloc[::-1]  # for horizontal bars (largest at top)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(d) + 1.5)))
    ax.barh(d["feature"], d["importance_mean"], xerr=d.get("importance_std", None))
    ax.set_xlabel("Permutation importance (ΔMSE)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_perm_importance_ci(
    df_ci: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    reward_only: list[str] | None = None,
    top_k: int = PLOT_TOP_K,
):
    """Plot importance mean with bootstrap CI (5–95%). Expects: feature, importance_mean, ci_low, ci_high."""
    d = df_ci.copy()
    if reward_only is not None:
        d = d[d["feature"].isin(reward_only)].copy()
    d = d.sort_values("importance_mean", ascending=False).head(top_k)
    d = d.iloc[::-1]

    x = d["importance_mean"].to_numpy()
    lo = d["ci_low"].to_numpy()
    hi = d["ci_high"].to_numpy()
    xerr = np.vstack([x - lo, hi - x])

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(d) + 1.5)))
    ax.barh(d["feature"], x, xerr=xerr)
    ax.set_xlabel("Permutation importance (ΔMSE)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def slice_importance(df: pd.DataFrame, reward_cols: list[str], target_col: str, *, min_rows: int = 45):
    """Permutation importance per (algorithm, difficulty) slice.

    Fits a small RF per slice using ONLY reward parameters as inputs.
    Returns per-slice importances + robustness summary.
    """
    results: list[dict] = []

    if "algorithm" not in df.columns or "difficulty" not in df.columns:
        print("Skipping slice importance: missing algorithm/difficulty columns.")
        return None, None

    algos = sorted(df["algorithm"].dropna().unique())
    diffs = sorted(df["difficulty"].dropna().unique())

    for algo in algos:
        for diff in diffs:
            sub = df[(df["algorithm"] == algo) & (df["difficulty"] == diff)].copy()
            if len(sub) < min_rows:
                continue

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

            model = make_rf_model(n_estimators=300)
            model.fit(Xs, ys)

            imp = permutation_importance(
                model,
                Xs,
                ys,
                n_repeats=PERM_REPEATS_SLICE,
                random_state=0,
                n_jobs=-1,
            )

            for i, c in enumerate(reward_cols):
                results.append(
                    {
                        "algorithm": str(algo),
                        "difficulty": float(diff),
                        "reward_param": c,
                        "importance": float(imp.importances_mean[i]),
                        "n_samples": int(len(Xs)),
                    }
                )

    per_slice = pd.DataFrame(results)
    if per_slice.empty:
        return per_slice, None

    robust = (
        per_slice.groupby("reward_param")
        .agg(
            mean=("importance", "mean"),
            std=("importance", "std"),
            count=("importance", "count"),
            mean_samples=("n_samples", "mean"),
        )
        .reset_index()
    )
    robust["robust_score"] = robust["mean"] - ROBUST_SIGMA_WEIGHT * robust["std"]
    robust = robust.sort_values("robust_score", ascending=False)

    return per_slice, robust


def recommend_values(df: pd.DataFrame, reward_cols: list[str], target_col: str, *, bins: int = 8):
    """Quantile-binning guidance for reward parameters.

    For each reward parameter, compute mean success within bins.
    Adds simple normal-approx CIs for the mean: mean ± 1.96*std/sqrt(n).
    """
    out_rows: list[dict] = []

    y = pd.to_numeric(df[target_col], errors="coerce")

    for c in reward_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() < 200:
            continue

        xs = x[ok]
        ys = y[ok]

        # Few unique values => group directly
        if xs.nunique() <= 20:
            g = (
                pd.DataFrame({"x": xs, "y": ys})
                .groupby("x")["y"]
                .agg(["mean", "count", "std"])
                .reset_index()
            )
            g["ci_low"] = g["mean"] - 1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1))
            g["ci_high"] = g["mean"] + 1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1))
            g = g.sort_values("mean", ascending=False).head(5)
            best = g.iloc[0]
            out_rows.append(
                {
                    "reward_param": c,
                    "best_value_or_bin": str(best["x"]),
                    "mean_success": float(best["mean"]),
                    "support": int(best["count"]),
                    "ci_low": float(best["ci_low"]),
                    "ci_high": float(best["ci_high"]),
                }
            )
        else:
            # Quantile binning
            try:
                b = pd.qcut(xs, q=bins, duplicates="drop")
            except ValueError:
                continue

            g = (
                pd.DataFrame({"bin": b, "y": ys})
                .groupby("bin")["y"]
                .agg(["mean", "count", "std"])
                .reset_index()
            )
            g["ci_low"] = g["mean"] - 1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1))
            g["ci_high"] = g["mean"] + 1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1))
            g = g.sort_values("mean", ascending=False).head(3)
            best = g.iloc[0]
            out_rows.append(
                {
                    "reward_param": c,
                    "best_value_or_bin": str(best["bin"]),
                    "mean_success": float(best["mean"]),
                    "support": int(best["count"]),
                    "ci_low": float(best["ci_low"]),
                    "ci_high": float(best["ci_high"]),
                }
            )

    rec = pd.DataFrame(out_rows)
    if not rec.empty:
        rec = rec.sort_values("mean_success", ascending=False)
    return rec


def partial_dependence_curve(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    feature: str,
    grid: np.ndarray,
    *,
    band_q_lo: float = PDP_BAND_Q_LO,
    band_q_hi: float = PDP_BAND_Q_HI,
):
    """PDP-style sensitivity curve: overwrite one feature for all rows, predict, aggregate."""
    if feature not in X.columns:
        raise KeyError(f"Feature '{feature}' not in X columns.")

    Xtmp = X.copy()
    mean_pred = np.empty(len(grid), dtype=float)
    lo_pred = np.empty(len(grid), dtype=float)
    hi_pred = np.empty(len(grid), dtype=float)

    for i, v in enumerate(grid):
        Xtmp[feature] = v
        yhat = model.predict(Xtmp)
        mean_pred[i] = float(np.mean(yhat))
        lo_pred[i] = float(np.quantile(yhat, band_q_lo))
        hi_pred[i] = float(np.quantile(yhat, band_q_hi))

    return mean_pred, lo_pred, hi_pred


def plot_pdp_curve(
    grid: np.ndarray,
    mean_pred: np.ndarray,
    lo_pred: np.ndarray,
    hi_pred: np.ndarray,
    *,
    out_path: Path,
    title: str,
    xlabel: str,
):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(grid, mean_pred, linewidth=2)
    ax.fill_between(grid, lo_pred, hi_pred, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Predicted success rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    df = load_df()

    # Choose reward shaping columns
    if REWARD_KEYS_HARDCODED is not None:
        reward_cols = [c for c in REWARD_KEYS_HARDCODED if c in df.columns]
        missing = sorted(set(REWARD_KEYS_HARDCODED) - set(df.columns))
        if missing:
            raise KeyError(
                "Missing hardcoded reward columns. "
                f"Missing: {missing}. Available cfg.* examples: {[c for c in df.columns if c.startswith('cfg.')][:20]}"
            )
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
    X, y, _keep = prepare_matrix(df, reward_cols, TARGET_METRIC)
    print(f"\nUsing {len(y)} runs after NaN filtering.")

    model, imp_reward, _imp_all = global_importance(X, y, reward_cols)

    print("\n=== GLOBAL reward shaping importance (permutation) ===")
    print(imp_reward.to_string(index=False))

    # Bootstrap CI for importance (global)
    print(f"\nComputing bootstrap importance CIs (n_boot={N_BOOT_CI})...")
    ci_df = bootstrap_importance_ci(make_rf_model(n_estimators=350), X, y, n_boot=N_BOOT_CI, perm_repeats=10)
    ci_reward = ci_df[ci_df["feature"].isin(reward_cols)].copy().sort_values("importance_mean", ascending=False)

    # Robustness across slices
    per_slice, robust = slice_importance(df, reward_cols, TARGET_METRIC, min_rows=45)
    if robust is not None:
        print("\n=== Robustness across (algorithm, difficulty) slices ===")
        print(robust.to_string(index=False))

    # Recommend values/ranges
    rec = recommend_values(df, reward_cols, TARGET_METRIC, bins=8)
    if not rec.empty:
        print("\n=== High-performing global values / bins (quick guidance) ===")
        print(rec.to_string(index=False))

    # Save outputs (CSVs)
    imp_reward.to_csv("reward_global_importance.csv", index=False)
    ci_reward.to_csv("reward_global_importance_bootstrap_ci.csv", index=False)

    if robust is not None:
        robust.to_csv("reward_robustness.csv", index=False)
    if per_slice is not None and not per_slice.empty:
        per_slice.to_csv("reward_importance_per_slice.csv", index=False)

    if rec is not None and not rec.empty:
        rec.to_csv("reward_value_recommendations.csv", index=False)

    # Plots
    plot_perm_importance_bar(
        imp_reward,
        out_path=PLOT_DIR / "perm_importance_global_reward.png",
        title="Global permutation importance (reward parameters)",
        top_k=min(PLOT_TOP_K, len(imp_reward)),
    )

    plot_perm_importance_ci(
        ci_reward,
        out_path=PLOT_DIR / "perm_importance_global_reward_bootstrap_ci.png",
        title="Global permutation importance with bootstrap CI (reward parameters)",
        reward_only=reward_cols,
        top_k=min(PLOT_TOP_K, len(ci_reward)),
    )

    if robust is not None:
        # Plot robustness as bar chart
        r = robust.copy().head(PLOT_TOP_K)
        r = r.iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(r) + 1.5)))
        ax.barh(r["reward_param"], r["robust_score"])
        ax.set_xlabel(f"Robust score = mean - {ROBUST_SIGMA_WEIGHT}·std")
        ax.set_title("Reward parameter robustness across algorithm/difficulty slices")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "robustness_reward_params.png", dpi=200)
        plt.close(fig)

    # PDP-style sensitivity curve for v_opt
    if PDP_FEATURE in X.columns:
        lo = float(X[PDP_FEATURE].quantile(PDP_Q_LO))
        hi = float(X[PDP_FEATURE].quantile(PDP_Q_HI))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            grid = np.linspace(lo, hi, PDP_GRID_POINTS)
            pdp_mean, pdp_lo, pdp_hi = partial_dependence_curve(model, X, PDP_FEATURE, grid)
            plot_pdp_curve(
                grid,
                pdp_mean,
                pdp_lo,
                pdp_hi,
                out_path=PLOT_DIR / "pdp_vopt_success.png",
                title="PDP: predicted success rate vs v_opt (cfg.optimal_speed)",
                xlabel="v_opt (cfg.optimal_speed)",
            )
            print(f"Wrote: {PLOT_DIR / 'pdp_vopt_success.png'}")
        else:
            print(f"Skipping PDP plot: invalid grid range for {PDP_FEATURE} (lo={lo}, hi={hi})")
    else:
        print(f"Skipping PDP plot: '{PDP_FEATURE}' not in X columns.")

    print("\nWrote:")
    print("  reward_global_importance.csv")
    print("  reward_global_importance_bootstrap_ci.csv")
    print("  reward_robustness.csv")
    print("  reward_importance_per_slice.csv (if computed)")
    print("  reward_value_recommendations.csv (if computed)")
    print(f"  plots in: {PLOT_DIR}/")


if __name__ == "__main__":
    main()