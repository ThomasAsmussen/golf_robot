#!/usr/bin/env python3
# load_ucb_final_tuning_runs.py
#
# Loads W&B runs for the *final tuning* UCB projects and exports a table
# containing (1) metadata, (2) the sweep tuning parameters, and (3) the metric.

import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import wandb
import yaml
from tqdm import tqdm

# -------------------------
# HARDCODED SETTINGS
# -------------------------
ENTITY = "rl_golf"

# These are the project names from your screenshot:
PROJECTS = [
    "golf_robot_ucb_v_5_final_tuning",
    "golf_robot_ucb_v_3_final_tuning",
    "golf_robot_ucb_v_1_final_tuning",
]

# Path to the attached sweep config (used to read tuning parameter names + metric name)
SWEEP_YAML = "/configs/wandb_sweep_ucb.yaml"  # if you run from repo root
# If you want to run it from elsewhere, set an absolute path:
# SWEEP_YAML = "/mnt/data/wandb_sweep_ucb.yaml"

ONLY_FINISHED = True

OUT_DIR = Path("data/wandb")
OUT_PARQUET = OUT_DIR / "wandb_ucb_final_tuning_runs.parquet"
OUT_CSV = OUT_DIR / "wandb_ucb_final_tuning_runs.csv"

# Accept both:
#   golf_robot_ucb_v_5_final_tuning
#   golf_robot_ucb_v5_final_tuning
PROJ_RE = re.compile(
    r"^golf_robot_(?P<algo>[^_]+)_v_?(?P<ver>\d+)_final_tuning$"
)


def parse_project_name(project_name: str):
    m = PROJ_RE.match(project_name)
    if not m:
        return None, None
    return m.group("algo"), int(m.group("ver"))


def safe_jsonable(x: Any):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    # numpy types
    try:
        import numpy as np

        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist() if x.size <= 200 else f"<ndarray shape={x.shape}>"
    except Exception:
        pass

    # containers
    if isinstance(x, (list, tuple)):
        return x if len(x) <= 200 else f"<list len={len(x)}>"
    if isinstance(x, dict):
        return x if len(x) <= 200 else f"<dict keys={len(x)}>"
    return str(x)


def main():
    # -------------------------
    # Read sweep yaml for metric + parameter keys
    # -------------------------
    project_root = Path(__file__).parents[3]
    sweep_path = project_root / "configs" / "wandb_sweep_ucb.yaml"

    if not sweep_path.exists():
        raise FileNotFoundError(
            f"Could not find sweep yaml at '{sweep_path}'. "
            f"Run from repo root or update SWEEP_YAML."
        )

    sweep_cfg = yaml.safe_load(sweep_path.read_text())
    metric_name = sweep_cfg.get("metric", {}).get("name", "final_success_rate")
    param_keys = list((sweep_cfg.get("parameters") or {}).keys())

    if not param_keys:
        raise RuntimeError(f"No parameters found in '{sweep_path}' under 'parameters:'")

    print("Sweep yaml:", sweep_path)
    print("Metric:", metric_name)
    print("Tuning parameters:", ", ".join(param_keys))

    # -------------------------
    # Load runs
    # -------------------------
    api = wandb.Api(timeout=60)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for project_name in PROJECTS:
        algo, ver = parse_project_name(project_name)
        if algo is None:
            raise ValueError(f"Project name does not match expected pattern: {project_name}")

        path = f"{ENTITY}/{project_name}"
        print(f"\nLoading runs from: {path}")
        runs = api.runs(path)

        kept = 0
        total = 0

        for run in tqdm(runs, desc=project_name, unit="run"):
            total += 1
            if ONLY_FINISHED and run.state != "finished":
                continue

            row: Dict[str, Any] = {}

            # Metadata
            row["entity"] = ENTITY
            row["project"] = project_name
            row["algorithm"] = algo
            row["version"] = ver
            row["run_id"] = run.id
            row["run_name"] = run.name
            row["state"] = run.state

            sweep_obj = getattr(run, "sweep", None)
            row["sweep_id"] = getattr(sweep_obj, "id", None) if sweep_obj is not None else None
            row["sweep_path"] = str(sweep_obj) if sweep_obj is not None else None

            try:
                row["created_at"] = run.created_at
            except Exception:
                row["created_at"] = None

            # -------------------------
            # Keep ONLY the tuning parameters from the sweep yaml
            # -------------------------
            for k in param_keys:
                # W&B sometimes stores nested keys as flat strings; we just read the flat key.
                row[f"param.{k}"] = safe_jsonable(run.config.get(k, None))

            # Metric (from summary)
            row[f"metric.{metric_name}"] = safe_jsonable(run.summary.get(metric_name, None))

            # (Optional) a couple of common useful summary fields if present:
            for k in ["_timestamp", "_runtime", "_step"]:
                if k in run.summary:
                    row[f"sum.{k}"] = safe_jsonable(run.summary.get(k))

            rows.append(row)
            kept += 1

        print(f"  Total runs seen: {total}")
        print(f"  Runs kept     : {kept} (ONLY_FINISHED={ONLY_FINISHED})")

    df = pd.DataFrame(rows)

    # Ensure version numeric
    df["version"] = pd.to_numeric(df["version"], errors="coerce").astype("Int64")

    # Final safety: stringify any remaining weird objects
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(safe_jsonable)

    metric_col = f"metric.{metric_name}"
    print("\nDataframe shape:", df.shape)
    print(f"Runs with '{metric_name}': {df[metric_col].notna().sum()} / {len(df)}")

    print("\nRuns per (algorithm, version):")
    print(df.groupby(["algorithm", "version"]).size().sort_index())

    # Save
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nWrote: {OUT_PARQUET}")
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()