#!/usr/bin/env python3
# load_all_wandb_runs.py

import re
from typing import Any, Dict, List

import pandas as pd
import wandb
from tqdm import tqdm

# -------------------------
# HARDCODED SETTINGS
# -------------------------
ENTITY = "rl_golf"
PROJECT_PREFIX = "golf_robot_"
METRIC = "final_success_rate"
ONLY_FINISHED = True

OUT_PARQUET = "data/wandb/wandb_all_runs.parquet"
OUT_CSV = "data/wandb/wandb_all_runs.csv"

PROJ_RE = re.compile(r"^(?P<prefix>golf_robot)_(?P<algo>[^_]+)_v(?P<diff>\d+)$")


def parse_project_name(project_name: str):
    m = PROJ_RE.match(project_name)
    if not m:
        return None, None
    return m.group("algo"), int(m.group("diff"))


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
    api = wandb.Api(timeout=60)

    # Discover projects
    projects = []
    for p in api.projects(ENTITY):
        if not p.name.startswith(PROJECT_PREFIX):
            continue
        algo, diff = parse_project_name(p.name)
        if algo is None:
            continue
        if diff == 0:
            continue
        projects.append((p.name, algo, diff))
    # projects = [
    #     ("golf_robot_ddpg_v3", "ddpg", 3)
    # ]
    if not projects:
        raise RuntimeError(
            f"No projects found under entity='{ENTITY}' matching "
            f"'{PROJECT_PREFIX}<algo>_v<diff>'."
        )

    projects = sorted(projects, key=lambda x: (x[2], x[1], x[0]))
    print("Found projects:")
    for name, algo, diff in projects:
        print(f"  - {name:25s}  algo={algo:10s}  diff={diff}")

    rows: List[Dict[str, Any]] = []

    for project_name, algo, diff in projects:
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
            row["difficulty"] = diff
            row["run_id"] = run.id
            row["run_name"] = run.name
            row["state"] = run.state

            # FIX: sweep as string id, not Sweep object
            sweep_obj = getattr(run, "sweep", None)
            row["sweep_id"] = getattr(sweep_obj, "id", None) if sweep_obj is not None else None
            row["sweep_path"] = str(sweep_obj) if sweep_obj is not None else None

            try:
                row["created_at"] = run.created_at
            except Exception:
                row["created_at"] = None

            # Config
            for k, v in run.config.items():
                if k.startswith("_"):
                    continue
                row[f"cfg.{k}"] = safe_jsonable(v)

            # Summary
            for k, v in run.summary.items():
                row[f"sum.{k}"] = safe_jsonable(v)

            # Convenience: metric
            row[f"metric.{METRIC}"] = safe_jsonable(run.summary.get(METRIC, None))

            rows.append(row)
            kept += 1

        print(f"  Total runs seen: {total}")
        print(f"  Runs kept     : {kept} (ONLY_FINISHED={ONLY_FINISHED})")

    df = pd.DataFrame(rows)

    # Ensure difficulty numeric
    df["difficulty"] = pd.to_numeric(df["difficulty"], errors="coerce").astype("Int64")

    # Final safety: stringify any remaining weird objects
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(safe_jsonable)

    metric_col = f"metric.{METRIC}"
    print("\nDataframe shape:", df.shape)
    print(f"Runs with '{METRIC}': {df[metric_col].notna().sum()} / {len(df)}")

    print("\nRuns per (algorithm, difficulty):")
    print(df.groupby(["algorithm", "difficulty"]).size().sort_index())

    # Save
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nWrote: {OUT_PARQUET}")
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()