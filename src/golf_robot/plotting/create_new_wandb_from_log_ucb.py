from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import wandb
import yaml


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield dicts from a JSONL file, skipping blank/malformed lines."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def relog_jsonl_to_wandb_ucb(
    jsonl_path: Path,
    *,
    project: str,
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[list[str]] = None,
    used_key: str = "used_for_training",
    only_used_for_training: bool = True,
    id: Optional[str] = None,
    mujoco_cfg: Optional[Dict[str, Any]] = None,
    rl_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Re-log UCB episodes from JSONL into a NEW W&B run with contiguous steps.

    Uses stored reward directly (no recompute).
    Logs ball_start_x/y (preferred) or ball_start_obs if present.
    """

    # IMPORTANT: make a NEW run with contiguous steps (do not resume training)
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=group,
        tags=tags,
        id=id,
        resume="never",
        config={
            "rl_config": rl_cfg,
            "mujoco_config": mujoco_cfg,
            "relog_source_jsonl": str(jsonl_path),
        },
    )

    step = 0
    logged = 0
    skipped = 0
    missing_reward = 0
    missing_ball_start = 0

    for ep in iter_jsonl(jsonl_path):
        if only_used_for_training and not ep.get(used_key, True):
            skipped += 1
            continue

        if "reward" not in ep or ep["reward"] is None:
            missing_reward += 1
            continue

        step += 1

        payload: Dict[str, Any] = {
            "reward": float(ep["reward"]),
        }

        # Ball start (UCB payload uses ball_start_x/y)
        if ep.get("ball_start_x", None) is not None and ep.get("ball_start_y", None) is not None:
            payload["ball_start/x"] = float(ep["ball_start_x"])
            payload["ball_start/y"] = float(ep["ball_start_y"])
        elif ep.get("ball_start_obs", None) is not None:
            try:
                x, y = ep["ball_start_obs"]
                payload["ball_start/x"] = float(x)
                payload["ball_start/y"] = float(y)
            except Exception:
                missing_ball_start += 1
        else:
            missing_ball_start += 1

        # Core UCB scalars (matches your log_payload)
        for k in (
            "in_hole",
            "out_of_bounds",
            "speed",
            "angle_deg",
            "chosen_hole",
            "q_mean",
            "q_std",
            "q_ucb",
            "ucb_beta",
            "replay_big_size",
            "replay_recent_size",
            "dist_at_hole",
            "speed_at_hole",
        ):
            if k in ep and ep[k] is not None:
                if k in ("chosen_hole", "replay_big_size", "replay_recent_size"):
                    try:
                        payload[k] = int(ep[k])
                    except Exception:
                        payload[k] = ep[k]
                else:
                    try:
                        payload[k] = float(ep[k]) if isinstance(ep[k], (int, float, str)) else ep[k]
                    except Exception:
                        payload[k] = ep[k]

        # Optional: keep original episode index
        if "episode" in ep and ep["episode"] is not None:
            try:
                payload["orig_episode"] = int(ep["episode"])
            except Exception:
                payload["orig_episode"] = ep["episode"]

        wandb.log(payload, step=step)
        logged += 1

    run.summary["relog_logged"] = logged
    run.summary["relog_skipped_not_used"] = skipped
    run.summary["relog_missing_reward"] = missing_reward
    run.summary["relog_missing_ball_start"] = missing_ball_start
    run.finish()

    print(
        f"[relog_ucb] logged={logged}, skipped_not_used={skipped}, "
        f"missing_reward={missing_reward}, missing_ball_start={missing_ball_start}"
    )


if __name__ == "__main__":
    # Repo root (same convention as your Thompson script)
    project_root = Path(__file__).parents[2]

    rl_config_path = project_root / "configs" / "rl_config.yaml"
    mujoco_config_path = project_root / "configs" / "mujoco_config.yaml"
    with open(rl_config_path, "r") as f:
        rl_cfg = yaml.safe_load(f)
    with open(mujoco_config_path, "r") as f:
        mujoco_cfg = yaml.safe_load(f)

    # UCB jsonl path
    log_path = project_root / "log" / "real_episodes" / "episode_logger_ucb.jsonl"

    # W&B target
    project_name = rl_cfg["training"].get("project_name", "rl_golf_wandb")
    run_id = rl_cfg["training"].get("wandb_run_id")  # optional; you can set None if you want a random id

    relog_jsonl_to_wandb_ucb(
        log_path,
        project=project_name,
        entity=None,
        run_name="relog_ucb_fixed_steps",
        group="en-ucb-relog",
        tags=["relog", "ucb"],
        only_used_for_training=True,
        id=run_id,
        mujoco_cfg=mujoco_cfg,
        rl_cfg=rl_cfg,
    )