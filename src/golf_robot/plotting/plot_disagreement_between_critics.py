# hole2_disagreement_per_action.py
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from thompson_bandit import (
    find_latest_doublecritic_checkpoint,
    load_doublecritics,
    cem_plan_action,
    MedianPlannerActor,
)

# -----------------------------
# Hardcoded paths (NO argparse)
# -----------------------------
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "configs").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

RL_CONFIG_PATH = PROJECT_ROOT / "configs" / "rl_config.yaml"
EPISODE_JSONL = PROJECT_ROOT / "log" / "real_episodes_eval" / "episode_logger_eval_thompson.jsonl"

MODEL_DIR = PROJECT_ROOT / "models" / "rl" / "dqn-bts"
OUT_DIR = PROJECT_ROOT / "log"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "hole2_disagreement_report_per_action.csv"
PLOT_QDIS = OUT_DIR / "hole2_qmin_across_heads_std.png"
PLOT_DA0 = OUT_DIR / "hole2_delta_action_a0_train_minus_eval.png"
PLOT_DA1 = OUT_DIR / "hole2_delta_action_a1_train_minus_eval.png"
PLOT_ABS_DA0 = OUT_DIR / "hole2_abs_delta_action_a0.png"
PLOT_ABS_DA1 = OUT_DIR / "hole2_abs_delta_action_a1.png"


def load_yaml(p: Path) -> dict:
    import yaml
    with p.open("r") as f:
        return yaml.safe_load(f)


def read_jsonl(p: Path) -> list[dict]:
    rows = []
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {p}")
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@torch.no_grad()
def qmin_per_head(critics1, critics2, s1: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
    vals = []
    for c1, c2 in zip(critics1, critics2):
        q1 = c1(s1, a1).squeeze()
        q2 = c2(s1, a1).squeeze()
        vals.append(torch.minimum(q1, q2))
    return torch.stack(vals, dim=0)


@torch.no_grad()
def q_absdiff_within_head(critics1, critics2, s1: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
    ds = []
    for c1, c2 in zip(critics1, critics2):
        q1 = c1(s1, a1).squeeze()
        q2 = c2(s1, a1).squeeze()
        ds.append(torch.abs(q1 - q2))
    return torch.stack(ds, dim=0)


@torch.no_grad()
def training_like_action(
    s1: torch.Tensor,  # [1, state_dim]
    critics1,
    critics2,
    rl_cfg: dict,
    device: torch.device,
    rng: np.random.RandomState,
    a_prev: torch.Tensor | None,
) -> tuple[torch.Tensor, dict, int, torch.Tensor | None]:
    cfg = rl_cfg["training"]
    K = len(critics1)
    head = int(rng.randint(K))

    cem_pop = int(cfg.get("cem_pop", 384))
    cem_iters = int(cfg.get("cem_iters", 2))
    cem_elite_frac = float(cfg.get("cem_elite_frac", 0.2))
    cem_init_std = float(cfg.get("cem_init_std", 0.4))
    cem_min_std = float(cfg.get("cem_min_std", 0.1))
    action_dim = int(rl_cfg["model"]["action_dim"])

    use_warm = bool(cfg.get("cem_warm_start", False))
    init_mu = a_prev.to(device) if (use_warm and a_prev is not None) else None

    s = s1.squeeze(0)

    def score_fn(s_batch, a_batch):
        q1 = critics1[head](s_batch, a_batch).squeeze(-1)
        q2 = critics2[head](s_batch, a_batch).squeeze(-1)
        return torch.minimum(q1, q2)

    a, stats = cem_plan_action(
        s=s,
        score_fn=score_fn,
        action_dim=action_dim,
        device=device,
        cem_iters=cem_iters,
        cem_pop=cem_pop,
        cem_elite_frac=cem_elite_frac,
        init_std=cem_init_std,
        min_std=cem_min_std,
        init_mu=init_mu,
        return_stats=True,
    )
    a1 = a.unsqueeze(0)
    new_a_prev = a.detach().cpu() if use_warm else None
    return a1, stats, head, new_a_prev


def main():
    rl_cfg = load_yaml(RL_CONFIG_PATH)
    device = torch.device(rl_cfg["training"].get("device", "cpu"))

    _, tag_stem = find_latest_doublecritic_checkpoint(MODEL_DIR, prefix=None)
    critics1, critics2, loaded_tag = load_doublecritics(MODEL_DIR, rl_cfg, tag_stem, device)

    for c in critics1:
        c.eval()
    for c in critics2:
        c.eval()

    eval_actor = MedianPlannerActor(critics1=critics1, critics2=critics2, rl_cfg=rl_cfg, device=device).to(device)
    eval_actor.eval()

    rows = read_jsonl(EPISODE_JSONL)
    hole2 = [r for r in rows if int(r.get("chosen_hole", -1)) == 2]
    if not hole2:
        raise RuntimeError(f"No episodes with chosen_hole==2 found in {EPISODE_JSONL}")

    rng = np.random.RandomState(0)
    a_prev = None

    report = []
    qstd_eval_list = []
    qstd_train_list = []

    delta_a0_list = []
    delta_a1_list = []
    abs_delta_a0_list = []
    abs_delta_a1_list = []

    for i, r in enumerate(hole2):
        if "state_norm" not in r:
            continue

        s = torch.tensor(r["state_norm"], dtype=torch.float32, device=device).unsqueeze(0)

        a_eval = eval_actor(s)  # [1, action_dim]
        a_train, cem_stats, head, a_prev = training_like_action(s, critics1, critics2, rl_cfg, device, rng, a_prev)

        # ---- per-component differences
        # supports action_dim >= 1,2 (your case is 2)
        da = (a_train - a_eval).squeeze(0).detach().cpu().numpy()  # [action_dim]
        delta_a0 = float(da[0])
        delta_a1 = float(da[1]) if len(da) > 1 else float("nan")
        abs_delta_a0 = float(abs(delta_a0))
        abs_delta_a1 = float(abs(delta_a1)) if len(da) > 1 else float("nan")

        delta_a0_list.append(delta_a0)
        abs_delta_a0_list.append(abs_delta_a0)
        if len(da) > 1:
            delta_a1_list.append(delta_a1)
            abs_delta_a1_list.append(abs_delta_a1)

        # ---- critic disagreement at each action
        v_eval = qmin_per_head(critics1, critics2, s, a_eval)
        d_eval = q_absdiff_within_head(critics1, critics2, s, a_eval)

        v_train = qmin_per_head(critics1, critics2, s, a_train)
        d_train = q_absdiff_within_head(critics1, critics2, s, a_train)

        qstd_eval = float(v_eval.std().item())
        qstd_train = float(v_train.std().item())
        qabs_eval = float(d_eval.mean().item())
        qabs_train = float(d_train.mean().item())

        qstd_eval_list.append(qstd_eval)
        qstd_train_list.append(qstd_train)

        report.append({
            "idx": i,
            "episode": r.get("episode", None),
            "chosen_hole": 2,
            "train_sampled_head": head,

            "action_eval_0": float(a_eval[0, 0].item()),
            "action_eval_1": float(a_eval[0, 1].item()) if a_eval.shape[1] > 1 else np.nan,
            "action_train_0": float(a_train[0, 0].item()),
            "action_train_1": float(a_train[0, 1].item()) if a_train.shape[1] > 1 else np.nan,

            # per-dim differences
            "delta_a0_train_minus_eval": delta_a0,
            "delta_a1_train_minus_eval": delta_a1,
            "abs_delta_a0": abs_delta_a0,
            "abs_delta_a1": abs_delta_a1,

            # critic disagreement
            "qmin_std_across_heads_eval_action": qstd_eval,
            "qmin_std_across_heads_train_action": qstd_train,
            "mean_abs_q1_q2_diff_eval_action": qabs_eval,
            "mean_abs_q1_q2_diff_train_action": qabs_train,

            "cem_best_score_train": float(cem_stats.get("best_score", np.nan)),
            "cem_std_mean_train": float(cem_stats.get("cem_std").mean().item()) if "cem_std" in cem_stats else np.nan,
        })

    if not report:
        raise RuntimeError(
            "Found hole==2 episodes, but none had 'state_norm'. "
            "Log 'state_norm' in evaluation JSONL the same way training logs it."
        )

    # ---- write CSV
    import csv
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)

    # ---- plots
    plt.figure()
    plt.plot(qstd_eval_list, label="std_k min(Q)_at_eval_action")
    plt.plot(qstd_train_list, label="std_k min(Q)_at_train_action")
    plt.xlabel("Hole-2 sample index")
    plt.ylabel("Std across heads")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_QDIS, dpi=160)
    plt.close()

    plt.figure()
    plt.plot(delta_a0_list, label="delta a0 = train - eval")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Hole-2 sample index")
    plt.ylabel("Signed delta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DA0, dpi=160)
    plt.close()

    if delta_a1_list:
        plt.figure()
        plt.plot(delta_a1_list, label="delta a1 = train - eval")
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("Hole-2 sample index")
        plt.ylabel("Signed delta")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DA1, dpi=160)
        plt.close()

    plt.figure()
    plt.plot(abs_delta_a0_list, label="|delta a0|")
    plt.xlabel("Hole-2 sample index")
    plt.ylabel("Absolute delta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_ABS_DA0, dpi=160)
    plt.close()

    if abs_delta_a1_list:
        plt.figure()
        plt.plot(abs_delta_a1_list, label="|delta a1|")
        plt.xlabel("Hole-2 sample index")
        plt.ylabel("Absolute delta")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_ABS_DA1, dpi=160)
        plt.close()

    print(f"[OK] Loaded critics tag: {loaded_tag}")
    print(f"[OK] Hole-2 rows used: {len(report)} / {len(hole2)}")
    print(f"[OK] Wrote: {OUT_CSV}")
    print(f"[OK] Plots: {PLOT_QDIS.name}, {PLOT_DA0.name}, {PLOT_DA1.name}, {PLOT_ABS_DA0.name}, {PLOT_ABS_DA1.name}")


if __name__ == "__main__":
    main()