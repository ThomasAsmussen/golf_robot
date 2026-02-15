#!/usr/bin/env python3
# make_angle_plus1deg_in_norm.py
#
# Copies a JSONL episode log, but shifts the *angle* component of action_norm
# by +1 degree expressed in normalized action space.
#
# In your code, action_norm = [speed_norm, angle_norm] in [-1,1]^2, and
# angle_deg = squash_to_range(angle_norm, angle_low, angle_high).
# Therefore +1 deg in physical corresponds to:
#   delta_norm = 2 * 1deg / (angle_high - angle_low)
#
# Also (optionally) updates the logged "angle_deg" field for consistency.

import json
from pathlib import Path

# -------------------------
# EDIT THESE IF NEEDED
# -------------------------
root = Path(__file__).parents[3]  # repo root= Path(__file__).parents[3]  # repo root
IN_JSONL  = root / "log/real_episodes/episode_logger_ucb.jsonl"
OUT_JSONL = IN_JSONL.with_name(IN_JSONL.stem + "_angle_plus1deg_norm.jsonl")

# Try to load bounds from your repo config (same pattern as thompson_bandit.py).
# If this file doesn't exist in your environment, set ANGLE_LOW/HIGH manually below.
RL_CONFIG_CANDIDATES = [
    root / "configs/rl_config.yaml",
]

# Manual fallback (only used if rl_config.yaml not found)
ANGLE_LOW_FALLBACK  = None  # e.g. -10.0
ANGLE_HIGH_FALLBACK = None  # e.g.  10.0

# -------------------------
# Helpers
# -------------------------
def _load_angle_bounds():
    for p in RL_CONFIG_CANDIDATES:
        if p.exists():
            import yaml
            cfg = yaml.safe_load(p.read_text())
            angle_low = float(cfg["model"]["angle_low"])
            angle_high = float(cfg["model"]["angle_high"])
            return angle_low, angle_high, p

    if ANGLE_LOW_FALLBACK is None or ANGLE_HIGH_FALLBACK is None:
        raise FileNotFoundError(
            "Could not find configs/rl_config.yaml via RL_CONFIG_CANDIDATES, and "
            "ANGLE_LOW_FALLBACK/ANGLE_HIGH_FALLBACK are not set.\n"
            "Either run this from your repo root (so configs/rl_config.yaml exists), "
            "or set the fallback constants at the top of this script."
        )
    return float(ANGLE_LOW_FALLBACK), float(ANGLE_HIGH_FALLBACK), None


def _clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def main():
    angle_low, angle_high, cfg_path = _load_angle_bounds()
    if angle_high == angle_low:
        raise ValueError("angle_high == angle_low; cannot compute normalization delta.")

    delta_norm = 2.0 * 1.0 / (angle_high - angle_low)  # +1 degree in normalized space
    print(f"[info] angle_low={angle_low}, angle_high={angle_high}")
    print(f"[info] delta_norm for +1 deg = {delta_norm}")
    if cfg_path is not None:
        print(f"[info] loaded bounds from: {cfg_path}")

    if not IN_JSONL.exists():
        raise FileNotFoundError(IN_JSONL)

    n_total = 0
    n_changed = 0

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with IN_JSONL.open("r", encoding="utf-8") as fin, OUT_JSONL.open("w", encoding="utf-8") as fout:
        for line in fin:
            n_total += 1
            raw = line.rstrip("\n")
            if not raw.strip():
                fout.write(line)
                continue

            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                # Keep malformed lines as-is (mirrors your "robust logging" style)
                fout.write(line)
                continue

            a = rec.get("action_norm", None)
            if isinstance(a, list) and len(a) >= 2:
                try:
                    angle_norm_old = float(a[1])
                    angle_norm_new = _clip(angle_norm_old + delta_norm, -1.0, 1.0)
                    if angle_norm_new != angle_norm_old:
                        a[1] = angle_norm_new
                        rec["action_norm"] = a
                        n_changed += 1

                    # Optional: keep the logged angle_deg consistent if present
                    # if "angle_deg" in rec and rec["angle_deg"] is not None:
                    #     try:
                    #         ang_deg_old = float(rec["angle_deg"])
                    #         ang_deg_new = _clip(ang_deg_old + 1.0, angle_low, angle_high)
                    #         rec["angle_deg"] = ang_deg_new
                    #     except (TypeError, ValueError):
                    #         pass
                except (TypeError, ValueError):
                    # action_norm[1] not parseable; leave as-is
                    pass

            fout.write(json.dumps(rec) + "\n")

    print(f"[done] wrote: {OUT_JSONL}")
    print(f"[done] total lines: {n_total}, modified action_norm lines: {n_changed}")


if __name__ == "__main__":
    main()