#!/usr/bin/env python3
"""
estimate_lookahead.py

Estimate the best "lookahead" (time-shift) to align measured execution with the
precomputed planned reference.

It does two complementary estimates:

A) Position-based lookahead:
   Find delta that minimizes RMS(q_meas(t) - q_plan(t + delta)) over a grid.

B) Velocity-based lookahead (optional, recommended if you logged commanded dq):
   Find delta that maximizes correlation between dq_meas(t) and dq_cmd(t - delta).
   (Equivalent to minimizing lag between commanded and measured velocity.)

Outputs:
- best delta per joint and global (median)
- plots: RMS vs delta (per joint + aggregate), correlation vs delta (if cmd available)
- JSON summary saved in OUT_DIR

Assumed CSVs:
  planned:    log/trajectory_sim.csv           (t,q0..q5,dq0..dq5)
  measured:   log/streamed_measurements.csv    (t,q0..q5,dq0..dq5)   [event-driven]
  commanded:  log/commanded_velocity.csv       (t,dq_cmd0..dq_cmd5)  [per tick]
             (header can also be t,dq0..dq5)

Run:
  python estimate_lookahead.py

Tune knobs at top: DELTA_RANGE, DELTA_STEP, WINDOW, USE_CMD
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ------------------------------ Config ------------------------------
SHOW_PLOTS = True
OUT_DIR = "log"

PLANNED_CSV = "log/trajectory_sim.csv"
MEASURED_CSV = "log/streamed_measurements.csv"
COMMANDED_CSV = "log/commanded_velocity.csv"  # optional

USE_CMD = True  # set False if you don't have commanded_velocity.csv

# Search grid for delta (seconds). Positive delta means "look ahead":
# compare q_meas(t) to q_plan(t + delta).
DELTA_MIN = -0.040
DELTA_MAX = +0.040
DELTA_STEP = 0.0005  # 0.5 ms resolution. (DT=8 ms -> fine to start)

# Optional time window (seconds) to avoid startup/stop transients.
# If None, use full overlap.
WINDOW_START = None  # e.g. 0.3
WINDOW_END = None    # e.g. 2.2

# Robustness: clip insane velocities / filter quiet segments
MIN_DQ_STD = 0.05  # if a joint barely moves, correlation is meaningless
EPS = 1e-12

# -------------------------------------------------------------------

def _try_set_backend(show: bool) -> None:
    if show:
        for bk in ("TkAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
            try:
                matplotlib.use(bk)
                return
            except Exception:
                pass
    else:
        matplotlib.use("Agg")

_try_set_backend(SHOW_PLOTS)


def _sorted_unique_time(t: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, float)
    X = np.asarray(X, float)
    order = np.argsort(t)
    t_s = t[order]
    X_s = X[order]
    keep = np.ones_like(t_s, dtype=bool)
    keep[1:] = t_s[1:] > t_s[:-1]
    return t_s[keep], X_s[keep]


def load_planned_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """planned: t,q0..q5,dq0..dq5"""
    t, Q, dQ = [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        hdr = next(reader, None)
        for row in reader:
            if not row:
                continue
            row = [c.strip() for c in row]
            if len(row) < 13:
                continue
            try:
                vals = [float(x) for x in row[:13]]
            except ValueError:
                continue
            t.append(vals[0])
            Q.append(vals[1:7])
            dQ.append(vals[7:13])
    t = np.asarray(t, float)
    Q = np.asarray(Q, float).reshape(-1, 6)
    dQ = np.asarray(dQ, float).reshape(-1, 6)
    if len(t) < 2:
        raise RuntimeError(f"Not enough rows parsed from planned CSV: {path}")
    return t, Q, dQ


def load_streamer_log(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """measured: t,q0..q5,dq0..dq5 (event-driven). Stops at first malformed after valid data."""
    t_list, Q_list, dQ_list = [], [], []

    def _is_header(row):
        if not row:
            return True
        s = ",".join(row).lower()
        return any(k in s for k in ("t", "time", "q0", "dq0"))

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is not None and not _is_header(first):
            rows = [first] + list(reader)
        else:
            rows = list(reader)

    seen_valid = False
    for row in rows:
        if not row:
            continue
        row = [c.strip() for c in row]
        while row and row[-1] == "":
            row.pop()
        if len(row) < 13:
            if seen_valid:
                break
            continue
        try:
            t_val = float(row[0])
            q_vals = [float(x) for x in row[1:7]]
            dq_vals = [float(x) for x in row[7:13]]
        except ValueError:
            if seen_valid:
                break
            continue
        t_list.append(t_val)
        Q_list.append(q_vals)
        dQ_list.append(dq_vals)
        seen_valid = True

    t = np.asarray(t_list, float)
    Q = np.asarray(Q_list, float).reshape(-1, 6)
    dQ = np.asarray(dQ_list, float).reshape(-1, 6)
    if len(t) < 2:
        raise RuntimeError(f"Not enough rows parsed from measured log: {path}")
    return t, Q, dQ


def load_commanded_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    commanded: t,dq_cmd0..dq_cmd5 or t,dq0..dq5 (or no header: 7 cols)
    """
    t_list, dQ_list = [], []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            raise RuntimeError(f"Empty commanded CSV: {path}")

        def looks_like_header(r):
            s = ",".join([c.strip().lower() for c in r])
            return ("t" in s) or ("dq" in s) or ("time" in s)

        has_header = looks_like_header(first)

        if has_header:
            header = [c.strip() for c in first]
            col = {name: i for i, name in enumerate(header)}
            t_idx = col.get("t", 0)

            # Prefer dq_cmd*, else dq*
            dq_idxs = []
            for i in range(6):
                if f"dq_cmd{i}" in col:
                    dq_idxs.append(col[f"dq_cmd{i}"])
            if len(dq_idxs) < 6:
                dq_idxs = []
                for i in range(6):
                    if f"dq{i}" in col:
                        dq_idxs.append(col[f"dq{i}"])
            if len(dq_idxs) < 6:
                # fallback assume 1..6
                dq_idxs = list(range(1, 7))

            for row in reader:
                if not row:
                    continue
                row = [c.strip() for c in row]
                if len(row) <= max([t_idx] + dq_idxs):
                    continue
                try:
                    t = float(row[t_idx])
                    dq = [float(row[k]) for k in dq_idxs[:6]]
                except ValueError:
                    continue
                t_list.append(t)
                dQ_list.append(dq)

        else:
            rows = [first] + list(reader)
            for row in rows:
                if not row:
                    continue
                row = [c.strip() for c in row]
                if len(row) < 7:
                    continue
                try:
                    t = float(row[0])
                    dq = [float(x) for x in row[1:7]]
                except ValueError:
                    continue
                t_list.append(t)
                dQ_list.append(dq)

    t = np.asarray(t_list, float)
    dQ = np.asarray(dQ_list, float).reshape(-1, 6)
    if len(t) < 2:
        raise RuntimeError(f"Not enough rows parsed from commanded CSV: {path}")
    return t, dQ


def _apply_window(t: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if WINDOW_START is None and WINDOW_END is None:
        return t, X
    t0 = WINDOW_START if WINDOW_START is not None else float(t[0])
    t1 = WINDOW_END if WINDOW_END is not None else float(t[-1])
    mask = (t >= t0) & (t <= t1)
    return t[mask], X[mask]


def interp_matrix(t_src: np.ndarray, X_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Linear interpolate each column of X_src onto t_query (clipped to overlap)."""
    X_out = np.zeros((len(t_query), X_src.shape[1]), dtype=float)
    for j in range(X_src.shape[1]):
        X_out[:, j] = np.interp(t_query, t_src, X_src[:, j])
    return X_out


def estimate_delta_rms_q(
    t_plan: np.ndarray, Q_plan: np.ndarray,
    t_meas: np.ndarray, Q_meas: np.ndarray,
    deltas: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each delta, compute per-joint RMS of q_meas(t) - q_plan(t+delta) over overlap.
    Returns:
      rms_per_joint: (len(deltas),6)
      rms_total:     (len(deltas),)
    """
    rms_j = np.zeros((len(deltas), 6), dtype=float)
    rms_tot = np.zeros((len(deltas),), dtype=float)

    for k, delta in enumerate(deltas):
        # we compare Q_meas at t to Q_plan at (t + delta)
        t_query = t_meas + delta

        # overlap: t_query must be within t_plan range
        mask = (t_query >= t_plan[0]) & (t_query <= t_plan[-1])
        if mask.sum() < 10:
            rms_j[k, :] = np.nan
            rms_tot[k] = np.nan
            continue

        t_q = t_query[mask]
        Qm = Q_meas[mask]

        Qp = interp_matrix(t_plan, Q_plan, t_q)
        err = Qm - Qp
        rms = np.sqrt(np.mean(err * err, axis=0))
        rms_j[k, :] = rms
        rms_tot[k] = float(np.sqrt(np.mean(err * err)))

    return rms_j, rms_tot


def estimate_delta_corr_dq(
    t_cmd: np.ndarray, dQ_cmd: np.ndarray,
    t_meas: np.ndarray, dQ_meas: np.ndarray,
    deltas: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each delta, compute correlation between dQ_meas(t) and dQ_cmd(t - delta)
    (i.e. command leads measurement by delta). We interpolate cmd onto (t_meas - delta).
    Returns:
      corr_per_joint: (len(deltas),6)   (Pearson r)
      corr_total:     (len(deltas),)    (mean across valid joints)
    """
    corr_j = np.zeros((len(deltas), 6), dtype=float)
    corr_tot = np.zeros((len(deltas),), dtype=float)

    # normalize measured per joint once (we'll recompute after masking, but keep fast)
    for k, delta in enumerate(deltas):
        # command evaluated at t - delta (command earlier)
        t_query = t_meas - delta

        mask = (t_query >= t_cmd[0]) & (t_query <= t_cmd[-1])
        if mask.sum() < 10:
            corr_j[k, :] = np.nan
            corr_tot[k] = np.nan
            continue

        t_q = t_query[mask]
        y = dQ_meas[mask, :]  # measured
        x = interp_matrix(t_cmd, dQ_cmd, t_q)  # commanded aligned

        r_list = []
        for j in range(6):
            xj = x[:, j]
            yj = y[:, j]

            if np.std(xj) < MIN_DQ_STD or np.std(yj) < MIN_DQ_STD:
                corr_j[k, j] = np.nan
                continue

            xj0 = xj - np.mean(xj)
            yj0 = yj - np.mean(yj)
            denom = (np.std(xj0) * np.std(yj0)) + EPS
            r = float(np.mean(xj0 * yj0) / denom)
            corr_j[k, j] = r
            r_list.append(r)

        corr_tot[k] = float(np.nanmean(r_list)) if r_list else np.nan

    return corr_j, corr_tot


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    if not os.path.exists(PLANNED_CSV):
        raise FileNotFoundError(PLANNED_CSV)
    if not os.path.exists(MEASURED_CSV):
        raise FileNotFoundError(MEASURED_CSV)

    print(f"[INFO] Loading planned:  {PLANNED_CSV}")
    t_plan, Q_plan, dQ_plan = load_planned_csv(PLANNED_CSV)
    print(f"[INFO] Loading measured: {MEASURED_CSV}")
    t_meas, Q_meas, dQ_meas = load_streamer_log(MEASURED_CSV)

    # Sort/unique time
    t_plan, Q_plan = _sorted_unique_time(t_plan, Q_plan)
    t_meas, Q_meas = _sorted_unique_time(t_meas, Q_meas)
    # (velocities follow same ordering)
    _, dQ_plan = _sorted_unique_time(t_plan, dQ_plan)  # safe: same t_plan
    _, dQ_meas = _sorted_unique_time(t_meas, dQ_meas)

    # Apply optional windows
    t_meas_w, Q_meas_w = _apply_window(t_meas, Q_meas)
    _, dQ_meas_w = _apply_window(t_meas, dQ_meas)
    # planned stays full; windowing is done via overlap masks in the estimators

    deltas = np.arange(DELTA_MIN, DELTA_MAX + 0.5 * DELTA_STEP, DELTA_STEP)

    # ------------------ A) Position-based RMS ------------------
    print("[INFO] Estimating delta by minimizing RMS position error...")
    rms_j, rms_tot = estimate_delta_rms_q(t_plan, Q_plan, t_meas_w, Q_meas_w, deltas)

    best_idx_tot = int(np.nanargmin(rms_tot))
    best_delta_tot = float(deltas[best_idx_tot])

    best_delta_j = np.zeros(6, dtype=float)
    for j in range(6):
        best_delta_j[j] = float(deltas[int(np.nanargmin(rms_j[:, j]))])

    print(f"[RMS-Q] Best global delta: {best_delta_tot*1e3:.2f} ms")
    for j in range(6):
        print(f"[RMS-Q] Joint {j}: delta={best_delta_j[j]*1e3:.2f} ms")

    # ------------------ B) Velocity-based correlation (optional) ------------------
    corr_j = corr_tot = None
    best_delta_corr_tot = None
    best_delta_corr_j = None

    if USE_CMD and os.path.exists(COMMANDED_CSV):
        print(f"[INFO] Loading commanded: {COMMANDED_CSV}")
        t_cmd, dQ_cmd = load_commanded_csv(COMMANDED_CSV)
        t_cmd, dQ_cmd = _sorted_unique_time(t_cmd, dQ_cmd)

        t_meas_v, dQ_meas_v = _apply_window(t_meas, dQ_meas)

        print("[INFO] Estimating delta by maximizing corr(dq_meas, dq_cmd shifted)...")
        corr_j, corr_tot = estimate_delta_corr_dq(t_cmd, dQ_cmd, t_meas_v, dQ_meas_v, deltas)

        best_idx_corr = int(np.nanargmax(corr_tot))
        best_delta_corr_tot = float(deltas[best_idx_corr])

        best_delta_corr_j = np.zeros(6, dtype=float)
        for j in range(6):
            best_delta_corr_j[j] = float(deltas[int(np.nanargmax(corr_j[:, j]))])

        print(f"[CORR-dQ] Best global delta: {best_delta_corr_tot*1e3:.2f} ms")
        for j in range(6):
            print(f"[CORR-dQ] Joint {j}: delta={best_delta_corr_j[j]*1e3:.2f} ms")
    else:
        print("[INFO] Skipping commanded correlation (USE_CMD=False or commanded CSV missing).")

    # ------------------ Summary recommendation ------------------
    # Choose a single recommended delta: prefer RMS-Q (works even without cmd),
    # but if corr exists, blend by taking median of the two global estimates.
    rec_candidates = [best_delta_tot]
    if best_delta_corr_tot is not None:
        rec_candidates.append(best_delta_corr_tot)
    rec_delta = float(np.median(rec_candidates))

    print(f"\n[RECOMMEND] Suggested lookahead delta ≈ {rec_delta*1e3:.2f} ms")
    print("           (Use +delta in reference: q_ref(t)=q_plan(t+delta))")

    # ------------------ Save plots ------------------
    # RMS plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(deltas * 1e3, rms_tot, "-", linewidth=2, label="RMS total")
    for j in range(6):
        ax.plot(deltas * 1e3, rms_j[:, j], "-", linewidth=1, label=f"joint {j}")
    ax.axvline(best_delta_tot * 1e3, linestyle="--", linewidth=2, label=f"best total {best_delta_tot*1e3:.1f} ms")
    ax.set_xlabel("delta (ms)  [compare q_meas(t) vs q_plan(t + delta)]")
    ax.set_ylabel("RMS position error (rad)")
    ax.grid(True)
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    rms_png = os.path.join(OUT_DIR, f"lookahead_rms_q_{ts}.png")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.savefig(rms_png, dpi=150)
    plt.close(fig)

    corr_png = None
    if corr_tot is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(deltas * 1e3, corr_tot, "-", linewidth=2, label="mean corr")
        for j in range(6):
            ax.plot(deltas * 1e3, corr_j[:, j], "-", linewidth=1, label=f"joint {j}")
        ax.axvline(best_delta_corr_tot * 1e3, linestyle="--", linewidth=2,
                   label=f"best mean {best_delta_corr_tot*1e3:.1f} ms")
        ax.set_xlabel("delta (ms)  [compare dq_meas(t) vs dq_cmd(t - delta)]")
        ax.set_ylabel("Pearson correlation")
        ax.grid(True)
        ax.legend(ncol=2, fontsize=9)
        plt.tight_layout()

        corr_png = os.path.join(OUT_DIR, f"lookahead_corr_dq_{ts}.png")
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.savefig(corr_png, dpi=150)
        plt.close(fig)

    # ------------------ Save JSON summary ------------------
    summary = {
        "time": ts,
        "planned_csv": PLANNED_CSV,
        "measured_csv": MEASURED_CSV,
        "commanded_csv": COMMANDED_CSV if (USE_CMD and os.path.exists(COMMANDED_CSV)) else None,
        "grid": {
            "delta_min_s": DELTA_MIN,
            "delta_max_s": DELTA_MAX,
            "delta_step_s": DELTA_STEP,
            "window_start_s": WINDOW_START,
            "window_end_s": WINDOW_END,
        },
        "rms_q": {
            "best_delta_total_s": best_delta_tot,
            "best_delta_per_joint_s": best_delta_j.tolist(),
        },
        "corr_dq": None if corr_tot is None else {
            "best_delta_total_s": best_delta_corr_tot,
            "best_delta_per_joint_s": best_delta_corr_j.tolist(),
        },
        "recommended_delta_s": rec_delta,
        "plots": {
            "rms_png": None if SHOW_PLOTS else rms_png,
            "corr_png": None if (SHOW_PLOTS or corr_png is None) else corr_png,
        }
    }

    out_json = os.path.join(OUT_DIR, f"lookahead_estimate_{ts}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] Wrote summary: {out_json}")
    if not SHOW_PLOTS:
        print(f"[DONE] Wrote plots: {rms_png}" + ("" if corr_png is None else f", {corr_png}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
