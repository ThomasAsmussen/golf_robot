#!/usr/bin/env python3
"""
Estimate per-joint actuator time constant tau from 5 repeated step tests.

Plot:
  - step reference u_ref (dashed)
  - mean response (shifted 3 samples left)
  - ±2 std band (shifted)
  - step edge vertical lines
  - 63.2% and 36.8% horizontal lines
  - crossing markers + vertical lines at t_cross_up/down

Notes:
  - Tau/lag are computed on the shifted mean so plot + numbers match.
"""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SHIFT_SAMPLES = 3  # shift measurements 3 samples to the left


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "t" not in df.columns:
        raise ValueError(f"Missing 't' column in {path}")
    return df


def _interp_to(t_ref: np.ndarray, t_meas: np.ndarray, y_meas: np.ndarray) -> np.ndarray:
    m = np.isfinite(t_meas) & np.isfinite(y_meas)
    t = np.asarray(t_meas[m], dtype=float)
    y = np.asarray(y_meas[m], dtype=float)
    if t.size < 2:
        return np.full_like(t_ref, np.nan, dtype=float)

    order = np.argsort(t)
    t, y = t[order], y[order]

    # Drop duplicate times (keep last)
    _, idx = np.unique(t, return_index=True)
    t, y = t[idx], y[idx]

    return np.interp(t_ref, t, y, left=np.nan, right=np.nan)


def shift_left_nan(y: np.ndarray, k: int) -> np.ndarray:
    """Shift array left by k samples (advance signal). Fill tail with NaN."""
    y = np.asarray(y, dtype=float)
    if k <= 0:
        return y.copy()
    out = np.full_like(y, np.nan, dtype=float)
    if k < y.size:
        out[:-k] = y[k:]
    return out


def _find_step_edges(u: np.ndarray, thresh: float = 0.5) -> tuple[int, int]:
    u = np.asarray(u, dtype=float)
    above = u > thresh
    d = np.diff(above.astype(int))

    up_candidates = np.where(d == 1)[0] + 1
    down_candidates = np.where(d == -1)[0] + 1

    if up_candidates.size == 0 or down_candidates.size == 0:
        raise ValueError("Could not detect both up and down edges in command.")

    idx_up = int(up_candidates[0])
    downs_after = down_candidates[down_candidates > idx_up]
    if downs_after.size == 0:
        raise ValueError("Detected an up edge but no down edge after it.")
    idx_down = int(downs_after[0])
    return idx_up, idx_down


def estimate_tau_from_mean_step(t: np.ndarray, u: np.ndarray, y: np.ndarray) -> dict:
    """
    Estimate tau from a 0->1->0 step test using 63.2% (up) and 36.8% (down)
    crossings, with sub-sample crossing times via linear interpolation.

    Assumes forced levels: low=0, high=1.
    """
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)

    idx_up, idx_down = _find_step_edges(u)
    n = len(t)
    if n < 3:
        raise ValueError("t is too short")

    target_up = 0.632
    target_down = 0.368

    def interp_crossing_time(i0, i1, target) -> float:
        y0, y1 = y[i0], y[i1]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            raise ValueError("Non-finite samples around crossing.")
        if y1 == y0:
            return float(t[i1])
        alpha = (target - y0) / (y1 - y0)
        return float(t[i0] + alpha * (t[i1] - t[i0]))

    # Rising: first y >= target_up after idx_up (before idx_down)
    up_search = np.arange(idx_up, min(n, idx_down))
    up_search = up_search[np.isfinite(y[up_search])]
    if up_search.size == 0:
        raise ValueError("No finite samples on rising search window.")

    ge = np.where(y[up_search] >= target_up)[0]
    if ge.size == 0:
        raise ValueError("No 63.2% crossing found on rising edge.")
    i1_up = int(up_search[ge[0]])
    i0_up = i1_up - 1
    while i0_up >= idx_up and (not np.isfinite(y[i0_up]) or y[i0_up] >= target_up):
        i0_up -= 1
    t_cross_up = float(t[i1_up]) if i0_up < idx_up else interp_crossing_time(i0_up, i1_up, target_up)
    tau_up = float(t_cross_up - t[idx_up])

    # Falling: first y <= target_down after idx_down
    down_search = np.arange(idx_down, n)
    down_search = down_search[np.isfinite(y[down_search])]
    if down_search.size == 0:
        raise ValueError("No finite samples on falling search window.")

    le = np.where(y[down_search] <= target_down)[0]
    if le.size == 0:
        raise ValueError("No 36.8% crossing found on falling edge.")
    i1_down = int(down_search[le[0]])
    i0_down = i1_down - 1
    while i0_down >= idx_down and (not np.isfinite(y[i0_down]) or y[i0_down] <= target_down):
        i0_down -= 1
    t_cross_down = float(t[i1_down]) if i0_down < idx_down else interp_crossing_time(i0_down, i1_down, target_down)
    tau_down = float(t_cross_down - t[idx_down])

    return {
        "idx_up": idx_up,
        "idx_down": idx_down,
        "target_up": target_up,
        "target_down": target_down,
        "t_cross_up": t_cross_up,
        "t_cross_down": t_cross_down,
        "tau_up": tau_up,
        "tau_down": tau_down,
        "tau": 0.5 * (tau_up + tau_down),
    }


def find_measurement_files_for_joint(folder: Path, joint: int) -> list[Path]:
    pat = re.compile(rf"^streamed_measurements_{joint}(\d)\.csv$")
    files = []
    for p in folder.glob(f"streamed_measurements_{joint}*.csv"):
        m = pat.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files[:5]]


def estimate_lag_from_edge_threshold(t, y, idx_edge, step_from, step_to, frac=0.05):
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    if step_to > step_from:
        thr = step_from + frac * (step_to - step_from)  # e.g. 0.05
        search = np.arange(idx_edge, len(t))
        search = search[np.isfinite(y[search])]
        hit = search[np.where(y[search] >= thr)[0]]
    else:
        thr = step_to + (1.0 - frac) * (step_from - step_to)  # e.g. 0.95
        search = np.arange(idx_edge, len(t))
        search = search[np.isfinite(y[search])]
        hit = search[np.where(y[search] <= thr)[0]]

    if hit.size == 0:
        return np.nan, thr
    return float(t[hit[0]] - t[idx_edge]), float(thr)


def main() -> None:
    here = Path(__file__).resolve()
    repo = here.parents[3]
    data_dir = repo / "log" / "tau_tuning"
    print(f"[INFO] Loading data from {data_dir}")

    ref_path = data_dir / "commanded_velocity.csv"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference file: {ref_path}")

    ref = _read_csv(ref_path)
    t_ref = ref["t"].to_numpy(dtype=float)
    u_ref = ref["dq_cmd4"].to_numpy(dtype=float)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.ravel()

    taus = []
    delays = []

    for j in range(6):
        meas_files = find_measurement_files_for_joint(data_dir, j)
        if len(meas_files) < 5:
            raise FileNotFoundError(
                f"Found {len(meas_files)} measurement files for joint {j}, expected 5. "
                f"Looked for streamed_measurements_{j}?.csv in {data_dir}"
            )

        Ys = []
        for mf in meas_files:
            dfm = _read_csv(mf)
            tm = dfm["t"].to_numpy(dtype=float)
            qj = dfm[f"q{j}"].to_numpy(dtype=float)

            m = np.isfinite(tm) & np.isfinite(qj)
            tm, qj = tm[m], qj[m]
            if tm.size < 2:
                Ys.append(np.full_like(t_ref, np.nan, dtype=float))
                continue

            order = np.argsort(tm)
            tm, qj = tm[order], qj[order]

            _, idx = np.unique(tm, return_index=True)
            tm, qj = tm[idx], qj[idx]

            # dq = Δq/Δt on measurement grid
            dq_raw = np.zeros_like(qj)
            dtm = np.diff(tm)
            good = dtm > 1e-9
            dq_raw[0] = 0.0
            dq_raw[1:] = 0.0
            dq_raw[1:][good] = (qj[1:][good] - qj[:-1][good]) / dtm[good]

            Ys.append(_interp_to(t_ref, tm, dq_raw))

        Y = np.vstack(Ys)  # (5, N)
        y_mean = np.nanmean(Y, axis=0)
        y_std = np.nanstd(Y, axis=0)

        # Shift measurements left by 3 samples
        y_mean_s = shift_left_nan(y_mean, SHIFT_SAMPLES)
        y_std_s = shift_left_nan(y_std, SHIFT_SAMPLES)

        # Tau/lag estimation on shifted mean (so plot & numbers match)
        est = estimate_tau_from_mean_step(t_ref, u_ref, y_mean_s)
        taus.append(est["tau"])

        lag_up, _ = estimate_lag_from_edge_threshold(
            t_ref, y_mean_s, idx_edge=est["idx_up"], step_from=0.0, step_to=1.0, frac=0.05
        )
        lag_down, _ = estimate_lag_from_edge_threshold(
            t_ref, y_mean_s, idx_edge=est["idx_down"], step_from=1.0, step_to=0.0, frac=0.05
        )
        delays.append((
            lag_up * 1000.0 if np.isfinite(lag_up) else np.nan,
            lag_down * 1000.0 if np.isfinite(lag_down) else np.nan
        ))

        ax = axes[j]

        # Mean + ±2 std band
        ax.plot(t_ref, u_ref, linestyle="--", linewidth=1.5, label="Step reference")
        ax.plot(t_ref, y_mean_s, linewidth=2.5, label="Mean response")
        ax.fill_between(
            t_ref,
            y_mean_s - 2.0 * y_std_s,
            y_mean_s + 2.0 * y_std_s,
            alpha=0.20,
            linewidth=0,
            label="±2 $\sigma$",
        )

        # Step reference
        

        # Step edges
        t_up = t_ref[est["idx_up"]]
        t_down = t_ref[est["idx_down"]]
        ax.axvline(t_up, linestyle=":", linewidth=1)
        ax.axvline(t_down, linestyle=":", linewidth=1)

        # 63.2% / 36.8% thresholds
        # ax.axhline(est["target_up"], linestyle="--", linewidth=1, label="63.2% / 36.8% thresholds")
        # ax.axhline(est["target_down"], linestyle="--", linewidth=1)

        # Crossing markers + verticals at crossing times
        ax.plot([est["t_cross_up"]], [est["target_up"]], marker="o", label="63.2% crossing")
        ax.plot([est["t_cross_down"]], [est["target_down"]], marker="o", label="36.8% crossing")
        ax.axvline(est["t_cross_up"], linestyle=":", linewidth=1)
        ax.axvline(est["t_cross_down"], linestyle=":", linewidth=1)

        ax.set_title(f"Joint {j}: τ={est['tau']:.4f}s (up={est['tau_up']:.4f}, down={est['tau_down']:.4f})")
        ax.set_xlabel("t [s]")
        ax.set_ylabel(f"dq{j} [rad/s]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    plt.tight_layout()
    plt.show()

    print("\nEstimated taus (shifted mean):")
    for j, tau in enumerate(taus):
        print(f"  joint {j}: {tau:.6f} s")

    print("\nEstimated lags from step edge (5% threshold, shifted mean):")
    for j, (up_ms, down_ms) in enumerate(delays):
        print(f"  joint {j}: lag_up={up_ms:.2f} ms, lag_down={down_ms:.2f} ms")


if __name__ == "__main__":
    main()