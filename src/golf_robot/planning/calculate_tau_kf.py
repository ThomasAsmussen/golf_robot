#!/usr/bin/env python3
import os
import pathlib
import numpy as np
import pandas as pd

def find_repo_root(start: pathlib.Path) -> pathlib.Path:
    """
    Walk upwards until we find a folder that looks like the repo root.
    Adjust markers if needed.
    """
    cur = start.resolve()
    for _ in range(10):
        if (cur / "log").exists() or (cur / "src").exists() or (cur / ".git").exists():
            return cur
        cur = cur.parent
    return start.resolve()

def estimate_tau_from_logs(traj: pd.DataFrame, meas: pd.DataFrame, thr=0.02):
    """
    Fit tau per joint from:
      dq_{k+1} - u_k = a (dq_k - u_k)
      a = exp(-dt/tau)  => tau = -dt / ln(a)
    """
    # Basic checks
    if "t" not in traj.columns or "t" not in meas.columns:
        raise ValueError("Both CSVs must contain a 't' column.")

    t_traj = traj["t"].to_numpy(dtype=float)
    t_meas = meas["t"].to_numpy(dtype=float)

    # Ensure sorted by time (important for interp and diff)
    if not np.all(np.diff(t_traj) >= 0):
        traj = traj.sort_values("t").reset_index(drop=True)
        t_traj = traj["t"].to_numpy(dtype=float)

    if not np.all(np.diff(t_meas) >= 0):
        meas = meas.sort_values("t").reset_index(drop=True)
        t_meas = meas["t"].to_numpy(dtype=float)

    dt = np.diff(t_meas)

    details = {}  # joint -> array of tau samples
    summary_rows = []

    for j in range(6):
        col_cmd = f"dq{j}"
        col_meas = f"dq{j}"
        if col_cmd not in traj.columns:
            raise ValueError(f"trajectory_sim.csv missing column '{col_cmd}'")
        if col_meas not in meas.columns:
            raise ValueError(f"streamed_measurements.csv missing column '{col_meas}'")

        # Interpolate command velocities to measurement timestamps
        dq_cmd_full = np.interp(t_meas, t_traj, traj[col_cmd].to_numpy(dtype=float))
        dq_meas_full = meas[col_meas].to_numpy(dtype=float)

        # Assume u_k held from k to k+1
        u = dq_cmd_full[:-1]
        dq0 = dq_meas_full[:-1]
        dq1 = dq_meas_full[1:]

        r0 = dq0 - u
        r1 = dq1 - u

        # Valid samples:
        # - positive dt
        # - enough excitation (|r0| > thr)
        # - r0 and r1 same sign (pure exponential decay towards 0)
        mask = (dt > 1e-6) & (np.abs(r0) > thr) & (r0 * r1 > 0)

        ratio = np.empty_like(r0)
        ratio[:] = np.nan
        ratio[mask] = r1[mask] / r0[mask]

        # ratio must be in (0, 1) for stable first-order decay
        mask2 = mask & (ratio > 1e-6) & (ratio < 0.999999)

        a = ratio[mask2]
        dtm = dt[mask2]

        # tau = -dt / ln(a)
        tau = -dtm / np.log(a)

        # Drop nonsense (can happen if velocity is super noisy)
        tau = tau[np.isfinite(tau)]
        tau = tau[(tau > 1e-4) & (tau < 1.0)]  # keep within [0.1ms, 1s] sanity window

        details[j] = tau

        if len(tau) > 0:
            q25, q50, q75 = np.percentile(tau, [25, 50, 75])
            summary_rows.append((j, q50, q25, q75, len(tau)))
        else:
            summary_rows.append((j, np.nan, np.nan, np.nan, 0))

    summary = pd.DataFrame(
        summary_rows,
        columns=["joint", "tau_median_s", "tau_q25_s", "tau_q75_s", "n_samples"],
    )

    return details, summary

def main():
    here = pathlib.Path(__file__).resolve()
    repo = find_repo_root(here.parents[2])
    print(f"[INFO] Running from {repo}")

    traj_path = repo / "log" / "trajectory_sim.csv"
    meas_path = repo / "log" / "streamed_measurements.csv"
    print(f"[INFO] Loading trajectory from {traj_path}")
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing: {traj_path}")
    if not meas_path.exists():
        raise FileNotFoundError(f"Missing: {meas_path}")

    traj = pd.read_csv(traj_path)
    meas = pd.read_csv(meas_path)

    details, summary = estimate_tau_from_logs(traj, meas, thr=0.02)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print("\n[TAU SUMMARY]")
    print(summary)

    # Print a C++ array using the median values (fallback to 0.03 if missing)
    taus = []
    for j in range(6):
        tau_med = summary.loc[summary["joint"] == j, "tau_median_s"].values[0]
        if not np.isfinite(tau_med):
            tau_med = 0.03
        taus.append(float(tau_med))

    cpp = ", ".join(f"{t:.6f}" for t in taus)
    print("\n[C++]")
    print(f"static const std::array<double,6> TAU_CMD = {{ {cpp} }};")

    # Optional: save per-joint tau samples
    out_dir = repo / "log"
    out_csv = out_dir / "tau_samples.csv"
    rows = []
    for j, arr in details.items():
        for v in arr:
            rows.append((j, v))
    pd.DataFrame(rows, columns=["joint", "tau_s"]).to_csv(out_csv, index=False)
    print(f"\n[INFO] Wrote tau samples to {out_csv}")

if __name__ == "__main__":
    main()
