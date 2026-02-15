#!/usr/bin/env python3
"""
measure_jitter.py

Measures:
  1) Local tick jitter for a 125 Hz loop (8 ms): how late/early do we run?
  2) Network/application RTT jitter to a UR robot via Dashboard Server (TCP 29999)

Dashboard Server facts:
  - Connect to robot IP on TCP port 29999
  - Commands end with newline '\n'
  (UR docs) :contentReference[oaicite:1]{index=1}

Outputs:
  log/jitter_local.csv
  log/jitter_rtt.csv

Notes:
  - This is "best effort" user-space timing (no RT kernel needed).
  - RTT via Dashboard includes some controller processing time (still useful).
"""

from __future__ import annotations

import csv
import os
import socket
import statistics as stats
import time
from pathlib import Path
from typing import Optional, Tuple, List


# =========================
# HARD-CODED SETTINGS
# =========================
ROBOT_IP = "192.38.66.227"     # <- your robot IP (same as traj_streamer.cpp)
DASHBOARD_PORT = 29999

HZ = 125.0
DT = 1.0 / HZ                 # 0.008s
DURATION_S = 60.0             # measure for this long

OUT_DIR = Path("log")
OUT_LOCAL = OUT_DIR / "jitter_local.csv"
OUT_RTT = OUT_DIR / "jitter_rtt.csv"

# Dashboard command candidates (different firmwares can vary).
# We'll try these in order and keep the first that yields a line response.
CMD_CANDIDATES = [
    "robotmode\n",
    "get robot mode\n",
    "running\n",
    "programState\n",
]


# =========================
# Helpers
# =========================
def ensure_outdir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def percentile(sorted_vals: List[float], p: float) -> float:
    """p in [0,100]. simple linear interpolation percentile."""
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def summarize(name: str, values_s: List[float], dt_target_s: Optional[float] = None) -> None:
    if not values_s:
        print(f"{name}: no samples")
        return

    v = values_s
    v_sorted = sorted(v)

    mean = stats.fmean(v)
    stdev = stats.pstdev(v) if len(v) > 1 else 0.0
    vmin = v_sorted[0]
    vmax = v_sorted[-1]
    p50 = percentile(v_sorted, 50)
    p90 = percentile(v_sorted, 90)
    p95 = percentile(v_sorted, 95)
    p99 = percentile(v_sorted, 99)

    print(f"\n=== {name} ===")
    print(f"samples: {len(v)}")
    print(f"min/mean/std/max: {vmin*1e3:.3f} / {mean*1e3:.3f} / {stdev*1e3:.3f} / {vmax*1e3:.3f} ms")
    print(f"p50/p90/p95/p99: {p50*1e3:.3f} / {p90*1e3:.3f} / {p95*1e3:.3f} / {p99*1e3:.3f} ms")

    if dt_target_s is not None:
        over = sum(1 for x in v if x > dt_target_s)
        print(f"> {dt_target_s*1e3:.3f} ms: {over} ({100.0*over/len(v):.2f}%)")


def connect_dashboard(ip: str, port: int, timeout_s: float = 1.0) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    # Disable Nagle for lower latency bursts
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.connect((ip, port))
    # After connect, use a slightly shorter timeout per command
    s.settimeout(0.5)
    return s


def recv_line(sock: socket.socket, max_bytes: int = 4096) -> str:
    """Read until newline or timeout; returns whatever we got (stripped)."""
    data = bytearray()
    while True:
        chunk = sock.recv(1)
        if not chunk:
            break
        data += chunk
        if chunk == b"\n" or len(data) >= max_bytes:
            break
    return data.decode("utf-8", errors="replace").strip()


def pick_working_command(sock: socket.socket) -> Tuple[str, str]:
    """
    Try CMD_CANDIDATES and return (cmd, first_response_line).
    Raises RuntimeError if none work.
    """
    # Some robots send a welcome line on connect—drain it if present.
    try:
        _ = recv_line(sock)
    except Exception:
        pass

    for cmd in CMD_CANDIDATES:
        try:
            sock.sendall(cmd.encode("ascii", errors="ignore"))
            resp = recv_line(sock)
            if resp:
                return cmd, resp
        except Exception:
            continue

    raise RuntimeError("No dashboard command responded. Are you in Remote Mode / is Dashboard available?")


# =========================
# Main measurement
# =========================
def main() -> None:
    ensure_outdir()

    # 1) Local tick jitter: measure actual period and deadline miss
    local_rows = []
    actual_periods = []
    lateness = []

    t0 = time.perf_counter()
    next_t = t0 + DT
    last_t = t0

    end_t = t0 + DURATION_S
    i = 0

    while True:
        now = time.perf_counter()
        if now >= end_t:
            break

        # Sleep most of the remaining time, then spin a tiny bit for accuracy
        remaining = next_t - now
        if remaining > 0.002:
            time.sleep(remaining - 0.001)

        while True:
            now = time.perf_counter()
            if now >= next_t:
                break

        # Tick
        period = now - last_t
        miss = now - next_t  # >=0 if we are late; negative if early (should be ~0)
        actual_periods.append(period)
        lateness.append(miss)

        local_rows.append({
            "i": i,
            "t_s": now - t0,
            "period_s": period,
            "lateness_s": miss,
        })

        last_t = now
        next_t += DT
        i += 1

    # 2) Dashboard RTT jitter
    rtt_rows = []
    rtts = []

    try:
        sock = connect_dashboard(ROBOT_IP, DASHBOARD_PORT)
        cmd, first_resp = pick_working_command(sock)
        print(f"Dashboard connected to {ROBOT_IP}:{DASHBOARD_PORT}. Using command: {cmd.strip()} (first resp: {first_resp})")
    except Exception as e:
        print(f"\n[WARN] Could not measure Dashboard RTT jitter: {e}")
        sock = None
        cmd = ""

    if sock is not None:
        t0r = time.perf_counter()
        end_tr = t0r + DURATION_S
        j = 0

        while True:
            now = time.perf_counter()
            if now >= end_tr:
                break

            t_send = time.perf_counter()
            try:
                sock.sendall(cmd.encode("ascii", errors="ignore"))
                resp = recv_line(sock)
                t_recv = time.perf_counter()
                rtt = t_recv - t_send
                rtts.append(rtt)
                rtt_rows.append({
                    "i": j,
                    "t_s": t_send - t0r,
                    "rtt_s": rtt,
                    "resp": resp,
                })
            except Exception as e:
                rtt_rows.append({
                    "i": j,
                    "t_s": time.perf_counter() - t0r,
                    "rtt_s": float("nan"),
                    "resp": f"ERROR: {e}",
                })

            # Don’t hammer the GUI; 50 Hz is plenty for RTT jitter visibility
            time.sleep(0.02)
            j += 1

        try:
            sock.close()
        except Exception:
            pass

    # Write CSVs
    with OUT_LOCAL.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["i", "t_s", "period_s", "lateness_s"])
        w.writeheader()
        w.writerows(local_rows)

    if rtt_rows:
        with OUT_RTT.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["i", "t_s", "rtt_s", "resp"])
            w.writeheader()
            w.writerows(rtt_rows)

    # Print summary
    summarize("Local actual period (should be ~8.000 ms)", actual_periods, dt_target_s=DT)
    # lateness: target is ~0; show abs for intuition too
    summarize("Local deadline lateness (positive = late)", lateness, dt_target_s=0.0)
    summarize("Dashboard RTT (TCP 29999)", rtts)

    print(f"\nWrote:\n  {OUT_LOCAL}\n  {OUT_RTT if rtt_rows else '(no RTT file)'}")


if __name__ == "__main__":
    main()