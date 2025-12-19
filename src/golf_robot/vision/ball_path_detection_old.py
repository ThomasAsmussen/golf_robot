#!/usr/bin/env python3
"""
Outputs (in the video’s folder):
- <stem>_trajectory.csv                (time_s, x_px, y_px, x_m, y_m, vx_mps, vy_mps, speed_mps)
- <stem>_xt.png, _yt.png, _traj.png    (positions)
- <stem>_vx.png, _vy.png, _speed.png   (velocities)
- <stem>_annotated.mp4                 (if --write_preview)

Requires: Python 3.8+, opencv-python, numpy, matplotlib, pandas
"""

import argparse
import os
import sys
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# Config & CLI
# -----------------------

@dataclass
class Config:
    video_path: str
    hsv_lower: Tuple[int, int, int]
    hsv_upper: Tuple[int, int, int]
    morph_kernel: int
    min_area: float
    max_area: float
    min_circularity: float
    blur_ksize: int
    write_preview: bool
    fill_gaps: bool
    gap_max_frames: int
    outlier_window: int
    outlier_k: float
    smooth_window: int

    # Checkerboard calibration 
    cb_cols: int          # inner corners per row
    cb_rows: int          # inner corners per col
    square_size_m: float  # checkerboard square size in meters



def parse_triplet(s: str) -> Tuple[int, int, int]:
    try:
        a, b, c = s.split(",")
        return int(a), int(b), int(c)
    except Exception:
        raise argparse.ArgumentTypeError("HSV triplet must be 'h,s,v' (e.g., 5,80,80).")


def get_args() -> Config:
    ap = argparse.ArgumentParser(description="Track orange golf ball, remove outliers, and plot position/velocity over time.")
    ap.add_argument("--video", dest="video_path", help="Path to input video", default="C:/Users/marti/OneDrive - Danmarks Tekniske Universitet/DTU/GitHub/golf_robot/data/tuning_videos/test1.mp4")
    ap.add_argument("--hsv_lower", type=parse_triplet, default=(5, 80, 80),
                    help="Lower HSV for orange (default 5,80,80)")
    ap.add_argument("--hsv_upper", type=parse_triplet, default=(30, 255, 255),
                    help="Upper HSV for orange (default 30,255,255)")
    ap.add_argument("--morph_kernel", type=int, default=5, help="Morph kernel size (odd). Default 5")
    ap.add_argument("--min_area", type=float, default=30.0, help="Min contour area to consider (px^2)")
    ap.add_argument("--max_area", type=float, default=50000.0, help="Max contour area to consider (px^2)")
    ap.add_argument("--min_circularity", type=float, default=0.6, help="Min circularity 0..1")
    ap.add_argument("--blur_ksize", type=int, default=5, help="Median blur kernel (odd). Default 5")
    ap.add_argument("--write_preview", action="store_true", help="Write annotated preview MP4", default=True)
    ap.add_argument("--fill_gaps", action="store_true",
                    help="Linearly interpolate short missing segments")
    ap.add_argument("--gap_max_frames", type=int, default=10,
                    help="Max missing frames to interpolate when --fill_gaps is set")

    # Outliers & smoothing
    ap.add_argument("--outlier_window", type=int, default=11,
                    help="Odd window (frames) for outlier detection (Hampel)")
    ap.add_argument("--outlier_k", type=float, default=3.0,
                    help="Aggressiveness in MADs for Hampel (larger = less aggressive)")
    ap.add_argument("--smooth_window", type=int, default=5,
                    help="Odd window for rolling-median smoothing (1 disables)")
    
    # Checkerboard calibration
    ap.add_argument("--cb_cols", type=int, default=8,
                    help="Number of inner corners per row (OpenCV patternSize width)")

    ap.add_argument("--cb_rows", type=int, default=6,
                    help="Number of inner corners per column (OpenCV patternSize height)")

    ap.add_argument("--square_size_m", type=float, default=(26.9 / 9.0) / 100.0,
                    help="Checkerboard square size in meters "
                         "(default assumes 26.9 cm board width / 9 squares)")


    args = ap.parse_args()

    cfg = Config(
        video_path=args.video_path,
        hsv_lower=args.hsv_lower,
        hsv_upper=args.hsv_upper,
        morph_kernel=args.morph_kernel,
        min_area=args.min_area,
        max_area=args.max_area,
        min_circularity=args.min_circularity,
        blur_ksize=args.blur_ksize,
        write_preview=args.write_preview,
        fill_gaps=args.fill_gaps,
        gap_max_frames=args.gap_max_frames,
        outlier_window=args.outlier_window,
        outlier_k=args.outlier_k,
        smooth_window=args.smooth_window,

        # NEW
        cb_cols=args.cb_cols,
        cb_rows=args.cb_rows,
        square_size_m=args.square_size_m,
    )

    return cfg


# -----------------------
# Utilities
# -----------------------

def compute_homography_from_checkerboard(
        frame: np.ndarray,
        pattern_size: Tuple[int, int],
        square_size_m: float) -> np.ndarray:
    """
    Detect a checkerboard and compute a homography from image pixels -> plane (X,Y in meters).

    pattern_size: (cols, rows) = number of inner corners per row/column (OpenCV convention).
    square_size_m: physical size of one square in meters.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # pattern_size = (cb_cols, cb_rows)
    found, corners = cv2.findChessboardCorners(gray, pattern_size,
                                               flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_FAST_CHECK +
                                                     cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        raise RuntimeError("Could not find checkerboard in the first frame.")

    # Refine corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Build the corresponding world points (on the plane, Z=0)
    cb_cols, cb_rows = pattern_size
    objp = np.zeros((cb_cols * cb_rows, 3), np.float32)
    # mgrid order: x from 0..cb_cols-1, y from 0..cb_rows-1
    objp[:, :2] = np.mgrid[0:cb_cols, 0:cb_rows].T.reshape(-1, 2)
    objp[:, :2] *= square_size_m  # in meters

    # Compute homography: image (u,v) -> world (X,Y)
    image_pts = corners_sub.reshape(-1, 2)
    world_pts = objp[:, :2]
    H, _ = cv2.findHomography(image_pts, world_pts)
    if H is None:
        raise RuntimeError("findHomography failed for checkerboard.")

    # After you compute H in compute_homography_from_checkerboard
    cb_cols, cb_rows = pattern_size
    board_width_m  = (cb_cols - 1) * square_size_m
    board_height_m = (cb_rows - 1) * square_size_m

    return H


def circularity(cnt: np.ndarray) -> float:
    per = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    if per <= 1e-6:
        return 0.0
    return 4 * math.pi * (area / (per * per))


def detect_ball_center(bgr: np.ndarray,
                       hsv_lower: Tuple[int, int, int],
                       hsv_upper: Tuple[int, int, int],
                       blur_ksize: int,
                       morph_kernel: int,
                       min_area: float,
                       max_area: float,
                       min_circularity: float) -> Optional[Tuple[float, float, float]]:
    """
    Returns (cx, cy, radius) in pixels if found, else None
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array(hsv_lower, dtype=np.uint8), np.array(hsv_upper, dtype=np.uint8))

    if blur_ksize % 2 == 0:
        blur_ksize += 1
    if blur_ksize > 1:
        mask = cv2.medianBlur(mask, blur_ksize)

    ksize = max(1, morph_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        circ = circularity(cnt)
        if circ < min_circularity:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        score = 0.7 * circ + 0.3 * min(1.0, area / max_area)
        if score > best_score:
            best_score = score
            best = (float(x), float(y), float(radius))

    return best


def interpolate_gaps(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Linear interpolate NaNs in a 1D array for gaps up to max_gap length.
    Longer gaps remain NaN.
    """
    out = arr.copy()
    n = len(out)
    isnan = np.isnan(out)
    if not isnan.any():
        return out

    idx = np.arange(n)
    valid = ~isnan
    if valid.sum() < 2:
        return out

    i = 0
    while i < n:
        if isnan[i]:
            j = i
            while j < n and isnan[j]:
                j += 1
            gap_len = j - i
            if gap_len <= max_gap and i > 0 and j < n:
                out[i:j] = np.interp(idx[i:j], idx[valid], out[valid])[0:gap_len]
            i = j
        else:
            i += 1
    return out


def hampel_filter_1d(x: np.ndarray, window_size: int, n_sigma: float) -> np.ndarray:
    """
    Robust outlier detection: mark points as NaN if they deviate from the rolling median
    by more than n_sigma * 1.4826 * MAD, within a centered window.
    Returns a copy with outliers set to NaN (does not interpolate).
    """
    if window_size < 3:
        return x.copy()
    if window_size % 2 == 0:
        window_size += 1

    x = x.astype(float)
    x_clean = x.copy()
    n = len(x)
    half = window_size // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w = x[lo:hi]
        wv = w[~np.isnan(w)]
        if wv.size < 3:
            continue
        med = np.median(wv)
        mad = np.median(np.abs(wv - med))
        if mad <= 1e-12:
            continue
        sigma = 1.4826 * mad
        if not np.isnan(x[i]) and np.abs(x[i] - med) > n_sigma * sigma:
            x_clean[i] = np.nan
    return x_clean


def rolling_median_1d(x: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling median smoothing (odd window). NaNs are preserved; short NaN runs can be
    pre-filled using interpolate_gaps() before/after.
    """
    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w = x[lo:hi]
        wv = w[~np.isnan(w)]
        if wv.size == 0:
            continue
        out[i] = np.median(wv)
    return out


def fill_ends(a: np.ndarray) -> np.ndarray:
    """Forward/back fill NaNs at array ends with nearest valid value."""
    a = a.copy()
    if np.isnan(a[0]):
        first = np.flatnonzero(~np.isnan(a))
        if first.size:
            a[:first[0]] = a[first[0]]
    if np.isnan(a[-1]):
        last = np.flatnonzero(~np.isnan(a))
        if last.size:
            a[last[-1]:] = a[last[-1]]
    return a


# -----------------------
# Main
# -----------------------

def main():
    cfg = get_args()
    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {cfg.video_path}", file=sys.stderr)
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1e-3:
        print("Warning: FPS not found; defaulting to 30.")
        src_fps = 30.0

    ok, first = cap.read()
    if not ok:
        print("Could not read first frame.", file=sys.stderr)
        sys.exit(1)

    # Use checkerboard
    H = None          # homography image -> plane (m)

    pattern_size = (cfg.cb_cols, cfg.cb_rows)
    print(f"Using checkerboard: pattern_size={pattern_size}, "
        f"square_size={cfg.square_size_m:.6f} m")
    H = compute_homography_from_checkerboard(first, pattern_size, cfg.square_size_m)
    print("Computed homography from checkerboard.")


    # Output paths
    # Force all outputs into ../../data/tuning_videos relative to script launch
    output_root = os.path.abspath(os.path.join("..", "..", "data", "tuning_videos"))
    os.makedirs(output_root, exist_ok=True)

    stem = os.path.splitext(os.path.basename(cfg.video_path))[0]

    out_csv   = os.path.join(output_root, f"{stem}_trajectory.csv")
    out_xt    = os.path.join(output_root, f"{stem}_xt.png")
    out_yt    = os.path.join(output_root, f"{stem}_yt.png")
    out_traj  = os.path.join(output_root, f"{stem}_traj.png")
    out_vx    = os.path.join(output_root, f"{stem}_vx.png")
    out_vy    = os.path.join(output_root, f"{stem}_vy.png")
    out_speed = os.path.join(output_root, f"{stem}_speed.png")
    out_preview = os.path.join(output_root, f"{stem}_annotated.mp4")


    # Optional writer
    writer = None
    if cfg.write_preview:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_fps = src_fps
        w, h = first.shape[1], first.shape[0]
        writer = cv2.VideoWriter(out_preview, fourcc, writer_fps, (w, h))

    # Allocate arrays
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    xs = np.full(frame_count, np.nan, dtype=np.float64)
    ys = np.full(frame_count, np.nan, dtype=np.float64)
    ts_video = np.zeros(frame_count, dtype=np.float64)

    # Process
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts_video[idx] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        det = detect_ball_center(frame,
                                 cfg.hsv_lower, cfg.hsv_upper,
                                 cfg.blur_ksize, cfg.morph_kernel,
                                 cfg.min_area, cfg.max_area, cfg.min_circularity)

        if det is not None:
            cx, cy, r = det
            xs[idx] = cx
            ys[idx] = cy

        # Preview overlay
        if writer is not None:
            show = frame.copy()

            # Draw ball in original image coords
            if not np.isnan(xs[idx]) and not np.isnan(ys[idx]):
                x, y = xs[idx], ys[idx]
                cv2.circle(show, (int(round(x)), int(round(y))), 8, (0, 0, 255), 2)

            # Warp the whole frame (with drawn ball) into the rectified view
            dewarped = cv2.warpPerspective(show, H, (w, h))

            writer.write(dewarped)


        idx += 1
        # Allow ESC to abort early
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if writer is not None:
        writer.release()

    # Trim arrays to actual number of frames processed
    xs = xs[:idx]
    ys = ys[:idx]
    ts_video = ts_video[:idx]

    # -----------------------
    # Post-processing
    # -----------------------

    # Step 1: optional interpolation of short detection gaps
    if cfg.fill_gaps:
        xs = interpolate_gaps(xs, cfg.gap_max_frames)
        ys = interpolate_gaps(ys, cfg.gap_max_frames)

    # Step 2: robust outlier removal (Hampel) -> set outliers to NaN
    xs = hampel_filter_1d(xs, cfg.outlier_window, cfg.outlier_k)
    ys = hampel_filter_1d(ys, cfg.outlier_window, cfg.outlier_k)

    # Step 3: fill NaNs introduced by outlier removal (short gaps only)
    xs = interpolate_gaps(xs, cfg.gap_max_frames)
    ys = interpolate_gaps(ys, cfg.gap_max_frames)

    # Step 4: optional smoothing (rolling median)
    xs_s = rolling_median_1d(xs, cfg.smooth_window)
    ys_s = rolling_median_1d(ys, cfg.smooth_window)

    # If smoothing yielded NaNs (edges), backfill from unsmoothed
    xs = np.where(np.isnan(xs_s), xs, xs_s)
    ys = np.where(np.isnan(ys_s), ys, ys_s)
    # Convert to meters using either checkerboard homography

    # Homography: [X;Y;W] = H * [u;v;1]
    x_m = np.full_like(xs, np.nan, dtype=np.float64)
    y_m = np.full_like(ys, np.nan, dtype=np.float64)

    valid = ~np.isnan(xs) & ~np.isnan(ys)
    if valid.any():
        pts = np.vstack((xs[valid], ys[valid], np.ones(valid.sum())))
        world = H @ pts
        world /= world[2, :]  # normalize homogeneous
        x_m[valid] = world[0, :]
        y_m[valid] = world[1, :]

    # Final gap cleanup for velocity stability
    x_m = interpolate_gaps(x_m, cfg.gap_max_frames)
    y_m = interpolate_gaps(y_m, cfg.gap_max_frames)
    x_m = fill_ends(x_m)
    y_m = fill_ends(y_m)

    # Velocities via gradient (central differences)
    vx = np.gradient(x_m, ts_video)  # m/s
    vy = np.gradient(y_m, ts_video)  # m/s
    speed = np.sqrt(vx**2 + vy**2)

    # -----------------------
    # Save CSV
    # -----------------------
    df = pd.DataFrame({
        "time_s": ts_video,
        "x_px": xs,
        "y_px": ys,
        "x_m": x_m,
        "y_m": y_m,
        "vx_mps": vx,
        "vy_mps": vy,
        "speed_mps": speed,
    })
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # -----------------------
    # Plots
    # -----------------------
    plt.figure()
    plt.plot(ts_video, x_m, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("X position (m)")
    plt.title("Ball X vs Time")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_xt, dpi=160)
    print(f"Saved: {out_xt}")
    plt.close()

    plt.figure()
    plt.plot(ts_video, y_m, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Y position (m)  (image down is +)")
    plt.title("Ball Y vs Time")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_yt, dpi=160)
    print(f"Saved: {out_yt}")
    plt.close()

    plt.figure()
    plt.plot(x_m, y_m, linewidth=2)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)  (image down is +)")
    plt.title("2D Trajectory")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_traj, dpi=160)
    print(f"Saved: {out_traj}")
    plt.close()

    # Velocity plots
    plt.figure()
    plt.plot(ts_video, vx, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Vx (m/s)")
    plt.title("Horizontal Velocity Vx vs Time")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_vx, dpi=160)
    print(f"Saved: {out_vx}")
    plt.close()

    plt.figure()
    plt.plot(ts_video, vy, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Vy (m/s)")
    plt.title("Vertical Velocity Vy vs Time")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_vy, dpi=160)
    print(f"Saved: {out_vy}")
    plt.close()

    plt.figure()
    plt.plot(ts_video, speed, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed vs Time")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_speed, dpi=160)
    print(f"Saved: {out_speed}")
    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
