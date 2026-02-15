import cv2
import numpy as np
import sys
import os
import glob
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import csv

# --- Diameter check parameters ---
GOLF_BALL_DIAM_M = 0.07   # official golf ball diameter ≈ 42.7 mm
DIAM_TOL_M       = 0.06    # allow ±15 mm tolerance (tighten later if stable)
INIT_CONSEC      = 2        # require N consecutive "ball-sized" detections before using meas
STOP_X_M = 2.4


class KalmanFilterCV2D:
    """
    2D constant-velocity Kalman filter:
    state = [x, y, vx, vy]^T in meters and m/s

    Optional outlier rejection via Mahalanobis gating.
    """
    def __init__(self, dt,
                 sigma_a=1.0,      # [m/s^2] accel noise std
                 meas_std=0.005,   # [m]
                 gate_threshold=20):
        """
        dt            : nominal time step [s]
        sigma_a       : std dev of (white) acceleration noise [m/s^2]
        meas_std      : measurement std dev [m]
        gate_threshold: Mahalanobis distance^2 for gating (≈9 ≈ 3σ in 2D)
        """
        self.dt = float(dt)
        self.sigma_a = float(sigma_a)
        self.init_buffer = []
        self.init_N = 5

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.R = (meas_std**2) * np.eye(2, dtype=np.float32)

        self.x = None  # state (4x1)
        self.P = None  # covariance (4x4)
        self.gate_threshold = gate_threshold

        # Initialize F and Q from dt
        self._set_dt(self.dt)

    def _set_dt(self, dt):
        dt = float(dt)
        self.dt = dt

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float32)

        dt2 = dt * dt
        dt3 = dt2 * dt
        sa2 = self.sigma_a * self.sigma_a

        self.Q = sa2 * np.array([
            [1/3 * dt3, 0.0,         0.5 * dt2,  0.0],
            [0.0,        1/3 * dt3,  0.0,        0.5 * dt2],
            [0.5 * dt2,  0.0,         dt,        0.0],
            [0.0,        0.5 * dt2,   0.0,        dt],
        ], dtype=np.float32)

    def step(self, z, dt=None, t=None):
        """
        Perform one predict+update step.
        z : measurement (x_m, y_m) in meters, or None for predict-only.
        dt: optional timestep override [s] (useful if frame intervals vary)

        Returns:
            x, y, vx, vy, used_measurement (bool)
        """
        if dt is not None:
            self._set_dt(dt)
        if t is not None:
            self.t = t
        
        if self.x is None:

            if z is None:
                return None, None, None, None, False

            # Store measurement and time
            x_m, y_m = z
            self.init_buffer.append((x_m, y_m, self.t))

            if len(self.init_buffer) < self.init_N:
                return None, None, None, None, False

            # Estimate velocity from buffer
            x0, y0, t0 = self.init_buffer[0]
            xN, yN, tN = self.init_buffer[-1]

            vx0 = (xN - x0) / (tN - t0)
            vy0 = (yN - y0) / (tN - t0)

            # Initialize KF state
            self.x = np.array([
                [xN],
                [yN],
                [vx0],
                [vy0]
            ], dtype=np.float32)

            # High velocity uncertainty
            self.P = np.diag([0.005**2, 0.005**2, 1e-1**2, 1e-1**2])

            return float(xN), float(yN), float(vx0), float(vy0), True


        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        if z is None:
            self.x, self.P = x_pred, P_pred
            return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0]), False

        # Measurement update with gating
        z_vec = np.array([[z[0]], [z[1]]], dtype=np.float32)
        y_k = z_vec - (self.H @ x_pred)  # innovation
        S = self.H @ P_pred @ self.H.T + self.R

        try:
            S_inv = np.linalg.inv(S)
            mahalanobis2 = float((y_k.T @ S_inv @ y_k)[0, 0])
        except np.linalg.LinAlgError:
            # If S is singular, fall back to accepting measurement (or reject—your choice)
            mahalanobis2 = 0.0
            S_inv = None

        if mahalanobis2 < self.gate_threshold and S_inv is not None:
            K_gain = P_pred @ self.H.T @ S_inv
            self.x = x_pred + K_gain @ y_k
            self.P = (np.eye(4, dtype=np.float32) - K_gain @ self.H) @ P_pred
            used = True
        else:
            self.x, self.P = x_pred, P_pred
            used = False

        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0]), used


def load_camera_params(path):
    location = os.path.join(path, 'camParamsForPython.mat')
    data = sio.loadmat(location)
    K = data['K']  # 3x3
    distCoeffs = data['distCoeffs'].astype(np.float64).ravel()
    imageSize = data['imageSize'].flatten().astype(int)  # (height, width)
    h = imageSize[0]
    w = imageSize[1]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 1, (w, h))
    return K, distCoeffs, imageSize, newK, roi


def rectify_with_chessboard(image,
                            cb_cols=8,
                            cb_rows=6,
                            debug=False,
                            win_size=(11, 11),
                            refine_eps=0.001,
                            refine_iters=30):
    """
    Returns:
        rectified : warped image
        H_plane   : homography mapping original undistorted pixels -> rectified plane pixels
        corners   : refined chessboard corners (N,2) in original undistorted pixels
        out_wh    : (out_w, out_h)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern_size = (cb_cols, cb_rows)

    found, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        raise RuntimeError("Chessboard not found.")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, refine_iters, refine_eps)
    corners = cv2.cornerSubPix(gray, corners, win_size, (-1, -1), criteria).reshape(-1, 2)

    corners_grid = corners.reshape(cb_rows, cb_cols, 2)
    TL = corners_grid[0, 0]
    TR = corners_grid[0, -1]
    BR = corners_grid[-1, -1]
    BL = corners_grid[-1, 0]

    src_quad = np.float32([TL, TR, BR, BL])

    bw = np.linalg.norm(TR - TL)
    bh = np.linalg.norm(BL - TL)

    dst_quad = np.float32([
        [0,      0],
        [bw - 1, 0],
        [bw - 1, bh - 1],
        [0,      bh - 1]
    ])

    H, _ = cv2.findHomography(src_quad, dst_quad)

    h, w = gray.shape[:2]
    image_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(image_corners, H)

    xs = warped[:, 0, 0]
    ys = warped[:, 0, 1]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    Tx, Ty = -min_x, -min_y
    T = np.array([[1, 0, Tx],
                  [0, 1, Ty],
                  [0, 0, 1]], dtype=np.float32)

    H_plane = T @ H
    out_w = int(np.ceil(max_x - min_x))
    out_h = int(np.ceil(max_y - min_y))

    rectified = cv2.warpPerspective(image, H_plane, (out_w, out_h))
    return rectified, H_plane, corners, (out_w, out_h)


def pixel_to_plane(u, v, H_plane):
    p = np.array([u, v, 1.0], dtype=np.float32)
    P = H_plane @ p
    X = P[0] / P[2]
    Y = P[1] / P[2]
    return float(X), float(Y)


def compute_wb_gains_from_corners(image, corners, cb_cols=8, cb_rows=6, erode_ksize=5, target_gray=None):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    corners_grid = corners.reshape(cb_rows, cb_cols, 2)
    TL = corners_grid[0, 0]
    TR = corners_grid[0, -1]
    BR = corners_grid[-1, -1]
    BL = corners_grid[-1, 0]

    board_poly = np.array([TL, TR, BR, BL], dtype=np.int32)
    cv2.fillConvexPoly(mask, board_poly, 255)

    if erode_ksize > 0:
        kernel = np.ones((erode_ksize, erode_ksize), np.uint8)
        mask = cv2.erode(mask, kernel)

    pixels = image[mask == 255].reshape(-1, 3).astype(np.float32)
    mean_bgr = pixels.mean(axis=0)

    target = mean_bgr.mean() if target_gray is None else float(target_gray)
    gains = (target / mean_bgr).astype(np.float32)
    return gains, mask


def apply_white_balance(image, gains):
    img_f = image.astype(np.float32)
    img_f *= gains
    img_f = np.clip(img_f, 0, 255)
    return img_f.astype(np.uint8)


def find_most_circular_blob(mask, min_area=50, circularity_thresh=0.45):
    mask_bin = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_circularity = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 1e-9:
            continue

        circularity = 4 * np.pi * area / (peri * peri)
        if circularity > circularity_thresh and circularity > best_circularity:
            best_circularity = circularity
            best_contour = cnt

    if best_contour is None:
        return None, None, None

    (cx, cy), radius = cv2.minEnclosingCircle(best_contour)
    center = (int(cx), int(cy))
    radius = int(radius)
    return center, radius, best_contour


def estimate_diameter_m(u, v, r_px, H_plane, meters_per_pixel):
    """
    Estimate blob diameter in meters by:
      - mapping center and (center + r_px) into rectified plane pixels
      - converting that pixel radius to meters via meters_per_pixel
    """
    Xc, Yc = pixel_to_plane(u, v, H_plane)
    Xe, Ye = pixel_to_plane(u + r_px, v, H_plane)
    r_plane_px = float(np.hypot(Xe - Xc, Ye - Yc))
    r_m = r_plane_px * float(meters_per_pixel)
    return 2.0 * r_m


def process_video(video_path, real_time_show=True):
    prev_t_s = None
    video_folder = os.path.abspath(os.path.join("data", "tuning_videos"))
    os.makedirs(video_folder, exist_ok=True)

    K, distCoeffs, imageSize, newK, roi = load_camera_params(os.path.abspath("data"))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1e-3:
        print("Warning: FPS not found; defaulting to 30.")
        src_fps = 30.0

    ret, first_frame_raw = cap.read()
    if not ret:
        print("Could not read first frame")
        sys.exit(1)

    h, w = first_frame_raw.shape[:2]

    first_undistorted = cv2.undistort(first_frame_raw, K, distCoeffs)
    map1, map2 = cv2.initUndistortRectifyMap(K, distCoeffs, None, K, (w, h), cv2.CV_16SC2)

    rectified, H_plane_non_corrected, corners, (out_w, out_h) = rectify_with_chessboard(first_undistorted, debug=False)

    # Correct the orientation of x and y
    S = np.array([[-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]], dtype=np.float64)

    H_plane = S @ H_plane_non_corrected

    gains, wb_mask = compute_wb_gains_from_corners(image=first_undistorted, corners=corners)
    print("WB gains:", gains)

    # --- Metric scaling (keep your existing calibration) ---
    bw = 137.9586      # chessboard width in rectified-pixel units (your existing constant)
    cb_cols = 8
    square_size = (26.9 / 9.0) / 100.0  # meters per square (your existing)
    pixels_per_square = bw / (cb_cols - 1)
    meters_per_pixel = square_size / pixels_per_square

    dt = 1.0 / src_fps
    dt = 1.0 / src_fps
    kf = KalmanFilterCV2D(
        dt=dt,
        sigma_a=0.1,  # m/s²
        meas_std=0.005,  # meters (≈ 5 mm)
        gate_threshold=20
    )

    init_streak = 0

    t_start = time.time()
    ball_pixels = []
    ball_world = []
    ball_times = []
    ball_vel = []
    frame_idx = 0

    x0_raw = None        # reference for plotted x
    stop_triggered = False
    stop_speed = None
    stop_x_plot = None
    stop_time = None


    while True:
        ret, frame_raw = cap.read()
        if not ret:
            break

        frame_idx += 1
        green_perimeter_mask = np.ones((h, w), dtype=np.uint8) * 255
        # green_perimeter_mask[0:120, :] = 0
        green_perimeter_mask[0:370, :] = 0
        green_perimeter_mask[950:1080, :] = 0
        #cv2.imshow("Green perimeter mask", cv2.resize(green_perimeter_mask, (800, 600)))
        #cv2.waitKey(0)
        frame_raw = cv2.bitwise_and(frame_raw, frame_raw, mask=green_perimeter_mask)

        frame_u = cv2.remap(frame_raw, map1, map2, interpolation=cv2.INTER_LINEAR)
        frame_wb = apply_white_balance(frame_u, gains)

        blurred = cv2.medianBlur(frame_wb, 5)
        blurred = cv2.GaussianBlur(blurred, (3, 3), 0)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, (8, 150, 100), (18, 255, 255))

        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)

        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        center, radius, contour = find_most_circular_blob(closed, min_area=50, circularity_thresh=0.1)

        debug = frame_u.copy()

        # --- decide measurement ---
        meas = None
        diam_m = None
        ball_sized = False

        if center is not None and radius is not None and contour is not None:
            u, v = center

            diam_m = estimate_diameter_m(u, v, radius, H_plane, meters_per_pixel)
            ball_sized = abs(diam_m - GOLF_BALL_DIAM_M) <= DIAM_TOL_M

            if ball_sized:
                init_streak += 1
                if init_streak >= INIT_CONSEC:
                    X_plane, Y_plane = pixel_to_plane(u, v, H_plane)
                    meas = (X_plane * meters_per_pixel, Y_plane * meters_per_pixel)
            else:
                init_streak = 0
        else:
            init_streak = 0

        # --- Kalman step ---
        t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Compute actual dt
        if prev_t_s is None:
            dt_k = 1.0 / src_fps
        else:
            dt_k = t_s - prev_t_s
            # Clamp dt to avoid blow-ups on bad timestamps / seeks
            dt_k = float(np.clip(dt_k, 1e-3, 0.2))  # 1ms..200ms, tune if needed

        prev_t_s = t_s
        # --- one single KF call per frame ---
        filt_x, filt_y, filt_vx, filt_vy, used_meas = kf.step(meas, dt=dt_k, t=t_s)
        # KF state is in meters; convert -> plane-pixels -> image pixels
        # ---------------- STOP CONDITION (matches plotted x) ----------------
        if filt_x is not None:

            # Initialize reference for plotted coordinate
            if x0_raw is None:
                x0_raw = float(filt_x)

            # Same x used in plotting: xs -= xs[0]; xs *= -1
            x_plot = -(float(filt_x) - x0_raw)

            # Stop everything once ball reaches STOP_X_M
            if x_plot >= STOP_X_M:
                stop_speed = float(np.hypot(filt_vx, filt_vy))
                stop_x_plot = x_plot
                stop_time = t_s

                print("\n==============================")
                print("STOP CONDITION REACHED")
                print(f"x_plot = {x_plot:.3f} m   (>= {STOP_X_M})")
                print(f"speed  = {stop_speed:.3f} m/s")
                print("==============================\n")

                stop_triggered = True
                break   # ✅ stops ALL processing immediately
        # -------------------------------------------------------------------


        # time base (video time)
        t_s = frame_idx / src_fps

        # store only if KF has a state
        if filt_x is not None:
            ball_times.append(t_s)
            ball_world.append((filt_x, filt_y))
            ball_vel.append((filt_vx, filt_vy))

        # --- drawing ---
        if center is not None and radius is not None:
            col = (0, 255, 0) if ball_sized else (0, 0, 255)
            cv2.circle(debug, center, radius, col, 2)
            cv2.circle(debug, center, 3, (0, 0, 255), -1)
            if diam_m is not None:
                cv2.putText(
                    debug,
                    f"{diam_m*1000:.1f}mm",
                    (center[0] + 10, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    col,
                    1
                )

        # path in pixel space (for visualization)
        if center is not None and ball_sized and init_streak >= INIT_CONSEC:
            ball_pixels.append(center)
            for i in range(1, len(ball_pixels)):
                cv2.line(debug, ball_pixels[i - 1], ball_pixels[i], (255, 0, 0), 2)

        # FPS print
        t = time.time() - t_start
        print("Processing FPS:", f"{frame_idx / t:.2f}", end="\r")

        if real_time_show:
            cv2.imshow("Ball tracking (undistorted)", cv2.resize(debug, (800, 600)))
            key = cv2.waitKey(int(1000 / src_fps)) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # --- Plot & save ---
    if not ball_world:
        print("\nNo ball trajectory collected.")
        return None, None, None, None, None, None, None

    xs = np.array([p[0] for p in ball_world])
    ys = np.array([p[1] for p in ball_world])
    ts = np.array(ball_times)

    # normalize start + flip to match your previous convention
    xs -= xs[0]
    ys -= ys[0]
    xs *= -1
    ys *= -1

    vxs = np.gradient(xs, ts)
    vys = np.gradient(ys, ts)
    speed = np.sqrt(vxs**2 + vys**2)

    processed_folder = os.path.join(os.path.dirname(video_folder), "tuning_videos_processed")
    os.makedirs(processed_folder, exist_ok=True)
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(processed_folder, f"{video_base}_trajectory.csv")

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "x_m", "y_m", "vx_m_s", "vy_m_s", "speed_m_s"])
        for t_, x_, y_, vx_, vy_, s_ in zip(ts, xs, ys, vxs, vys, speed):
            writer.writerow([t_, x_, y_, vx_, vy_, s_])

    print(f"\nSaved CSV to: {csv_path}")

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Ball trajectory on plane (Kalman filtered)")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(ts, xs, marker="o", label="x")
    plt.plot(ts, ys, marker="o", label="y")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Ball position over time")
    plt.legend()
    plt.grid(True)
    plt.show()

    # plt.figure()
    # plt.plot(ts, vxs, marker="o", label="vx")
    # plt.plot(ts, vys, marker="o", label="vy")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Velocity [m/s]")
    # plt.title("Ball velocity over time")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    plt.figure()
    plt.plot(ts, speed, marker="o")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("Ball speed over time")
    plt.grid(True)
    plt.show()

    return ts, xs, ys, vxs, vys, speed, csv_path

def plane_to_pixel(X_plane, Y_plane, H_plane):
    """
    X_plane, Y_plane are in rectified-plane pixel units (same units as pixel_to_plane output).
    Returns (u,v) in the original undistorted image pixel coordinates.
    """
    H_inv = np.linalg.inv(H_plane)
    p = np.array([X_plane, Y_plane, 1.0], dtype=np.float32)
    U = H_inv @ p
    u = U[0] / U[2]
    v = U[1] / U[2]
    return float(u), float(v)


if __name__ == "__main__":
    video_path = "data/OBS_variability_quintic_imp_nocam_hole1/2026-02-05_15-27-54.mp4"
    ts, xs, ys, vxs, vys, speed, csv_path = process_video(video_path, real_time_show=True)
    print("Trajectory CSV:", csv_path)
