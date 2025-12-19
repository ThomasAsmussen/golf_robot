import cv2
import numpy as np
import sys
import os
import glob
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import csv



class KalmanFilterCV2D:
    """
    2D constant-velocity Kalman filter:
    state = [x, y, vx, vy]^T in meters and m/s

    Optional outlier rejection via Mahalanobis gating.
    """
    def __init__(self, dt,
                 q_pos=1e-4,
                 q_vel=1e-3,
                 meas_std=0.005,
                 gate_threshold=0.005):
        """
        dt            : time step [s] (e.g. 1 / src_fps)
        q_pos         : process noise on position
        q_vel         : process noise on velocity
        meas_std      : measurement std dev [m]
        gate_threshold: Mahalanobis distance^2 for gating (≈9 ≈ 3σ in 2D)
        """

        self.dt = dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float32)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)
        self.R = (meas_std**2) * np.eye(2, dtype=np.float32)

        self.x = None  # state (4x1)
        self.P = None  # covariance (4x4)
        self.gate_threshold = gate_threshold

    def step(self, z):
        """
        Perform one predict+update step.
        z: measurement (x_m, y_m) in meters, or None for predict-only.

        Returns:
            x, y, vx, vy, used_measurement (bool)
        """
        if self.x is None:
            if z is None:
                # No measurement, no state yet: nothing to do
                return None, None, None, None, False

            # Initialize at first measurement with zero velocity
            x_m, y_m = z
            self.x = np.array([[x_m],
                               [y_m],
                               [0.0],
                               [0.0]], dtype=np.float32)
            self.P = np.eye(4, dtype=np.float32) * 1e-3
            return float(x_m), float(y_m), 0.0, 0.0, True

        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        if z is None:
            # No measurement: just use prediction
            self.x = x_pred
            self.P = P_pred
            return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0]), False

        # Measurement update with gating
        z_vec = np.array([[z[0]], [z[1]]], dtype=np.float32)
        y_k = z_vec - (self.H @ x_pred)        # innovation
        S = self.H @ P_pred @ self.H.T + self.R

        try:
            S_inv = np.linalg.inv(S)
            mahalanobis2 = float((y_k.T @ S_inv @ y_k)[0, 0])
        except np.linalg.LinAlgError:
            mahalanobis2 = 0.0

        if mahalanobis2 < self.gate_threshold:
            # Accept measurement
            K_gain = P_pred @ self.H.T @ S_inv
            self.x = x_pred + K_gain @ y_k
            self.P = (np.eye(4, dtype=np.float32) - K_gain @ self.H) @ P_pred
            used = True
        else:
            # Reject measurement → keep prediction
            self.x = x_pred
            self.P = P_pred
            used = False

        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0]), used



def undistort_camera(image_glob: str = "data/calibration_images/*.jpg", 
                                 img_shape: tuple = (1920, 1080),
                                 cb_rows: int = 7, cb_cols: int = 9,
                                 square_size: float = 0.036):
    
    # Prepare object points
    objp = np.zeros((cb_rows*cb_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cb_cols, 0:cb_rows].T.reshape(-1, 2)
    objp *= square_size
    obj_points = []  # 3D points
    img_points = []  # 2D points
    images = glob.glob(image_glob)  # your images
    gray = None
    #print(len(images))
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        
        
        ret, corners = cv2.findChessboardCorners(gray, (cb_cols, cb_rows), None)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)

    # Calibrate
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )
    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, img_shape, 1, img_shape
    )
    
    return ret, mtx, dist, newcameramtx, roi

def load_camera_params(path):
    # Best calibration was done with MATLAB
    # Load the data saved from MATLAB
    location = os.path.join(path, 'camParamsForPython.mat')
    data = sio.loadmat(location)
    K = data['K']                 # 3x3 intrinsic matrix
    distCoeffs = data['distCoeffs'].astype(np.float64).ravel()  # 1D array
    
    imageSize = data['imageSize'].flatten().astype(int)  # (height, width)
    h = imageSize[0]
    w = imageSize[1]
    
    # (Optional) get optimal new camera matrix
    newK, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 1, (w, h))
    return K, distCoeffs, imageSize, newK, roi

def rectify_with_chessboard(image,
                            cb_cols = 8,
                            cb_rows = 6,
                            square_size_m=0.030,
                            debug=True,
                            win_size=(11,11),
                            refine_eps=0.001,
                            refine_iters=30
                            ):
    """
    Rectify an image using a detected chessboard pattern.

    Parameters:
        img          : input BGR image
        cb_cols      : number of internal corners horizontally
        cb_rows      : number of internal corners vertically
        debug        : whether to display debug images
        win_size     : window size for cornerSubPix
        refine_eps   : epsilon for corner refinement
        refine_iters : max iterations for corner refinement

    Returns:
        rectified    : rectified full image (complete warped canvas)
        H_full       : 3×3 homography mapping original -> rectified coords
        corners      : refined chessboard corners (N×2)
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern_size = (cb_cols, cb_rows)

    # 1) Detect chessboard
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        raise RuntimeError("Chessboard not found.")

    # 2) Refine corners
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        refine_iters,
        refine_eps
    )
    corners = cv2.cornerSubPix(gray, corners, win_size, (-1, -1), criteria)
    corners = corners.reshape(-1, 2)          # (N,2)

    # 3) Gridify corner array
    corners_grid = corners.reshape(cb_rows, cb_cols, 2)

    TL = corners_grid[0, 0]
    TR = corners_grid[0, -1]
    BR = corners_grid[-1, -1]
    BL = corners_grid[-1, 0]

    src_quad = np.float32([TL, TR, BR, BL])

    # Debug: visualize TL/TR/BR/BL
    if debug:
        dbg = image.copy()
        colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)]
        for p, c in zip([TL, TR, BR, BL], colors):
            cv2.circle(dbg, tuple(p.astype(int)), 10, c, -1)
            print("Corner:", p)
        cv2.imshow("Quad corners", cv2.resize(dbg, (600,400)))
        cv2.waitKey(0)

    # 4) Compute board size in the original image
    bw = np.linalg.norm(TR - TL)
    bh = np.linalg.norm(BL - TL)

    dst_quad = np.float32([
        [0,      0],
        [bw - 1, 0],
        [bw - 1, bh - 1],
        [0,      bh - 1]
    ])

    # 5) Homography for plane rectification
    H, _ = cv2.findHomography(src_quad, dst_quad)

    # 6) Expand canvas for full warp
    h, w = gray.shape[:2]
    image_corners = np.float32([
        [0, 0],
        [w-1, 0],
        [w-1, h-1],
        [0, h-1]
    ]).reshape(-1, 1, 2)

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
    # Convert to homogeneous
    p = np.array([u, v, 1.0], dtype=np.float32)
    
    # Transform
    P = H_plane @ p
    
    # Normalize
    X = P[0] / P[2]
    Y = P[1] / P[2]
    return X, Y

def compute_wb_gains_from_corners(
    image,
    corners,
    cb_cols = 8,
    cb_rows = 6,
    erode_ksize=5,
    target_gray=None
):
    """
    Given an image and known chessboard corners (already refined),
    build a chessboard mask and compute white-balance gains.

    Parameters:
        image        : BGR image
        corners      : refined corners, shape (N, 2)
        cb_rows      : number of internal corners vertically
        cb_cols      : number of internal corners horizontally
        erode_ksize  : mask erosion kernel size (to avoid edge bleeding)
        target_gray  : if None → gray-world; otherwise, enforce fixed brightness (e.g. 180)

    Returns:
        gains        : np.array([gainB, gainG, gainR])
        mask         : uint8 mask of chessboard region (255 = board)
    """

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Reshape corner list into grid shape
    corners_grid = corners.reshape(cb_rows, cb_cols, 2)

    # Extract outer quadrilateral of the chessboard
    TL = corners_grid[0, 0]
    TR = corners_grid[0, -1]
    BR = corners_grid[-1, -1]
    BL = corners_grid[-1, 0]

    board_poly = np.array([TL, TR, BR, BL], dtype=np.int32)

    # Fill chessboard polygon
    cv2.fillConvexPoly(mask, board_poly, 255)

    # Optional erosion to avoid mixed black/white edges
    if erode_ksize > 0:
        kernel = np.ones((erode_ksize, erode_ksize), np.uint8)
        mask = cv2.erode(mask, kernel)

    # Extract board pixels
    pixels = image[mask == 255].reshape(-1, 3).astype(np.float32)
    mean_bgr = pixels.mean(axis=0)

    # Determine the target gray level
    if target_gray is None:
        target = mean_bgr.mean()   # gray-world
    else:
        target = float(target_gray)

    gains = (target / mean_bgr).astype(np.float32)

    return gains, mask

def apply_white_balance(image, gains):
    """
    Apply computed WB gains to the whole image.
    """
    img_f = image.astype(np.float32)
    img_f *= gains  # broadcast over channel dimension
    img_f = np.clip(img_f, 0, 255)
    return img_f.astype(np.uint8)

def find_most_circular_blob(mask, min_area=50, circularity_thresh=0.45):
    """
    Finds the most circular blob in a binary mask using circularity = 4πA / P².

    Parameters:
        mask : uint8 binary image (0 or 255)
        min_area : minimum area for considering a blob
        circularity_thresh : minimum circularity required (0→1)

    Returns:
        center  : (x, y) of circle center, or None
        radius  : radius in pixels, or None
        contour : the contour of the best blob, or None
    """
    # Ensure mask is correct type
    mask_bin = (mask > 0).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_circularity = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        circularity = 4 * np.pi * area / (peri * peri)

        if circularity > circularity_thresh and circularity > best_circularity:
            best_circularity = circularity
            best_contour = cnt

    # No valid contour found
    if best_contour is None:
        return None, None, None

    # Compute enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(best_contour)

    center = (int(cx), int(cy))
    radius = int(radius)

    return center, radius, best_contour


def process_video(
    video_path,
    real_time_show=True,
):
    """
    Run ball detection + Kalman tracking on a single video.

    Returns:
        ts, xs, ys, vxs, vys, speed, csv_path
    """

    # ---- SETUP ----
    video_folder = os.path.abspath(os.path.join("data", "tuning_videos"))
    os.makedirs(video_folder, exist_ok=True)
    #video_path = os.path.join(video_folder, "test_20251204_111648.avi")

    K, distCoeffs, imageSize, newK, roi = load_camera_params(os.path.abspath("data"))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1e-3:
        print("Warning: FPS not found; defaulting to 30.")
        src_fps = 30.0

    # Read first frame 
    ret, first_frame_raw = cap.read()
    if not ret:
        print("Could not read first frame")
        sys.exit(1)

    h, w = first_frame_raw.shape[:2]

    # 1) Undistort the first frame
    first_undistorted = cv2.undistort(first_frame_raw, K, distCoeffs)
    # For faster computing in loop:
    map1, map2 = cv2.initUndistortRectifyMap(
        K, distCoeffs, None, K, (w, h), cv2.CV_16SC2
    )

    # 2) Rectify / find homography on the UNDISTORTED image
    rectified, H_plane, corners, (out_w, out_h) = rectify_with_chessboard(first_undistorted)

    # 3) Compute WB gains also on UNDISTORTED image
    gains, wb_mask = compute_wb_gains_from_corners(
        image=first_undistorted,
        corners=corners
    )

    print("WB gains:", gains)

    # 4) Metric scaling (as before, but now bw should be measured on the rectified or undistorted board)
    bw = 137.9586      # board width in *this* domain (update if needed)
    cb_cols = 8
    square_size = (26.9/9.0)/100.0  # meters per square

    pixels_per_square = bw / (cb_cols - 1)
    meters_per_pixel = square_size / pixels_per_square


    # ---- KALMAN FILTER INSTANCE ----
    dt = 1.0 / src_fps
    kf = KalmanFilterCV2D(
        dt=dt,
        q_pos=1e-3,      # tune these
        q_vel=1e-4,
        meas_std=0.005,  # meters (≈ 5 mm)
        gate_threshold=15.0
    )


    # ---- PROCESS VIDEO ----
    t_start = time.time()
    ball_pixels = []
    ball_world = []
    ball_times = []
    ball_vel = []
    frame_idx = 0

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            break

        frame_idx += 1  # count frames processed
        
        # A) Undistort current frame
        #frame_u = cv2.undistort(frame_raw, K, distCoeffs)
        frame_u = cv2.remap(frame_raw, map1, map2, interpolation=cv2.INTER_LINEAR)

        # B) Apply the same WB gains
        frame_wb = apply_white_balance(frame_u, gains)

        # C) Blur
        blurred = cv2.medianBlur(frame_wb, 5)
        blurred = cv2.GaussianBlur(blurred, (3, 3), 0)

        # D) HSV + masks
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        mask_sat = sat >= 150

        hue = hsv[:, :, 0]
        low_thres, up_thres = 1, 10 # 2, 10
        mask_hue = (hue >= low_thres) & (hue <= up_thres)

        blurred[~(mask_sat & mask_hue)] = (0, 0, 0) # Filtered

        # E) Morphology
        # Open to remove noise
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        
        # Close to fill holes
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(opened,cv2.MORPH_CLOSE, kernel)

        # F) Blob + circularity
        gray_closed = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(gray_closed, 1, 255, cv2.THRESH_BINARY)

        center, radius, contour = find_most_circular_blob(mask_bin)

        debug = frame_u.copy()  # draw on undistorted image (or frame_raw if you prefer)

        if center is not None:
            u, v = center

            # Plane coords in pixels
            X_plane, Y_plane = pixel_to_plane(u, v, H_plane)

            # Plane coords in meters (measurement)
            meas_x = X_plane * meters_per_pixel
            meas_y = Y_plane * meters_per_pixel

            # Time for this detection (based on video FPS, not processing speed)
            t_s = frame_idx / src_fps
            ball_times.append(t_s)

            # Kalman filter step with measurement
            filt_x, filt_y, filt_vx, filt_vy, used_meas = kf.step((meas_x, meas_y))

            # If for some reason KF isn't initialized yet, skip
            if filt_x is None:
                continue

            # Store filtered position and velocity
            ball_world.append((filt_x, filt_y))
            ball_vel.append((filt_vx, filt_vy))

            # Keep raw pixel center list for drawing path if you like
            ball_pixels.append(center)

            # Visualization (still drawn at detected center in pixels)
            cv2.circle(debug, center, radius, (0, 255, 0), 2)
            cv2.circle(debug, center, 3, (0, 0, 255), -1)

            for i in range(1, len(ball_pixels)):
                cv2.line(debug, ball_pixels[i-1], ball_pixels[i], (255, 0, 0), 2)
        else:
            # No detection → prediction only
            filt_x, filt_y, filt_vx, filt_vy, used_meas = kf.step(None)

            if filt_x is not None:
                # Still update trajectory with predicted state
                t_s = frame_idx / src_fps
                ball_times.append(t_s)
                ball_world.append((filt_x, filt_y))
                ball_vel.append((filt_vx, filt_vy))

        # FPS
        t = time.time()-t_start
        print("Processing FPS:", f"{frame_idx / t:.2f}", end='\r')
        
        if real_time_show:
            cv2.imshow("Ball tracking (undistorted)", cv2.resize(debug, (800,600)))
            # Wait according to video FPS (approx real-time playback)
            key = cv2.waitKey(int(1000 / src_fps)) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit early
                break
        
    cap.release()
    cv2.destroyAllWindows()


    # --- Plotting the results ---
    # --- Convert to arrays and compute velocities ---
    if ball_world:
        xs = np.array([p[0] for p in ball_world])
        ys = np.array([p[1] for p in ball_world])
        ts = np.array(ball_times)
        #vxs = np.array([v[0] for v in ball_vel])
        #vys = np.array([v[1] for v in ball_vel])
        # Process:
        xs -= xs[0] # Start at (0,0)
        ys -= ys[0]
        xs *= -1
        ys *= -1  # Flip to match real-world coords
        #vxs *= -1
        #vys *= -1
        
        # Scalar speed
        # finite-difference velocity from filtered positions
        vxs = np.gradient(xs, ts)
        vys = np.gradient(ys, ts)

        # if you want speed from these:
        speed = np.sqrt(vxs**2 + vys**2)

        # --- Save to CSV ---
        processed_folder = os.path.join(os.path.dirname(video_folder), "tuning_videos_processed")
        os.makedirs(processed_folder, exist_ok=True)

        video_base = os.path.splitext(os.path.basename(video_path))[0]
        csv_path = os.path.join(processed_folder, f"{video_base}_trajectory.csv")
   
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "x_m", "y_m", "vx_m_s", "vy_m_s", "speed_m_s"])
            for t_, x_, y_, vx_, vy_, s_ in zip(ts, xs, ys, vxs, vys, speed):
                writer.writerow([t_, x_, y_, vx_, vy_, s_])

        print(f"Saved CSV to: {csv_path}")

        # --- Optional: plotting ---
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.gca().set_aspect("equal", "box")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Ball trajectory on plane (Kalman filtered)")
        plt.show()
    
    return ts, xs, ys, vxs, vys, speed, csv_path
    
if __name__ == "__main__":
    video_path = "data/tuning_videos/test_1.0-5.avi"

    ts, xs, ys, vxs, vys, speed, csv_path = process_video(
        video_path,
        real_time_show=True,   # turn off GUI if running batch
    )

    print("Trajectory CSV:", csv_path)