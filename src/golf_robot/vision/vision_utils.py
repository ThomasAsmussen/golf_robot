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
        self.init_N = 3

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


def rectify_with_chessboard(
    image,
    cb_cols=8,
    cb_rows=6,
    square_size_m=0.030,
    debug=False,
    win_size=(11, 11),
    refine_eps=0.001,
    refine_iters=30
):
    """
    Detect chessboard, canonicalize its orientation, and compute a metric homography.

    Returns:
        H_plane             : 3x3 homography mapping image pixels (u,v,1) -> board meters (X,Y,1)
        corners_px          : refined chessboard corners in image pixels (N,2), canonicalized
        meters_per_pixel_x  : diagnostic resolution along board X (m/px)
        meters_per_pixel_y  : diagnostic resolution along board Y (m/px)
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
    corners_px = corners.reshape(-1, 2)  # (N,2)

    # 3) Canonicalize grid orientation so:
    #    TL=g[0,0], TR=g[0,-1], BR=g[-1,-1], BL=g[-1,0]
    g = corners_px.reshape(cb_rows, cb_cols, 2)

    # Ensure +X goes right on the top row
    if g[0, -1, 0] < g[0, 0, 0]:
        g = g[:, ::-1, :]

    # Ensure +Y goes down on the left column
    if g[-1, 0, 1] < g[0, 0, 1]:
        g = g[::-1, :, :]

    corners_px = g.reshape(-1, 2)

    TL = g[0, 0]
    TR = g[0, -1]
    BR = g[-1, -1]
    BL = g[-1, 0]

    # Debug: visualize TL/TR/BR/BL
    if debug:
        dbg = image.copy()
        for p, c in zip([TL, TR, BR, BL], [(0,0,255), (0,255,0), (255,0,0), (0,255,255)]):
            cv2.circle(dbg, tuple(p.astype(int)), 10, c, -1)
        cv2.imshow("TL/TR/BR/BL used for H (canonical)", dbg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 4) Build metric correspondences for ALL corners (best scale stability)
    obj = np.zeros((cb_rows * cb_cols, 2), np.float32)
    obj[:, 0] = np.tile(np.arange(cb_cols), cb_rows) * square_size_m   # X meters
    obj[:, 1] = np.repeat(np.arange(cb_rows), cb_cols) * square_size_m # Y meters

    # 5) Homography: image(px) -> board(m)
    H_plane, _ = cv2.findHomography(corners_px.astype(np.float32), obj, method=0)

    # 6) Diagnostic resolution (m/px) from median adjacent spacing in image
    dx_px = np.linalg.norm(g[:, 1:, :] - g[:, :-1, :], axis=2).ravel()
    dy_px = np.linalg.norm(g[1:, :, :] - g[:-1, :, :], axis=2).ravel()

    meters_per_pixel_x = float(square_size_m / np.median(dx_px))
    meters_per_pixel_y = float(square_size_m / np.median(dy_px))

    return H_plane, corners_px, meters_per_pixel_x, meters_per_pixel_y




def pixel_to_plane(u, v, H_plane):
    # Convert to homogeneous
    p = np.array([u, v, 1.0], dtype=np.float32)
    
    # Transform
    P = H_plane @ p
    
    # Normalize
    X = P[0] / P[2]
    Y = P[1] / P[2]
    return X, Y

def plane_to_pixel(X, Y, H_plane):
    # INVERSE OF pixel_to_plane
    H_inv = np.linalg.inv(H_plane)
    pt_plane = np.array([X, Y, 1.0], dtype=np.float64)
    pt_pix = H_inv @ pt_plane
    pt_pix /= pt_pix[2]  # normalize
    return pt_pix[0], pt_pix[1]

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