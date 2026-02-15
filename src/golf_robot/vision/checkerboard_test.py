import cv2
import numpy as np
import sys
import os
import glob
import scipy.io as sio
import matplotlib.pyplot as plt


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

def rectify_with_chessboard_old(image,
                            cb_cols=6,
                            cb_rows=8,
                            debug=True,
                            win_size=(11, 11),
                            refine_eps=0.001,
                            refine_iters=30):
    """
    Rectify an image using a detected chessboard pattern.

    Parameters:
        image        : input BGR image
        cb_cols      : number of internal corners horizontally
        cb_rows      : number of internal corners vertically
        debug        : whether to display debug images
        win_size     : window size for cornerSubPix
        refine_eps   : epsilon for corner refinement
        refine_iters : max iterations for corner refinement

    Returns:
        rectified    : rectified full image (complete warped canvas)
        H_plane      : 3×3 homography mapping original -> rectified coords
        corners      : refined chessboard corners (N×2)
        (out_w,out_h): output image size
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
    corners = corners.reshape(-1, 2)  # (N, 2)

    # 3) Gridify corner array (for TL / TR / BR / BL)
    corners_grid = corners.reshape(cb_rows, cb_cols, 2)

    TL = corners_grid[0, 0]
    TR = corners_grid[0, -1]
    BR = corners_grid[-1, -1]
    BL = corners_grid[-1, 0]

    src_quad = np.float32([TL, TR, BR, BL])

    # Debug: visualize ALL corners + TL/TR/BR/BL
    if debug:
        dbg = image.copy()

        # draw all inner corners as small yellow points
        for p in corners:
            cv2.circle(dbg, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)

        # highlight the four outer corners with larger, distinct colors
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        for p, c in zip([TL, TR, BR, BL], colors):
            cv2.circle(dbg, (int(p[0]), int(p[1])), 8, c, 2)

        cv2.imshow("Chessboard corners (all + outer quad)", dbg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
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

def rectify_with_chessboard(image,
                            cb_cols=6,
                            cb_rows=8,
                            win_size=(11, 11),
                            refine_eps=0.001,
                            refine_iters=30,
                            square_size_m=0.030,   # checker square size [m]
                            draw_debug_rectified=True,
                            bar_length_m=1.0):
    """
    Rectify an image using a detected chessboard pattern and draw:
      - all inner corners
      - highlighted outer corners
      - 1 m scale bars in x and y on the rectified image

    Parameters:
        image         : input BGR image
        cb_cols       : number of internal corners horizontally
        cb_rows       : number of internal corners vertically
        win_size      : window size for cornerSubPix
        refine_eps    : epsilon for corner refinement
        refine_iters  : max iterations for corner refinement
        square_size_m : physical size of one checker square [m]
        draw_debug_rectified : if True, return rectified image with markers + scale
        bar_length_m  : physical length for scale bars (here we’ll use 1.0 m)

    Returns:
        rectified     : rectified image (with corners + scale bars if draw_debug_rectified=True)
        H_plane       : 3×3 homography mapping original -> rectified coords
        corners       : refined chessboard corners in original image (N×2)
        (out_w,out_h) : output image size in pixels
        mpx_x, mpx_y  : meters per pixel in x and y
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
    corners = corners.reshape(-1, 2)  # (N, 2)

    # 3) Gridify corner array (for TL / TR / BR / BL + scale)
    corners_grid = corners.reshape(cb_rows, cb_cols, 2)

    TL = corners_grid[0, 0]
    TR = corners_grid[0, -1]
    BR = corners_grid[-1, -1]
    BL = corners_grid[-1, 0]

    src_quad = np.float32([TL, TR, BR, BL])

    # --- Compute meters-per-pixel from checkerboard spacing ---
    dx_px = np.linalg.norm(corners_grid[:, 1:, :] - corners_grid[:, :-1, :], axis=2).ravel()
    dy_px = np.linalg.norm(corners_grid[1:, :, :] - corners_grid[:-1, :, :], axis=2).ravel()

    mean_dx_px = np.median(dx_px)
    mean_dy_px = np.median(dy_px)

    meters_per_pixel_x = square_size_m / mean_dx_px
    meters_per_pixel_y = square_size_m / mean_dy_px

    # 4) Compute board size in the original image (in pixels)
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

    # 6) Expand canvas for full warp (so we don’t crop the image)
    h, w = gray.shape[:2]
    image_corners = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
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

    # --- Draw corners + 1 m scale bars on the RECTIFIED image ---
    if draw_debug_rectified:
        # Transform all original corner points into rectified coordinates
        corners_h = corners.reshape(-1, 1, 2).astype(np.float32)
        corners_rect = cv2.perspectiveTransform(corners_h, H_plane)
        corners_rect = corners_rect.reshape(-1, 2)

        # Draw all corners as small yellow dots
        dbg = rectified.copy()
        for p in corners_rect:
            cv2.circle(dbg, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        # Highlight the four extreme corners with larger colored markers
        outer_original = np.float32([TL, TR, BR, BL]).reshape(-1, 1, 2)
        outer_rect = cv2.perspectiveTransform(outer_original, H_plane).reshape(-1, 2)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        for p, c in zip(outer_rect, colors):
            cv2.circle(dbg, (int(p[0]), int(p[1])), 8, c, 2)

        # Draw 1 m scale bars using the rectified metric scale
        dbg = draw_scale_bars_1m(
            dbg,
            m_per_px_x=meters_per_pixel_x,
            m_per_px_y=meters_per_pixel_y,
            margin_px=40,
            thickness=4
        )

        rectified = dbg

    return rectified, H_plane, corners, (out_w, out_h), meters_per_pixel_x, meters_per_pixel_y
import cv2
import numpy as np

def draw_scale_bars_1m(image,
                       m_per_px_x,
                       m_per_px_y,
                       margin_px=40,
                       thickness=4,
                       color=(255, 0, 0),
                       text_color=(255, 0, 0),
                       font_scale=0.7):
    """
    Draw 1 m horizontal and vertical scale bars on the image.

    Parameters:
        image       : BGR image
        m_per_px_x  : meters per pixel in x-direction
        m_per_px_y  : meters per pixel in y-direction
        margin_px   : margin from image border
        thickness   : line thickness
        color       : BGR color of the bars
        text_color  : BGR color of the text
        font_scale  : font scale of the labels

    Returns:
        image_with_bars : copy of image with 1 m scale bars
    """
    img = image.copy()
    h, w = img.shape[:2]

    # 1 meter in pixels for each direction
    bar_len_px_x = int(round(1.0 / m_per_px_x))  # width direction
    bar_len_px_y = int(round(1.0 / m_per_px_y))  # height direction

    # ---- Horizontal 1 m bar (bottom-left) ----
    x1 = margin_px
    y1 = h - margin_px
    x2 = x1 + bar_len_px_x
    y2 = y1

    cv2.line(img, (x1, y1-100), (x2, y2-100), color, thickness)
    cv2.putText(img, "1.00 m", (x1, y1 - 10-100),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)

    # ---- Vertical 1 m bar (bottom-right) ----
    x3 = w - margin_px
    y3 = h - margin_px
    x4 = x3
    y4 = y3 - bar_len_px_y

    cv2.line(img, (x3, y3), (x4, y4), color, thickness)
    cv2.putText(img, "1.00 m",
                (x3 - 110, y3 - bar_len_px_y // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)

    # Optional: annotate total image size in meters
    total_w_m = w * m_per_px_x
    total_h_m = h * m_per_px_y
    info = f"W = {total_w_m:.2f} m, H = {total_h_m:.2f} m"
    cv2.putText(img, info, (margin_px, margin_px + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

    return img


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

def find_most_circular_blob(mask, min_area=50, circularity_thresh=0.7):
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


def capture_single_frame(
    camera_index: int, 
    cap=None,
    operating_system: str = "windows",
    frame_width: int = 1920,
    frame_height: int = 1080,
    n_warmup: int = 15,
    n_attempts: int = 5,
):
    """
    Open a camera, warm it up, grab ONE clean frame, then close it again.
    Includes basic sanity checks on the returned frame size.
    """
    shutdown_cap = False
    if cap is None:
        shutdown_cap = True
        # Choose backend
        if operating_system.lower() == "windows":
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            # Explicit V4L2 backend on Linux
            cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    # Configure resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # MJPG is nice but can be flaky on some setups; keep it if it behaves,
    # otherwise comment this line out.
    if operating_system.lower() == "windows":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    try:
        # Warmup: let exposure / WB settle & flush old buffers
        for _ in range(n_warmup):
            cap.grab()

        # Try a few times to get a correctly sized frame
        for _ in range(n_attempts):
            ret, img = cap.read()
            if not ret or img is None:
                continue

            h, w = img.shape[:2]
            if h == frame_height and w == frame_width:
                return img

            # If size is wrong, flush a bit and retry
            for _ in range(5):
                cap.grab()

        raise RuntimeError(
            f"Failed to capture a valid {frame_width}x{frame_height} frame "
            f"from camera {camera_index}"
        )
    finally:
        ret, img = cap.read()
        print("img.shape =", img.shape)
        if shutdown_cap:
            cap.release()


#---- SETUP ----begin#
#ret, mtx, dist, newcameramtx, roi = undistort_camera()

# path = os.path.abspath(os.path.join("data", "tuning_videos"))
# os.makedirs(path, exist_ok=True)
# video_path = os.path.join(path, "test1.mp4")
K, distCoeffs, imageSize, newK, roi = load_camera_params(os.path.abspath("data")) #Load lens calibration from matlab

# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print(f"Failed to open video: {video_path}", file=sys.stderr)
#     sys.exit(1)
    
# src_fps = cap.get(cv2.CAP_PROP_FPS)
# if not src_fps or src_fps <= 1e-3:
#     print("Warning: FPS not found; defaulting to 30.")
#     src_fps = 30.0
#---- SETUP ----end#

# check, frame = cap.read()
# if not check:
#     print(f"Failed to read first frame of video: {video_path}", file=sys.stderr)
#     sys.exit(1)

# h, w = frame.shape[:2]
h, w = 1080, 1920


# --- Load picture ---
#frame1 = cv2.imread("data/ball_hue.jpg")
frame_height = 1080
frame_width = 1920
camera_index = 0
use_cam=True
if use_cam:
    frame = capture_single_frame(
        camera_index=camera_index,
        cap=None,
        operating_system="windows",
        frame_width=w,
        frame_height=h,
    )
else:
    frame = cv2.imread("data/green_with_ball_example.jpg")

# --- Show distorted frame ---
scale=0.5
frame_resized = cv2.resize(frame, (int(w*scale), int(h*scale)))
cv2.imshow("distorted", frame_resized)

# --- Undistort the frame ---
undistorted = cv2.undistort(frame, K, distCoeffs)
scale=0.5
undistorted_resized = cv2.resize(undistorted, (int(w*scale), int(h*scale)))
cv2.imshow("undistorted", undistorted_resized)
#cv2.waitKey(0)

# --- Rectify (homography) with chessboard ---
# --- Rectify (homography) with chessboard ---
rectified, H_plane, corners, (out_w, out_h), mpx, mpy = rectify_with_chessboard(frame)

cb_rows, cb_cols = 8, 6
corners_grid = corners.reshape(cb_rows, cb_cols, 2)

# Transform corners into rectified image coordinates
corners_h    = corners.reshape(-1, 1, 2).astype(np.float32)
corners_rect = cv2.perspectiveTransform(corners_h, H_plane).reshape(cb_rows, cb_cols, 2)

# Example: horizontal distance between TL and TR
TL = corners_rect[0, 0]
TR = corners_rect[0, -1]

dx_px = TR[0] - TL[0]      # pixel distance in x
dy_px = TR[1] - TL[1]      # (should be ≈ 0 after rectification)

# Convert to meters (mainly using x-scale here)
dist_m = np.sqrt((dx_px * mpx) ** 2 + (dy_px * mpy) ** 2)
print("Horizontal distance TL–TR [m]:", dist_m)

# Example: vertical distance between TL and BL
BL = corners_rect[-1, 0]
dx_px_v = BL[0] - TL[0]
dy_px_v = BL[1] - TL[1]
dist_m_v = np.sqrt((dx_px_v * mpx) ** 2 + (dy_px_v * mpy) ** 2)
print("Vertical distance TL–BL [m]:", dist_m_v)

print("Meters per pixel: ", mpx, mpy)
#cv2.imwrite("../../Mini-golf robot Master/Vision Report Material/rectified_end_output.jpg", rectified)
scale=0.5
rectified_resized = cv2.resize(rectified, (int(w*scale), int(h*scale)))
cv2.imshow("Rectified", rectified_resized)
#cv2.waitKey(0)
#cv2.imwrite("../../Mini-golf robot Master/Vision Report Material/undistorted.jpg", undistorted)

# --- White-balancing with chessboard corners ---
gains, wb_mask = compute_wb_gains_from_corners(
    image=frame,
    corners=corners)
print("WB gains:", gains)
whitebalanced = apply_white_balance(undistorted, gains)
scale=0.5
whitebalanced_resized = cv2.resize(whitebalanced, (int(w*scale), int(h*scale)))
cv2.imshow("Whitebalanced", whitebalanced_resized)
cv2.waitKey(0)
#cv2.imwrite("../../Mini-golf robot Master/Vision Report Material/whitebalanced.jpg", whitebalanced)



# Histogram
# --- Gaussian blur (tune kernel size if needed) ---
blurred = cv2.medianBlur(whitebalanced, 5) #remove salt-and-pepper noise
blurred = cv2.GaussianBlur(blurred, (3, 3), 0) # remove high-frequency noise
#blurred = cv2.medianBlur(blurred, 5)
#blurred = cv2.medianBlur(blurred, 5)


# --- Convert BGR → HSV ---
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

# Saturation channel
sat = hsv[:, :, 1]

# Threshold: keep only pixels with S >=  threshold
thres = 220 #120 sometimes      # <-- adjust this value (0–255 range)

mask_sat = sat >= thres   # boolean mask
mask_val = hsv[:, :, 2] >= 110  # boolean mask

# Create output image
output = blurred.copy()
output[~mask_sat] = (0,0,0)   # set low-saturation pixels to black
output[~mask_val] = (0,0,0)   # set low-value pixels to black

scale=0.5
output_resized = cv2.resize(output, (int(w*scale), int(h*scale)))
cv2.imshow("Masked Saturation and Value", output_resized)
cv2.imwrite("../../Mini-golf robot Master/Vision Report Material/masked_sat_val.jpg", output)
cv2.waitKey(0)


# Hue channel is hsv[:,:,0], values 0–179
hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
hue = hsv[:, :, 0]

# --- Compute histogram ---
hist = cv2.calcHist([hsv], [0], None, [30], [1, 30])

# --- Plot histogram ---
plt.plot(hist)
plt.title("Hue Histogram")
plt.xlabel("Hue Value (0–179)")
plt.ylabel("Pixel Count")
plt.show()

#12-35
# Threshold:
up_thres = 15
low_thres = 2 


mask_hue = (hue >= low_thres) & (hue <= up_thres)# boolean mask

# Create output image
output = blurred.copy()
output[~mask_hue] = (0,0,0)   # set low-saturation pixels to black

scale=0.5
output_resized = cv2.resize(output, (int(w*scale), int(h*scale)))
cv2.imshow("Masked Hue", output_resized)
cv2.waitKey(0)


# Apply filtering and processing
filtered = blurred.copy()
filtered[~(mask_sat & mask_hue & mask_val)] = (0,0,0) 

# Morphology closing
kernel = np.ones((5,5), np.uint8)
closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
scale=0.5
closed_resized = cv2.resize(closed, (int(w*scale), int(h*scale)))
cv2.imshow("Morphology Closed", closed_resized)
cv2.waitKey(0)

# Blob analysis with circularity
gray_closed = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY)
_, mask_bin = cv2.threshold(gray_closed, 1, 255, cv2.THRESH_BINARY) # Binary mask
center, radius, contour = find_most_circular_blob(mask_bin)
print("Center:", center, "Radius:", radius)


debug = frame.copy()
if contour is not None:
    cv2.drawContours(debug, [contour], -1, (0,0,255), 2)
    cv2.circle(debug, center, radius, (0,255,0), 2)
    cv2.circle(debug, center, 3, (255,0,0), -1)

cv2.imshow("Ball with Circularity Filter", debug)
cv2.waitKey(0)


#hsv[]




# Example corners:
#Corner: [1341.0642  708.0873]
#Corner: [1479.0186  707.0035]
#Corner: [1482.2751  805.8451]
#Corner: [1343.6626  806.8338]
bw = 137.9586
bh = 98.7807
cb_cols = 8
cb_rows = 6
square_size = (26.9/9.0)/100.0  # meters

X, Y = pixel_to_plane(u=1341.0642, v=708.0873, H_plane=H_plane)
print(H_plane)
print("World XY:", X, Y)
pixels_per_square = bw / (cb_cols - 1)
meters_per_pixel = square_size / pixels_per_square
X_meters = X * meters_per_pixel
Y_meters = Y * meters_per_pixel
print("World XY in meters:", X_meters, Y_meters)

X, Y = pixel_to_plane(u=1479.0186, v=707.0035, H_plane=H_plane)
print("World XY:", X, Y)
pixels_per_square = bw / (cb_cols - 1)
meters_per_pixel = square_size / pixels_per_square
X_meters = X * meters_per_pixel
Y_meters = Y * meters_per_pixel
print("World XY in meters:", X_meters, Y_meters)