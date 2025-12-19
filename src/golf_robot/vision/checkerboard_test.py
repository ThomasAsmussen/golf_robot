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

def rectify_with_chessboard(image,
                            cb_cols = 8,
                            cb_rows = 6,
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
        cv2.imshow("Quad corners", dbg)
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

#---- SETUP ----begin#
#ret, mtx, dist, newcameramtx, roi = undistort_camera()

path = os.path.abspath(os.path.join("data", "tuning_videos"))
os.makedirs(path, exist_ok=True)
video_path = os.path.join(path, "test1.mp4")
K, distCoeffs, imageSize, newK, roi = load_camera_params(os.path.abspath("data")) #Load lens calibration from matlab

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open video: {video_path}", file=sys.stderr)
    sys.exit(1)
    
src_fps = cap.get(cv2.CAP_PROP_FPS)
if not src_fps or src_fps <= 1e-3:
    print("Warning: FPS not found; defaulting to 30.")
    src_fps = 30.0
#---- SETUP ----end#

check, frame = cap.read()
if not check:
    print(f"Failed to read first frame of video: {video_path}", file=sys.stderr)
    sys.exit(1)

h, w = frame.shape[:2]


# --- Load picture ---
#frame1 = cv2.imread("data/ball_hue.jpg")
frame1 = cv2.imread("data/green_with_ball_example.jpg")

# --- Show distorted frame ---
scale=0.5
frame_resized = cv2.resize(frame1, (int(w*scale), int(h*scale)))
cv2.imshow("distorted", frame_resized)

# --- Undistort the frame ---
undistorted = cv2.undistort(frame1, K, distCoeffs)
scale=0.5
undistorted_resized = cv2.resize(undistorted, (int(w*scale), int(h*scale)))
cv2.imshow("undistorted", undistorted_resized)
#cv2.waitKey(0)

# --- Rectify (homography) with chessboard ---
rectified, H_plane, corners, (out_w, out_h) = rectify_with_chessboard(frame)
scale=0.5
rectified_resized = cv2.resize(rectified, (int(w*scale), int(h*scale)))
cv2.imshow("Rectified", rectified_resized)
#cv2.waitKey(0)

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
thres = 150 #120 sometimes      # <-- adjust this value (0–255 range)

mask_sat = sat >= thres   # boolean mask

# Create output image
output = blurred.copy()
output[~mask_sat] = (0,0,0)   # set low-saturation pixels to black

scale=0.5
output_resized = cv2.resize(output, (int(w*scale), int(h*scale)))
cv2.imshow("Masked", output_resized)
cv2.waitKey(0)

# Hue channel is hsv[:,:,0], values 0–179
hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
hue = hsv[:, :, 0]

# --- Compute histogram ---
hist = cv2.calcHist([hsv], [0], None, [180], [1, 180])

# --- Plot histogram ---
plt.plot(hist)
plt.title("Hue Histogram")
plt.xlabel("Hue Value (0–179)")
plt.ylabel("Pixel Count")
plt.show()

#12-35
# Threshold:
up_thres = 10
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
filtered[~(mask_sat & mask_hue)] = (0,0,0) 

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


debug = frame1.copy()
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