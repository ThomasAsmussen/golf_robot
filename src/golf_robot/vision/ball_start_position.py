import cv2
import os
import numpy as np
try:
    from vision.ball_position_detection import detect_ball_position
    from vision.vision_utils import apply_white_balance, compute_wb_gains_from_corners, rectify_with_chessboard, load_camera_params, pixel_to_plane, plane_to_pixel
except ImportError:
    from ball_position_detection import detect_ball_position
    from vision_utils import apply_white_balance, compute_wb_gains_from_corners, rectify_with_chessboard, load_camera_params, pixel_to_plane, plane_to_pixel
# from ball_position_detection import detect_ball_position
# from vision_utils import apply_white_balance, compute_wb_gains_from_corners, rectify_with_chessboard, load_camera_params, pixel_to_plane, plane_to_pixel

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
        # print("img.shape =", img.shape)
        if shutdown_cap:
            cap.release()

def get_ball_start_position(debug=True, return_debug_image=False, debug_raw=False, use_cam=True, camera_index = 1, cap=None, operating_system="windows"):
    # Read image
    frame_height = 1080
    frame_width = 1920
    if use_cam:
        img = capture_single_frame(
            camera_index=camera_index,
            cap=cap,
            operating_system=operating_system,
            frame_width=frame_width,
            frame_height=frame_height,
        )
    else:
        img = cv2.imread("data/ball_start_test2.jpg")

    if debug_raw:

        cv2.imshow("debug pic", img)
        cv2.waitKey(0)

    h, w = img.shape[:2]
    # ------ DO ONCE (START) ------- #
    # 0) Camera setup
    K, distCoeffs, imageSize, newK, roi = load_camera_params(os.path.abspath("data"))

    # 1) Undistort init:
    map1, map2 = cv2.initUndistortRectifyMap(
        K, distCoeffs, None, K, (w, h), cv2.CV_16SC2
    )
    # Undistort
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    # 2) Rectify / find homography on the UNDISTORTED image
    H_plane, corners, meters_per_pixel_x, meters_per_pixel_y = rectify_with_chessboard(undistorted, debug=False)

    # print("Resolution meters_per_pixel (x,y):", meters_per_pixel_x, meters_per_pixel_y)


    # 3) Compute WB gains also on UNDISTORTED image
    gains, wb_mask = compute_wb_gains_from_corners(
        image=undistorted,
        corners=corners
    )
    # ------ DO ONCE (END) ------- #


    # ------ DO IN LOOP (START) ----#
    # A) Undistort current frame
    img_ud = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    # B) Apply the same WB gains
    img_wb = apply_white_balance(img_ud, gains)

    # Correct the orientation of x and y
    S = np.array([[0, -1, 0],
                [-1, 0, 0],
                [0, 0, 1]], dtype=np.float64)

    H_plane_corrected = S @ H_plane


    # Find ball position
    ball_position = detect_ball_position(img_wb, debug=False,hue_high=20, sat_low=200, val_low=130) # low_hue=1,high_hue=10,sat_thresh=150,min_area=50, circularity_thresh=0.45,

    # print(ball_position)
    if ball_position is None:
        cv2.imshow("Error image", cv2.resize(img_wb, (640, 480)))
        cv2.waitKey(0)
        raise RuntimeError("Could not detect ball position.")

    # Convert to real world
    bu, bv = ball_position
    ball_x, ball_y = pixel_to_plane(bu, bv, H_plane_corrected)

    # Find checkerboard postion
    corners2 = corners.reshape(-1, 2)
    u = corners2[:, 0]
    v = corners2[:, 1]

    # Top-left (in image pixel sense)
    idx_tl = np.argmin(u + v)

    # Top-right
    idx_tr = np.argmax(u - v)

    # Bottom-left
    idx_bl = np.argmin(u - v)

    # Bottom-right
    idx_br = np.argmax(u + v)

    u_tr, v_tr = corners2[idx_tr]
    # print("TR corner pixel:", u_tr, v_tr)

    ref_x, ref_y = pixel_to_plane(u_tr, v_tr, H_plane_corrected)


    # print("Ref plane:", ref_x, ref_y)

    # Distance to origo from plane:
    x_tr_to_0 = 0.548
    y_tr_to_0 = -0.088
    x0 = ref_x + x_tr_to_0
    y0 = ref_y + y_tr_to_0
    # For visual
    u0, v0 = plane_to_pixel(x0, y0, H_plane_corrected)


    # Now you can express ball relative to that reference corner in plane coords
    # print("origo:", x0, y0)
    # print("ball pos:", ball_x, ball_y)
    dx = ball_x - x0
    dy = ball_y - y0

    # print("Difference from origo:", dx, dy)

    if debug:
        # Debug visual image
        dbg = img_wb.copy()
        # Draw ball
        cv2.circle(dbg, (int(bu), int(bv)), 6, (0, 255, 0), -1)
        # Draw reference corner (TR)
        cv2.circle(dbg, (int(u_tr), int(v_tr)), 6, (0, 0, 255), -1)
        # Draw origin
        cv2.circle(dbg, (int(u0), int(v0)), 6, (255, 0, 0), -1)
        # Draw arrows
        a_l = 0.1 #in meters
        # Origin in metric space
        # +X direction (unit vector)
        u_x, v_x = plane_to_pixel(x0 + a_l, y0, H_plane_corrected)
        # +Y direction (unit vector)
        u_y, v_y = plane_to_pixel(x0, y0 + a_l, H_plane_corrected)
        # Convert to int
        p0 = (int(u0), int(v0))
        px = (int(u_x), int(v_x))
        py = (int(u_y), int(v_y))
        # print(p0,px,py)
        cv2.arrowedLine(dbg, p0, px, (0, 0, 255), 3, tipLength=0.15)  # +X (red)
        cv2.arrowedLine(dbg, p0, py, (0, 255, 0), 3, tipLength=0.15)  # +Y (green)
        # Labels
        cv2.putText(dbg, "+X", (px[0] + 5, px[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(dbg, "+Y", (py[0] + 5, py[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        if return_debug_image:
            return (dx, dy, dbg)
        else:   
            cv2.imshow("ball + ref + coordinate frame", dbg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    return (dx, dy)

if __name__ == "__main__":
    dx, dy = get_ball_start_position(debug = True, use_cam=True, debug_raw=True, camera_index=4, operating_system="linux")
    
    print("Ball start position (dx, dy) from origo:", dx, dy)
