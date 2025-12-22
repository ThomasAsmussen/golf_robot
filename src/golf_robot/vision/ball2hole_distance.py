import cv2
import os
import json
import sys
import numpy as np
from vision.hole_position_detection import find_holes, save_holes_config
from vision.ball_position_detection import detect_ball_position
from vision.vision_utils import apply_white_balance, compute_wb_gains_from_corners, rectify_with_chessboard, load_camera_params, pixel_to_plane, plane_to_pixel

def get_ball_final_position(camera_index=0, chosen_hole=None, use_cam=True, debug=True):
    # Read image
    frame_height = 1080
    frame_width = 1920
    if use_cam:
        # --- Capture one image from camera instead of reading from disk ---
        cap = cv2.VideoCapture(camera_index)  # try 0, or 1 if you have multiple cameras
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if not cap.isOpened():
            raise RuntimeError("Could not open camera")

        # Warm up a few frames (helps exposure/white balance settle)
        for _ in range(10):
            cap.read()

        ret, img = cap.read()
        cap.release()

        if not ret or img is None:
            raise RuntimeError("Failed to capture image from camera")
    else:
        img = cv2.imread("data/hole_test_50x30.jpg")
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
    #rectified, H_plane, corners, (out_w, out_h), meters_per_pixel = rectify_with_chessboard(undistorted)
    H_plane, corners, meters_per_pixel_x, meters_per_pixel_y = rectify_with_chessboard(undistorted, debug=False)

    # Correct the orientation of x and y
    S = np.array([[0, -1, 0],
                [-1, 0, 0],
                [0, 0, 1]], dtype=np.float64)

    H_plane_corrected = S @ H_plane

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



    # Get reference corner in plane coords (origo)
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

    u_tl, v_tl = corners2[idx_tl]
    print("TR corner pixel:", u_tl, v_tl)

    # Origo is the top-left corner in plane coords
    ref_x, ref_y = pixel_to_plane(u_tl, v_tl, H_plane_corrected)
    print("Ref plane before offset:", ref_x, ref_y)

    # From start camera origo (measured from camera origo to hole then hole to checkerboard tl corner):
    offset_x = 3.85 + 0.01566469927311509 # in meters
    offset_y = 0.0 - 0.13509415128381785 # in meters

    print("Ref plane:", ref_x, ref_y)

    # Find hole positions
    new_holes = True
    if new_holes:
        hole_positions = find_holes(img_wb)
        print(hole_positions)  # [(cx, cy), ...]

        # If you want to save a config JSON
        save_holes_config(
            detections=hole_positions,
            frame_shape=img.shape,
            path="data/holes_pixel_config.json",
            pixels_per_cm=6.5
        )
    else:
        with open("data/holes_pixel_config.json", "r") as f:
            cfg = json.load(f)
        hole_positions = [tuple(h["center_px"]) for h in cfg["holes"]]

    print(hole_positions)

    # Choose a hole:
    
    if chosen_hole is None:
        print("No hole chosen, aborting")
        sys.exit(0)
    else:
        hole_position = hole_positions[chosen_hole-1][0]
    print(hole_position)
    # Convert to real world
    hu, hv = hole_position
    hole_x, hole_y = pixel_to_plane(hu, hv, H_plane_corrected)

    # Find ball position
    ball_position = detect_ball_position(img_wb, debug=False)
    # Convert to real world
    bu, bv = ball_position
    ball_x, ball_y = pixel_to_plane(bu, bv, H_plane_corrected)

    # Now express hole and ball relative to that reference corner in plane coords
    print("corner tl:", ref_x, ref_y)
    print("hole pos:", hole_x, hole_y)
    print("ball pos:", ball_x, ball_y)
    hx, hy = hole_x - ref_x + offset_x, hole_y - ref_y + offset_y
    bx, by = ball_x - ref_x + offset_x, ball_y - ref_y + offset_y

    print("Hole pos relative to origo:", hx, hy)
    print("Ball pos relative to origo:", bx, by)

    # Distance:
    delta_x = float(np.abs(hx-bx))
    delta_y = float(np.abs(hy-by))

    magnitude = np.linalg.norm((delta_x, delta_y))

    print("Distance: ", magnitude)
    print("x and y contribution (x height, y width): ", (delta_x, delta_y))

    if debug:
        # Draw overlay
        vis = img_wb.copy()

        # Ball (green)
        cv2.circle(vis, (bu, bv), 12, (0, 255, 0), 2)
        cv2.circle(vis, (bu, bv), 3, (0, 255, 0), -1)
        cv2.putText(vis, "BALL", (bu + 10, bv - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Hole (red)
        cv2.circle(vis, (hu, hv), 12, (0, 0, 255), 2)
        cv2.circle(vis, (hu, hv), 3, (0, 0, 255), -1)
        cv2.putText(vis, "HOLE", (hu + 10, hv - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Line between them (yellow)
        cv2.line(vis, (hu, hv), (bu, bv), (0, 255, 255), 2)

        # Show delta text (top-left)
        cv2.putText(vis, f"dx={delta_x:.3f} m, dy={delta_y:.3f} m", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Show origo (blue):
        cv2.circle(vis, (int(u_tl), int(v_tl)), 12, (255, 0, 0), 2)
        cv2.circle(vis, (int(u_tl), int(v_tl)), 3, (255, 0, 0), -1)

        # Draw arrows
        a_l = 0.1 #in meters
        # Origin in metric space
        # +X direction (unit vector)
        u_x, v_x = plane_to_pixel(ref_x + a_l, ref_y, H_plane_corrected)
        # +Y direction (unit vector)
        u_y, v_y = plane_to_pixel(ref_x, ref_y + a_l, H_plane_corrected)
        # Convert to int
        p0 = (int(u_tl), int(v_tl))
        px = (int(u_x), int(v_x))
        py = (int(u_y), int(v_y))
        print(p0,px,py)
        cv2.arrowedLine(vis, p0, px, (0, 0, 255), 3, tipLength=0.15)  # +X (red)
        cv2.arrowedLine(vis, p0, py, (0, 255, 0), 3, tipLength=0.15)  # +Y (green)

        # Display
        resized = cv2.resize(vis, (960, 540))
        cv2.imshow("Ball + Hole overlay", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    ball_final_position = np.array([bx, by])
    return ball_final_position
 
if __name__ == "__main__":
    ball_final_position = get_ball_final_position(camera_index=0, chosen_hole=1, use_cam=False)
    print("Ball final position (bx, by) from origo:", ball_final_position)
