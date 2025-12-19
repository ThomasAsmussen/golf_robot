import cv2
import os
import json
import numpy as np
from hole_position_detection import find_holes, save_holes_config
from ball_position_detection import detect_ball_position
from vision_utils import apply_white_balance, compute_wb_gains_from_corners, rectify_with_chessboard, load_camera_params, pixel_to_plane


# Read image
img = cv2.imread("data/hole_test_38x32.jpg")
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
H_plane, corners, meters_per_pixel_x, meters_per_pixel_y = rectify_with_chessboard(undistorted, debug=True)


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

new_holes = True
if new_holes:
    hole_positions = find_holes(img_wb)
    print(hole_positions)  # [(cx, cy), ...]

    # If you want to save a config JSON
    save_holes_config(
        detections=hole_positions,
        frame_shape=img.shape,
        path="data/holes_config.json",
        pixels_per_cm=6.5
    )
else:
    with open("data/holes_config.json", "r") as f:
        cfg = json.load(f)
    hole_positions = [tuple(h["center_px"]) for h in cfg["holes"]]


print(hole_positions)

ball_position = detect_ball_position(img_wb, debug=True)

print(ball_position)


# Convert to real world
bu, bv = ball_position
ball_x, ball_y = pixel_to_plane(bu, bv, H_plane)

hole_position = hole_positions[0][0]
hu, hv = hole_position
hole_x, hole_y = pixel_to_plane(hu, hv, H_plane)


# Distance:
delta_x = float(np.abs(hole_x-ball_x))
delta_y = float(np.abs(hole_y-ball_y))

magnitude = np.linalg.norm((delta_x, delta_y))

print("Distance: ", magnitude)
print("x and y contribution (x height, y width): ", (delta_x, delta_y))

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
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Display
cv2.imshow("Ball + Hole overlay", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
# 38x32
