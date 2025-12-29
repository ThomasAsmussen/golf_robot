import cv2
import numpy as np
import sys
import os
import glob
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import csv
import json
from vision.vision_utils import (
    KalmanFilterCV2D,
    load_camera_params,
    rectify_with_chessboard,
    pixel_to_plane,
    compute_wb_gains_from_corners,
    apply_white_balance,
    plane_to_pixel
)


def get_ref_tl_corner_px(corners_px):
    """
    corners_px is (N,2) refined chessboard corners in image pixels.
    Return the top-left corner pixel (u_tl, v_tl).
    """
    u = corners_px[:, 0]
    v = corners_px[:, 1]
    idx_tl = np.argmin(u + v)
    return float(u[idx_tl]), float(v[idx_tl])

def load_hole_center_px(chosen_hole, path="data/holes_pixel_config.json"):
    with open(path, "r") as f:
        cfg = json.load(f)
    holes = [tuple(h["center_px"]) for h in cfg["holes"]]
    if chosen_hole is None or chosen_hole < 1 or chosen_hole > len(holes):
        raise ValueError(f"chosen_hole={chosen_hole} out of range. Found {len(holes)} holes in config.")
    return holes[chosen_hole - 1]

def to_origo_frame(X, Y, ref_x, ref_y, offset_x, offset_y):
    """
    Express plane (X,Y) in a shared "origo" frame:
    (X,Y) -> (X - ref_x + offset_x, Y - ref_y + offset_y)
    """
    return (float(X - ref_x + offset_x), float(Y - ref_y + offset_y))

def crossed_x_plane(x_prev, x_curr, x_plane):
    # True if segment crosses y=y_plane (either direction)
    return (x_prev - x_plane) * (x_curr - x_plane) <= 0 and (x_prev != x_curr)


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


def estimate_diameter_m(u, v, r_px, H_plane):
    # center in meters
    Xc, Yc = pixel_to_plane(u, v, H_plane)

    # measure scale locally by mapping a point r_px to the right
    Xe, Ye = pixel_to_plane(u + r_px, v, H_plane)

    r_m = float(np.hypot(Xe - Xc, Ye - Yc))
    return 2.0 * r_m

def process_video(
    video_path, 
    chosen_hole=None,
    real_time_show=True,
):
    """
    Run ball detection + Kalman tracking on a single video.

    Returns:
        ts, xs, ys, vxs, vys, speed, csv_path
    """

    # ---- SETUP ----
    video_folder = os.path.abspath(os.path.join("data", "path_tests"))
    os.makedirs(video_folder, exist_ok=True)
    #video_path = os.path.join(video_folder, "test_20251204_111648.avi")

    K, distCoeffs, imageSize, newK, roi = load_camera_params(os.path.abspath("data"))

    print("Processing video:", video_path)
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
    
    # Build a binary mask for the allowed "green perimeter" area (255 = allowed)
    green_perimeter_mask = np.ones((h, w), dtype=np.uint8) * 255
    green_perimeter_mask[0:120, :] = 0
    green_perimeter_mask[0:350, 0:800] = 0
    green_perimeter_mask[:, 1800:1920] = 0
    green_perimeter_mask[950:1080, :] = 0

    # 1) Undistort the first frame
    first_undistorted = cv2.undistort(first_frame_raw, K, distCoeffs)
    # For faster computing in loop:
    map1, map2 = cv2.initUndistortRectifyMap(
        K, distCoeffs, None, K, (w, h), cv2.CV_16SC2
    )

    # 2) Rectify / find homography on the UNDISTORTED image
    grey_temp = cv2.cvtColor(first_undistorted, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    bgr_temp = cv2.cvtColor(clahe.apply(grey_temp), cv2.COLOR_GRAY2BGR)
    try:
        H_plane, corners, mppx, mppy = rectify_with_chessboard(bgr_temp, 
                                                                            cb_cols=8,
                                                                            cb_rows=6,
                                                                            square_size_m=0.030,
                                                                            debug=False
                                                                            )
    except RuntimeError as e:
        print("Error in rectify_with_chessboard:", e)
        print("Skip this shot")
        return None, None

    # Orientation correction (same as ball2hole_distance.py)
    S = np.array([[0, -1, 0],
                [-1, 0, 0],
                [0,  0, 1]], dtype=np.float64)

    H_plane_corrected = S @ H_plane

    # Reference TL corner in pixel coords
    corners2 = corners.reshape(-1, 2)
    u_tl, v_tl = get_ref_tl_corner_px(corners2)

    # Reference TL corner in plane coords (meters)
    ref_x, ref_y = pixel_to_plane(u_tl, v_tl, H_plane_corrected)

    # Offsets (same values you used before)
    offset_x = 3.85 + 0.01566469927311509
    offset_y = 0.0 - 0.13509415128381785

    # Load hole center from saved config and convert to "origo" frame
    hu, hv = load_hole_center_px(chosen_hole, path="data/holes_pixel_config.json")
    hole_x, hole_y = pixel_to_plane(hu, hv, H_plane_corrected)
    hole_xo, hole_yo = to_origo_frame(hole_x, hole_y, ref_x, ref_y, offset_x, offset_y)

    print(f"Hole (origo frame): x={hole_xo:.4f} m, y={hole_yo:.4f} m")


    # 3) Compute WB gains also on UNDISTORTED image
    gains, wb_mask = compute_wb_gains_from_corners(
        image=first_undistorted,
        corners=corners,
        cb_cols=8,
        cb_rows=6,
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
    outside_count = 0
    out_cnt_to_reset = 5
    xs_hist, ys_hist, ts_hist = [], [], []
    prev_fx = None
    crossed_once = False
    ROI_R = 140  # pixels (tune)
    GOLF_BALL_DIAM_M = 0.05
    DIAM_TOL_M = 0.02
    INIT_CONSEC = 2

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            break
        
        frame_idx += 1  # count frames processed
        
        # A) Undistort current frame
        #frame_u = cv2.undistort(frame_raw, K, distCoeffs)
        frame_u = cv2.remap(frame_raw, map1, map2, interpolation=cv2.INTER_LINEAR)
        
        
        # B) ROI selection: process only a small window around the Kalman position
        # Default ROI = full frame (used when KF not initialized yet)
        roi_x0, roi_y0, roi_x1, roi_y1 = 0, 0, w, h

        if kf.x is not None:
            # Convert current KF state (meters in origo frame) -> pixel prediction,
            # so we can crop around where we expect the ball to be.
            Xp, Yp = float(kf.x[0,0] + ref_x - offset_x), float(kf.x[1,0] + ref_y - offset_y)
            u_hat, v_hat = plane_to_pixel(Xp, Yp, H_plane_corrected)
            ui, vi = int(round(u_hat)), int(round(v_hat))
            # Clamp ROI to image bounds
            roi_x0 = max(0, ui - ROI_R); roi_x1 = min(w, ui + ROI_R)
            roi_y0 = max(0, vi - ROI_R); roi_y1 = min(h, vi + ROI_R)

        # Crop both the frame and the green-perimeter mask to the ROI.
        # All detection work below should use these smaller arrays.
        frame_roi = frame_u[roi_y0:roi_y1, roi_x0:roi_x1]
        mask_roi  = green_perimeter_mask[roi_y0:roi_y1, roi_x0:roi_x1]
        
        # C) Apply WB ONLY on ROI (faster than full frame)
        frame_roi_wb = apply_white_balance(frame_roi, gains)  # ROI only
        # C) Blur ONLY on ROI
        blurred = cv2.medianBlur(frame_roi_wb, 5)
        blurred = cv2.GaussianBlur(blurred, (3, 3), 0)
        # D) Apply the green mask ONLY on ROI (mask_roi matches ROI shape)
        blurred = cv2.bitwise_and(blurred, blurred, mask=mask_roi)  # ROI mask
        #cv2.imshow("green", cv2.resize(blurred, (800,600)))
        #cv2.waitKey(0)
        
        # E) HSV + masks
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        #sat = hsv[:, :, 1]
        #mask_sat = sat >= 150

        #hue = hsv[:, :, 0]
        #low_thres, up_thres = 1, 10 # 2, 10
        #mask_hue = (hue >= low_thres) & (hue <= up_thres)
         
        #blurred[~(mask_sat & mask_hue)] = (0, 0, 0) # Filtered
        
        mask_hsv = cv2.inRange(hsv, (1, 150, 0), (10, 255, 255))
        #filtered = cv2.bitwise_and(blurred, blurred, mask=mask_hsv) # Dont do this, use binary mask directly for computation speed
        
        # F) Morphology
        # Open to remove noise
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)
        
        # Close to fill holes
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(opened,cv2.MORPH_CLOSE, kernel)

        # G) Blob + circularity
        #gray_closed = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(closed, 1, 255, cv2.THRESH_BINARY)

        center, radius, contour = find_most_circular_blob(mask_bin)
        
        if center is not None:
            center = (center[0] + roi_x0, center[1] + roi_y0)
        
        debug = frame_u.copy()  # draw on undistorted image (or frame_raw if you prefer)

        
        # Moved outside the loop
        #GOLF_BALL_DIAM_M = 0.05  # 42.67 mm + 7.33 mm margin for the outer circle
        #DIAM_TOL_M = 0.02     # 10mm tolerance (start here; widen to 0.015 if needed)
        #INIT_CONSEC = 2        # require 2 consecutive ball-sized detections to start

        if "init_streak" not in locals():
            init_streak = 0

        meas = None
        diam_m = None
        
        if center is not None and radius is not None and contour is not None:
            u, v = center

            # Estimate real diameter in meters using metric homography
            diam_m = estimate_diameter_m(u, v, radius, H_plane_corrected)  # or estimate_diameter_m

            # Optional: area sanity check so tiny specks don't start it
            area = cv2.contourArea(contour)
            area_ok = area >= 40  # tune

            ball_sized = (abs(diam_m - GOLF_BALL_DIAM_M) <= DIAM_TOL_M) and area_ok

            if ball_sized:
                # Convert center to meters (measurement position)
                X_m, Y_m = pixel_to_plane(u, v, H_plane_corrected)
                Xo, Yo = to_origo_frame(X_m, Y_m, ref_x, ref_y, offset_x, offset_y)
                meas = (Xo, Yo)

                # init logic
                if kf.x is None:
                    init_streak += 1
                    if init_streak >= INIT_CONSEC:
                        # start KF using measurement (your KF init happens inside step)
                        init_streak = 0
                        # (meas will be used below)
                    else:
                        # Not enough consecutive confirms yet; treat as no measurement for KF
                        meas = None
            else:
                init_streak = 0
        else:
            init_streak = 0
            
        # --- Kalman step ---
        filt_x, filt_y, filt_vx, filt_vy, used_meas = kf.step(meas)
        t_s = frame_idx / src_fps
        
        if filt_x is not None:
            # 1) store filtered positions
            xs_hist.append(float(filt_x))
            ys_hist.append(float(filt_y))
            ts_hist.append(float(t_s))

            # 2) crossing detection on filtered X
            if (not crossed_once) and (prev_fx is not None):
                if (prev_fx - hole_xo) * (float(filt_x) - hole_xo) <= 0 and float(filt_x) != prev_fx:

                    # Choose which sample to call "the crossing" (no interpolation):
                    # pick the closer of (prev sample) and (current sample) to the plane x=hole_xo
                    i_curr = len(xs_hist) - 1
                    i_prev = len(xs_hist) - 2

                    if abs(xs_hist[i_prev] - hole_xo) <= abs(xs_hist[i_curr] - hole_xo):
                        idx_cross = i_prev
                    else:
                        idx_cross = i_curr

                    # 3) velocity from gradient of FILTERED positions (same definition as your CSV)
                    # need at least 3 samples for a meaningful gradient
                    if len(xs_hist) >= 3:
                        # Convert to arrays
                        ts = np.array(ts_hist)
                        xs = np.array(xs_hist)
                        ys = np.array(ys_hist)

                        # Compute velocities (still needed to build the regression data)
                        vxs = np.gradient(xs, ts)
                        vys = np.gradient(ys, ts)

                        # --- Linear regression: v = a*t + b ---
                        # Fit vx(t)
                        ax, bx = np.polyfit(ts, vxs, 1)

                        # Fit vy(t)
                        ay, by = np.polyfit(ts, vys, 1)

                        # Time at crossing
                        t_cross = ts[idx_cross]

                        # Velocity at crossing (from regression)
                        vx = float(ax * t_cross + bx)
                        vy = float(ay * t_cross + by)

                        # Speed
                        spd = float(np.hypot(vx, vy))
                        bx = float(xs_hist[idx_cross])
                        by = float(ys_hist[idx_cross])
                        dist = float(np.hypot(bx - hole_xo, by - hole_yo))

                        print("\n=== Crossing detected (x crosses hole_x) ===")
                        print(f"idx={idx_cross}, t={ts_hist[idx_cross]:.3f}s")
                        print(f"Ball pos (filtered): x={bx:.4f} m, y={by:.4f} m")
                        print(f"Distance to hole: {dist:.4f} m")
                        print(f"Velocity (gradient of filtered pos): vx={vx:.3f} m/s, vy={vy:.3f} m/s, speed={spd:.3f} m/s\n")
                        
                        crossed_once = True
                        
                        if crossed_once:
                            #load_hole_center_px
                            fig = plt.figure()
                            plt.plot(xs, ys, marker="o")
                            plt.scatter(
                                hole_xo,
                                hole_yo,
                                s=150,
                                c="black",
                                marker="o",
                                edgecolors="white",
                                linewidths=2,
                                label="Holes",
                                zorder=5
                            )
                            plt.scatter(
                                bx,
                                by,
                                s=150,
                                c="orange",
                                marker="o",
                                edgecolors="white",
                                linewidths=1,
                                label="Holes",
                                zorder=5
                            )
                            plt.gca().set_aspect("equal", "box")
                            plt.xlabel("X [m]")
                            plt.ylabel("Y [m]")
                            plt.title("Ball trajectory on plane (Kalman filtered)")
                            plt.grid()
                            plt.draw()
                            plt.waitforbuttonpress(0)
                            plt.close(fig)
                            
                            return dist, spd
                        
                    else:
                        # not enough samples for gradient yet; you can delay printing until you have >=3
                        pass

            prev_fx = float(filt_x)

        # --- Check if ball left green perimeter ---
        if filt_x is not None:
            Xp, Yp = float(filt_x + ref_x - offset_x), float(filt_y + ref_y - offset_y)
            u_hat, v_hat = plane_to_pixel(Xp, Yp, H_plane_corrected)
            ui, vi = int(round(u_hat)), int(round(v_hat))

            inside_green = (0 <= ui < w and 0 <= vi < h and green_perimeter_mask[vi, ui] > 0)

            if not inside_green:
                outside_count += 1
            else:
                outside_count = 0

            if outside_count >= out_cnt_to_reset:
                print("Ball left green -> resetting Kalman filter")
                kf.x = None
                kf.P = None
                outside_count = 0

        # --- store only after KF has initialized ---
        if filt_x is not None:
            t_s = frame_idx / src_fps
            ball_times.append(t_s)
            ball_world.append((filt_x, filt_y))
            ball_vel.append((filt_vx, filt_vy))

        # --- drawing ---
        if center is not None and radius is not None:
            col = (0, 255, 0) if (diam_m is not None and abs(diam_m - GOLF_BALL_DIAM_M) <= DIAM_TOL_M) else (0, 0, 255)
            cv2.circle(debug, center, radius, col, 2)
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
    
    return None, None

    # # --- Plotting the results ---
    # # --- Convert to arrays and compute velocities ---
    # if ball_world:
    #     xs = np.array([p[0] for p in ball_world])
    #     ys = np.array([p[1] for p in ball_world])
    #     ts = np.array(ball_times)
    #     #vxs = np.array([v[0] for v in ball_vel])
    #     #vys = np.array([v[1] for v in ball_vel])
    #     # Process:
    #     #xs -= xs[0] # Start at (0,0)
    #     #ys -= ys[0]
    #     #vxs *= -1
    #     #vys *= -1
        
    #     # Scalar speed
    #     # finite-difference velocity from filtered positions
    #     vxs = np.gradient(xs, ts)
    #     vys = np.gradient(ys, ts)

    #     # if you want speed from these:
    #     speed = np.sqrt(vxs**2 + vys**2)

    #     # --- Save to CSV ---
    #     processed_folder = os.path.join(os.path.dirname(video_folder), "needs_deletion")
    #     os.makedirs(processed_folder, exist_ok=True)

    #     video_base = os.path.splitext(os.path.basename(video_path))[0]
    #     csv_path = os.path.join(processed_folder, f"{video_base}_trajectory.csv")
   
    #     with open(csv_path, mode="w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["time_s", "x_m", "y_m", "vx_m_s", "vy_m_s", "speed_m_s"])
    #         for t_, x_, y_, vx_, vy_, s_ in zip(ts, xs, ys, vxs, vys, speed):
    #             writer.writerow([t_, x_, y_, vx_, vy_, s_])

    #     print(f"Saved CSV to: {csv_path}")

    #     # --- Optional: plotting ---
    #     plt.figure()
    #     plt.plot(xs, ys, marker="o")
    #     plt.gca().set_aspect("equal", "box")
    #     plt.xlabel("X [m]")
    #     plt.ylabel("Y [m]")
    #     plt.title("Ball trajectory on plane (Kalman filtered)")
    #     plt.show()
    
    # #return ts, xs, ys, vxs, vys, speed, csv_path
    
if __name__ == "__main__":
    video_path = "data/trajectory_recording_20251228_181821_last10s.avi"

    dist, spd = process_video(
        video_path,
        chosen_hole=2,
        real_time_show=False,   # turn off GUI if running batch
    )

    #print("Trajectory CSV:", csv_path)