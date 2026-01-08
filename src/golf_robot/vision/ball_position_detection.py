import cv2
import numpy as np

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
        if peri == 0:
            continue

        circularity = 4 * np.pi * area / (peri * peri)
        if circularity > circularity_thresh and circularity > best_circularity:
            best_circularity = circularity
            best_contour = cnt

    if best_contour is None:
        return None, None, None

    (cx, cy), radius = cv2.minEnclosingCircle(best_contour)
    return (int(cx), int(cy)), int(radius), best_contour

def detect_ball_position(
    image_bgr,
    hue_low=1,
    hue_high=15,
    val_low=150,
    val_high=255,
    sat_low=150,
    sat_high=255,
    min_area=50,
    circularity_thresh=0.45,
    debug=False
):
    """
    Uses the SAME ball procedure from your video loop, but assumes `image_bgr`
    is already undistorted.

    Returns:
        center_px: (u, v) or None
    """
    # C) Blur
    blurred = cv2.medianBlur(image_bgr, 5)
    blurred = cv2.GaussianBlur(blurred, (3, 3), 0)

    # D) HSV + masks
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, (hue_low, sat_low, val_low), (hue_high, sat_high, val_high))
    

    # E) Morphology
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((11, 11), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # F) Binary + circular blob
    _, mask_bin = cv2.threshold(closed, 1, 255, cv2.THRESH_BINARY)

    center, radius, _ = find_most_circular_blob(
        mask_bin, min_area=min_area, circularity_thresh=circularity_thresh
    )

    if debug:
        dbg = image_bgr.copy()
        if center is not None:
            cv2.circle(dbg, center, radius, (0, 255, 0), 2)
            cv2.circle(dbg, center, 3, (0, 0, 255), -1)

        cv2.imshow("filtered", cv2.resize(mask_hsv, (800, 600)))
        cv2.imshow("mask_bin", cv2.resize(mask_bin, (800, 600)))
        cv2.imshow("debug", cv2.resize(dbg, (800, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return center  # (u, v) or None


# ---------- example usage ----------
#import os
#from ball_position_detection import detect_ball_position   

#folder = os.path.abspath("data")
#os.makedirs(folder, exist_ok=True)

#path = os.path.join(folder, "hole_test_img.jpg")
#frame = cv2.imread(path)
#center = detect_ball_position(frame, debug=True)

#print(center)
