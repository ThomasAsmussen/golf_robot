import cv2
import numpy as np
import json
from datetime import datetime

def find_holes(
    image_bgr,
    pixels_per_cm=6.5,
    target_diam_cm=10.8,
    tol_cm=2.0,
    min_area=500,
    circularity_thresh=0.5,
    debug=False
):
    """
    Uses your EXACT hole detection procedure, but assumes `image_bgr` is already undistorted.

    Returns:
        detections: [((cx, cy), radius_px, diam_cm), ...]
        mask: uint8 mask image
        blur: debug image used in processing
    """
    min_diam_cm = target_diam_cm - tol_cm
    max_diam_cm = target_diam_cm + tol_cm

    # Smooth a bit
    blur = cv2.GaussianBlur(image_bgr, (15, 15), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    mask_sat = sat < 60  # low saturation
    blur[mask_sat]  = [255, 255, 255]  # make low-sat areas white
    blur[~mask_sat] = [0, 0, 0]        # make other areas black

    val = hsv[:, :, 2]
    mask_val = val < 40
    blur[mask_val]  = [255, 255, 255]  # make very dark areas white (as in your code)
    blur[~mask_val] = [0, 0, 0]        # make other areas black

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    mask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < circularity_thresh:
            continue

        (cx, cy), radius_px = cv2.minEnclosingCircle(cnt)
        diam_cm = (2 * radius_px) / float(pixels_per_cm)

        if min_diam_cm <= diam_cm <= max_diam_cm:
            detections.append(((int(cx), int(cy)), float(radius_px), float(diam_cm)))

    if debug:
        vis = image_bgr.copy()
        for (c, r_px, d_cm) in detections:
            cv2.circle(vis, c, int(r_px), (0, 255, 0), 2)
            cv2.circle(vis, c, 3, (0, 0, 255), -1)
            cv2.putText(vis, f"{d_cm:.1f} cm", (c[0] + 5, c[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("hole_blur", cv2.resize(blur, (960, 540)))
        cv2.imshow("hole_mask", cv2.resize(mask, (960, 540)))
        cv2.imshow("hole_detections", cv2.resize(vis, (960, 540)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detections


def save_holes_config(detections, frame_shape, path, pixels_per_cm=6.5):
    """
    Same JSON structure you used before. Expects detections from find_holes_undistorted().
    """
    H, W = frame_shape[:2]
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "image_size_px": {"width": int(W), "height": int(H)},
        "pixels_per_cm": float(pixels_per_cm),
        "holes": []
    }

    for i, (center, r_px, d_cm) in enumerate(detections):
        cx, cy = center
        data["holes"].append({
            "id": int(i),
            "center_px": [int(cx), int(cy)],
            "radius_px": float(r_px),
            "diameter_cm": float(d_cm)
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Example usage
import cv2
#from hole_detector import detect_hole_positions_pixels_undistorted
if __name__ == "__main__":
    #img = cv2.imread("data/hole_test_img.jpg")
    img = cv2.imread("data/hole_pic_09_02.png")
    hole_positions = find_holes(img, pixels_per_cm=6.5)
    print(hole_positions)  # [((cx, cy), radius_px, diam_cm), ...]

    # If you want to save a config JSON
    save_holes_config(
        detections=hole_positions,
        frame_shape=img.shape,
        path="data/holes_pixel_config.json",
        pixels_per_cm=6.5
    )
