import cv2
import os
import time

def record_from_camera(
    camera_index=0,
    output_folder=os.path.abspath("data"),
    filename_prefix="hole_test",
    fps=30,
    frame_width=1920,
    frame_height=1080,
    time_length=10.0,
    show_preview=False,
    convert_to_mp4=False,
):
    os.makedirs(output_folder, exist_ok=True)

    # No CAP_DSHOW on Linux
    cap = cv2.VideoCapture(camera_index)

    # Optionally: force V4L2 backend
    # cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"Could not open camera {camera_index}.")
        return

    # Try to set camera properties (some drivers may ignore these)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    print("Requested FPS:", fps)
    print("Camera reports FPS:", cap.get(cv2.CAP_PROP_FPS))
    # Often not needed, and can cause issues on some cams, so you can comment this out
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    avi_path = os.path.join(output_folder, f"{filename_prefix}_{timestamp}.avi")

    # On Ubuntu, MJPG sometimes fails. Try XVID or mp4v.
    # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # or "mp4v" with .mp4 extension

    writer = cv2.VideoWriter(avi_path, fourcc, fps, (frame_width, frame_height))

    if not writer.isOpened():
        print("Could not open VideoWriter. Check codec/fourcc and file path.")
        cap.release()
        return

    print(f"Recording raw AVI to {avi_path}")



    t0 = time.time()
    frame_count = 0

    while time.time() - t0 < time_length:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Make sure frame size matches what VideoWriter expects
        h, w = frame.shape[:2]
        if (w, h) != (frame_width, frame_height):
            # Resize if needed
            frame = cv2.resize(frame, (frame_width, frame_height))

        writer.write(frame)
        frame_count += 1

        if show_preview:
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Recorded {frame_count} raw frames.")
    print("Done.")
    
    actual_fps = frame_count / (time.time() - t0)
    print(f"Recorded {frame_count} frames in {time.time() - t0:.2f} s "
        f"(~{actual_fps:.2f} fps)")

if __name__ == "__main__":
    record_from_camera(camera_index=2)
