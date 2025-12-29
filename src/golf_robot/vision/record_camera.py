import cv2
import os
import time
from collections import deque

def record_from_camera(
    camera_index=0,
    output_folder=os.path.abspath("data"),
    filename_prefix="trajectory_recording",
    fps=30,
    frame_width=1920,
    frame_height=1080,
    keep_last_seconds=5.0,
    show_preview=False,
    stop_event=None,  # <-- pass this in from the training script
):
    os.makedirs(output_folder, exist_ok=True)


        # Choose backend
    if True:   #operating_system.lower() == "windows":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        # Explicit V4L2 backend on Linux
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    if True: #operating_system.lower() == "windows":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    buffer_size = max(1, int(round(fps * keep_last_seconds)))
    frame_buffer = deque(maxlen=buffer_size)

    print(f"[rec] running. Keeping last {keep_last_seconds:.1f}s (~{buffer_size} frames).")

    while True:
        if stop_event is not None and stop_event.is_set():
            break

        ret, frame = cap.read()
        if not ret:
            print("[rec] Failed to grab frame.")
            break

        h, w = frame.shape[:2]
        if (w, h) != (frame_width, frame_height):
            frame = cv2.resize(frame, (frame_width, frame_height))

        frame_buffer.append(frame)

        if show_preview:
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    if not frame_buffer:
        print("[rec] No frames buffered; nothing to save.")
        return None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        output_folder, f"{filename_prefix}_{timestamp}_last{int(keep_last_seconds)}s.avi"
    )
    print("path saved: ", out_path)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        print("[rec] Could not open VideoWriter.")
        return None

    for f in frame_buffer:
        writer.write(f)
    writer.release()

    print(f"[rec] Saved last ~{len(frame_buffer)/fps:.2f}s to {out_path}")
    return out_path


if __name__ == "__main__":
    # Example: run capture for 10s, but only save the last 5s
    record_from_camera(camera_index=2, time_length=10.0, keep_last_seconds=5.0, show_preview=False)
