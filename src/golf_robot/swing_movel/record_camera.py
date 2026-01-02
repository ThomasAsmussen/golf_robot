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


import os
import time
import cv2
from collections import deque
import multiprocessing as mp


def _record_worker(cfg, stop_event, out_queue):
    camera_index = cfg["camera_index"]
    operating_system = cfg["operating_system"]
    fps = cfg["fps"]
    frame_width = cfg["frame_width"]
    frame_height = cfg["frame_height"]
    keep_last_seconds = cfg["keep_last_seconds"]
    output_folder = cfg["output_folder"]
    filename_prefix = cfg["filename_prefix"]

    os.makedirs(output_folder, exist_ok=True)

    # Backend
    if operating_system == "windows":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    elif operating_system == "linux":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    else:
        out_queue.put(("error", f"Unsupported OS: {operating_system}"))
        return

    # Try to reduce buffering / latency
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Request MJPG + mode
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        out_queue.put(("error", f"Could not open camera {camera_index}"))
        return

    # Warmup/flush (important for “first run good” behavior)
    t_end = time.perf_counter() + 1.0
    while time.perf_counter() < t_end:
        cap.read()

    buffer_size = max(1, int(round(fps * keep_last_seconds)))
    buf = deque(maxlen=buffer_size)

    # For diagnostics
    last_t = None
    worst_gap = 0.0

    while not stop_event.is_set():
        ret, frame = cap.read()
        t = time.perf_counter()
        if not ret:
            continue

        if last_t is not None:
            dt = t - last_t
            if dt > worst_gap:
                worst_gap = dt
        last_t = t

        # Avoid resizing if possible; if camera sometimes returns other sizes, resize here
        h, w = frame.shape[:2]
        if (w, h) != (frame_width, frame_height):
            frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

        buf.append(frame)

    cap.release()

    if not buf:
        out_queue.put(("error", "No frames captured"))
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        output_folder,
        f"{filename_prefix}_{timestamp}_last{int(keep_last_seconds)}s.avi"
    )

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))
    if not writer.isOpened():
        out_queue.put(("error", "Could not open VideoWriter"))
        return

    for f in buf:
        writer.write(f)
    writer.release()

    out_queue.put(("ok", out_path, len(buf), worst_gap))


class RollingVideoRecorder:
    """
    Start/stop a camera recording in a separate process.
    Keeps last N seconds in RAM; saves to disk when stopped.
    """
    def __init__(
        self,
        camera_index: int,
        operating_system: str,
        fps: int = 30,
        frame_width: int = 1920,
        frame_height: int = 1080,
        keep_last_seconds: float = 15.0,
        output_folder: str = "data",
        filename_prefix: str = "trajectory_recording",
    ):
        self.cfg = dict(
            camera_index=camera_index,
            operating_system=operating_system,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
            keep_last_seconds=keep_last_seconds,
            output_folder=output_folder,
            filename_prefix=filename_prefix,
        )
        self.stop_event = mp.Event()
        self.out_queue = mp.Queue(maxsize=1)
        self.proc = None

    def start(self):
        if self.proc is not None and self.proc.is_alive():
            return
        self.stop_event.clear()
        self.proc = mp.Process(target=_record_worker, args=(self.cfg, self.stop_event, self.out_queue), daemon=True)
        self.proc.start()

    def stop_and_save(self, timeout: float = 30.0):
        if self.proc is None:
            return None

        self.stop_event.set()
        self.proc.join(timeout=timeout)

        # If it didn't exit, terminate
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join()

        self.proc = None

        if self.out_queue.empty():
            return None

        status = self.out_queue.get()
        return status



if __name__ == "__main__":
    record_from_camera(camera_index=2)
