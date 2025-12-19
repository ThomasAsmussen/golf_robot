import cv2
import os
import time
import subprocess

def record_from_camera(
    camera_index=0,
    output_folder=os.path.abspath("data"),
    #output_folder=os.path.abspath(os.path.join("data", "tuning_videos")),
    filename_prefix="hole_test",
    fps=30,
    frame_width=1920,
    frame_height=1080,
    time_length=10.0,
    show_preview=False,
    convert_to_mp4=True
):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        print("Could not open camera.")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    avi_path = os.path.join(output_folder, f"{filename_prefix}_{timestamp}.avi")
    mp4_path = avi_path.replace(".avi", ".mp4")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # very light encoder
    writer = cv2.VideoWriter(avi_path, fourcc, fps, (frame_width, frame_height))

    print(f"Recording raw AVI to {avi_path}")

    t0 = time.time()
    frame_count = 0
    
    while time.time() - t0 < time_length:
        ret, frame = cap.read()
        if not ret:
            break

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

    # Convert to MP4 afterwards
    if convert_to_mp4:
        pass #No need
        #print("Converting to MP4…")
        #ffmpeg_cmd = [
        #    "ffmpeg", "-y",
        #    "-i", avi_path,
        #    "-vcodec", "libx264",
        #    "-preset", "slow",
        #    "-crf", "18",
        #    mp4_path
        #]
        #subprocess.run(ffmpeg_cmd)
        #print(f"MP4 saved: {mp4_path}")

    print("Done.")


if __name__ == "__main__":
    record_from_camera(camera_index=1)