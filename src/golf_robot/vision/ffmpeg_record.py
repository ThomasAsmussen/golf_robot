import os
import subprocess
import json
from pathlib import Path

FFMPEG_EXE = str(Path(__file__).resolve().parent / "ffmpeg.exe")

def start_ffmpeg_record_windows(camera_alt, out_path, w=1920, h=1080, fps=30):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd = [
        FFMPEG_EXE,
        "-hide_banner", "-loglevel", "error",
        "-f", "dshow",
        "-rtbufsize", "1024M",
        "-framerate", str(fps),
        "-video_size", f"{w}x{h}",
        "-i", f"video={camera_alt}",   # SAFE here
        "-c:v", "copy",                # IMPORTANT: minimal CPU
        out_path
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )

    # Lower priority AFTER launch
    try:
        import psutil
        psutil.Process(proc.pid).nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception:
        print("Could not set ffmpeg process priority.")
        pass

    return proc

def stop_ffmpeg_record(proc, timeout=5):
    if proc is None:
        return
    try:
        proc.stdin.write(b"q\n")
        proc.stdin.flush()
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=timeout)


def get_video_frame_count(video_path):
    base = Path(__file__).resolve().parent.parent   # one level up
    video_location = base / video_path
    ffprobe_exe = str(base / "vision" / "ffprobe.exe")
    print(ffprobe_exe)
    def run(cmd):
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

    # A) Try counting frames
    cmdA = [
        ffprobe_exe, "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "json",
        video_path
    ]
    try:
        outA = run(cmdA)
        dataA = json.loads(outA)
        n = dataA["streams"][0].get("nb_read_frames")
        if n is not None:
            return int(n)
    except subprocess.CalledProcessError:
        pass  # fall through

    # B) Try stream nb_frames (metadata)
    cmdB = [
        ffprobe_exe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "json",
        video_path
    ]
    outB = run(cmdB)
    dataB = json.loads(outB)
    n2 = dataB["streams"][0].get("nb_frames")
    if n2 is not None:
        return int(n2)

    # C) Fallback: duration * avg_frame_rate
    cmdC = [
        ffprobe_exe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,avg_frame_rate",
        "-of", "json",
        video_path
    ]
    outC = run(cmdC)
    dataC = json.loads(outC)
    s = dataC["streams"][0]
    dur = float(s.get("duration") or 0.0)
    fr = s.get("avg_frame_rate") or "0/0"
    num, den = fr.split("/")
    fps = (float(num) / float(den)) if float(den) else 0.0
    if dur > 0 and fps > 0:
        return int(round(dur * fps))

    raise RuntimeError("Could not determine frame count (ffprobe gave no usable info).")

def trim_last_seconds_reencode(ffmpeg_exe, in_path, out_path, seconds=15):
    cmd = [
        ffmpeg_exe, "-hide_banner", "-loglevel", "error",
        "-sseof", f"-{seconds}",
        "-i", str(in_path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(out_path)
    ]
    subprocess.check_call(cmd)

if __name__ == "__main__":
    import glob
    data_dir = "data"
    prefix ="trajectory_recording"
    pattern = os.path.join(data_dir, f"{prefix}_*") #_last15s.avi")
    files = glob.glob(pattern)
    video_path = max(files, key=os.path.getmtime)
    last15_path = video_path.replace(".avi", "_last15.mp4")  # mp4 is nicer for tools
    trim_last_seconds_reencode(FFMPEG_EXE, video_path, last15_path, seconds=15)
    print("Saved last 15s:", last15_path)
    
    print("Recorded video path:", last15_path)
    frames_captured = get_video_frame_count(last15_path)
    expected_frames = int(15 * 30)
    print(frames_captured)

    