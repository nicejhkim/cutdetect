import subprocess
import re

from scenedetect import detect, ContentDetector

# scenedetect(https://www.scenedetect.com/docs/)를 이용해서 원본 영상을 scene 단위로 분리
def split_scenes_scenedetect(video_path: str, threshold: float = 30.0) -> list:
    scenes = detect(video_path, ContentDetector(threshold=threshold), show_progress=True)
    return scenes

# ffmpeg을 이용해서 원본 영상을 scene 단위로 분리
def split_scenes_ffmpeg(video_path: str) -> list:
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", "select='gt(scene,0.4)',metadata=print",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    log = result.stderr

    pattern = r"frame:(\d+)\s+pts:\d+\s+pts_time:(\d+\.\d+)"
    matches = re.findall(pattern, log)

    scene_changes = []
    for frame, pts_time in matches:
        scene_changes.append({
            "frame": int(frame),
            "time": float(pts_time)
        })

    return scene_changes
