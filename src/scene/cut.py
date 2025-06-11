import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# ffmpeg을 이용해서 영상의 주어진 구간을 저장
import subprocess


def cut_video_ffmpeg(video_path: str, start_sec: float, end_sec: float, output_path: str):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "copy", output_path ]
    
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)


def cut_video_ffmpeg_parallel(tasks: list, max_workers=None):
    if not max_workers:
        max_workers = os.cpu_count()//2

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(cut_video_ffmpeg, *args) for args in tasks]
        for future in as_completed(futures):
            future.result()

def cut_video_ffmpeg_seqential(tasks: list):
    for task in tasks:
        cut_video_ffmpeg(task[0], task[1], task[2], task[3])
