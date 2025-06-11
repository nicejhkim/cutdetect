import os, sys

root_dir = os.getcwd()
sys.path.insert(0, str(root_dir))

from src.scene.split import split_scenes_ffmpeg, split_scenes_scenedetect
from src.common import utils

# scenes = utils.measure_time('split_scenes', split_scenes_ffmpeg, os.path.join(root_dir, "samples/san.mp4"))
scenes = utils.measure_time('split_scenes', split_scenes_scenedetect, os.path.join(root_dir, "samples/san.mp4"))