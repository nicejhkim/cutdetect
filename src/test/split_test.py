import os, sys

root_dir = os.getcwd()
sys.path.insert(0, str(root_dir))

from src.cut.detect import split_cuts_ffmpeg, split_cuts_scenedetect
from src.common import utils

# scenes = utils.measure_time('split_scenes', split_cuts_ffmpeg, os.path.join(root_dir, "samples/san.mp4"))
scenes = utils.measure_time('split_scenes', split_cuts_scenedetect, os.path.join(root_dir, "samples/san.mp4"))