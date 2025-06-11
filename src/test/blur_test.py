
import os, sys

from pathlib import Path

root_dir = os.getcwd()
sys.path.insert(0, str(root_dir))
from src.common import utils
from src.frame import frames, yolo

blur_src_path = "test"

for file in Path(blur_src_path).glob("*.mp4"):
    blured_file = utils.add_postfix_to_filename(file, "blured")
    yolo.blur_yolo(file, blured_file)