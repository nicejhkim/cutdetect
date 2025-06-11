
import os, sys

import torch
root_dir = os.getcwd()
sys.path.insert(0, str(root_dir))
from src.frame import frames, yolo

f = frames.extract_frame('samples/health.mp4', 300)
frames.save_frame(f, 'test/frame_health.png')

results = yolo.detect_labels_with_yolo(f)

labels = []
frame_area = f.shape[0] * f.shape[1]
for box, cls_id in zip(results.boxes.xywh, results.boxes.cls):
    label = yolo.yolo.model.names[int(cls_id)]
    labels.append(label)
    if label == "person":
        _, _, w, h = box
        # person_area = w * h
        print(f"person exists: prop: {h / f.shape[0]}")

f2 = yolo.draw_person_boxes(f, results)   
frames.save_frame(f2, 'test/frame_person.png')
print(labels)