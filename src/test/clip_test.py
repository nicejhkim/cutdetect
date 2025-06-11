import os, sys

import torch
root_dir = os.getcwd()
sys.path.insert(0, str(root_dir))
from src.frame import frames, clip


f = frames.extract_frame('samples/health.mp4', 1500)
PROMPTS = [
    "a traditional Korean building",
    "a traditional Asian building",
    "a western-style modern building",
    "a road",
    "a mountain landscape",
    "a beach or sea",
    "a river",
    "a field",
    "a person with traditional Korean costume",
    "a person with modern western costume"
]

frames.save_frame(f, 'frame_health.png')

probs = clip.classify_with_clip(f, PROMPTS)
probs_f = [t.tolist() for t in probs]
best_label = PROMPTS[torch.argmax(torch.tensor(probs_f))]
