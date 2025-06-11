import cv2
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)

# CLIP 모델로 frame 판독
def classify_with_clip(frame, prompts: list) -> list:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).squeeze()
    return list(probs)

def get_best_from_probs(probs:list, prompts):
    best_label = prompts[torch.argmax(torch.tensor(probs))]
    best_prob = max(probs)
    return best_label, best_prob