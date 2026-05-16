"""
cnn_inversion_check.py
======================
Run on ONE known-fake and ONE known-real video to confirm whether
the CNN models need 1.0-sigmoid or plain sigmoid.

Paste output here before updating the pipeline.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from timm import create_model
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── UPDATE THESE ─────────────────────────────────────────────────────
FAKE_VIDEO = r"D:\Projects\Major Project\Deepfake Detection\datasets\dfd\manipulated\01_02__exit_phone_room__YVGY8LOK.mp4"
REAL_VIDEO = r"D:\Projects\Major Project\Deepfake Detection\datasets\dfd\original\01__exit_phone_room.mp4"

MODEL_CONFIGS = {
    "efficientnet_b3": {
        "path":       r"D:\Projects\Major Project\Deepfake Detection\models\new\new_effb3_finetuned.pth",
        "input_size": (300, 300),
        "model_name": "efficientnet_b3",
        "norm_mean":  [0.485, 0.456, 0.406],
        "norm_std":   [0.229, 0.224, 0.225],
    },
    "xception": {
        "path":       r"D:\Projects\Major Project\Deepfake Detection\models\new\new_xception_finetuned.pth",
        "input_size": (299, 299),
        "model_name": "legacy_xception",
        "norm_mean":  [0.5, 0.5, 0.5],
        "norm_std":   [0.5, 0.5, 0.5],
    },
    "vit": {
        "path":       r"D:\Projects\Major Project\Deepfake Detection\models\new\new_vit_finetuned.pth",
        "input_size": (224, 224),
        "model_name": "vit_small_patch16_224",
        "norm_mean":  [0.485, 0.456, 0.406],
        "norm_std":   [0.229, 0.224, 0.225],
    },
    "convnext": {
        "path":       r"D:\Projects\Major Project\Deepfake Detection\models\new\new_convnext_finetuned.pth",
        "input_size": (224, 224),
        "model_name": "convnext_small",
        "norm_mean":  [0.485, 0.456, 0.406],
        "norm_std":   [0.229, 0.224, 0.225],
    },
}
# ─────────────────────────────────────────────────────────────────────

def load_model(name, config):
    model = create_model(config["model_name"], pretrained=False, num_classes=1)
    ckpt  = torch.load(config["path"], map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(DEVICE).eval()

def extract_frames(video_path, n=10):
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total-1, n, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def score_frames(model, frames, input_size, norm_mean, norm_std):
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    sigs = []
    with torch.no_grad():
        for f in frames:
            x    = tf(f).unsqueeze(0).to(DEVICE)
            logit = model(x)
            sigs.append(torch.sigmoid(logit).squeeze().item())
    return float(np.mean(sigs))

print(f"\nDevice: {DEVICE}")
print("="*65)
print(f"{'Model':<20} {'sigmoid fake':>14} {'sigmoid real':>14} {'gap':>8}  {'Verdict'}")
print("="*65)

fake_frames = extract_frames(FAKE_VIDEO)
real_frames = extract_frames(REAL_VIDEO)

for name, config in MODEL_CONFIGS.items():
    model       = load_model(name, config)
    sig_fake    = score_frames(model, fake_frames, config["input_size"],
                               config["norm_mean"], config["norm_std"])
    sig_real    = score_frames(model, real_frames, config["input_size"],
                               config["norm_mean"], config["norm_std"])
    gap         = sig_fake - sig_real

    if gap > 0.05:
        verdict = "sigmoid = P(Fake)  → NO inversion"
    elif gap < -0.05:
        verdict = "sigmoid = P(Real)  → USE 1.0-sigmoid"
    else:
        verdict = "ambiguous"

    print(f"  {name:<18} {sig_fake:>14.4f} {sig_real:>14.4f} {gap:>8.4f}  {verdict}")

print("="*65)