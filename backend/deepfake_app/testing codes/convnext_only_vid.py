import torch
import torch.nn.functional as F
import cv2
import numpy as np
import logging
from pathlib import Path
from PIL import Image
from torchvision import transforms
from timm import create_model
from facenet_pytorch import MTCNN
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Config ──
CHECKPOINT_PATH = "convnext/checkpoints/best_convnext_epoch_2_acc99.59.pth"  # or convnext_epoch_2.pth
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_KEYFRAMES = 100

# ── Model ──
model = create_model("convnext_tiny", pretrained=False, num_classes=2)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()
print("ConvNeXt loaded successfully from epoch", checkpoint['epoch'])

# ── Face detector (same as hybrid) ──
mtcnn = MTCNN(keep_all=False, device=DEVICE)

# ── Preprocess ──
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def crop_face(frame):
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        boxes, _ = mtcnn.detect(img_pil)
        if boxes is None or len(boxes) == 0:
            return None
        box = boxes[0].astype(int)
        x1, y1, x2, y2 = box
        padding = 20
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
        face = img_rgb[y1:y2, x1:x2]
        return face
    except:
        return None

def analyze_video_convnext_only(video_path):
    print(f"\nTesting ConvNeXt alone on: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames, {fps:.1f} fps, {total_frames/fps:.2f}s")
    
    # Smart keyframe selection (same as hybrid)
    candidate_count = min(NUM_KEYFRAMES * 3, total_frames)
    indices = np.linspace(0, total_frames - 1, candidate_count, dtype=int)
    
    faces = []
    for idx in tqdm(indices, desc="Extracting keyframes"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        face = crop_face(frame)
        if face is not None:
            faces.append(face)
    
    cap.release()
    
    if not faces:
        print("No faces detected in video")
        return {"label": "No faces", "fake_prob": 0.5}
    
    print(f"Found {len(faces)} face crops")
    
    # Batch inference
    fake_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(faces), BATCH_SIZE):
            batch = faces[i:i+BATCH_SIZE]
            inputs = torch.stack([transform(f) for f in batch]).to(DEVICE)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # fake prob
            fake_probs.extend(probs)
    
    avg_fake_prob = np.mean(fake_probs)
    print(f"Average fake probability: {avg_fake_prob:.4f}")
    
    label = "Fake" if avg_fake_prob >= 0.60 else "Real"
    confidence = avg_fake_prob if label == "Fake" else 1 - avg_fake_prob
    
    return {
        "label": label,
        "confidence": confidence,
        "avg_fake_prob": avg_fake_prob,
        "num_faces": len(faces),
        "per_frame_probs": fake_probs
    }

if __name__ == '__main__':
    video = r"D:\Projects\Major Project\Deepfake Detection\datasets\dfd\original\01__talking_angry_couch.mp4"
    result = analyze_video_convnext_only(video)
    print("\nFinal result (ConvNeXt alone):")
    print(f"Label: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    print(f"Avg fake prob: {result['avg_fake_prob']:.4f}")
    print(f"Faces analyzed: {result['num_faces']}")