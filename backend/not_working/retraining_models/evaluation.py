import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

# -------------------------
# CONFIG
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_CLASSES = 2

TRAIN_DIR = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/train"
VAL_DIR   = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/valid"

CHECKPOINT_PATH = (
    "D:/Projects/Major Project/Deepfake Detection/"
    "backend/retraining_models/checkpoints/efficientnet_best.pth"
)

# -------------------------
# TRANSFORMS (MUST MATCH TRAINING)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# DATASETS & LOADERS
# -------------------------
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_ds   = datasets.ImageFolder(VAL_DIR, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Train classes:", train_ds.class_to_idx)
print("Val classes:", val_ds.class_to_idx)
print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))

# -------------------------
# MODEL (TIMM — THIS IS THE FIX)
# -------------------------
model = timm.create_model(
    "efficientnet_b3",
    pretrained=False,
    num_classes=NUM_CLASSES
)

# Load checkpoint safely
state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
if "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
else:
    model.load_state_dict(state)

model.to(DEVICE)

for p in model.parameters():
    p.requires_grad = False

model.eval()

# -------------------------
# VALIDATION
# -------------------------
correct = 0
total = 0

with torch.no_grad():
    loop = tqdm(val_loader, desc="Validating")
    for images, labels in loop:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        acc = 100 * correct / total
        loop.set_postfix(acc=f"{acc:.2f}%")

print(f"\nFinal Validation Accuracy: {100 * correct / total:.2f}%")
