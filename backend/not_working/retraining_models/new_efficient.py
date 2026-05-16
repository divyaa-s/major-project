
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
'''
# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 15
LR = 5e-5
'''
from torchvision import datasets, transforms

TRAIN_DIR = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/train"
VAL_DIR   = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/valid"

# Minimal transform (no augmentation needed for this check)
tfm = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=tfm)
val_ds   = datasets.ImageFolder(VAL_DIR, transform=tfm)

print("Train classes:", train_ds.class_to_idx)
print("Val classes:", val_ds.class_to_idx)
print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))


'''
# -----------------------------
# MODEL (2-CLASS, CONTINUATION)
# -----------------------------
model = timm.create_model(
    "efficientnet_b3",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)

# -----------------------------
# FREEZE / UNFREEZE STRATEGY
# -----------------------------
# Freeze entire backbone
# Freeze entire model
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last 3 EfficientNet blocks
for param in model.blocks[-3:].parameters():
    param.requires_grad = True

# Classifier always trainable
for param in model.classifier.parameters():
    param.requires_grad = True


# -----------------------------
# LOSS, OPTIMIZER, SCHEDULER
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=3, factor=0.3
)

# -----------------------------
# TRAINING LOOP
# -----------------------------
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

    for imgs, labels in loop:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100 * correct / total:.2f}%"
        )

    train_acc = correct / total

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    scheduler.step(val_acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss/len(train_loader):.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # -------------------------
    # CHECKPOINTS
    # -------------------------
    torch.save(
        model.state_dict(),
        f"{CHECKPOINT_DIR}/efficientnet_epoch_{epoch+1}.pth"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            model.state_dict(),
            f"{CHECKPOINT_DIR}/efficientnet_best.pth"
        )

print("✅ Training complete")
print("🏆 Best Validation Accuracy:", best_val_acc)
'''