import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from tqdm import tqdm
import pandas as pd
import argparse
import matplotlib.pyplot as plt

# ==================== CONFIG ====================
DATA_ROOT = "D:/Projects/Minor Project/Deepfake Detection/datasets/New folder/real-vs-fake"
MODEL_NAME = "convnext_tiny"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20                           # change as needed
LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Main output folder ──
MAIN_SAVE_DIR = "convnext"
os.makedirs(MAIN_SAVE_DIR, exist_ok=True)

# ── Subfolders ──
CHECKPOINT_DIR = os.path.join(MAIN_SAVE_DIR, "checkpoints")
RESULTS_DIR    = os.path.join(MAIN_SAVE_DIR, "results")
GRAPHS_DIR     = os.path.join(MAIN_SAVE_DIR, "graphs")

for d in [CHECKPOINT_DIR, RESULTS_DIR, GRAPHS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"→ Checkpoints → {CHECKPOINT_DIR}")
print(f"→ Results/CSV → {RESULTS_DIR}")
print(f"→ Graphs     → {GRAPHS_DIR}")

# ==================== ARGUMENTS (for resuming) ====================
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
args = parser.parse_args()

# ==================== TRANSFORMS ====================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ==================== DATA ====================
train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_ROOT, "valid"), transform=val_transform)

print(f"Train samples: {len(train_dataset):,}")
print(f"Valid samples: {len(val_dataset):,}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

# ==================== MODEL, LOSS, OPTIMIZER ====================
print(f"Loading {MODEL_NAME} ...")
model = create_model(MODEL_NAME, pretrained=True, num_classes=2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

start_epoch = 0
best_val_acc = 0.0

# ==================== RESUME (if requested) ====================
if args.resume and os.path.exists(args.resume):
    print(f"Resuming from: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_acc = checkpoint.get('val_acc', 0.0)
    print(f"→ Resumed at epoch {start_epoch}, previous best val acc: {best_val_acc:.2f}%")
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6, last_epoch=start_epoch-1)

# ==================== METRICS HISTORY ====================
train_losses = []
train_accs = []
val_losses = []
val_accs = []

# ==================== MAIN TRAINING LOGIC ====================
def train_model():
    global start_epoch, best_val_acc

    for epoch in range(start_epoch, EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        pbar = tqdm(train_loader, desc=f"Training epoch {epoch+1}", leave=False)

        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= total
        train_acc = 100. * correct / total

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        val_loss /= total
        val_acc = 100. * correct / total

        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}   | Acc: {val_acc:.2f}%")

        scheduler.step()

        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
        }
        epoch_path = os.path.join(CHECKPOINT_DIR, f"convnext_epoch_{epoch+1}.pth")
        torch.save(checkpoint, epoch_path)
        print(f"→ Checkpoint saved: {epoch_path}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(CHECKPOINT_DIR, f"best_convnext_epoch_{epoch+1}_acc{val_acc:.2f}.pth")
            torch.save(checkpoint, best_path)
            print(f"→ New best model: {best_path} (val acc: {val_acc:.2f}%)")

    # ==================== FINAL SUMMARY ====================
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("\nEpoch | Train Loss | Train Acc | Val Loss | Val Acc")
    print("-"*70)
    for e in range(len(train_losses)):
        print(f"{e+1:5d} | {train_losses[e]:10.4f} | {train_accs[e]:9.2f}% | "
              f"{val_losses[e]:8.4f} | {val_accs[e]:7.2f}%")

    # Save history to CSV
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses)+1),
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    })
    history_path = os.path.join(RESULTS_DIR, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"\nMetrics history saved to: {history_path}")

    # ==================== GENERATE & SAVE GRAPHS ====================
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 6))

    # Plot 1: Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o', color='blue')
    plt.plot(epochs_range, val_losses, label='Validation Loss', marker='o', color='orange')
    plt.title('Training & Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label='Train Accuracy', marker='o', color='green')
    plt.plot(epochs_range, val_accs, label='Validation Accuracy', marker='o', color='red')
    plt.title('Training & Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save graphs
    loss_plot_path = os.path.join(GRAPHS_DIR, "loss_curve.png")
    acc_plot_path  = os.path.join(GRAPHS_DIR, "accuracy_curve.png")

    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    print(f"→ Loss curve saved: {loss_plot_path}")

    plt.savefig(acc_plot_path, dpi=150, bbox_inches='tight')
    print(f"→ Accuracy curve saved: {acc_plot_path}")

    plt.show()  # Displays if running interactively
    plt.close()

    print("\nAll training outputs organized in 'convnext/' folder.")
    print("  - Checkpoints → convnext/checkpoints/")
    print("  - Results/CSV → convnext/results/")
    print("  - Graphs     → convnext/graphs/")

# ==================== ENTRY POINT ====================
if __name__ == '__main__':
    train_model()