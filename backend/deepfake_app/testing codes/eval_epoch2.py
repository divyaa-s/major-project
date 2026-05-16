import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==================== CONFIG ====================
DATA_ROOT = "D:/Projects/Minor Project/Deepfake Detection/datasets/New folder/real-vs-fake"
MODEL_NAME = "convnext_tiny"
IMG_SIZE = 224
BATCH_SIZE = 64                   # larger is fine for eval
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to epoch 2 checkpoint (update if needed)
CHECKPOINT_PATH = "convnext/checkpoints/convnext_epoch_2.pth"

# Results folder (same as training)
RESULTS_DIR = "convnext/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== TRANSFORMS ====================
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ==================== DATA ====================
val_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "valid"), transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)  # workers=0 to avoid Windows issues

print(f"Validation samples: {len(val_dataset):,}")
print(f"Using device: {DEVICE}")

# ==================== LOAD MODEL & CHECKPOINT ====================
def load_model_and_checkpoint():
    print(f"Loading model from: {CHECKPOINT_PATH}")
    model = create_model(MODEL_NAME, pretrained=False, num_classes=2)
    
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return model, checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None

# ==================== EVALUATION ====================
def evaluate_model(model):
    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []

    print("\nRunning validation...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= total
    val_acc = 100. * correct / total

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix (0=Real, 1=Fake):")
    print(cm)

    # Plot and save
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Epoch 2')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_plot_path = os.path.join(RESULTS_DIR, "confusion_matrix_epoch2.png")
    plt.savefig(cm_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"→ Confusion matrix saved: {cm_plot_path}")

    return val_loss, val_acc, correct, total

# ==================== MAIN ====================
if __name__ == '__main__':
    model, checkpoint = load_model_and_checkpoint()
    if model is None:
        print("Failed to load model — exiting.")
    else:
        model = model.to(DEVICE)
        model.eval()
        
        val_loss, val_acc, correct, total = evaluate_model(model)
        
        print("\n" + "="*60)
        print("EPOCH 2 VALIDATION RESULTS (re-computed)")
        print("="*60)
        print(f"Validation Loss:    {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"Correct predictions: {correct:,} / {total:,}")
        
        print("\nSaved checkpoint info:")
        print(f"  Epoch:          {checkpoint['epoch']}")
        print(f"  Val accuracy:   {checkpoint.get('val_acc', 'not saved')}%")
        print(f"  Train loss:     {checkpoint.get('train_loss', 'not saved'):.4f}")
        print(f"  Train accuracy: {checkpoint.get('train_acc', 'not saved'):.2f}%")
        
        print("\nDone!")