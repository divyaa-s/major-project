import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Device configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# --- Image transformation ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # For ViT model
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --- Test dataset path ---
test_path = "D:/Projects/Minor Project/Deepfake Detection/datasets/New folder/real-vs-fake/test"
test_dataset = datasets.ImageFolder(test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# --- Load trained ViT model ---
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(test_dataset.classes))
model.load_state_dict(torch.load("vit_deit_tiny_5epochs.pth", map_location=device))
model.to(device)
model.eval()

# --- Store predictions and labels ---
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- Classification Report ---
print("📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# --- Confusion Matrix ---
conf_mat = confusion_matrix(all_labels, all_preds)
print("\n🔄 Confusion Matrix:")
print(conf_mat)

# --- Plot Confusion Matrix ---
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
