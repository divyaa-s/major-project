import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import timm  # pip install timm

def train_efficientnet_b0():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet expects 224x224 input
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Paths to dataset
    train_path = "D:/Projects/Minor Project/Deepfake Detection/datasets/New folder/real-vs-fake/train"
    val_path = "D:/Projects/Minor Project/Deepfake Detection/datasets/New folder/real-vs-fake/valid"

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    val_dataset = datasets.ImageFolder(val_path, transform=transform)

    # Reduce batch size for CPU performance
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    # Create EfficientNet-B0 model
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(train_dataset.classes))
    model.to(device)

    # Optional: Freeze base layers to speed up CPU training
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():  # Only train the classifier head
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct = 0.0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            running_loss += loss.item()
            acc = correct / len(train_loader.dataset)
            loop.set_postfix(loss=loss.item(), acc=f"{acc*100:.2f}%")

        print(f"✅ Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {acc*100:.2f}%")

    torch.save(model.state_dict(), "efficientnet_b0_model.pth")

if __name__ == "__main__":
    train_efficientnet_b0()
