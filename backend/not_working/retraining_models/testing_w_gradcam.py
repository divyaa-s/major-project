import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image

# -------------------------
# CONFIG
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_PATH = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/fake/justin_fake.png"   # <-- put ONE image here
CHECKPOINT_PATH = "D:/Projects/Major Project/Deepfake Detection/backend/retraining_models/checkpoints/efficientnet_best.pth"
SAVE_PATH = "D:/Projects/Major Project/Deepfake Detection/backend/retraining_models/fake3_gradcam_output.jpg"

# -------------------------
# LOAD MODEL
# -------------------------
import timm
import torch.nn as nn

model = timm.create_model(
    "efficientnet_b3",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()


# -------------------------
# HOOKS FOR GRAD-CAM
# -------------------------
features = None
gradients = None

def forward_hook(module, input, output):
    global features
    features = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

target_layer = model.blocks[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# -------------------------
# IMAGE PREPROCESS
# -------------------------
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(DEVICE)

# -------------------------
# FORWARD + BACKWARD
# -------------------------
output = model(input_tensor)
pred_class = output.argmax(dim=1)

model.zero_grad()
output[0, pred_class].backward()

# -------------------------
# GRAD-CAM COMPUTATION
# -------------------------
weights = gradients.mean(dim=(2, 3), keepdim=True)
cam = (weights * features).sum(dim=1)
cam = torch.relu(cam)
cam = cam.squeeze().detach().cpu().numpy()

cam = cv2.resize(cam, (300, 300))
cam = (cam - cam.min()) / (cam.max() + 1e-8)

# -------------------------
# OVERLAY
# -------------------------
img_np = np.array(img.resize((300, 300)))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

cv2.imwrite(SAVE_PATH, overlay)
print(f"Grad-CAM saved to {SAVE_PATH}")
