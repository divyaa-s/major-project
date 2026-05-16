import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch

# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Face detector (loaded ONCE)
# --------------------------------------------------

mtcnn = MTCNN(
    keep_all=False,
    device=DEVICE
)

# --------------------------------------------------
# Face cropping utility
# --------------------------------------------------

def crop_face_and_get_original(img_path):
    """
    Args:
        img_path (str): path to image

    Returns:
        face_crop (np.ndarray): cropped face (RGB)
        original_image (np.ndarray): full image (RGB)
    """

    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    boxes, _ = mtcnn.detect(img)

    if boxes is None or len(boxes) == 0:
        return None, img_np

    x1, y1, x2, y2 = boxes[0].astype(int)

    # Clamp to image bounds
    h, w = img_np.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face_crop = img_np[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None, img_np

    return face_crop, img_np
