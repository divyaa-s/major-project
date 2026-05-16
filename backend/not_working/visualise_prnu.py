import cv2
import numpy as np
import matplotlib.pyplot as plt
from prnu.noise_extraction import extract_noise
from backend.not_working.prnu.face_background_prnu import face_cascade

def visualize_prnu(img_path, show_face=True, overlay=True):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = extract_noise(img_path)

    # Suppress structural leakage for visualization
    noise_vis = noise - cv2.GaussianBlur(noise, (5, 5), 0)

    overlay_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect face once
    if show_face:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(64,64)
        )
        for (x,y,w,h) in faces:
            cv2.rectangle(overlay_img, (x,y), (x+w,y+h), (0,255,0), 2)

    plt.figure(figsize=(15,5))

    # Original image
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # PRNU residual (suppressed structure)
    plt.subplot(1,3,2)
    plt.imshow(noise_vis, cmap='seismic', vmin=-0.01, vmax=0.01)
    plt.colorbar(label='PRNU intensity')
    plt.title("PRNU Residual (Visualized)")
    plt.axis('off')

    # Overlay
    plt.subplot(1,3,3)
    plt.imshow(overlay_img, alpha=0.7)
    if overlay:
        plt.imshow(np.abs(noise_vis), cmap='hot', alpha=0.4, vmin=0, vmax=0.008)
    plt.title("PRNU Overlay + Face")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img_path = r"C:\Users\Divyaa Sriram\OneDrive\Pictures\Camera Roll\WIN_20241009_19_14_01_Pro.jpg"
    visualize_prnu(img_path)
