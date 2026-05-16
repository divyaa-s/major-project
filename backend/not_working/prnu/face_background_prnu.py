import cv2
import numpy as np
from prnu.noise_extraction import extract_noise_from_array

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def face_background_prnu_diff(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Skip tiny images
    if gray.shape[0] < 100 or gray.shape[1] < 100:
        return None

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(64, 64)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    # Safety check
    if w < 32 or h < 32:
        return None

    face = gray[y:y+h, x:x+w]
    background = gray.copy()
    background[y:y+h, x:x+w] = 0

    face_noise = extract_noise_from_array(face)
    bg_noise = extract_noise_from_array(background)

    if face_noise is None or bg_noise is None:
        return None

    corr = np.corrcoef(face_noise.flatten(), bg_noise.flatten())[0, 1]
    return abs(corr)
