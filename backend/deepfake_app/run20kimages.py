import os
import random
import django
import sys
import csv
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------
# DJANGO SETUP
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

# --------------------------------------------------
# IMPORT YOUR NEW ENSEMBLE PIPELINE
# --------------------------------------------------
from deepfake_detection import generate_gradcam_and_ensemble_predict

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
FAKE_TEST_FOLDER = r"D:\Projects\Major Project\Deepfake Detection\datasets\new_combined\fake"
REAL_TEST_FOLDER = r"D:\Projects\Major Project\Deepfake Detection\datasets\new_combined\real"

# SETTING CAP: 1000 per class = 2000 total
MAX_SAMPLES_PER_CLASS = 1000  
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR / "new_20k_results.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

class DummyRequest:
    def build_absolute_uri(self, path=""):
        return None

dummy_request = DummyRequest()

# --------------------------------------------------
# COLLECT IMAGES
# --------------------------------------------------
def collect_images(folder_path, max_samples=None):
    images = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                images.append(os.path.join(root, f))

    if max_samples and len(images) > max_samples:
        random.seed(42)
        images = random.sample(images, max_samples)

    print(f"Collected {len(images)} images from {folder_path}")
    return images

print("Collecting images (capped at 2k total)...")
real_images = collect_images(REAL_TEST_FOLDER, MAX_SAMPLES_PER_CLASS)
fake_images = collect_images(FAKE_TEST_FOLDER, MAX_SAMPLES_PER_CLASS)

all_samples = (
    [(img, "Real") for img in real_images] +
    [(img, "Fake") for img in fake_images]
)

random.seed(42)
random.shuffle(all_samples)
print(f"\nTotal images to process: {len(all_samples)}")

# --------------------------------------------------
# RUN EVALUATION
# --------------------------------------------------
correct, real_correct, fake_correct = 0, 0, 0
real_total, fake_total = 0, 0

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # UPDATED HEADER to include new forensic and fusion metrics
    writer.writerow([
        "image_path",
        "true_label",
        "predicted_label",
        "confidence",
        "ensemble_fake_prob",  # Simple Average
        "final_fake_prob",     # Meta-Learner output
        "watermark_prob",      # Forensic signal
        "decision_source",     # Meta-learner vs Fallback
        "convnext_fake",
        "xception_fake",
        "efficientnet_fake",
        "vit_fake"
    ])

    for img_path, true_label in tqdm(all_samples, ncols=100):
        try:
            result = generate_gradcam_and_ensemble_predict(
                request=dummy_request,
                img_path=img_path
            )

            # Extracting values from your dictionary return structure
            model_probs = result["models"]

            writer.writerow([
                img_path,
                true_label,
                result["label"],
                result["confidence"],
                result["ensemble_fake_probability"],
                result["final_fake_probability"],
                result["watermark_probability"],
                result["decision_source"],
                model_probs["convnext"]["prob_fake"],
                model_probs["xception"]["prob_fake"],
                model_probs["efficientnet_b3"]["prob_fake"],
                model_probs["vit"]["prob_fake"]
            ])

            # Force disk write
            file.flush()
            os.fsync(file.fileno())
            
            # Accuracy counting
            if result["label"] == true_label:
                correct += 1
                if true_label == "Real": real_correct += 1
                else: fake_correct += 1

            if true_label == "Real": real_total += 1
            else: fake_total += 1

        except Exception as e:
            print(f"Error on {img_path}: {e}")

# --------------------------------------------------
# FINAL METRICS
# --------------------------------------------------
total = real_total + fake_total
if total > 0:
    print("\n==============================")
    print("EXPANDED ENSEMBLE RESULTS (Capped at 2k)")
    print("==============================")
    print(f"Total Accuracy: {correct/total:.4f} ({correct}/{total})")
    print(f"Real Accuracy : {real_correct/real_total:.4f}")
    print(f"Fake Accuracy : {fake_correct/fake_total:.4f}")
    print(f"CSV saved at: {OUTPUT_CSV}")
    print("==============================\n")