import os
import random
import csv
from tqdm import tqdm
from deepfake_detection import generate_gradcam_and_ensemble_predict

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
# Point this to the folder you want to test (Real or Fake)
TEST_FOLDER = r"D:\Projects\Major Project\Deepfake Detection\datasets\new_combined\fake"
OUTPUT_CSV = "fake_final_pipeline_results.csv"

# Tell the script what the actual images are so you can calculate accuracy later
GROUND_TRUTH = "Fake"  # Change to "Fake" when testing your fake folder

MAX_SAMPLES = 500  # THE 500 IMAGE CAP

# --------------------------------------------------
# RUN BATCH EVALUATION
# --------------------------------------------------
print(f"🚀 Running V5 Meta-Learner Batch Analysis on: {TEST_FOLDER}")

# 1. Collect all valid images
images = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

# 2. Apply the Cap
if len(images) > MAX_SAMPLES:
    random.seed(42)  # For reproducible results
    images = random.sample(images, MAX_SAMPLES)
    print(f"⚠️ Folder contains multiple images. Randomly sampled {MAX_SAMPLES} for testing.")
else:
    print(f"✅ Processing all {len(images)} images found in folder.")

# 3. Process Images
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    # NEW HEADERS: We are now tracking the PRNU, Watermark, and Meta-Learner!
    writer.writerow([
        "image_name", 
        "true_label", 
        "meta_learner_label", 
        "meta_learner_prob", 
        "cnn_simple_avg", 
        "watermark_prob",
        "vit_score", 
        "convnext_score", 
        "xception_score", 
        "effnet_score"
    ])

    for filename in tqdm(images, ncols=100):
        img_path = os.path.join(TEST_FOLDER, filename)
        
        try:
            # Call the updated pipeline
            # (Pass None for request; GradCAMs/Plots are skipped automatically if not needed)
            raw_res = generate_gradcam_and_ensemble_predict(None, img_path)
            
            models = raw_res["models"]
            
            # Extract Base CNN Scores
            vit_score = models["vit"]["prob_fake"]
            convnext_score = models["convnext"]["prob_fake"]
            xception_score = models["xception"]["prob_fake"]
            effnet_score = models["efficientnet_b3"]["prob_fake"]
            
            # Write row to CSV
            writer.writerow([
                filename, 
                GROUND_TRUTH,                           # What the image actually is
                raw_res["label"],                       # The final decision (Real/Fake)
                raw_res["final_fake_probability"],      # The V5 Meta-Learner Score
                raw_res["ensemble_fake_probability"],   # The old CNN Average
                raw_res.get("watermark_probability", 0),# Watermark detector
                vit_score,
                convnext_score,
                xception_score,
                effnet_score
            ])

            # Force write to disk periodically in case it crashes midway
            file.flush()

        except Exception as e:
            print(f"\n❌ Error on {filename}: {e}")

print(f"\n✅ Done! V5 Meta-Learner results saved to: {OUTPUT_CSV}")