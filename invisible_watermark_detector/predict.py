import os
import cv2
import numpy as np
import joblib
from tqdm import tqdm
from features.dct_features import extract_dct_features, extract_frequency_statistics

# Load model & scaler
clf    = joblib.load("models/new_watermark_classifier.pkl")
scaler = joblib.load("models/new_feature_scaler.pkl")

# ────────────────────────────────────────────────
# Tunable zones (based on your probability histogram & threshold tuning)
# You can adjust these numbers after more testing
# ────────────────────────────────────────────────
CONF_VERY_REAL     = 0.25   # ≤ this → almost certainly real
CONF_LIKELY_REAL   = 0.45   # ≤ this → probably real
CONF_UNCERTAIN_LOW = 0.60   # below this but above likely_real → uncertain low
CONF_UNCERTAIN_HIGH = 0.80  # above this but below very high → uncertain high
CONF_HIGH_AI       = 0.82   # ≥ this → high confidence AI/watermarked

def predict_image(image_path, verbose=True):
    name = os.path.splitext(os.path.basename(image_path))[0]

    dct_feats  = extract_dct_features(image_path)
    freq_stats = extract_frequency_statistics(image_path)

    if dct_feats is None or freq_stats is None:
        return {"error": "Could not process image"}

    combined = np.hstack([dct_feats, freq_stats])
    scaled   = scaler.transform(combined.reshape(1, -1))
    probas   = clf.predict_proba(scaled)[0]
    ai_prob  = probas[1]  # probability of being watermarked / AI

    # ────────────────────────────────────────────────
    # Multi-zone decision logic
    # ────────────────────────────────────────────────
    if ai_prob >= CONF_HIGH_AI:
        decision = "WATERMARKED / AI-GENERATED (high confidence)"
        confidence = ai_prob
    elif ai_prob >= CONF_UNCERTAIN_HIGH:
        decision = "LIKELY WATERMARKED / AI (medium confidence)"
        confidence = ai_prob
    elif ai_prob <= CONF_VERY_REAL:
        decision = "REAL (high confidence)"
        confidence = 1 - ai_prob
    elif ai_prob <= CONF_LIKELY_REAL:
        decision = "LIKELY REAL / camera image"
        confidence = 1 - ai_prob
    else:
        decision = "UNCERTAIN / possible editing or weak watermark"
        confidence = 0.50

    result = {
        "filename"         : name,
        "ai_probability"   : round(ai_prob, 4),
        "decision"         : decision,
        "confidence"       : round(confidence, 3),
        "raw_probas"       : probas.tolist(),
    }

    if verbose:
        print(f"\n{'─'*70}")
        print(f"Image          : {name}")
        print(f"AI probability : {ai_prob:.4f}")
        print(f"Decision       : {decision}")
        print(f"Confidence     : {confidence:.3f}")
        print(f"{'─'*70}")

    return result


def batch_predict(folder_path, verbose=False):
    results = []
    images = [f for f in os.listdir(folder_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    if not images:
        print(f"No images found in {folder_path}")
        return results

    for fname in tqdm(images, desc="Predicting"):
        path = os.path.join(folder_path, fname)
        try:
            res = predict_image(path, verbose=verbose)
            results.append(res)
        except Exception as e:
            print(f"Failed {fname}: {e}")

    return results


if __name__ == "__main__":
    # Single image test (change path if needed)
    test_image = r"data/ai_watermarked/0CJ704HU19.jpg"   # ← use raw string for Windows paths

    if os.path.isfile(test_image):
        print("Testing single image...")
        predict_image(test_image, verbose=True)
    else:
        print(f"Test image not found: {test_image}")
        print("Current working dir:", os.getcwd())

    # Batch example (uncomment when ready)
    # results = batch_predict("data/ai_watermarked", verbose=False)
    # for r in results[:5]: print(r)   # show first 5