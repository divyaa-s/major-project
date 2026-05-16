import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import numpy as np
import os
import uuid
import logging
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from timm import create_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import joblib
from dct_features import extract_dct_features, extract_frequency_statistics
# -------------------------
# CONFIG
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

# ============================================================
# LOAD META-LEARNER
# ============================================================
META_LEARNER_PATH = os.path.join(PROJECT_ROOT, 'models', 'final', 'smart_meta_learner_v5.pkl')

try:
    meta_learner = joblib.load(META_LEARNER_PATH)
    logger.info("✅ Successfully loaded Meta-Learner Ensemble!")
except Exception as e:
    logger.error(f"❌ Failed to load Meta-Learner: {e}")
    meta_learner = None

# ============================================================
# LOAD MODELS
# ============================================================
def load_model(model_name, path):
    model = create_model(model_name, pretrained=False, num_classes=1)
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

MODELS = {
    "convnext": (
        load_model(
            "convnext_small",
            r"D:\Projects\Major Project\Deepfake Detection\models\v5\convnext_finetuned_v5.pth"
        ),
        224
    ),
    "xception": (
        load_model(
            "legacy_xception",
            r"D:\Projects\Major Project\Deepfake Detection\models\v5\xception_finetuned_v5.pth"
        ),
        299
    ),
    "efficientnet_b3": (
        load_model(
            "efficientnet_b3",
            r"D:\Projects\Major Project\Deepfake Detection\models\v5\efficientnet_b3_finetuned_v5.pth"
        ),
        300
    ),
    "vit": (
        load_model(
            "vit_small_patch16_224",
            r"D:\Projects\Major Project\Deepfake Detection\models\v5\vit_finetuned_v5.pth"
        ),
        224
    )
}

print("Loaded models:", list(MODELS.keys()))
print("Running on device:", DEVICE)

# ============================================================
# LOAD WATERMARK MODEL
# ============================================================
try:
    WATERMARK_CLF = joblib.load(
        "D:/Projects/Major Project/Deepfake Detection/invisible_watermark_detector/models/new_watermark_classifier.pkl"
    )
    WATERMARK_SCALER = joblib.load(
        "D:/Projects/Major Project/Deepfake Detection/invisible_watermark_detector/models/new_feature_scaler.pkl"
    )
    logger.info("Watermark model loaded successfully.")
except Exception as e:
    logger.error(f"Watermark model load failed: {e}")
    WATERMARK_CLF = None
    WATERMARK_SCALER = None

# ============================================================
# WATERMARK DETECTION
# ============================================================
def detect_watermark(img_path):
    watermark_prob = 0.0

    if WATERMARK_CLF is None or WATERMARK_SCALER is None:
        return watermark_prob

    try:
        dct_feats  = extract_dct_features(img_path)
        freq_stats = extract_frequency_statistics(img_path)

        if dct_feats is None or freq_stats is None:
            return watermark_prob

        combined = np.hstack([dct_feats, freq_stats])
        scaled   = WATERMARK_SCALER.transform(combined.reshape(1, -1))
        probas   = WATERMARK_CLF.predict_proba(scaled)[0]
        watermark_prob = float(probas[1])

    except Exception as e:
        logger.warning(f"Watermark detection failed: {e}")

    return watermark_prob

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================
def generate_gradcam(img_path, model, input_size, predicted_label, confidence, request):
    target_layer = None
    for name, m in reversed(list(model.named_modules())):
        if isinstance(m, nn.Conv2d):
            target_layer = m
            break

    if target_layer is None:
        return None

    img_pil = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    x = transform(img_pil).unsqueeze(0).to(DEVICE)

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(
            input_tensor=x,
            targets=[ClassifierOutputTarget(0)]
        )[0]

    img_resized   = np.array(img_pil.resize((input_size, input_size))).astype(np.float32) / 255.0
    visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

    output_dir = os.path.join(BASE_DIR, "static", "grad_cams")
    os.makedirs(output_dir, exist_ok=True)

    filename  = f"gradcam_{predicted_label}_{confidence:.4f}_{uuid.uuid4().hex[:8]}.jpg"
    save_path = os.path.join(output_dir, filename)
    Image.fromarray(visualization).save(save_path)

    return f"/static/grad_cams/{filename}"

def generate_radar_plot(ensemble_fake, watermark_prob, fake_probs, label, confidence, request):
    gan_fingerprint = float(np.mean([
        fake_probs.get("vit", 0.0),
        fake_probs.get("convnext", 0.0),
    ]))
    frequency    = float(watermark_prob)
    cnn_ensemble = float(ensemble_fake)
    color_anomaly = float(np.mean([
        fake_probs.get("efficientnet_b3", 0.0),
        fake_probs.get("xception", 0.0),
    ]))

    categories  = ["GAN Fingerprint", "Frequency", "CNN Ensemble", "Color Anomaly"]
    values      = [gan_fingerprint, frequency, cnn_ensemble, color_anomaly]
    N           = len(categories)

    values_plot = values + [values[0]]
    angles      = [n / float(N) * 2 * np.pi for n in range(N)]
    angles     += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f9f9f9")

    ref_values = [1.0] * N + [1.0]
    ax.plot(angles, ref_values, color="#cccccc", linewidth=1, linestyle="--")
    ax.fill(angles, ref_values, color="#e8e8e8", alpha=0.3)

    color = "#cc0000" if label == "Fake" else "#007700"
    ax.plot(angles, values_plot, color=color, linewidth=2.5)
    ax.fill(angles, values_plot, color=color, alpha=0.20)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight="bold")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=7, color="#888888")
    ax.set_ylim(0, 1.0)
    ax.grid(color="#cccccc", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.spines["polar"].set_visible(False)

    ax.set_title(
        f"Signal Profile — {label} ({confidence * 100:.1f}%)",
        size=13, fontweight="bold", pad=20, color=color
    )

    for angle, val in zip(angles[:-1], values):
        ax.annotate(
            f"{val:.2f}",
            xy=(angle, val),
            xytext=(angle, val + 0.08),
            ha="center", va="center",
            fontsize=8, color=color, fontweight="bold"
        )

    plt.tight_layout()

    output_dir = os.path.join(BASE_DIR, "static", "radar_plots")
    os.makedirs(output_dir, exist_ok=True)

    filename  = f"radar_{label.lower()}_{confidence:.4f}_{uuid.uuid4().hex[:8]}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return f"/static/radar_plots/{filename}"

def generate_bar_chart(fake_probs, ensemble_fake, watermark_prob, final_fake, label, confidence, request):
    """
    Generates a horizontal bar chart of the individual model probabilities, 
    forensic scores, and the final Meta-Learner output.
    """
    # Now we have 8 Categories!
    categories = [
        'Final',
        'Watermark',
        'CNN ens.',
        'vit',
        'convnext',
        'xception',
        'eff_b3'
    ]
    
    values = [
        final_fake,
        watermark_prob,        # Your watermark detector!
        ensemble_fake,
        fake_probs.get('vit', 0.0),
        fake_probs.get('convnext', 0.0),
        fake_probs.get('xception', 0.0),
        fake_probs.get('efficientnet_b3', 0.0)
    ]
    
    # Colors: Final=LightBlue, Watermark=Purple, Ensemble=Orange, Base Models=DarkBlue
    colors = ['#3498db', '#9b59b6', '#e67e22', '#2980b9', '#2980b9', '#2980b9', '#2980b9']

    # Made the figure slightly taller (4.5) to comfortably fit the 8th bar
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Create horizontal bars
    bars = ax.barh(categories, values, color=colors, height=0.5)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Fake probability / score", fontsize=10)
    
    title_color = "#cc0000" if label == "Fake" else "#007700"
    ax.set_title(f"Deepfake Analysis - {label} ({confidence * 100:.1f}%)", fontsize=12, pad=15, color=title_color, fontweight="bold")
    
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.invert_yaxis()

    # Annotate the values at the end of each bar
    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', ha='left', fontsize=9, color='#333333', fontweight='bold')

    plt.tight_layout()

    output_dir = os.path.join(BASE_DIR, "static", "bar_charts")
    os.makedirs(output_dir, exist_ok=True)

    filename  = f"barchart_{label.lower()}_{confidence:.4f}_{uuid.uuid4().hex[:8]}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return f"/static/bar_charts/{filename}"


# ============================================================
# MAIN PIPELINE
# ============================================================
def generate_gradcam_and_ensemble_predict(request, img_path, true_label="Unknown"):

    logger.warning(">>> 4-MODEL CNN ENSEMBLE invoked")

    img = Image.open(img_path).convert("RGB")

    fake_probs = {}
    real_probs = {}

    best_model      = None
    best_input_size = None
    best_confidence = -1
    best_model_name = None

    # ── Run all models ────────────────────────────────────────────
    for name, (model, input_size) in MODELS.items():
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits    = model(x)
            prob_fake = torch.sigmoid(logits).item()
            prob_real = 1.0 - prob_fake

        logger.warning(f"{name}: raw_sigmoid={torch.sigmoid(logits).item():.4f}  prob_fake={prob_fake:.4f}")
        fake_probs[name] = prob_fake
        real_probs[name] = prob_real

        model_label = "Fake" if prob_fake > 0.5 else "Real"
        model_conf  = prob_fake if model_label == "Fake" else prob_real

        if model_conf > best_confidence:
            best_confidence = model_conf
            best_model      = model
            best_input_size = input_size
            best_model_name = name

    # ── Ensemble Calculation ───────────────────────────────────────
    ensemble_fake = float(np.mean(list(fake_probs.values())))
    watermark_prob = detect_watermark(img_path)
# ── Meta-Learner Fusion (V5) ───────────────────────────────────────
    if meta_learner is not None:
        # Extract base scores
        convnext_score = fake_probs.get("convnext", 0.5)
        xception_score = fake_probs.get("xception", 0.5)
        effnet_score = fake_probs.get("efficientnet_b3", 0.5)
        vit_score = fake_probs.get("vit", 0.5)
        
        base_scores = [convnext_score, xception_score, effnet_score, vit_score]
        
        # Calculate Smart Features
        disagreement = max(base_scores) - min(base_scores)
        max_confidence = max(base_scores)
        
        # V5 Features: 4 CNNs + 2 Smart Stats + 1 Watermark
        features = np.array([[
            convnext_score,
            xception_score,
            effnet_score,
            vit_score,
            disagreement,
            max_confidence,
            watermark_prob 
        ]])
        
        final_fake = float(meta_learner.predict_proba(features)[0][1])
        decision_source = "smart_meta_learner_v5"
    else:
        logger.warning("Meta-Learner not found! Falling back to raw average.")
        final_fake = ensemble_fake 
        decision_source = "fallback_average"

    # Set Final Label & Confidence
    final_real = 1.0 - final_fake
    label      = "Fake" if final_fake >= 0.50 else "Real"
    confidence = final_fake if label == "Fake" else final_real

    # ── GradCAM ───────────────────────────────────────────────────
    if best_model is not None:
        logger.warning(f"GradCAM model used: {best_model_name}")
        gradcam_url = generate_gradcam(
            img_path, best_model, best_input_size,
            label, confidence, request
        )
    else:
        gradcam_url = None

    # ── Radar plot ────────────────────────────────────────────────
    try:
        radar_url = generate_radar_plot(
            ensemble_fake=ensemble_fake,
            watermark_prob=watermark_prob,
            fake_probs=fake_probs,
            label=label,
            confidence=confidence,
            request=request
        )
    except Exception as e:
        logger.warning(f"Radar plot generation failed: {e}")
        radar_url = None

    # ── Bar Chart ─────────────────────────────────────────────────
    try:
        barchart_url = generate_bar_chart(
            fake_probs=fake_probs,
            ensemble_fake=ensemble_fake,
            watermark_prob=watermark_prob,
            final_fake=final_fake,
            label=label,
            confidence=confidence,
            request=request
        )
    except Exception as e:
        logger.warning(f"Bar chart generation failed: {e}")
        barchart_url = None

    # ── Per-model details ─────────────────────────────────────────
    model_details = {}
    for name in fake_probs:
        model_fake  = fake_probs[name]
        model_real  = real_probs[name]
        model_label = "Fake" if model_fake > 0.5 else "Real"
        model_conf  = model_fake if model_label == "Fake" else model_real

        model_details[name] = {
            "label":      model_label,
            "confidence": round(model_conf, 4),
            "prob_fake":  round(model_fake, 4),
            "prob_real":  round(model_real, 4)
        }

    # ── Logging ───────────────────────────────────────────────────
    logger.warning("\n" + "=" * 70)
    logger.warning("ENSEMBLE FINAL RESULT (META-LEARNER)")
    logger.warning(f"Final Label       : {label}")
    logger.warning(f"Final Confidence  : {confidence:.4f}")
    logger.warning(f"CNN Base Avg Prob : {ensemble_fake:.4f}")
    logger.warning(f"Watermark Prob    : {watermark_prob:.4f}")
    logger.warning(f"Decision Source   : {decision_source}")

    for name, details in model_details.items():
        logger.warning(
            f"{name.upper():15} → "
            f"Label: {details['label']}, "
            f"Conf: {details['confidence']:.4f}, "
            f"Fake: {details['prob_fake']:.4f}, "
            f"Real: {details['prob_real']:.4f}"
        )
    logger.warning("=" * 70 + "\n")

    # ── Return ────────────────────────────────────────────────────
    return {
        "label":                    label,
        "confidence":               round(confidence, 4),
        "gradcam_url":              gradcam_url,
        "radar_url":                radar_url,
        "barchart_url":             barchart_url,             # <-- NEW URL RETURNED HERE
        "ensemble_fake_probability": round(ensemble_fake, 4),
        "final_fake_probability":   round(final_fake, 4),
        "watermark_probability":    round(watermark_prob, 4),
        "decision_source":          decision_source,
        "models":                   model_details,
        "watermark": {
            "probability":         round(watermark_prob, 4),
            "override_triggered":  False 
        }
    }