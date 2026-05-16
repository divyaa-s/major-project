import matplotlib
matplotlib.use('Agg')           # Must be BEFORE importing pyplot
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import uuid
import logging
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from timm import create_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from django.conf import settings
from scipy.stats import entropy
import csv
import joblib

# Force Python to look in the current directory (fixes Django/Windows import caching)
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import normally
from dct_features import extract_dct_features, extract_frequency_statistics

# -------------------------
# LOGGING (Django-safe)
# -------------------------
logger = logging.getLogger(__name__)

# -------------------------
# Watermark model loaded ONCE at module level
# -------------------------
try:
    WATERMARK_CLF = joblib.load("D:/Projects/Major Project/Deepfake Detection/invisible_watermark_detector/models/new_watermark_classifier.pkl")
    WATERMARK_SCALER = joblib.load("D:/Projects/Major Project/Deepfake Detection/invisible_watermark_detector/models/new_feature_scaler.pkl")
    logger.info("Watermark classifier and scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load watermark models: {e}")
    WATERMARK_CLF = None
    WATERMARK_SCALER = None



# -------------------------
# CONFIG
# -------------------------
STRICT_MODE = True   # True = research / False = user-safe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIGS = {
    "efficientnet_b3": {
        "path": "D:/Projects/Major Project/Deepfake Detection/models/efficientnet_b3_model.pth",
        "input_size": (300, 300),
        "model_name": "efficientnet_b3"
    },
    "xception": {
        "path": "D:/Projects/Major Project/Deepfake Detection/models/xceptionnet_20k.pth",
        "input_size": (299, 299),
        "model_name": "xception"
    },
    "vit": {
        "path": "D:/Projects/Major Project/Deepfake Detection/models/vit_deit_tiny_5epochs.pth",
        "input_size": (224, 224),
        "model_name": "vit_tiny_patch16_224"
    },
    "convnext": {
        "path": "D:/Projects/Major Project/Deepfake Detection/models/convnext_tiny_deepfake.pth",
        "input_size": (224, 224),
        "model_name": "convnext_tiny"
    }
}

mtcnn = MTCNN(keep_all=False, device=DEVICE)

# CSV logging path
CSV_PATH = os.path.join(settings.BASE_DIR, "deepfake_app/working_debug_results.csv")

# -------------------------
# FACE CROP
# -------------------------
def crop_face(img_path):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    boxes, _ = mtcnn.detect(img)

    if boxes is None:
        return None, None

    x1, y1, x2, y2 = boxes[0].astype(int)
    face = img_np[y1:y2, x1:x2]
    return face, img_np

# -------------------------
# PREPROCESS
# -------------------------
def preprocess(face, size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(face).unsqueeze(0).to(DEVICE)

# -------------------------
# FORENSIC ANALYSIS
# -------------------------
def frequency_artifact_score(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    f = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log(np.abs(f) + 1)

    h, w = magnitude.shape
    high_freq = magnitude[int(0.6*h):, int(0.6*w):]
    score = np.mean(high_freq) / (np.mean(magnitude) + 1e-8)

    if score > 0.85:
        score = 0.45

    return float(np.clip(score, 0, 1))

def gan_fingerprint_detection(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dct = np.abs(cv2.dct(gray))

    h, w = dct.shape
    mid = dct[h//4:3*h//4, w//4:3*w//4]
    high = dct[3*h//4:, 3*w//4:]

    ratio = np.mean(mid) / (np.mean(high) + 1e-8)
    return float(np.clip(ratio / 10.0, 0, 1))

def color_distribution_score(face):
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    entropies = []

    for ch in cv2.split(lab):
        hist, _ = np.histogram(ch, bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-8)
        entropies.append(entropy(hist))

    avg_entropy = np.mean(entropies)
    max_entropy = np.log2(256)
    score = 1 - avg_entropy / max_entropy

    if score < 0.35:
        score = 0.0

    return float(np.clip(score, 0, 1))

# -------------------------
# LOAD MODEL
# -------------------------
def load_model(model_name, path):
    model = create_model(model_name, pretrained=False, num_classes=2, in_chans=3)
    
    state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
    
    if not isinstance(state_dict, dict):
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        else:
            raise ValueError("Checkpoint format not recognized")
    
    if list(state_dict.keys())[0].startswith('module.'):
        logger.info(f"Stripping 'module.' prefix from {model_name} checkpoint")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing or unexpected:
        logger.warning(f"Key mismatch loading {model_name} - missing: {len(missing)}, unexpected: {len(unexpected)}")
    
    model.to(DEVICE).eval()
    
    target_layer = None
    for name, m in reversed(list(model.named_modules())):
        if isinstance(m, nn.Conv2d):
            target_layer = m
            logger.debug(f"Grad-CAM target layer for {model_name}: {name}")
            break
    
    return model, target_layer

# -------------------------
# MODEL PREDICTION
# -------------------------
def get_model_prediction(model, face, size, name):
    with torch.no_grad():
        if name == "convnext":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])

        x = transform(face).unsqueeze(0).to(DEVICE)

        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

        if name == "convnext":
            fake_prob = probs[0].item()     # class 0 = Fake for ConvNeXt
        else:
            fake_prob = probs[1].item()     # class 1 = Fake for others

        real_prob = 1.0 - fake_prob

        if name == "convnext":
            logits_np = logits[0].cpu().numpy()
            probs_np  = probs.cpu().numpy()
            print(f"\n=== ConvNeXt Debug (ImageNet norm) ===")
            print(f"Raw logits       : {logits_np}")
            print(f"Softmax probs    : {probs_np}")
            print(f"Real prob      : {real_prob:.4f}")
            print(f"Fake prob      : {fake_prob:.4f}")
            print(f"Argmax class     : {np.argmax(probs_np)}")
            print("=======================================\n")

        return {
            "model_name": name,
            "fake_probability": fake_prob,
            "confidence": max(probs).item(),
            "label": "Fake" if fake_prob > 0.5 else "Real"
        }

# -------------------------
# GENERATE GRAD-CAM
# -------------------------
def generate_gradcam(model, target_layer, face, full_image, input_size, predicted_label, model_name, confidence, request):
    target_class = 1 if predicted_label == "Fake" else 0

    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(
        input_tensor=preprocess(face, input_size),
        targets=[ClassifierOutputTarget(target_class)]
    )[0]

    resized = cv2.resize(full_image, input_size)
    normalized = resized.astype(np.float32) / 255.0

    visualization = show_cam_on_image(normalized, grayscale_cam, use_rgb=True)

    output_dir = os.path.join(settings.MEDIA_ROOT, "grad_cams")
    os.makedirs(output_dir, exist_ok=True)

    filename = f"gradcam_{model_name}_{predicted_label}_{confidence:.4f}_{uuid.uuid4().hex}.jpg"
    path = os.path.join(output_dir, filename)

    cv2.imwrite(path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    return request.build_absolute_uri(settings.MEDIA_URL + "grad_cams/" + filename)

# -------------------------
# RADAR PLOT FUNCTION
# -------------------------
def plot_radar_summary(freq, gan, color, cnn_ensemble, label, confidence, request):
    try:
        categories = ['Frequency', 'GAN fingerprint', 'Color anomaly', 'CNN ensemble']
        values = [freq, gan, color, cnn_ensemble]
        values += values[:1]  # close polygon
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        ax.fill(angles, values, color='red', alpha=0.25)
        ax.plot(angles, values, color='red', linewidth=2)

        ax.fill(angles, [1.0] * len(angles), color='gray', alpha=0.07)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(0, 1.05)

        ax.set_title(f"Signal Profile — {label} ({confidence:.1%})", size=13, y=1.1, pad=20)

        plot_dir = os.path.join(settings.MEDIA_ROOT, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        radar_filename = f"radar_{uuid.uuid4().hex[:12]}.png"
        radar_path = os.path.join(plot_dir, radar_filename)
        plt.savefig(radar_path, dpi=110, bbox_inches="tight")
        plt.close(fig)

        radar_url = request.build_absolute_uri(settings.MEDIA_URL + "plots/" + radar_filename)
        logger.info(f"Saved radar plot - {radar_url}")
        return radar_url

    except Exception as e:
        logger.error(f"Failed to generate radar plot: {str(e)}")
        return None

# ----------------
# MAIN PIPELINE
# ----------------
def generate_gradcam_and_ensemble_predict(request, img_path, true_label="Unknown"):
    logger.warning(">>> Deepfake pipeline invoked")

    face, full = crop_face(img_path)
    if face is None:
        return {"error": "No face detected"}

    preds, models = [], {}

    for key, cfg in MODEL_CONFIGS.items():
        model, layer = load_model(cfg["model_name"], cfg["path"])
        pred = get_model_prediction(model, face, cfg["input_size"], key)
        preds.append(pred)
        models[key] = (model, layer, cfg)

    freq = frequency_artifact_score(face)
    gan = gan_fingerprint_detection(face)
    color = color_distribution_score(face)

    forensic = 0.25 * freq + 0.40 * gan + 0.35 * color

    portrait_gan_suspect = False
    if color > 0.50 and freq >= 0.40 and gan < 0.25:
        logger.warning("Portrait GAN heuristic triggered (+0.10 forensic boost)")
        forensic = min(forensic + 0.10, 1.0)
        portrait_gan_suspect = True

    # ────────────────────────────────────────────────
    # WATERMARK DETECTION (integrated here)
    # ────────────────────────────────────────────────
    watermark_prob = 0.0
    watermark_detected = False
    try:
        if WATERMARK_CLF is not None and WATERMARK_SCALER is not None:
            dct_feats = extract_dct_features(img_path)
            freq_stats = extract_frequency_statistics(img_path)
            if dct_feats is not None and freq_stats is not None:
                combined = np.hstack([dct_feats, freq_stats])
                scaled = WATERMARK_SCALER.transform(combined.reshape(1, -1))
                probas = WATERMARK_CLF.predict_proba(scaled)[0]
                watermark_prob = probas[1]  # prob of watermarked/AI
                watermark_detected = watermark_prob >= 0.75

                logger.info(f"Watermark prob: {watermark_prob:.4f}, detected: {watermark_detected}")
                print(f"[WATERMARK] prob = {watermark_prob:.4f}, detected = {watermark_detected}")

                # Optional small forensic boost if strong watermark signal
                if watermark_prob >= 0.75:
                    forensic = min(forensic + 0.10 * watermark_prob, 0.95)
                    logger.warning(f"Watermark boost applied: +{0.10 * watermark_prob:.3f} to forensic")
    except Exception as e:
        logger.warning(f"Watermark extraction failed: {e}")
        print(f"[WATERMARK ERROR] {e}")

    logger.warning(f"Forensic -> freq={freq:.3f}, gan={gan:.3f}, color={color:.3f}, combined={forensic:.3f}")

    probs = {p["model_name"]: p["fake_probability"] for p in preds}

    cnn_ensemble = (
        0.20 * probs["efficientnet_b3"] +
        0.10 * probs["xception"] +
        0.35 * probs["convnext"] +
        0.35 * probs["vit"]
    )

    logger.warning(f"CNN ensemble (raw) -> {cnn_ensemble:.3f}")

    # Initial fusion
    if STRICT_MODE:
        ensemble = 0.50 * cnn_ensemble + 0.50 * forensic
    else:
        ensemble = 0.60 * cnn_ensemble + 0.40 * forensic
    
    ensemble = float(np.clip(ensemble, 0.05, 0.95))
    logger.warning(f"Initial ensemble score -> {ensemble:.3f}")

    # Decision logic
    FAKE_THRESHOLD = 0.55 if STRICT_MODE else 0.65
    cnn_forensic_disagreement = abs(cnn_ensemble - forensic)
    logger.warning(f"CNN-Forensic disagreement: {cnn_forensic_disagreement:.3f}")

    label = None
    confidence = None
    decision_source = None

    # Priority 0.5 - WATERMARK strong override (new!)
    if watermark_prob >= 0.50:
        logger.warning("*** WATERMARK HIGH-CONF OVERRIDE TRIGGERED ***")
        logger.warning(f"   Watermark prob = {watermark_prob:.3f} → Forcing Fake")
        label = "Fake"
        confidence = max(confidence or 0, watermark_prob)
        decision_source = "watermark_high_conf_override"

    # Priority 0 - CNN high-confidence override
    vit_prob = probs["vit"]
    if label is None and vit_prob >= 0.92 and cnn_ensemble >= 0.92 and forensic >= 0.55:
        logger.warning("*** HIGH-CONFIDENCE CNN OVERRIDE TRIGGERED ***")
        logger.warning(f"   ViT={vit_prob:.3f}, CNN ensemble={cnn_ensemble:.3f} → Forcing Fake")
        label = "Fake"
        confidence = cnn_ensemble
        decision_source = "cnn_high_conf_override"

    # Priority 1: Forensic override
    if label is None and forensic >= 0.45 and cnn_ensemble < 0.40:
        logger.warning("*** FORENSIC OVERRIDE TRIGGERED ***")
        label = "Fake"
        base_confidence = 0.65 * forensic + 0.35 * ensemble
        if forensic >= 0.55:
            base_confidence = min(0.90, base_confidence * 1.15)
        confidence = min(0.85, base_confidence)
        decision_source = "forensic_override"

    # Priority 1.5 - WATERMARK strong safety push to Real (if very low prob)
    if label is None and watermark_prob <= 0.12 and ensemble > 0.50:
        logger.warning(f"*** WATERMARK SAFETY PUSH TO REAL *** prob={watermark_prob:.3f}")
        ensemble = min(ensemble, 0.48)  # pull back from Fake
        decision_source = decision_source or "watermark_safety_pull_real"

    # Priority 2: High disagreement
    if label is None and cnn_forensic_disagreement > 0.20:
        logger.warning(f"*** HIGH DISAGREEMENT ({cnn_forensic_disagreement:.3f})")
        if cnn_ensemble > 0.60 and forensic < 0.55:
            ensemble_adjusted = 0.50 * cnn_ensemble + 0.50 * forensic
            ensemble = ensemble_adjusted * 0.80
            logger.warning(f"   Adjusted ensemble: {ensemble:.3f}")
        else:
            ensemble = 0.50 * cnn_ensemble + 0.50 * forensic
            ensemble = float(np.clip(ensemble, 0.05, 0.95))
        label = None

    if label is None:
        logger.warning(f"Final ensemble score -> {ensemble:.3f}")
        
        if abs(ensemble - FAKE_THRESHOLD) < 0.10:
            logger.warning(f"*** BORDERLINE CASE: {ensemble:.3f} near {FAKE_THRESHOLD}")
            
            if ensemble >= FAKE_THRESHOLD and forensic < 0.45:
                logger.warning("   Borderline fake but forensics weak - Reclassified Real")
                label = "Real"
                confidence = max(0.55, 1 - ensemble)
                decision_source = "borderline_protection_real"
            
            elif ensemble < FAKE_THRESHOLD and forensic >= 0.50:
                logger.warning("   Borderline real but forensics strong - Reclassified Fake")
                label = "Fake"
                confidence = max(0.55, forensic)
                decision_source = "borderline_protection_fake"
            
            else:
                if ensemble >= FAKE_THRESHOLD:
                    label = "Fake"
                    confidence = ensemble
                    decision_source = "ensemble_fake_borderline"
                else:
                    label = "Real"
                    confidence = 1 - ensemble
                    decision_source = "ensemble_real_borderline"

        elif ensemble < 0.45 and forensic < 0.45 and not portrait_gan_suspect:
            logger.warning("Relaxed real-safety veto triggered")
            label = "Real"
            confidence = 1 - ensemble
            decision_source = "real_safety_veto_relaxed"
        
        else:
            if ensemble >= FAKE_THRESHOLD:
                label = "Fake"
                confidence = ensemble
                decision_source = "ensemble_fake"
            else:
                label = "Real"
                confidence = 1 - ensemble
                decision_source = "ensemble_real"

    logger.warning(f"*** FINAL DECISION: {label} ({confidence:.3f}) via '{decision_source}'")

    # GRAD-CAM: Use consensus-based winner
    winner = min(preds, key=lambda x: abs(x["fake_probability"] - cnn_ensemble))
    logger.warning(f"Winning model for Grad-CAM: {winner['model_name']} (closest to ensemble {cnn_ensemble:.3f})")
    model, layer, cfg = models[winner["model_name"]]

    gradcam_url = generate_gradcam(
        model, layer, face, full,
        cfg["input_size"], label,
        winner["model_name"], confidence, request
    )

    # CONSOLE SUMMARY (ASCII-safe)
    print("\n" + "="*70)
    print(f"DEEPFAKE DETECTION RESULT - {label.upper()} ({confidence:.1%} confidence)")
    print("-"*70)
    print(f"Decision came from : {decision_source}")
    print(f"Winning model      : {winner['model_name']}")
    print()

    print("CNN Predictions:")
    for p in sorted(preds, key=lambda x: x["fake_probability"], reverse=True):
        prob = p["fake_probability"]
        bar = "#" * int(prob * 20) + "-" * (20 - int(prob * 20))
        print(f"  {p['model_name']:<12} : {prob:.1%}  [{bar}]  {p['label']}")

    print("\nForensic Signals:")
    print(f"  Frequency artifacts  : {freq:.3f}   {'high' if freq > 0.50 else 'med' if freq > 0.35 else 'low'}")
    print(f"  GAN fingerprint      : {gan:.3f}   {'strong' if gan > 0.40 else 'moderate' if gan > 0.20 else 'weak'}")
    print(f"  Color entropy anomaly: {color:.3f}   {'suspicious' if color > 0.50 else 'mild' if color > 0.35 else 'normal'}")
    print(f"  Combined forensic    : {forensic:.3f}")
    if portrait_gan_suspect:
        print("Portrait-style GAN heuristic activated (+0.10 boost)")

    if watermark_prob > 0.0:
        print(f"  Watermark probability: {watermark_prob:.4f}   {'STRONG' if watermark_detected else 'low/weak'}")

    print("\nEnsemble scores:")
    print(f"  CNN ensemble         : {cnn_ensemble:.3f}")
    print(f"  Final fused score    : {ensemble:.3f}  - {label}")
    print("="*70 + "\n")

    # ────────────────────────────────────────────────
    # CSV Logging - updated with watermark columns
    # ────────────────────────────────────────────────
    try:
        csv_dir = os.path.dirname(CSV_PATH)
        print(f"[CSV DEBUG] Target path: {CSV_PATH}")
        print(f"[CSV DEBUG] Directory: {csv_dir}")
        print(f"[CSV DEBUG] Dir exists? {os.path.exists(csv_dir)}")

        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
            print(f"[CSV DEBUG] Created missing directory")

        header = [
            "image_path", "true_label", "predicted_label", "confidence",
            "ensemble_score", "cnn_ensemble", "forensic_score",
            "freq_artifact", "gan_fingerprint", "color_entropy",
            "portrait_gan_boost", "winning_model", "decision_source",
            "vit_prob", "convnext_prob", "eff_b3_prob", "xception_prob",
            "watermark_prob", "watermark_detected"   # ← new columns
        ]

        file_exists = os.path.exists(CSV_PATH)
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
                print(f"[CSV DEBUG] Wrote header")

            row = [
                os.path.basename(img_path),
                true_label,
                label,
                round(confidence, 4) if confidence is not None else None,
                round(ensemble, 4),
                round(cnn_ensemble, 4),
                round(forensic, 4),
                round(freq, 4),
                round(gan, 4),
                round(color, 4),
                "Yes" if portrait_gan_suspect else "No",
                winner["model_name"] if 'winner' in locals() else "N/A",
                decision_source if decision_source else "N/A",
                round(probs.get("vit", 0), 4),
                round(probs.get("convnext", 0), 4),
                round(probs.get("efficientnet_b3", 0), 4),
                round(probs.get("xception", 0), 4),
                round(watermark_prob, 4),
                "Yes" if watermark_detected else "No"
            ]

            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

        print(f"[CSV DEBUG] Successfully appended row for {os.path.basename(img_path)}")

    except Exception as csv_err:
        print(f"[CSV CRITICAL ERROR] {str(csv_err)}")
        print(f"[CSV CRITICAL ERROR] Path attempted: {CSV_PATH}")
        logger.exception("CSV write failed completely")

    # BAR PLOT
    plot_url = None
    try:
        models_list = ["eff_b3", "xception", "convnext", "vit", "CNN ens.", "Forensic", "Final"]
        values = [
            probs["efficientnet_b3"],
            probs["xception"],
            probs["convnext"],
            probs["vit"],
            cnn_ensemble,
            forensic,
            ensemble
        ]
        colors = ["#3498db"] * 4 + ["#e67e22", "#e74c3c" if ensemble >= 0.5 else "#2ecc71"]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(models_list, values, color=colors, height=0.65)

        ax.set_xlim(0, 1)
        ax.set_xlabel("Fake probability / score")
        ax.set_title(f"Deepfake Analysis — {label} ({confidence:.1%})", fontsize=14, pad=15)
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{width:.3f}", va="center", fontsize=10)

        ax.axvline(FAKE_THRESHOLD, color="gray", linestyle="--", alpha=0.7, linewidth=1.2)
        ax.text(FAKE_THRESHOLD + 0.01, 0.02, f"threshold {FAKE_THRESHOLD}", color="gray", fontsize=9)

        plt.tight_layout()

        plot_dir = os.path.join(settings.MEDIA_ROOT, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = f"analysis_{uuid.uuid4().hex[:12]}.png"
        plot_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        plot_url = request.build_absolute_uri(settings.MEDIA_URL + "plots/" + plot_filename)
        logger.info(f"Saved bar plot - {plot_url}")

    except Exception as e:
        logger.error(f"Failed to generate bar plot: {str(e)}")

    # RADAR PLOT
    radar_url = plot_radar_summary(
        freq=freq,
        gan=gan,
        color=color,
        cnn_ensemble=cnn_ensemble,
        label=label,
        confidence=confidence,
        request=request
    )

    # FINAL RESPONSE
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "gradcam_url": gradcam_url,
        "plot_url": plot_url,
        "radar_url": radar_url,
        "winning_model": winner["model_name"],
        "decision_source": decision_source,
        "forensic_scores": {
            "frequency": round(freq, 4),
            "gan_fingerprint": round(gan, 4),
            "color_entropy": round(color, 4),
            "combined": round(forensic, 4)
        },
        "cnn_scores": {
            "ensemble": round(cnn_ensemble, 4),
            "efficientnet_b3": round(probs["efficientnet_b3"], 4),
            "xception": round(probs["xception"], 4),
            "convnext": round(probs["convnext"], 4),
            "vit": round(probs["vit"], 4)
        },
        "all_predictions": preds
    }