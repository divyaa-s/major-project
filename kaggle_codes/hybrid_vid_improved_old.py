"""
hybrid_vid_improved.py - FIXED & ENSEMBLE VERSION (Updated Feb 2026)

Changes vs previous version:
- FIXED: _load_cnn_models was iterating over `convnext_model` (a torch.nn.Module object)
  instead of `MODEL_CONFIGS` (the dict). This silently prevented ALL models from loading.
- REMOVED: Redundant global convnext_model/convnext_transform block (lines 32-43 in old
  version). The class loads all models via MODEL_CONFIGS — the global was unused and would
  crash if the path was wrong.
- ENSEMBLE: All 4 models (EfficientNet-B3, Xception, ViT, ConvNeXt) now run on every
  keyframe and scores are logged per-model in the console.
- Per-model normalization: each model now gets its own correct ImageNet or 0.5-centered
  normalization based on how it was trained.
- Ensemble weights are now equal (0.25 each) instead of ConvNeXt-dominated (0.55).
  Override via MODEL_ENSEMBLE_WEIGHTS if you want custom weighting.
- Fusion logic: `convnext_score` guard replaced with `top_model_score` (max across all
  loaded models) so the logic works even if ConvNeXt fails to load.
- Console output: per-model scores printed during analysis for easy inspection.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import logging
from pathlib import Path
from PIL import Image
from torchvision import transforms
from timm import create_model
from facenet_pytorch import MTCNN
from tqdm import tqdm
from temporal_analysis import TemporalAnalyzer
from quality_forensic import QualityForensicsAnalyzer

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model registry
# Each entry defines the model architecture, checkpoint path, input size,
# temperature scaling value, and the normalization stats used during training.
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "efficientnet_b3": {
        "path": "/kaggle/input/models/divyaasriram/models/pytorch/default/1/models/effb3_best_final.pth",
        "input_size": (300, 300),
        "model_name": "efficientnet_b3",
        "num_classes": 1,          # Trained with BCEWithLogitsLoss / sigmoid output
        "temperature": 3.5,
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std":  [0.229, 0.224, 0.225],
    },
    "xception": {
        "path": "/kaggle/input/models/divyaasriram/models/pytorch/default/1/models/xceptionnet_best.pth",
        "input_size": (299, 299),
        "model_name": "legacy_xception",
        "num_classes": 1,          # Trained with BCEWithLogitsLoss / sigmoid output
        "temperature": 4.0,
        "norm_mean": [0.5, 0.5, 0.5],
        "norm_std":  [0.5, 0.5, 0.5],
    },
    "vit": {
        # NOTE: filename has a space — make sure the actual file on disk matches exactly
        "path": "/kaggle/input/models/divyaasriram/models/pytorch/default/1/models/vit_best_final.pth",
        "input_size": (224, 224),
        "model_name": "vit_small_patch16_224",  # dim=384 — matches checkpoint (was vit_tiny=192)
        "num_classes": 1,
        "temperature": 4.5,
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std":  [0.229, 0.224, 0.225],
    },
    "convnext": {
        "path": "/kaggle/input/models/divyaasriram/models/pytorch/default/1/models/convnext_best.pth",
        "input_size": (224, 224),
        "model_name": "convnext_small",
        "num_classes": 1,          # Trained with BCEWithLogitsLoss / sigmoid output
        "temperature": 3.0,
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std":  [0.229, 0.224, 0.225],
    },
}

# ---------------------------------------------------------------------------
# Ensemble weights — adjust to taste.
# These are normalised internally so they don't need to sum to 1.0.
# Set all equal for a pure unweighted ensemble.
# ---------------------------------------------------------------------------
MODEL_ENSEMBLE_WEIGHTS = {
    "efficientnet_b3": 0.25,
    "xception":        0.25,
    "vit":             0.25,
    "convnext":        0.25,
}


class ImprovedHybridAnalyzer:
    """
    Improved hybrid analyzer with full 4-model ensemble and per-model console logging.
    """

    def __init__(self, num_keyframes=100, temporal_max_frames=30, temporal_skip=2,
                 smart_keyframe_selection=True):
        self.num_keyframes = num_keyframes
        self.smart_keyframe_selection = smart_keyframe_selection
        self.faces_detected_count = 0

        self.temporal_analyzer = TemporalAnalyzer(
            max_frames=temporal_max_frames,
            skip_frames=temporal_skip
        )
        self.quality_analyzer = QualityForensicsAnalyzer()

        self._init_face_detectors()
        self.models = {}
        self._load_cnn_models()

        logger.info(f"Initialized ImprovedHybridAnalyzer with {len(self.models)} CNN model(s)")
        logger.info(f"Loaded models: {list(self.models.keys())}")
        logger.info(f"Smart keyframe selection: {self.smart_keyframe_selection}")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_face_detectors(self):
        try:
            # Force CPU — facenet_pytorch 2.2.9 has a CUDA index bug with torch 2.9+
            self.mtcnn = MTCNN(keep_all=False, device="cpu")
            self.mtcnn_available = True
            logger.info("✅ MTCNN initialized (CPU — CUDA disabled due to torch 2.9 compatibility)")
        except Exception as e:
            logger.warning(f"⚠️  MTCNN initialization failed: {e}")
            self.mtcnn_available = False

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.error("❌ Failed to load Haar Cascade")
        else:
            logger.info("✅ OpenCV Haar Cascade initialized")


    def _load_cnn_models(self):
        """
        Load all CNN models defined in MODEL_CONFIGS.

        BUG FIXED: The original code iterated over `convnext_model` (a torch.nn.Module
        which has no .items() method in the dict sense) instead of `MODEL_CONFIGS`.
        This caused a crash or silently loaded nothing. Now correctly iterates over the
        MODEL_CONFIGS dictionary.

        A startup summary table is printed to console after all models are attempted,
        so you can immediately see which models loaded and which failed (with reasons).
        """
        load_results = {}   # name -> ("ok", None) or ("fail", error_str)

        for name, config in MODEL_CONFIGS.items():   # <-- was: convnext_model.items()
            try:
                model = create_model(
                    config["model_name"],
                    pretrained=False,
                    num_classes=config.get("num_classes", 1)
                )

                checkpoint = torch.load(config["path"], map_location=DEVICE)

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                        logger.info(f"  [{name}] Extracted 'model_state_dict' key")
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                        logger.info(f"  [{name}] Extracted 'state_dict' key")
                    else:
                        state_dict = checkpoint
                        logger.info(f"  [{name}] Using checkpoint directly as state_dict")
                else:
                    state_dict = checkpoint

                # Remove 'module.' prefix if saved from DataParallel
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

                model.load_state_dict(state_dict, strict=False)
                model.to(DEVICE).eval()

                self.models[name] = {
                    "model": model,
                    "config": config,
                }
                load_results[name] = ("ok", None)

            except Exception as e:
                load_results[name] = ("fail", str(e))
                logger.error(f"❌ Failed to load [{name}]: {e}")

        # ── Startup model health check (always printed to console) ────────
        print("\n" + "=" * 65)
        print("🧠 CNN MODEL LOAD STATUS")
        print("=" * 65)
        for name, (status, err) in load_results.items():
            cfg = MODEL_CONFIGS[name]
            if status == "ok":
                weight = MODEL_ENSEMBLE_WEIGHTS.get(name, 0.25)
                nc = MODEL_CONFIGS[name].get("num_classes", 1)
                print(f"  ✅ {name:<20s}  arch={cfg['model_name']:<25s}  num_classes={nc}  weight={weight:.2f}")
            else:
                # Truncate very long error messages for readability
                short_err = err if len(err) < 90 else err[:87] + "..."
                print(f"  ❌ {name:<20s}  FAILED — {short_err}")
                print(f"     path: {cfg['path']}")
        print(f"\n  Loaded {len(self.models)}/{len(MODEL_CONFIGS)} models  |  Device: {DEVICE}")
        print("=" * 65 + "\n")

        if len(self.models) == 0:
            logger.error("NO CNN MODELS LOADED — CNN analysis will return 0.5 (neutral) for all frames")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze_video(self, video_path):
        self.faces_detected_count = 0
        logger.info(f"🎬 Starting improved hybrid analysis: {video_path}")

        # ── Single pass replaces 3 separate video reads ───────────────
        temporal_frames, keyframes = self._single_pass_extract(video_path)

        if not temporal_frames and not keyframes:
            return {"error": "No faces detected in video"}

        # ── Temporal analysis (feed pre-extracted frames directly) ────
        temporal_results = self.temporal_analyzer.analyze_from_frames(
            temporal_frames, video_path
        )
        if "error" in temporal_results:
            return temporal_results

        if not keyframes:
            return {"error": "No keyframes could be extracted"}

        # ── CNN + Quality (unchanged) ──────────────────────────────────
        logger.info("🤖 Running CNN ensemble on keyframes...")
        cnn_results = self._analyze_keyframes_with_cnns(keyframes)

        logger.info("🔬 Running quality forensics analysis...")
        quality_results = self._analyze_quality_mismatch(keyframes)

        logger.info("⚡ Fusing predictions...")
        final_results = self._fuse_predictions(temporal_results, cnn_results, quality_results)

        return final_results

    # ------------------------------------------------------------------
    # Keyframe extraction
    # ------------------------------------------------------------------

    def _extract_smart_keyframes(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return []

        candidate_count = min(self.num_keyframes * 3, total_frames)
        candidate_indices = np.linspace(0, total_frames - 1, candidate_count, dtype=int)

        candidates = []
        for idx in tqdm(candidate_indices, desc="🎞 Extracting Keyframes", leave=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            face, quality_score, face_box = self._crop_face_with_quality(frame)
            if face is not None:
                self.faces_detected_count += 1
                candidates.append({
                    "frame_index": int(idx),
                    "face": face,
                    "frame": frame,
                    "quality": quality_score,
                    "face_box": face_box,
                })

        cap.release()

        if not candidates:
            logger.warning("No face candidates found")
            return []

        candidates.sort(key=lambda x: x["quality"], reverse=True)
        selected = candidates[: self.num_keyframes]
        logger.info(f"   Selected {len(selected)} highest-quality keyframes from {len(candidates)} candidates")
        return selected

    def _extract_keyframes_even(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return []

        indices = np.linspace(0, total_frames - 1, self.num_keyframes, dtype=int)

        keyframes = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            face, quality_score, face_box = self._crop_face_with_quality(frame)
            if face is not None:
                self.faces_detected_count += 1
                keyframes.append({
                    "frame_index": int(idx),
                    "face": face,
                    "frame": frame,
                    "quality": quality_score,
                    "face_box": face_box,
                })

        cap.release()
        logger.info(f"   Extracted {len(keyframes)}/{self.num_keyframes} frames with faces")
        return keyframes

    # ------------------------------------------------------------------
    # Face detection & preprocessing
    # ------------------------------------------------------------------

    def _crop_face_with_quality(self, frame):
        """Detect face, crop it, and return a quality estimate."""
        if self.mtcnn_available:
            try:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                boxes, probs = self.mtcnn.detect(img_pil)
                if boxes is None or len(boxes) == 0:
                    return None, 0.0, None
                box = boxes[0].astype(int)
                x1, y1, x2, y2 = box
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
                face = img_rgb[y1:y2, x1:x2]
                quality = self._estimate_face_quality(face)
                return face, quality, (x1, y1, x2, y2)
            except Exception as e:
                logger.debug(f"MTCNN failed: {e}")

        # Fallback: Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces_detected) == 0:
            return None, 0.0, None
        x, y, w, h = faces_detected[0]
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        quality = self._estimate_face_quality(face)
        return face, quality, (x1, y1, x2, y2)

    def _estimate_face_quality(self, face):
        """Laplacian-variance-based sharpness score, normalised to [0, 1]."""
        if face.size == 0:
            return 0.0
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(laplacian_var / 500.0, 1.0)

    def _preprocess_face(self, face, size, norm_mean, norm_std):
        """
        Resize, convert to tensor, and normalise a face crop.

        Each model gets its own normalization stats (passed in from MODEL_CONFIGS)
        so that ImageNet-trained models and 0.5-centred models are handled correctly.
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        return transform(face).unsqueeze(0).to(DEVICE)

    # ------------------------------------------------------------------
    # CNN ensemble inference
    # ------------------------------------------------------------------

    def _analyze_keyframes_with_cnns(self, keyframes):
        if not self.models:
            logger.error("No CNN models loaded — cannot run CNN analysis")
            return {
                "frame_predictions": [],
                "ensemble": {
                    "avg_fake_probability": 0.5,
                    "per_model_avg": {},
                    "frames_analyzed": 0,
                },
            }

        # Pre-initialize predictions dict for all frames
        frame_predictions = [
            {
                "frame_index": kf["frame_index"],
                "model_predictions": {},
            }
            for kf in keyframes
        ]

        for name, model_info in self.models.items():
            model  = model_info["model"]
            config = model_info["config"]
            temp   = config.get("temperature", 1.0)
            BATCH_SIZE = 16  # Safe for T4 — increase to 32 if no OOM

            try:
                # ── Step 1: Preprocess all frames for this model ──────────
                tensors = []
                valid_indices = []  # track which keyframe indices succeeded

                for i, kf in enumerate(keyframes):
                    try:
                        x = self._preprocess_face(
                            kf["face"],
                            config["input_size"],
                            config["norm_mean"],
                            config["norm_std"],
                        )
                        tensors.append(x.squeeze(0))  # remove batch dim for stacking
                        valid_indices.append(i)
                    except Exception as e:
                        logger.warning(f"  [{name}] Preprocessing failed for frame {kf['frame_index']}: {e}")

                if not tensors:
                    logger.warning(f"  [{name}] No frames preprocessed successfully")
                    continue

                # ── Step 2: Run batched inference ─────────────────────────
                all_probs = []

                for batch_start in range(0, len(tensors), BATCH_SIZE):
                    batch_tensors = tensors[batch_start : batch_start + BATCH_SIZE]
                    batch = torch.stack(batch_tensors).to(DEVICE)  # (B, C, H, W)

                    with torch.no_grad():
                        output = model(batch)

                        if temp != 1.0:
                            output = output / temp

                        if config.get("num_classes", 1) == 1:
                            probs = torch.sigmoid(output).squeeze(-1)  # (B,)
                        else:
                            probs = F.softmax(output, dim=1)[:, 1]     # (B,)

                    all_probs.extend(probs.cpu().tolist())

                # ── Step 3: Write scores back to frame_predictions ────────
                for idx, prob in zip(valid_indices, all_probs):
                    frame_predictions[idx]["model_predictions"][name] = float(prob)

                logger.info(f"  [{name}] Batched inference complete — {len(all_probs)} frames")

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"  [{name}] OOM with BATCH_SIZE={BATCH_SIZE} — retrying with BATCH_SIZE=8")
                # Retry with smaller batch
                try:
                    all_probs = []
                    for batch_start in range(0, len(tensors), 8):
                        batch = torch.stack(tensors[batch_start : batch_start + 8]).to(DEVICE)
                        with torch.no_grad():
                            output = model(batch)
                            if temp != 1.0:
                                output = output / temp
                            if config.get("num_classes", 1) == 1:
                                probs = torch.sigmoid(output).squeeze(-1)
                            else:
                                probs = F.softmax(output, dim=1)[:, 1]
                        all_probs.extend(probs.cpu().tolist())

                    for idx, prob in zip(valid_indices, all_probs):
                        frame_predictions[idx]["model_predictions"][name] = float(prob)

                except Exception as e:
                    logger.error(f"  [{name}] Retry also failed: {e}")

            except Exception as e:
                logger.error(f"  [{name}] Batched inference failed: {e}")

        cnn_ensemble = self._calculate_cnn_ensemble(frame_predictions)

        # ── Console summary ───────────────────────────────────────────────
        print("\n" + "─" * 60)
        print("🤖 CNN ENSEMBLE — PER-MODEL AVERAGE SCORES")
        print("─" * 60)
        for model_name, avg_score in cnn_ensemble["per_model_avg"].items():
            weight = MODEL_ENSEMBLE_WEIGHTS.get(model_name, 0.25)
            bar  = "█" * int(avg_score * 30)
            flag = "  ⚠️  EXTREME" if avg_score < 0.05 or avg_score > 0.95 else ""
            print(f"  {model_name:<20s}  score={avg_score:.4f}  weight={weight:.2f}  |{bar:<30s}|{flag}")
        print(f"\n  Weighted Ensemble Score : {cnn_ensemble['avg_fake_probability']:.4f}")
        print(f"  Frames analysed        : {cnn_ensemble['frames_analyzed']}")
        print("─" * 60 + "\n")

        return {
            "frame_predictions": frame_predictions,
            "ensemble": cnn_ensemble,
        }

    def _calculate_cnn_ensemble(self, frame_predictions):
        """
        Compute per-model averages across all frames, then compute a
        weighted ensemble score using MODEL_ENSEMBLE_WEIGHTS.
        """
        if not frame_predictions:
            return {
                "avg_fake_probability": 0.5,
                "per_model_avg": {},
                "frames_analyzed": 0,
            }

        # Aggregate per-model scores across frames
        model_scores = {name: [] for name in self.models.keys()}
        for frame_pred in frame_predictions:
            for model_name, fake_prob in frame_pred["model_predictions"].items():
                if model_name in model_scores:
                    model_scores[model_name].append(fake_prob)

        per_model_avg = {
            name: float(np.mean(scores))
            for name, scores in model_scores.items()
            if scores
        }

        if not per_model_avg:
            return {
                "avg_fake_probability": 0.5,
                "per_model_avg": {},
                "frames_analyzed": len(frame_predictions),
            }

        # Compute weighted ensemble (only for models that successfully ran)
        total_weight = sum(
            MODEL_ENSEMBLE_WEIGHTS.get(name, 0.25)
            for name in per_model_avg
        )
        if total_weight == 0:
            total_weight = len(per_model_avg)

        cnn_ensemble_score = sum(
            (MODEL_ENSEMBLE_WEIGHTS.get(name, 0.25) / total_weight) * score
            for name, score in per_model_avg.items()
        )

        return {
            "avg_fake_probability": float(round(cnn_ensemble_score, 4)),
            "per_model_avg": per_model_avg,
            "frames_analyzed": len(frame_predictions),
        }

    # ------------------------------------------------------------------
    # Quality forensics
    # ------------------------------------------------------------------

    def _analyze_quality_mismatch(self, keyframes):
        frames_with_boxes = []

        for kf in keyframes:
            frame    = kf["frame"]
            face_box = kf.get("face_box")

            if face_box is not None:
                frames_with_boxes.append((frame, face_box))
            else:
                # Fallback: approximate face region from frame centre
                h, w = frame.shape[:2]
                face_h = int(h * 0.4)
                face_w = int(w * 0.3)
                y1 = int(h * 0.2)
                y2 = y1 + face_h
                x1 = int(w * 0.35)
                x2 = x1 + face_w
                frames_with_boxes.append((frame, (x1, y1, x2, y2)))

        quality_results = self.quality_analyzer.analyze_keyframes(frames_with_boxes)
        logger.info(f"   Quality mismatch score: {quality_results['avg_quality_mismatch']:.4f}")
        return quality_results
    
    def _single_pass_extract(self, video_path):
        """
        Read the video ONCE and collect all frames needed for:
        - Temporal analysis (30 evenly spaced frames)
        - Keyframe selection (up to num_keyframes * 3 candidates)
        - Quality forensics (reuses keyframes)

        Returns frames_for_temporal, frames_for_keyframes
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0:
            cap.release()
            return [], []

        # ── Calculate which frame indices each component needs ────────
        temporal_count    = self.temporal_analyzer.max_frames
        candidate_count   = min(self.num_keyframes * 3, total_frames)

        temporal_indices  = set(np.linspace(0, total_frames - 1, temporal_count,  dtype=int).tolist())
        keyframe_indices  = set(np.linspace(0, total_frames - 1, candidate_count, dtype=int).tolist())

        all_indices = sorted(temporal_indices | keyframe_indices)

        # ── Single sequential read ─────────────────────────────────────
        raw_frames = {}  # index -> frame (BGR numpy)

        print(f"   📽️  Reading {len(all_indices)} unique frames (was 3 separate passes)...")

        for idx in tqdm(all_indices, desc="🎞 Single-pass read", leave=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                raw_frames[idx] = frame

        cap.release()

        # ── Batch MTCNN detection across ALL frames at once ───────────
        print(f"   🔍 Batch MTCNN detection on {len(raw_frames)} frames...")

        all_pil_images = []
        all_frame_keys = []

        for idx, frame in raw_frames.items():
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            all_pil_images.append(img_pil)
            all_frame_keys.append(idx)

        # MTCNN batch detect — processes all frames in one call
        # Returns list of (boxes, probs) per image
        MTCNN_BATCH = 32
        all_boxes = []

        for b_start in range(0, len(all_pil_images), MTCNN_BATCH):
            batch_imgs = all_pil_images[b_start : b_start + MTCNN_BATCH]
            try:
                # mtcnn.detect() accepts a list of PIL images
                batch_boxes, batch_probs = self.mtcnn.detect(batch_imgs)
                all_boxes.extend(batch_boxes if batch_boxes is not None else [None] * len(batch_imgs))
            except Exception as e:
                logger.warning(f"Batch MTCNN failed: {e} — using None for batch")
                all_boxes.extend([None] * len(batch_imgs))

        # ── Build face_box lookup ──────────────────────────────────────
        face_boxes = {}  # frame_index -> (x1, y1, x2, y2) or None

        for idx, boxes in zip(all_frame_keys, all_boxes):
            frame = raw_frames[idx]
            h, w  = frame.shape[:2]

            if boxes is not None and len(boxes) > 0 and boxes[0] is not None:
                x1, y1, x2, y2 = map(int, boxes[0])
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                if x2 > x1 and y2 > y1:
                    face_boxes[idx] = (x1, y1, x2, y2)
                    continue

            # Haar fallback for this frame
            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces  = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) > 0:
                x, y, fw, fh = faces[0]
                padding = 20
                face_boxes[idx] = (
                    max(0, x - padding),
                    max(0, y - padding),
                    min(w, x + fw + padding),
                    min(h, y + fh + padding),
                )
            else:
                face_boxes[idx] = None

        # ── Build temporal frames list ─────────────────────────────────
        temporal_frames = []
        for idx in sorted(temporal_indices):
            if idx not in raw_frames:
                continue
            frame    = raw_frames[idx]
            face_box = face_boxes.get(idx)
            if face_box is None:
                continue

            x1, y1, x2, y2 = face_box
            face       = frame[y1:y2, x1:x2]
            eye_regions = self.temporal_analyzer._detect_eye_regions(frame, face_box)

            temporal_frames.append({
                "frame_index": int(idx),
                "frame":       frame,
                "face":        face,
                "face_box":    face_box,
                "eye_regions": eye_regions,
            })

        # ── Build keyframe candidates list ────────────────────────────
        keyframe_candidates = []
        for idx in sorted(keyframe_indices):
            if idx not in raw_frames:
                continue
            frame    = raw_frames[idx]
            face_box = face_boxes.get(idx)
            if face_box is None:
                continue

            x1, y1, x2, y2 = face_box
            face_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            quality  = self._estimate_face_quality(face_rgb)

            keyframe_candidates.append({
                "frame_index": int(idx),
                "frame":       frame,
                "face":        face_rgb,
                "face_box":    face_box,
                "quality":     quality,
            })

        # Sort by quality and select top N keyframes
        keyframe_candidates.sort(key=lambda x: x["quality"], reverse=True)
        keyframes = keyframe_candidates[: self.num_keyframes]

        self.faces_detected_count = len(temporal_frames) + len(keyframes)

        print(f"   ✅ Single pass complete: {len(temporal_frames)} temporal frames, "
            f"{len(keyframes)} keyframes from {len(keyframe_candidates)} candidates")

        return temporal_frames, keyframes
    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _fuse_predictions(self, temporal_results, cnn_results, quality_results):
        temporal_score = temporal_results["temporal_consistency_score"]
        cnn_score      = cnn_results["ensemble"]["avg_fake_probability"]
        quality_score  = quality_results["avg_quality_mismatch"]

        blink_score    = temporal_results["blink_analysis"]["score"]
        landmark_score = temporal_results["landmark_stability"]["score"]
        flow_score     = temporal_results["optical_flow_analysis"]["score"]

        base_fusion    = 0.40 * temporal_score + 0.40 * cnn_score + 0.20 * quality_score
        final_score    = base_fusion
        decision_source = "hybrid_ensemble"

        per_model = cnn_results["ensemble"]["per_model_avg"]

        top_model_score  = max(per_model.values()) if per_model else 0.5
        convnext_score   = per_model.get("convnext", top_model_score)
        xception_score   = per_model.get("xception", top_model_score)
        effnet_score     = per_model.get("efficientnet_b3", top_model_score)
        vit_score        = per_model.get("vit", top_model_score)

        # ── PRIORITY 1: Strong FAKE signal ────────────────────────────────
        # Trigger only when quality confirms manipulation AND top model is extreme
        if quality_score > 0.05 and top_model_score > 0.90:
            final_score = 0.05 * temporal_score + 0.45 * cnn_score + 0.50 * quality_score
            decision_source = "ensemble_quality_boost_fake_strong"
            logger.info(
                f"Strong fake: top_model={top_model_score:.4f} convnext={convnext_score:.4f} "
                f"quality={quality_score:.4f}"
            )

        # ── PRIORITY 2: Strong REAL override ──────────────────────────────
        elif (
            temporal_score < 0.45
            and quality_score < 0.08
            and cnn_score < 0.70
            and blink_score < 0.75
            and landmark_score < 0.35
            and convnext_score < 0.80
        ):
            final_score = 0.20 * cnn_score + 0.65 * temporal_score + 0.15 * quality_score
            decision_source = "temporal_quality_real_override"
            logger.info("Strong real signal → overriding CNN ensemble confidence")

        # ── PRIORITY 3: Xception anchor — real video protection ──────────
        # Xception is the most conservative / least overconfident model.
        # When quality is clean AND xception is the only skeptical model,
        # the other models may be overfitting to compression artifacts.
        # Pull the score down to avoid false positives on real videos.
        elif (
            quality_score < 0.20
            and xception_score < 0.70          # xception clearly not convinced (< 0.70 = truly skeptical)
            and cnn_score >= 0.70              # but ensemble still flags it
        ):
            # Weight xception more heavily as a "real" anchor
            anchored_cnn = (
                0.15 * effnet_score +
                0.45 * xception_score +        # xception gets majority weight
                0.20 * vit_score +
                0.20 * convnext_score
            )
            final_score = 0.40 * temporal_score + 0.40 * anchored_cnn + 0.20 * quality_score
            decision_source = "xception_anchor_real_protection"
            logger.info(
                f"Xception anchor: xception={xception_score:.4f} anchored_cnn={anchored_cnn:.4f} "
                f"quality={quality_score:.4f}"
            )

        # ── Other decision pathways ────────────────────────────────────────
        elif temporal_score >= 0.60 and cnn_score >= 0.65:
            final_score = max(base_fusion, 0.62)
            decision_source = "strong_agreement_fake"

        elif cnn_score >= 0.75 and temporal_score < 0.50:
            final_score = 0.55 * cnn_score + 0.45 * temporal_score
            decision_source = "cnn_override"

        elif temporal_score >= 0.65 and cnn_score < 0.50:
            final_score = 0.60 * temporal_score + 0.40 * cnn_score
            decision_source = "temporal_override"

        elif abs(temporal_score - cnn_score) > 0.35:
            final_score = base_fusion * 0.88
            decision_source = "disagreement_uncertain"

        final_score = float(np.clip(final_score, 0.0, 1.0))

        # ── Global Real Confidence Guard ───────────────────────────────────
        # When CNN is high but quality is very clean, pull toward temporal.
        # IMPORTANT: uses a HIGHER threshold (0.72) when guarded, to prevent
        # real videos with high temporal scores from being falsely flagged.
        if cnn_score > 0.82 and quality_score < 0.18:
            final_score = 0.80 * temporal_score + 0.15 * cnn_score + 0.05 * quality_score
            decision_source += "_guarded_real"
            logger.info(
                f"Global real guard: cnn={cnn_score:.4f} quality={quality_score:.4f} "
                f"→ anchored to temporal={temporal_score:.4f}"
            )

        # ── Adaptive threshold ─────────────────────────────────────────────
        if "ensemble_quality_boost_fake_strong" in decision_source:
            THRESHOLD = 0.58 if "_guarded_real" in decision_source else 0.40

        elif "xception_anchor_real_protection" in decision_source:
            # More lenient threshold — xception is signalling real
            THRESHOLD = 0.72

        elif "cnn_override" in decision_source or "strong_agreement_fake" in decision_source:
            if "_guarded_real" in decision_source:
                # High CNN + clean quality + temporal fake → raise bar to 0.72
                # so real videos with misfiring temporal don't get tagged
                THRESHOLD = 0.72
            else:
                THRESHOLD = 0.55

        else:
            # Default — also raised slightly when guarded to protect real videos
            THRESHOLD = 0.72 if "_guarded_real" in decision_source else 0.62

        label      = "Fake" if final_score >= THRESHOLD else "Real"
        confidence = final_score if label == "Fake" else 1 - final_score

        needs_manual_review = abs(temporal_score - cnn_score) > 0.38
        review_reason = "High disagreement between temporal and CNN" if needs_manual_review else None

        logger.info(
            f"Fusion: temporal={temporal_score:.4f} cnn={cnn_score:.4f} "
            f"quality={quality_score:.4f} → final={final_score:.4f} "
            f"threshold={THRESHOLD} → {label}"
        )

        return {
            "label":             label,
            "confidence":        float(round(confidence, 4)),
            "final_score":       float(round(final_score, 4)),
            "decision_source":   decision_source,
            "temporal_score":    float(round(temporal_score, 4)),
            "cnn_score":         float(round(cnn_score, 4)),
            "quality_score":     float(round(quality_score, 4)),
            "temporal_analysis": temporal_results,
            "cnn_analysis":      cnn_results,
            "quality_analysis":  quality_results,
            "needs_manual_review": needs_manual_review,
            "review_reason":       review_reason,
            "fusion_weights": {"temporal": 0.40, "cnn": 0.40, "quality": 0.20},
        }


# Convenience alias (keeps test_hybrid_analysis.py working unchanged)
HybridVideoAnalyzer = ImprovedHybridAnalyzer