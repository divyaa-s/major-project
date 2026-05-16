"""
hybrid_vid_improved.py - FIXED & ENSEMBLE VERSION (Updated March 2026)
Optimized with Targeted Single-Pass Frame Extraction
"""

import os
import sys
import torch
import torch.nn as nn
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
import joblib

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# CNN Model registry
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "efficientnet_b3": {
        "path":       r"D:\Projects\Major Project\Deepfake Detection\models\new\new_effb3_finetuned.pth",
        "input_size": (300, 300),
        "model_name": "efficientnet_b3",
        "num_classes": 1,
        "norm_mean":  [0.485, 0.456, 0.406],
        "norm_std":   [0.229, 0.224, 0.225],
    },
    "xception": {
        "path":       r"D:\Projects\Major Project\Deepfake Detection\models\new\new_xception_finetuned.pth",
        "input_size": (299, 299),
        "model_name": "legacy_xception",
        "num_classes": 1,
        "norm_mean":  [0.5, 0.5, 0.5],
        "norm_std":   [0.5, 0.5, 0.5],
    },
    "vit": {
        "path":       r"D:\Projects\Major Project\Deepfake Detection\models\new\new_vit_finetuned.pth",
        "input_size": (224, 224),
        "model_name": "vit_small_patch16_224",
        "num_classes": 1,
        "norm_mean":  [0.485, 0.456, 0.406],
        "norm_std":   [0.229, 0.224, 0.225],
    },
    "convnext": {
        "path":       r"D:\Projects\Major Project\Deepfake Detection\models\new\new_convnext_finetuned.pth",
        "input_size": (224, 224),
        "model_name": "convnext_small",
        "num_classes": 1,
        "norm_mean":  [0.485, 0.456, 0.406],
        "norm_std":   [0.229, 0.224, 0.225],
    },
}

MODEL_ENSEMBLE_WEIGHTS = {
    "efficientnet_b3": 0.25,
    "xception":        0.25,
    "vit":             0.25,
    "convnext":        0.25,
}

# ---------------------------------------------------------------------------
# BiLSTM model paths
# ---------------------------------------------------------------------------
BILSTM_CHECKPOINT  = r"D:\Projects\Major Project\Deepfake Detection\models\v5\bilstm_v2_best.pth"
BILSTM_NORM_STATS  = r"D:\Projects\Major Project\Deepfake Detection\models\v5\norm_stats.pt"

# BiLSTM architecture constants
BILSTM_FLOW_DIM   = 12
BILSTM_HIDDEN_DIM = 64
BILSTM_NUM_LAYERS = 2
BILSTM_SEQ_LEN    = 48
BILSTM_N_CLIPS    = 3
BILSTM_DROPOUT    = 0.3
BILSTM_WINSIZE    = 9
BILSTM_FLOW_RESIZE = (320, 180)


# ---------------------------------------------------------------------------
# BiLSTM architecture
# ---------------------------------------------------------------------------
class OptFlowBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim * 2, 1)
        nn.init.constant_(self.fc.bias, -1.0)

    def forward(self, x):
        x      = self.norm(x)
        out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attn(out), dim=1)
        ctx    = (out * attn_w).sum(dim=1)
        return self.fc(self.drop(ctx)).squeeze(1)


class ImprovedHybridAnalyzer:
    def __init__(self, num_keyframes=100, temporal_max_frames=30, temporal_skip=2,
                 smart_keyframe_selection=True):
        self.num_keyframes            = num_keyframes
        self.smart_keyframe_selection = smart_keyframe_selection
        self.faces_detected_count     = 0

        self.temporal_analyzer = TemporalAnalyzer(
            max_frames=temporal_max_frames,
            skip_frames=temporal_skip,
        )
        self.quality_analyzer = QualityForensicsAnalyzer()

        self.meta_learner = None
        if os.path.exists('video_meta_learner_v2.pkl'):
            self.meta_learner = joblib.load('video_meta_learner_v2.pkl')
            print("✅ SUCCESS: Video Meta-Learner loaded for fusion!")
        else:
            print("⚠️ WARNING: video_meta_learner_v2.pkl not found. Falling back to heuristic math.")
            
        self._init_face_detectors()
        self.models = {}
        self._load_cnn_models()

        self.bilstm_model     = None
        self.bilstm_norm_mean = None
        self.bilstm_norm_std  = None
        self.bilstm_available = False
        self._load_bilstm_model()

        logger.info(f"Initialized ImprovedHybridAnalyzer")
        logger.info(f"  CNN models loaded : {list(self.models.keys())}")
        logger.info(f"  BiLSTM available  : {self.bilstm_available}")

    def _init_face_detectors(self):
        try:
            self.mtcnn           = MTCNN(keep_all=False, device=DEVICE)
            self.mtcnn_available = True
            logger.info("✅ MTCNN initialized")
        except Exception as e:
            logger.warning(f"⚠️  MTCNN init failed: {e}")
            self.mtcnn_available = False

        cascade_path      = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.error("❌ Haar Cascade failed to load")
        else:
            logger.info("✅ Haar Cascade initialized")

    def _load_cnn_models(self):
        load_results = {}

        for name, config in MODEL_CONFIGS.items():
            try:
                model      = create_model(config["model_name"], pretrained=False,
                                          num_classes=config.get("num_classes", 1))
                checkpoint = torch.load(config["path"], map_location=DEVICE,
                                        weights_only=False)

                if isinstance(checkpoint, dict):
                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
                model.to(DEVICE).eval()

                self.models[name] = {"model": model, "config": config}
                load_results[name] = ("ok", None)

            except Exception as e:
                load_results[name] = ("fail", str(e))
                logger.error(f"❌ Failed to load [{name}]: {e}")

        print("\n" + "=" * 65)
        print("🧠 CNN MODEL LOAD STATUS")
        print("=" * 65)
        for name, (status, err) in load_results.items():
            cfg = MODEL_CONFIGS[name]
            if status == "ok":
                w = MODEL_ENSEMBLE_WEIGHTS.get(name, 0.25)
                print(f"  ✅ {name:<20s}  arch={cfg['model_name']:<25s}  weight={w:.2f}")
            else:
                short_err = err if len(err) < 90 else err[:87] + "..."
                print(f"  ❌ {name:<20s}  FAILED — {short_err}")
        print(f"\n  Loaded {len(self.models)}/{len(MODEL_CONFIGS)} models  |  Device: {DEVICE}")
        print("=" * 65 + "\n")

    def _load_bilstm_model(self):
        print("=" * 65)
        print("⏱  BILSTM TEMPORAL BRANCH")
        print("=" * 65)

        if not os.path.exists(BILSTM_CHECKPOINT):
            print(f"  ⚠️  Checkpoint not found: {BILSTM_CHECKPOINT}")
            print(f"  BiLSTM branch disabled — fusion will use CNN+quality only")
            print("=" * 65 + "\n")
            return

        if not os.path.exists(BILSTM_NORM_STATS):
            print(f"  ⚠️  Norm stats not found: {BILSTM_NORM_STATS}")
            print(f"  BiLSTM branch disabled — norm stats required for inference")
            print("=" * 65 + "\n")
            return

        try:
            ckpt  = torch.load(BILSTM_CHECKPOINT, map_location=DEVICE, weights_only=False)
            model = OptFlowBiLSTM(BILSTM_FLOW_DIM, BILSTM_HIDDEN_DIM,
                                  BILSTM_NUM_LAYERS, BILSTM_DROPOUT)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(DEVICE).eval()
            self.bilstm_model = model

            norm  = torch.load(BILSTM_NORM_STATS, map_location="cpu", weights_only=False)
            self.bilstm_norm_mean = norm["mean"]
            self.bilstm_norm_std  = norm["std"]

            self.bilstm_available = True
            print(f"  ✅ BiLSTM loaded")
        except Exception as e:
            print(f"  ❌ BiLSTM load failed: {e}")
            self.bilstm_available = False

        print("=" * 65 + "\n")

    def analyze_video(self, video_path):
        self.faces_detected_count = 0
        logger.info(f"🎬 Starting hybrid analysis: {video_path}")

        # 1. External Temporal Analyzer (still does its own read)
        temporal_results = self.temporal_analyzer.analyze_video(video_path)
        if "error" in temporal_results:
            return temporal_results

        # 2. TARGETED SINGLE-PASS READ for CNNs and BiLSTM
        cnn_dict, bilstm_dict, total_frames = self._extract_all_needed_frames(video_path)
        
        if total_frames == 0:
            return {"error": "Could not read video frames", "temporal_results": temporal_results}

        # 3. BiLSTM Inference from pre-loaded dict
        bilstm_score = self._analyze_temporal_bilstm(bilstm_dict, total_frames)

        # 4. Keyframe Face Extraction from pre-loaded dict
        if self.smart_keyframe_selection:
            keyframes = self._extract_smart_keyframes(cnn_dict)
        else:
            keyframes = self._extract_keyframes_even(cnn_dict)

        if not keyframes:
            return {"error": "No keyframes extracted", "temporal_results": temporal_results}

        # 5. Spatial Inference & Fusion
        cnn_results     = self._analyze_keyframes_with_cnns(keyframes)
        quality_results = self._analyze_quality_mismatch(keyframes)
        final_results   = self._fuse_predictions(
            temporal_results, cnn_results, quality_results, bilstm_score
        )

        return final_results

    # ──────────────────────────────────────────────────────────────────
    # TARGETED SINGLE-PASS FRAME EXTRACTION
    # ──────────────────────────────────────────────────────────────────
    
    def _extract_all_needed_frames(self, video_path):
        """
        Reads the video EXACTLY ONCE. 
        Intelligently grabs only the specific frames needed by CNNs and BiLSTM.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return {}, {}, 0

        if self.smart_keyframe_selection:
            candidate_count = min(self.num_keyframes * 3, total_frames)
        else:
            candidate_count = self.num_keyframes
            
        cnn_indices = set(np.linspace(0, total_frames - 1, candidate_count, dtype=int))

        bilstm_indices = set()
        if self.bilstm_available:
            positions = np.linspace(0.15, 0.85, BILSTM_N_CLIPS)
            for pos in positions:
                start = max(0, int(pos * total_frames))
                for i in range(BILSTM_SEQ_LEN):
                    if start + i < total_frames:
                        bilstm_indices.add(start + i)

        all_needed = cnn_indices.union(bilstm_indices)
        max_needed = max(all_needed) if all_needed else 0

        cnn_frames = {}
        bilstm_frames = {}

        logger.info(f"🎞 Single-Pass Extraction: Fetching {len(all_needed)} specific frames out of {total_frames}...")

        current_idx = 0
        
        # --- TQDM ADDED HERE ---
        with tqdm(total=max_needed + 1, desc="🎞 Decoding Video", leave=False) as pbar:
            while current_idx <= max_needed:
                ret = cap.grab()
                if not ret:
                    break

                if current_idx in all_needed:
                    ret, frame = cap.retrieve()
                    if not ret:
                        current_idx += 1
                        pbar.update(1)
                        continue

                    if current_idx in cnn_indices:
                        cnn_frames[current_idx] = frame.copy()

                    if current_idx in bilstm_indices:
                        small = cv2.resize(frame, BILSTM_FLOW_RESIZE, interpolation=cv2.INTER_LINEAR)
                        bilstm_frames[current_idx] = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                current_idx += 1
                pbar.update(1)

        cap.release()
        return cnn_frames, bilstm_frames, total_frames

    # ──────────────────────────────────────────────────────────────────
    # BiLSTM inference 
    # ──────────────────────────────────────────────────────────────────

    def _extract_flow_features(self, g1, g2):
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None,
            pyr_scale=0.5, levels=3,
            winsize=BILSTM_WINSIZE,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        h, w   = mag.shape
        cy, cx = h // 2, w // 2
        r      = min(h, w) // 4
        center = mag[cy-r:cy+r, cx-r:cx+r]
        outer  = mag.copy(); outer[cy-r:cy+r, cx-r:cx+r] = 0
        boundary_score = float(center.mean()) / (float(outer.mean()) + 1e-6)

        hist, _ = np.histogram(mag.flatten(), bins=20, range=(0, mag.max()+1e-6))
        hist    = hist / (hist.sum() + 1e-6)
        entropy = float(-np.sum(hist * np.log(hist + 1e-10)))

        means = [mag[:cy,:cx].mean(), mag[:cy,cx:].mean(),
                 mag[cy:,:cx].mean(), mag[cy:,cx:].mean()]

        return np.array([
            float(mag.mean()), float(mag.std()), float(mag.max()),
            float(flow[...,0].mean()), float(flow[...,1].mean()),
            float(ang.std()), boundary_score,
            1.0 / (float(mag.std()) + 1e-6), entropy,
            float(mag[cy-r:cy+r,cx-r:cx+r].mean()) / (float(mag.mean()) + 1e-6),
            float(max(means) - min(means)),
            float(np.corrcoef(mag[:h//2].flatten(),
                              mag[h//2:h//2*2].flatten())[0,1]) if h>=2 else 0.0,
        ], dtype=np.float32)

    def _analyze_temporal_bilstm(self, bilstm_frames_dict, total_frames):
        if not self.bilstm_available:
            return 0.5

        try:
            positions  = np.linspace(0.15, 0.85, BILSTM_N_CLIPS)
            clip_probs = []

            for pos in positions:
                start_idx = max(0, int(pos * total_frames))
                grays = []
                
                # Retrieve sequential frames for this clip from dict
                for i in range(BILSTM_SEQ_LEN):
                    idx = start_idx + i
                    if idx in bilstm_frames_dict:
                        grays.append(bilstm_frames_dict[idx])
                
                # Padding fallback 
                while len(grays) < BILSTM_SEQ_LEN and grays:
                    grays.append(grays[-1])
                    
                if len(grays) < BILSTM_SEQ_LEN:
                    continue

                flow_seq = [self._extract_flow_features(grays[i], grays[i+1])
                            for i in range(len(grays) - 1)]

                seq_arr = np.array(flow_seq, dtype=np.float32)
                seq_t    = torch.tensor(seq_arr, dtype=torch.float32).unsqueeze(0)
                seq_norm = (seq_t - self.bilstm_norm_mean) / self.bilstm_norm_std
                seq_norm = torch.clamp(seq_norm, -10.0, 10.0)

                with torch.no_grad():
                    logit = self.bilstm_model(seq_norm.to(DEVICE))
                    prob  = torch.sigmoid(logit).item()

                clip_probs.append(prob)

            if not clip_probs:
                logger.warning("BiLSTM: no clips extracted — returning neutral 0.5")
                return 0.5

            bilstm_score = float(np.mean(clip_probs))
            logger.info(f"BiLSTM: {len(clip_probs)} clips → "
                        f"probs={[round(p,3) for p in clip_probs]} → "
                        f"avg={bilstm_score:.4f}")
            return bilstm_score

        except Exception as e:
            logger.warning(f"BiLSTM inference failed: {e} — returning neutral 0.5")
            return 0.5

    # ──────────────────────────────────────────────────────────────────
    # Keyframe extraction (From Pre-loaded Dict)
    # ──────────────────────────────────────────────────────────────────

    def _extract_smart_keyframes(self, cnn_frames_dict):
        candidates = []

        # --- TQDM ADDED HERE ---
        for idx in tqdm(sorted(cnn_frames_dict.keys()), desc="🧑 Tracking Faces (Smart)", leave=False):
            frame = cnn_frames_dict[idx]
            face, quality_score, face_box = self._crop_face_with_quality(frame)
            if face is not None:
                self.faces_detected_count += 1
                candidates.append({
                    "frame_index": int(idx),
                    "face":        face,
                    "frame":       frame,
                    "quality":     quality_score,
                    "face_box":    face_box,
                })

        if not candidates:
            logger.warning("No face candidates found")
            return []

        candidates.sort(key=lambda x: x["quality"], reverse=True)
        selected = candidates[: self.num_keyframes]
        logger.info(f"   Selected {len(selected)} keyframes from {len(candidates)} candidates")
        return selected

    def _extract_keyframes_even(self, cnn_frames_dict):
        keyframes = []

        # --- TQDM ADDED HERE ---
        for idx in tqdm(sorted(cnn_frames_dict.keys()), desc="🧑 Tracking Faces (Even)", leave=False):
            frame = cnn_frames_dict[idx]
            face, quality_score, face_box = self._crop_face_with_quality(frame)
            if face is not None:
                self.faces_detected_count += 1
                keyframes.append({
                    "frame_index": int(idx),
                    "face":        face,
                    "frame":       frame,
                    "quality":     quality_score,
                    "face_box":    face_box,
                })

        logger.info(f"   Extracted {len(keyframes)}/{self.num_keyframes} frames with faces")
        return keyframes
    
    
    def _extract_keyframes_even(self, cnn_frames_dict):
        keyframes = []

        for idx in sorted(cnn_frames_dict.keys()):
            frame = cnn_frames_dict[idx]
            face, quality_score, face_box = self._crop_face_with_quality(frame)
            if face is not None:
                self.faces_detected_count += 1
                keyframes.append({
                    "frame_index": int(idx),
                    "face":        face,
                    "frame":       frame,
                    "quality":     quality_score,
                    "face_box":    face_box,
                })

        logger.info(f"   Extracted {len(keyframes)}/{self.num_keyframes} frames with faces")
        return keyframes

    # ──────────────────────────────────────────────────────────────────
    # Face detection & preprocessing
    # ──────────────────────────────────────────────────────────────────

    def _crop_face_with_quality(self, frame):
        if self.mtcnn_available:
            try:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                boxes, probs = self.mtcnn.detect(img_pil)
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0].astype(int)
                    pad = 20
                    x1 = max(0, x1 - pad);  y1 = max(0, y1 - pad)
                    x2 = min(frame.shape[1], x2 + pad)
                    y2 = min(frame.shape[0], y2 + pad)
                    face    = img_rgb[y1:y2, x1:x2]
                    quality = self._estimate_face_quality(face)
                    return face, quality, (x1, y1, x2, y2)
            except Exception as e:
                logger.debug(f"MTCNN failed: {e}")

        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces  = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            return None, 0.0, None

        x, y, w, h = faces[0]
        pad = 20
        x1  = max(0, x - pad);  y1 = max(0, y - pad)
        x2  = min(frame.shape[1], x + w + pad)
        y2  = min(frame.shape[0], y + h + pad)
        face    = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        quality = self._estimate_face_quality(face)
        return face, quality, (x1, y1, x2, y2)

    def _estimate_face_quality(self, face):
        if face.size == 0:
            return 0.0
        gray          = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(laplacian_var / 500.0, 1.0)

    def _preprocess_face(self, face, size, norm_mean, norm_std):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        return transform(face).unsqueeze(0).to(DEVICE)

    # ──────────────────────────────────────────────────────────────────
    # CNN ensemble inference
    # ──────────────────────────────────────────────────────────────────

    def _analyze_keyframes_with_cnns(self, keyframes):
        if not self.models:
            logger.error("No CNN models loaded")
            return {
                "frame_predictions": [],
                "ensemble": {
                    "avg_fake_probability": 0.5,
                    "per_model_avg": {},
                    "frames_analyzed": 0,
                },
            }

        frame_predictions = []

        for kf in tqdm(keyframes, desc="🧠 CNN Processing Frames", leave=False):
            frame_preds = {
                "frame_index":       kf["frame_index"],
                "model_predictions": {},
            }

            for name, model_info in self.models.items():
                try:
                    import cv2
                    import numpy as np
                    model  = model_info["model"]
                    config = model_info["config"]
                    
                    face_img = kf["face"]

                    if name in ["vit", "convnext"]:
                        if isinstance(face_img, np.ndarray) and face_img.shape[-1] == 3:
                            face_color = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        else:
                            face_color = face_img
                    else:
                        face_color = face_img

                    x = self._preprocess_face(
                        face_color,
                        config["input_size"],
                        config["norm_mean"],
                        config["norm_std"],
                    )

                    with torch.no_grad():
                        output = model(x)
                        if config.get("num_classes", 1) == 1:
                            raw_prob = torch.sigmoid(output).squeeze().item()
                            fake_prob = 1.0 - raw_prob
                        else:
                            import torch.nn.functional as F
                            fake_prob = F.softmax(output, dim=1)[0][1].item()
                            
                    frame_preds["model_predictions"][name] = float(fake_prob)

                except Exception as e:
                    logger.warning(f"  [{name}] failed on frame {kf['frame_index']}: {e}")
                    frame_preds["model_predictions"][name] = 0.5

            frame_predictions.append(frame_preds)

        cnn_ensemble = self._calculate_cnn_ensemble(frame_predictions)

        print("\n" + "─" * 60)
        print("🤖 CNN ENSEMBLE — PER-MODEL AVERAGE SCORES")
        print("─" * 60)
        for model_name, avg_score in cnn_ensemble.get("per_model_avg", {}).items():
            try:
                weight = MODEL_ENSEMBLE_WEIGHTS.get(model_name, 0.25)
            except NameError:
                weight = 0.25
                
            bar    = "█" * int(avg_score * 30)
            flag   = "  ⚠️  EXTREME" if avg_score < 0.05 or avg_score > 0.95 else ""
            print(f"  {model_name:<20s}  score={avg_score:.4f}  weight={weight:.2f}"
                  f"  |{bar:<30s}|{flag}")
        print(f"\n  Weighted Ensemble Score : {cnn_ensemble.get('avg_fake_probability', 0.5):.4f}")
        print(f"  Frames analysed        : {cnn_ensemble.get('frames_analyzed', 0)}")
        print("─" * 60 + "\n")

        return {
            "frame_predictions": frame_predictions,
            "ensemble":          cnn_ensemble,
        }

    def _calculate_cnn_ensemble(self, frame_predictions):
        if not frame_predictions:
            return {"avg_fake_probability": 0.5, "per_model_avg": {},
                    "frames_analyzed": 0}

        model_scores = {name: [] for name in self.models.keys()}
        for fp in frame_predictions:
            for model_name, fake_prob in fp["model_predictions"].items():
                if model_name in model_scores:
                    model_scores[model_name].append(fake_prob)

        per_model_avg = {
            name: float(np.mean(scores))
            for name, scores in model_scores.items()
            if scores
        }

        if not per_model_avg:
            return {"avg_fake_probability": 0.5, "per_model_avg": {},
                    "frames_analyzed": len(frame_predictions)}

        active_models = {
            name: score for name, score in per_model_avg.items()
            if MODEL_ENSEMBLE_WEIGHTS.get(name, 0.25) > 0.0
        }

        if not active_models:
            return {"avg_fake_probability": 0.5, "per_model_avg": per_model_avg,
                    "frames_analyzed": len(frame_predictions)}

        total_weight       = sum(MODEL_ENSEMBLE_WEIGHTS.get(n, 0.25)
                                 for n in active_models)
        cnn_ensemble_score = sum(
            (MODEL_ENSEMBLE_WEIGHTS.get(n, 0.25) / total_weight) * s
            for n, s in active_models.items()
        )

        return {
            "avg_fake_probability": float(round(cnn_ensemble_score, 4)),
            "per_model_avg":        per_model_avg,
            "frames_analyzed":      len(frame_predictions),
        }

    def _analyze_quality_mismatch(self, keyframes):
        frames_with_boxes = []
        for kf in keyframes:
            frame    = kf["frame"]
            face_box = kf.get("face_box")
            if face_box is not None:
                frames_with_boxes.append((frame, face_box))
            else:
                h, w  = frame.shape[:2]
                x1    = int(w * 0.35); x2 = x1 + int(w * 0.3)
                y1    = int(h * 0.2);  y2 = y1 + int(h * 0.4)
                frames_with_boxes.append((frame, (x1, y1, x2, y2)))

        quality_results = self.quality_analyzer.analyze_keyframes(frames_with_boxes)
        logger.info(f"   Quality mismatch: {quality_results['avg_quality_mismatch']:.4f}")
        return quality_results

    def _fuse_predictions(self, temporal_results, cnn_results, quality_results, bilstm_score):
        import numpy as np

        temporal_score   = float(temporal_results.get("temporal_consistency_score", 0.5))
        quality_mismatch = float(quality_results.get("avg_quality_mismatch", 0.0))
        cnn_score        = float(cnn_results["ensemble"].get("avg_fake_probability", 0.5))

        per_model      = cnn_results["ensemble"].get("per_model_avg", {})
        effb3_score    = float(per_model.get("efficientnet_b3", 0.5))
        xception_score = float(per_model.get("xception",        0.5))
        vit_score      = float(per_model.get("vit",             0.5))
        convnext_score = float(per_model.get("convnext",        0.5))

        bilstm_fake_prob = float(bilstm_score)

        signals = {
            "bilstm":          bilstm_fake_prob,
            "efficientnet_b3": effb3_score,
            "xception":        xception_score,
            "vit":             vit_score,
            "convnext":        convnext_score,
            "quality":         quality_mismatch, 
            "temporal":        temporal_score,
        }

        # =========================================================
        # META-LEARNER FUSION 
        # =========================================================
        if hasattr(self, 'meta_learner') and self.meta_learner is not None:
            features = np.array([[
                effb3_score,
                xception_score,
                vit_score,
                convnext_score,
                quality_mismatch, 
                float(temporal_results.get('flow_score', temporal_score)),
                float(temporal_results.get('blink_score', 0.5)),
                float(temporal_results.get('landmark_score', 0.5))
            ]])
            
            final_score = float(self.meta_learner.predict_proba(features)[0, 1])
            decision_source = "logistic_meta_learner"
            logger.info("FUSION METHOD: Logistic Meta-Learner Override Active")
            
            confidences = {k: 0.0 for k in signals}
            total_conf = 1.0

        # =========================================================
        # FALLBACK: HEURISTIC MATH
        # =========================================================
        else:
            confidences = {k: 2.0 * abs(v - 0.5) for k, v in signals.items()}
            total_conf  = sum(confidences.values())

            if total_conf < 1e-8:
                final_score     = 0.5
                decision_source = "confidence_weighted_fusion_neutral"
            else:
                final_score     = sum(confidences[k] * signals[k] for k in signals) / total_conf
                decision_source = "confidence_weighted_fusion"
                
            logger.info("Conf-Weighted Fusion:")
            for k, s in signals.items():
                c = confidences[k]
                logger.info(f"  {k:<20} score={s:.4f}  conf={c:.4f}  "
                            f"contrib={c*s:.4f}  weight={c/total_conf*100:.1f}%")
            logger.info(f"  total_conf={total_conf:.4f}  final={final_score:.4f}")

        # =========================================================
        # FINAL LABEL & FORMATTING
        # =========================================================
        final_score = float(np.clip(final_score, 0.0, 1.0))

        THRESHOLD = 0.50   

        label      = "Fake" if final_score >= THRESHOLD else "Real"
        confidence = final_score if label == "Fake" else 1.0 - final_score

        logger.info(f"  FINAL PIPELINE OUTPUT -> {label} (Score: {final_score:.4f}, Threshold: {THRESHOLD})")

        return {
            "label":               label,
            "confidence":          float(round(confidence,    4)),
            "final_score":         float(round(final_score,   4)),
            "decision_source":     decision_source,
            "bilstm_score":        float(round(bilstm_score,  4)),
            "temporal_score":      float(round(temporal_score, 4)),
            "cnn_score":           float(round(cnn_score,      4)),
            "quality_score":       float(round(quality_mismatch, 4)),
            "temporal_analysis":   temporal_results,
            "cnn_analysis":        cnn_results,
            "quality_analysis":    quality_results,
            "needs_manual_review": abs(final_score - 0.5) < 0.10,
            "review_reason": (
                "Score within 0.10 of threshold — low confidence prediction"
                if abs(final_score - 0.5) < 0.10 else None
            ),
            "fusion_weights": {
                **{k: round(confidences[k] / total_conf, 6) for k in signals},
                "total_confidence": round(total_conf, 6),
                "threshold":        THRESHOLD,
                "method":           decision_source,
            },
        }

HybridVideoAnalyzer = ImprovedHybridAnalyzer