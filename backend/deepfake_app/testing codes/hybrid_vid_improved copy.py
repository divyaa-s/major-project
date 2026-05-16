"""
hybrid_vid_improved.py - FIXED & IMPROVED VERSION
Changes in this version:
- Added temperature scaling for efficientnet_b3
- Added strong real-signal override when temporal + quality agree on real
- Slightly relaxed some thresholds to reduce false positives
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

from temporal_analysis import TemporalAnalyzer
from quality_forensic import QualityForensicsAnalyzer

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIGS = {
    "efficientnet_b3": {
        "path": "D:/Projects/Major Project/Deepfake Detection/models/efficientnet_b3_model.pth",
        "input_size": (300, 300),
        "model_name": "efficientnet_b3",
        "temperature": 3.5,           # NEW: calibration for overconfident model
    },
    "xception": {
        "path": "D:/Projects/Major Project/Deepfake Detection/models/xceptionnet_20k.pth",
        "input_size": (299, 299),
        "model_name": "xception",
        "temperature": 4.0,
    },
    "vit": {
        "path": "D:/Projects/Major Project/Deepfake Detection/models/vit_deit_tiny_5epochs.pth",
        "input_size": (224, 224),
        "model_name": "vit_tiny_patch16_224",
        "temperature": 4.5,
    },
     "convnext": {
         "path": "D:/Projects/Major Project/Deepfake Detection/backend/deepfake_app/convnext/checkpoints/best_convnext_epoch_2_acc99.59.pth",
         "input_size": (224, 224),
         "model_name": "convnext_tiny",
         "temperature": 3.0,
     },
}


class ImprovedHybridAnalyzer:
    """
    Improved hybrid analyzer with fixes for overconfident CNNs and false positives
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
        
        logger.info(f"Initialized ImprovedHybridAnalyzer with {len(self.models)} CNN models")
        logger.info(f"Smart keyframe selection: {self.smart_keyframe_selection}")
    
    def _init_face_detectors(self):
        try:
            self.mtcnn = MTCNN(keep_all=False, device=DEVICE)
            self.mtcnn_available = True
            logger.info("✅ MTCNN initialized")
        except Exception as e:
            logger.warning(f"⚠️ MTCNN initialization failed: {e}")
            self.mtcnn_available = False
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        logger.info("✅ OpenCV Haar Cascade initialized")
    
    def _load_cnn_models(self):
        """Load all CNN models with better error handling"""
        for name, config in MODEL_CONFIGS.items():
            try:
                model = create_model(
                    config["model_name"],
                    pretrained=False,
                    num_classes=2
                )
                
                state_dict = torch.load(config["path"], map_location=DEVICE)
                
                # Special handling for ConvNeXt checkpoint (it's a full dict, not just state_dict)
                if name == "convnext":
                    if 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']  # ← extract the actual weights
                        logger.info(f"Extracted model_state_dict for ConvNeXt")
                    else:
                        logger.warning("ConvNeXt checkpoint does not contain 'model_state_dict' key")
                
                model.load_state_dict(state_dict)
                model.to(DEVICE).eval()
                
                self.models[name] = {
                    "model": model,
                    "config": config
                }
                logger.info(f"✅ Loaded {name} successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {name}: {e}")
                logger.warning(f"Model '{name}' will be skipped in analysis")
    
    def analyze_video(self, video_path):
        self.faces_detected_count = 0
        logger.info(f"🎬 Starting improved hybrid analysis: {video_path}")
        
        temporal_results = self.temporal_analyzer.analyze_video(video_path)
        if "error" in temporal_results:
            return temporal_results
        
        if self.smart_keyframe_selection:
            logger.info(f"🧠 Extracting {self.num_keyframes} keyframes (SMART selection)...")
            keyframes = self._extract_smart_keyframes(video_path)
        else:
            logger.info(f"🖼️  Extracting {self.num_keyframes} keyframes (even spacing)...")
            keyframes = self._extract_keyframes_even(video_path)
        
        if not keyframes:
            return {
                "error": "No keyframes could be extracted",
                "temporal_results": temporal_results
            }
        
        logger.info("🤖 Running CNN models on keyframes...")
        cnn_results = self._analyze_keyframes_with_cnns(keyframes)
        
        logger.info("🔬 Running quality forensics analysis...")
        quality_results = self._analyze_quality_mismatch(keyframes)
        
        logger.info("⚡ Fusing temporal + CNN + quality predictions...")
        final_results = self._fuse_predictions(temporal_results, cnn_results, quality_results)
        
        return final_results
    
    def _extract_smart_keyframes(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        candidate_count = min(self.num_keyframes * 3, total_frames)
        candidate_indices = np.linspace(0, total_frames - 1, candidate_count, dtype=int)
        
        candidates = []
        
        for idx in candidate_indices:
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
                    "face_box": face_box
                })
        
        cap.release()
        
        if not candidates:
            logger.warning("No face candidates found")
            return []
        
        candidates.sort(key=lambda x: x["quality"], reverse=True)
        selected = candidates[:self.num_keyframes]
        
        logger.info(f"   Selected {len(selected)} highest quality keyframes")
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
                    "face_box": face_box
                })
        
        cap.release()
        logger.info(f"   Extracted {len(keyframes)}/{self.num_keyframes} frames with faces")
        return keyframes
    
    def _crop_face_with_quality(self, frame):
        """Crop face and estimate simple quality score"""
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
                x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
                face = img_rgb[y1:y2, x1:x2]
                quality = self._estimate_face_quality(face)
                return face, quality, (x1, y1, x2, y2)
            except:
                pass
        
        # Fallback Haar
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            return None, 0.0, None
        x, y, w, h = faces[0]
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        quality = self._estimate_face_quality(face)
        return face, quality, (x1, y1, x2, y2)
    
    def _estimate_face_quality(self, face):
        """Simple quality estimate (higher = better)"""
        if face.size == 0:
            return 0.0
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(laplacian_var / 500.0, 1.0)  # rough normalization
    
    def _preprocess_face(self, face, size):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        return transform(face).unsqueeze(0).to(DEVICE)
    
    def _analyze_keyframes_with_cnns(self, keyframes):
        frame_predictions = []
        
        for kf in keyframes:
            frame_preds = {
                "frame_index": kf["frame_index"],
                "model_predictions": {}
            }
            
            for name, model_info in self.models.items():
                try:
                    model = model_info["model"]
                    config = model_info["config"]
                    
                    x = self._preprocess_face(kf["face"], config["input_size"])
                    
                    with torch.no_grad():
                        output = model(x)
                        # Apply temperature scaling if available
                        temp = config.get("temperature", 1.0)
                        if temp != 1.0:
                            output = output / temp
                        probs = F.softmax(output, dim=1)[0]
                        fake_prob = probs[1].item()
                    
                    frame_preds["model_predictions"][name] = float(fake_prob)
                    
                except Exception as e:
                    logger.warning(f"Model {name} failed on frame {kf['frame_index']}: {e}")
            
            frame_predictions.append(frame_preds)
        
        cnn_ensemble = self._calculate_cnn_ensemble(frame_predictions)
        
        return {
            "frame_predictions": frame_predictions,
            "ensemble": cnn_ensemble
        }
    
    def _calculate_cnn_ensemble(self, frame_predictions):
        if not frame_predictions:
            return {
                "avg_fake_probability": 0.5,
                "per_model_avg": {},
                "frames_analyzed": 0
            }
        
        model_scores = {name: [] for name in self.models.keys()}
        
        for frame_pred in frame_predictions:
            for model_name, fake_prob in frame_pred["model_predictions"].items():
                model_scores[model_name].append(fake_prob)
        
        per_model_avg = {}
        for model_name, scores in model_scores.items():
            if scores:
                per_model_avg[model_name] = float(np.mean(scores))
        
        # ── Updated: Explicit weights including ConvNeXt ──
        if len(per_model_avg) >= 2:
            weights = {
                "efficientnet_b3": 0.15,
                "xception":       0.15,
                "vit":            0.15,
                "convnext":       0.55,   # 50% — since it's your best single model
            }
            # Normalize weights so they sum to 1.0 (optional but cleaner)
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}
            
            cnn_ensemble = sum(
                weights.get(name, 0.25) * score   # fallback 0.25 if somehow missing
                for name, score in per_model_avg.items()
            )
        else:
            cnn_ensemble = np.mean(list(per_model_avg.values())) if per_model_avg else 0.5
        
        return {
            "avg_fake_probability": float(round(cnn_ensemble, 4)),
            "per_model_avg": per_model_avg,
            "frames_analyzed": len(frame_predictions)
        }
    
    def _analyze_quality_mismatch(self, keyframes):
        frames_with_boxes = []
        
        for kf in keyframes:
            frame = kf["frame"]
            face_box = kf.get("face_box")
            
            if face_box is not None:
                frames_with_boxes.append((frame, face_box))
            else:
                # Fallback estimation
                h, w = frame.shape[:2]
                face_h = int(h * 0.4)
                face_w = int(w * 0.3)
                y1 = int(h * 0.2)
                y2 = y1 + face_h
                x1 = int(w * 0.35)
                x2 = x1 + face_w
                face_box = (x1, y1, x2, y2)
                frames_with_boxes.append((frame, face_box))
        
        quality_results = self.quality_analyzer.analyze_keyframes(frames_with_boxes)
        
        logger.info(f"   Quality mismatch: {quality_results['avg_quality_mismatch']:.4f}")
        return quality_results
    
    def _fuse_predictions(self, temporal_results, cnn_results, quality_results):
        temporal_score = temporal_results["temporal_consistency_score"]
        cnn_score = cnn_results["ensemble"]["avg_fake_probability"]
        quality_score = quality_results["avg_quality_mismatch"]
        
        blink_score = temporal_results["blink_analysis"]["score"]
        optical_flow_score = temporal_results["optical_flow_analysis"]["score"]
        landmark_score = temporal_results["landmark_stability"]["score"]
        
        base_fusion = 0.40 * temporal_score + 0.40 * cnn_score + 0.20 * quality_score
        final_score = base_fusion
        decision_source = "hybrid_ensemble"
        
        # HIGHEST PRIORITY: Strong real override (run first)
        if (
            temporal_score < 0.48 and
            quality_score < 0.10 and
            cnn_score > 0.65 and
            blink_score < 0.80 and
            landmark_score < 0.40
        ):
            final_score = 0.20 * cnn_score + 0.65 * temporal_score + 0.15 * quality_score
            decision_source = "temporal_quality_real_override"
            logger.info("Strong real signal → overriding CNN confidence")
        
        # Quality + CNN suspicion override (strong fake bias)
        elif quality_score > 0.04 or ("convnext" in cnn_results["ensemble"]["per_model_avg"] and cnn_results["ensemble"]["per_model_avg"]["convnext"] > 0.85):
            final_score = 0.10 * temporal_score + 0.30 * cnn_score + 0.60 * quality_score
            decision_source = "convnext_quality_boost_fake_strong"
            logger.info("ConvNeXt high + any quality signal → very strong fake override (60% quality)")
        
        # Other pathways
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
        
        # Adaptive threshold: lower when quality or ConvNeXt is suspicious
        if (
            quality_score > 0.30 or
            ("convnext" in cnn_results["ensemble"]["per_model_avg"] and 
             cnn_results["ensemble"]["per_model_avg"]["convnext"] > 0.90) or
            "convnext_quality_boost_fake" in decision_source
        ):
            THRESHOLD = 0.50   # or 0.52 if you want slightly more conservative
        else:
            THRESHOLD = 0.62
        
        if final_score >= THRESHOLD:
            label = "Fake"
            confidence = final_score
        else:
            label = "Real"
            confidence = 1 - final_score
        
        needs_manual_review = abs(temporal_score - cnn_score) > 0.38
        review_reason = "High disagreement between temporal and CNN" if needs_manual_review else None
        
        return {
            "label": label,
            "confidence": float(round(confidence, 4)),
            "final_score": float(round(final_score, 4)),
            "decision_source": decision_source,
            "temporal_score": float(round(temporal_score, 4)),
            "cnn_score": float(round(cnn_score, 4)),
            "quality_score": float(round(quality_score, 4)),
            "temporal_analysis": temporal_results,
            "cnn_analysis": cnn_results,
            "quality_analysis": quality_results,
            "needs_manual_review": needs_manual_review,
            "review_reason": review_reason,
            "fusion_weights": {"temporal": 0.40, "cnn": 0.40, "quality": 0.20}
        }
# Convenience alias
HybridVideoAnalyzer = ImprovedHybridAnalyzer