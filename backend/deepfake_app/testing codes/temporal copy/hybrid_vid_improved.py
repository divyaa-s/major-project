"""
hybrid_vid_improved.py - FIXED VERSION
Improvements:
1. Fixed ConvNeXt extreme prediction handling
2. Added temperature scaling for ViT
3. Motion consistency detection for natural motion
4. Store actual face boxes for quality forensics
5. Fixed face count reporting
6. Better logging and debugging
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
        "model_name": "efficientnet_b3"
    },
    "xception": {
        "path": "D:/Projects/Major Project/Deepfake Detection/models/xceptionnet_20k.pth",
        "input_size": (299, 299),
        "model_name": "xception",
        "temperature": 3.0  # INCREASED: 2.5 → 4.0 for stronger calibration
    },
    "vit": {
        "path": "D:/Projects/Major Project/Deepfake Detection/models/vit_deit_tiny_5epochs.pth",
        "input_size": (224, 224),
        "model_name": "vit_tiny_patch16_224",
        "temperature": 3.5  # INCREASED: 3.0 → 4.5 for stronger calibration
    }#,
    #"convnext": {
    #    "path": "D:/Projects/Major Project/Deepfake Detection/models/convnext_tiny_deepfake.pth",
    #    "input_size": (224, 224),
    #    "model_name": "convnext_tiny"
    #}
}


class ImprovedHybridAnalyzer:
    """
    Improved hybrid analyzer with fixes for all critical issues
    """
    
    def __init__(self, num_keyframes=5, temporal_max_frames=30, temporal_skip=2,
                 smart_keyframe_selection=True):
        """
        Args:
            num_keyframes: Number of frames to analyze with CNNs
            temporal_max_frames: Max frames for temporal analysis
            temporal_skip: Skip rate for temporal analysis
            smart_keyframe_selection: Use quality-based selection (recommended)
        """
        self.num_keyframes = num_keyframes
        self.smart_keyframe_selection = smart_keyframe_selection
        
        # NEW: Counter for face detection
        self.faces_detected_count = 0
        
        self.temporal_analyzer = TemporalAnalyzer(
            max_frames=temporal_max_frames,
            skip_frames=temporal_skip
        )
        
        # Initialize quality forensics analyzer
        self.quality_analyzer = QualityForensicsAnalyzer()
        
        # Initialize face detectors (MTCNN + OpenCV fallback)
        self._init_face_detectors()
        
        # Load CNN models
        self.models = {}
        self._load_cnn_models()
        
        logger.info(f"Initialized ImprovedHybridAnalyzer with {len(self.models)} CNN models")
        logger.info(f"Smart keyframe selection: {self.smart_keyframe_selection}")
    
    def _init_face_detectors(self):
        """Initialize both MTCNN and OpenCV for redundancy"""
        # Try MTCNN first
        try:
            self.mtcnn = MTCNN(keep_all=False, device=DEVICE)
            self.mtcnn_available = True
            logger.info("✅ MTCNN initialized")
        except Exception as e:
            logger.warning(f"⚠️ MTCNN initialization failed: {e}")
            self.mtcnn_available = False
        
        # Always initialize OpenCV as backup
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        logger.info("✅ OpenCV Haar Cascade initialized")
    
    def _load_cnn_models(self):
        """Load all CNN models"""
        for name, config in MODEL_CONFIGS.items():
            try:
                model = create_model(
                    config["model_name"],
                    pretrained=False,
                    num_classes=2
                )
                model.load_state_dict(
                    torch.load(config["path"], map_location=DEVICE)
                )
                model.to(DEVICE).eval()
                
                self.models[name] = {
                    "model": model,
                    "config": config
                }
                logger.info(f"✅ Loaded {name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {name}: {e}")
    
    def analyze_video(self, video_path):
        """Complete hybrid analysis with smart keyframe selection"""
        # NEW: Reset face counter at start
        self.faces_detected_count = 0
        
        logger.info(f"🎬 Starting improved hybrid analysis: {video_path}")
        
        # 1. Temporal Analysis
        logger.info("📊 Running temporal consistency analysis...")
        temporal_results = self.temporal_analyzer.analyze_video(video_path)
        
        if "error" in temporal_results:
            return temporal_results
        
        # 2. Extract keyframes (SMART selection if enabled)
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
        
        # 3. CNN Analysis on keyframes
        logger.info("🤖 Running CNN models on keyframes...")
        cnn_results = self._analyze_keyframes_with_cnns(keyframes)
        
        # 4. Quality Forensics Analysis
        logger.info("🔬 Running quality forensics analysis...")
        quality_results = self._analyze_quality_mismatch(keyframes)
        
        # 5. Intelligent Fusion
        logger.info("⚡ Fusing temporal + CNN + quality predictions...")
        final_results = self._fuse_predictions(temporal_results, cnn_results, quality_results)
        
        return final_results
    
    def _extract_smart_keyframes(self, video_path):
        """
        IMPROVED: Extract keyframes based on quality, not just spacing
        NOW: Stores actual face boxes for quality forensics
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        # Sample 3x more frames than needed to have choices
        candidate_count = min(self.num_keyframes * 3, total_frames)
        candidate_indices = np.linspace(0, total_frames - 1, candidate_count, dtype=int)
        
        candidates = []
        
        for idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # FIXED: Now returns face_box too
            face, quality_score, face_box = self._crop_face_with_quality(frame)
            
            if face is not None:
                self.faces_detected_count += 1  # NEW: Increment counter
                
                candidates.append({
                    "frame_index": int(idx),
                    "face": face,
                    "frame": frame,
                    "quality": quality_score,
                    "face_box": face_box  # NEW: Store actual face box
                })
        
        cap.release()
        
        if not candidates:
            logger.warning("No face candidates found")
            return []
        
        # Sort by quality, take top N
        candidates.sort(key=lambda x: x["quality"], reverse=True)
        selected = candidates[:self.num_keyframes]
        
        logger.info(f"   Selected {len(selected)} highest quality keyframes")
        
        return selected
    
    def _extract_keyframes_even(self, video_path):
        """Fallback: Extract evenly spaced keyframes"""
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
            
            # FIXED: Now returns face_box too
            face, quality_score, face_box = self._crop_face_with_quality(frame)
            
            if face is not None:
                self.faces_detected_count += 1  # NEW: Increment counter
                
                keyframes.append({
                    "frame_index": int(idx),
                    "face": face,
                    "frame": frame,
                    "quality": quality_score,
                    "face_box": face_box  # NEW: Store actual face box
                })
        
        cap.release()
        
        logger.info(f"   Extracted {len(keyframes)}/{self.num_keyframes} frames with faces")
        
        return keyframes
    
    def _crop_face_with_quality(self, frame):
        """
        Detect and crop face with quality assessment
        FIXED: Now returns actual face box coordinates
        
        Returns:
            cropped_face: Face image
            quality_score: Quality metric
            face_box: (x1, y1, x2, y2) actual coordinates
        """
        
        # Try MTCNN first
        if self.mtcnn_available:
            try:
                boxes, probs = self.mtcnn.detect(frame)
                if boxes is not None and len(boxes) > 0:
                    box = boxes[0]
                    x1, y1, x2, y2 = [int(b) for b in box]
                    
                    # Expand slightly
                    h, w = frame.shape[:2]
                    margin = 20
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(w, x2 + margin)
                    y2 = min(h, y2 + margin)
                    
                    face = frame[y1:y2, x1:x2]
                    quality = self._assess_quality(face)
                    
                    return face, quality, (x1, y1, x2, y2)
            except Exception as e:
                logger.debug(f"MTCNN detection failed: {e}")
        
        # Fallback to Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            x, y, w_box, h_box = faces[0]
            x1, y1 = x, y
            x2, y2 = x + w_box, y + h_box
            
            face = frame[y1:y2, x1:x2]
            quality = self._assess_quality(face)
            
            return face, quality, (x1, y1, x2, y2)
        
        return None, 0.0, None
    
    def _assess_quality(self, face):
        """Assess face quality for keyframe selection"""
        if face is None or face.size == 0:
            return 0.0
        
        # Sharpness (Laplacian variance)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Size score
        h, w = face.shape[:2]
        size_score = min(h * w / (300 * 300), 1.0)
        
        # Brightness
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2
        
        # Combined quality
        quality = (
            0.50 * min(sharpness / 500, 1.0) +
            0.30 * size_score +
            0.20 * brightness_score
        )
        
        return float(quality)
    
    def _analyze_keyframes_with_cnns(self, keyframes):
        """Run all CNN models on keyframes"""
        frame_predictions = []
        
        for kf in keyframes:
            face = kf["face"]
            frame_idx = kf["frame_index"]
            
            predictions = self._analyze_single_keyframe(face, frame_idx)
            
            frame_predictions.append({
                "frame_index": frame_idx,
                "model_predictions": predictions
            })
        
        # Aggregate results
        ensemble = self._aggregate_cnn_results(frame_predictions)
        
        return {
            "ensemble": ensemble,
            "frame_predictions": frame_predictions
        }
    
    def _analyze_single_keyframe(self, face, frame_index):
        """
        Analyze one keyframe with all CNN models
        FIXED: Added temperature scaling for ViT
        FIXED: Added logging for extreme predictions
        """
        predictions = {}
        
        for model_name, model_data in self.models.items():
            model = model_data["model"]
            config = model_data["config"]
            
            # Transform
            img_pil = Image.fromarray(face)
            img_resized = img_pil.resize(config["input_size"])
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(img_resized).unsqueeze(0).to(DEVICE)
            
            # Prediction
            with torch.no_grad():
                logits = model(input_tensor)
                
                # NEW: Temperature scaling for ViT and Xception to reduce overconfidence
                if model_name in ["vit", "xception"] and "temperature" in config:
                    temperature = config["temperature"]
                    logits = logits / temperature
                    logger.debug(f"Applied temperature scaling ({temperature}) to {model_name}")
                
                probs = F.softmax(logits, dim=1)[0]
                fake_prob = float(probs[1].cpu())
                
                # NEW: Log extreme predictions for debugging
                if fake_prob > 0.95 or fake_prob < 0.05:
                    logger.debug(f"⚠️ {model_name} extreme prediction: {fake_prob:.4f} on frame {frame_index}")
                    logger.debug(f"   Raw logits: {logits[0].cpu().numpy()}")
                
                predictions[model_name] = fake_prob
        
        return predictions
    
    def _aggregate_cnn_results(self, frame_predictions):
        """
        Aggregate CNN predictions across frames
        FIXED: Better handling of ConvNeXt extreme predictions
        FIXED: Lower ViT cap to reduce false positives
        """
        # Collect per-model scores
        model_scores = {name: [] for name in self.models.keys()}
        
        for frame_pred in frame_predictions:
            preds = frame_pred["model_predictions"]
            for model_name, score in preds.items():
                model_scores[model_name].append(score)
        
        # Calculate per-model averages
        per_model_avg = {}
        for model_name, scores in model_scores.items():
            if scores:
                per_model_avg[model_name] = float(np.mean(scores))
        
        if len(per_model_avg) < 3:
            # If models failed, simple average
            cnn_ensemble = np.mean(list(per_model_avg.values())) if per_model_avg else 0.5
        else:
            # Cap extreme predictions - VERY AGGRESSIVE to prevent false positives
            eff_score = np.clip(per_model_avg.get("efficientnet_b3", 0.5), 0.15, 0.85)
            xce_score = np.clip(per_model_avg.get("xception", 0.5), 0.15, 0.75)  # UPDATED: Cap at 75%
            vit_score = np.clip(per_model_avg.get("vit", 0.5), 0.15, 0.70)      # UPDATED: Cap at 70%
            cnx_score = np.clip(per_model_avg.get("convnext", 0.5), 0.15, 0.85)
            
            # NEW: Detect if ConvNeXt is stuck at extremes
            convnext_is_extreme = (cnx_score < 0.10 or cnx_score > 0.90)
            
            if convnext_is_extreme:
                # ConvNeXt being unreliable - reweight ensemble
                logger.warning(f"⚠️ ConvNeXt extreme prediction: {cnx_score:.4f} - rebalancing ensemble")
                
                base_ensemble = (
                    0.30 * eff_score +    # Increased from 0.20
                    0.30 * xce_score +    # Increased from 0.20
                    0.20 * vit_score +    # Increased from 0.15
                    0.20 * cnx_score      # Decreased from 0.45
                )
            else:
                # Normal weighting (ConvNeXt most trusted)
                base_ensemble = (
                    0.20 * eff_score +
                    0.20 * xce_score +
                    0.15 * vit_score +
                    0.45 * cnx_score
                )
            
            # Ensemble boost logic
            high_confidence_models = sum([
                1 for score in [eff_score, xce_score, vit_score, cnx_score]
                if score > 0.70
            ])
            
            # Only boost if ConvNeXt also agrees (prevents false positives)
            should_boost = (high_confidence_models >= 2 and cnx_score > 0.30)
            
            if should_boost:
                boost = 0.10 * high_confidence_models
                cnn_ensemble = min(base_ensemble + boost, 0.95)
                logger.info(f"📈 Ensemble boost: {high_confidence_models} models confident (+{boost:.2f})")
            else:
                cnn_ensemble = base_ensemble
                if high_confidence_models >= 2 and cnx_score < 0.30:
                    logger.warning(f"⚠️ Boost BLOCKED: {high_confidence_models} models confident but ConvNeXt says real")
        
        return {
            "avg_fake_probability": float(cnn_ensemble),
            "per_model_avg": per_model_avg,
            "frames_analyzed": len(frame_predictions)
        }
    
    def _analyze_quality_mismatch(self, keyframes):
        """
        Analyze face-body quality mismatch across keyframes
        FIXED: Now uses actual stored face boxes instead of estimates
        """
        frames_with_boxes = []
        
        for kf in keyframes:
            frame = kf["frame"]
            face_box = kf.get("face_box")  # FIXED: Get stored box
            
            if face_box is not None:
                # Use actual detected face box
                frames_with_boxes.append((frame, face_box))
            else:
                # Fallback to estimation only if no box stored
                logger.warning("Face box not found, using estimation")
                h, w = frame.shape[:2]
                face_h = int(h * 0.4)
                face_w = int(w * 0.3)
                y1 = int(h * 0.2)
                y2 = y1 + face_h
                x1 = int(w * 0.35)
                x2 = x1 + face_w
                face_box = (x1, y1, x2, y2)
                frames_with_boxes.append((frame, face_box))
        
        # Analyze all keyframes
        quality_results = self.quality_analyzer.analyze_keyframes(frames_with_boxes)
        
        logger.info(f"   Quality mismatch: {quality_results['avg_quality_mismatch']:.4f}")
        
        return quality_results
    
    def _fuse_predictions(self, temporal_results, cnn_results, quality_results):
        """
        Intelligent fusion with quality forensics
        FIXED: Better natural motion detection
        FIXED: Uses actual face count
        """
        temporal_score = temporal_results["temporal_consistency_score"]
        cnn_score = cnn_results["ensemble"]["avg_fake_probability"]
        quality_score = quality_results["avg_quality_mismatch"]
        
        # Get component scores
        landmark_score = temporal_results["landmark_stability"]["score"]
        blink_score = temporal_results["blink_analysis"]["score"]
        optical_flow_score = temporal_results["optical_flow_analysis"]["score"]
        
        # Base fusion: temporal + CNN + quality
        base_fusion = (
            0.40 * temporal_score + 
            0.40 * cnn_score +
            0.20 * quality_score
        )
        
        logger.info(f"Fusion scores: temporal={temporal_score:.4f}, cnn={cnn_score:.4f}, quality={quality_score:.4f}")
        
        # Get ConvNeXt specific score
        cnx_score = cnn_results["ensemble"]["per_model_avg"].get("convnext", 0.5)
        
        decision_source = "hybrid_ensemble"
        final_score = base_fusion
        
        # UPDATED: Stronger penalty when ConvNeXt says real
        # ConvNeXt was often correct on real videos in testing
        if cnx_score < 0.15 and base_fusion >= 0.50:
            # ConvNeXt strongly disagrees - increase penalty
            penalty = 0.20  # Increased from 0.10
            final_score = base_fusion * (1 - penalty)
            decision_source = "convnext_strong_real_signal"
            logger.warning(f"⚠️ ConvNeXt strong real signal: applying {penalty*100}% penalty")
        elif cnx_score < 0.20 and base_fusion >= 0.50:
            # ConvNeXt moderately disagrees
            penalty = 0.10
            final_score = base_fusion * (1 - penalty)
            decision_source = "convnext_real_signal"
            logger.warning(f"⚠️ ConvNeXt real signal: applying {penalty*100}% penalty")
        
        # IMPROVED: Better natural motion detection
        # More lenient landmark threshold (0.45 instead of 0.65)
        natural_fast_motion = (
            landmark_score >= 0.45 and  # FIXED: Lowered from 0.65
            optical_flow_score >= 0.60 and  # High optical flow
            blink_score < 0.70
        )
        
        # Decision pathways
        if temporal_score >= 0.50 and cnn_score >= 0.50:
            if natural_fast_motion:
                # FIXED: Stronger penalty for natural motion false positives
                final_score = base_fusion * 0.70  # Reduced from 0.85
                decision_source = "borderline_natural_motion"
                logger.warning("⚠️ High confidence false positive risk: natural motion detected")
            else:
                final_score = max(base_fusion, 0.60)
                decision_source = "strong_agreement_fake"
        
        elif temporal_score < 0.35 and cnn_score < 0.35:
            final_score = min(base_fusion, 0.30)
            decision_source = "strong_agreement_real"
        
        # NEW: When CNN strongly says REAL despite high temporal score
        elif cnn_score < 0.47 and temporal_score > 0.50 and landmark_score >= 0.65:
            # CNN confidently says REAL, landmarks high (natural talking motion)
            final_score = 0.25 * temporal_score + 0.75 * cnn_score  # Trust CNN heavily
            decision_source = "cnn_real_natural_motion"
            logger.info(f"CNN says REAL despite high landmarks - likely natural motion")
        
        elif landmark_score >= 0.65 and cnn_score >= 0.45:
            if optical_flow_score < 0.40:
                # High landmarks but low flow = likely talking/natural motion
                # Give even more weight to CNN (which says REAL)
                final_score = 0.30 * temporal_score + 0.70 * cnn_score  # More weight to CNN
                decision_source = "uncertain_natural_motion"
                logger.info("Natural motion pattern: high landmarks but low optical flow")
            else:
                final_score = max(base_fusion, 0.55)
                decision_source = "temporal_landmark_cnn_confirm"
        
        elif cnn_score >= 0.70 and temporal_score < 0.50:
            final_score = 0.60 * cnn_score + 0.40 * temporal_score
            decision_source = "cnn_override"
        
        elif temporal_score >= 0.65 and cnn_score < 0.50:
            final_score = 0.60 * temporal_score + 0.40 * cnn_score
            decision_source = "temporal_override"
        
        elif abs(temporal_score - cnn_score) > 0.30:
            final_score = base_fusion * 0.90
            decision_source = "disagreement_uncertain"
        
        # NEW: Flag for manual review if high disagreement
        needs_manual_review = False
        review_reason = None
        
        if abs(temporal_score - cnn_score) > 0.40:
            needs_manual_review = True
            review_reason = "High component disagreement - models fundamentally disagree"
            logger.warning(f"⚠️ MANUAL REVIEW RECOMMENDED: {review_reason}")
        
        if cnx_score < 0.10 and (cnn_score > 0.70 or temporal_score > 0.70):
            needs_manual_review = True
            review_reason = "ConvNeXt strongly disagrees with other systems"
            logger.warning(f"⚠️ MANUAL REVIEW RECOMMENDED: {review_reason}")
        
        final_score = float(np.clip(final_score, 0.0, 1.0))
        
        # Determine label with adjusted threshold
        # UPDATED: Very conservative thresholds to reduce false positives
        if decision_source == "cnn_real_natural_motion":
            THRESHOLD = 0.60  # Change from 0.70
        elif decision_source in ["uncertain_natural_motion", ...]:
            THRESHOLD = 0.60  # Change from 0.65
        else:
            THRESHOLD = 0.55  # Change from 0.60
        
        if final_score >= THRESHOLD:
            label = "Fake"
            confidence = final_score
        else:
            label = "Real"
            confidence = 1 - final_score
        
        # FIXED: Update video_info with actual face count
        temporal_results_copy = temporal_results.copy()
        if "video_info" in temporal_results_copy:
            temporal_results_copy["video_info"]["faces_detected"] = self.faces_detected_count
        
        return {
            "label": label,
            "confidence": float(round(confidence, 4)),
            "final_score": float(round(final_score, 4)),
            "decision_source": decision_source,
            "temporal_score": float(round(temporal_score, 4)),
            "cnn_score": float(round(cnn_score, 4)),
            "quality_score": float(round(quality_score, 4)),
            "temporal_analysis": temporal_results_copy,  # FIXED: Updated with face count
            "cnn_analysis": cnn_results,
            "quality_analysis": quality_results,
            "needs_manual_review": needs_manual_review,  # NEW
            "review_reason": review_reason,  # NEW
            "fusion_weights": {
                "temporal": 0.40,
                "cnn": 0.40,
                "quality": 0.20
            }
        }


# Convenience alias
HybridVideoAnalyzer = ImprovedHybridAnalyzer

