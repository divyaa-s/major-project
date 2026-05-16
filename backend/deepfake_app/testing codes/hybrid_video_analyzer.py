"""
Hybrid Video Deepfake Detector
Combines Temporal Analysis + CNN Models for superior detection
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

# Import your temporal analyzer
from temporal_analysis import TemporalAnalyzer
#from backend.deepfake_app.temporal_analysis_copy import TemporalAnalyzer

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Your model configurations
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


class HybridVideoAnalyzer:
    """
    Combines temporal consistency analysis with CNN-based frame analysis
    for comprehensive deepfake detection
    """
    
    def __init__(self, num_keyframes=100, temporal_max_frames=30, temporal_skip=2):
        """
        Args:
            num_keyframes: Number of frames to analyze with CNN models
            temporal_max_frames: Max frames for temporal analysis
            temporal_skip: Skip rate for temporal analysis
        """
        self.num_keyframes = num_keyframes
        self.temporal_analyzer = TemporalAnalyzer(
            max_frames=temporal_max_frames,
            skip_frames=temporal_skip
        )
        
        # Initialize face detector
        try:
            self.mtcnn = MTCNN(keep_all=False, device=DEVICE)
            self.use_mtcnn = True
            logger.info("✅ MTCNN initialized")
        except Exception as e:
            logger.warning(f"⚠️ MTCNN initialization failed: {e}")
            logger.warning("Will use OpenCV Haar Cascade as fallback")
            self.use_mtcnn = False
            # Load OpenCV face detector as fallback
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load CNN models
        self.models = {}
        self._load_cnn_models()
        
        logger.info(f"Initialized HybridVideoAnalyzer with {len(self.models)} CNN models")
    
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
        """
        Complete hybrid analysis of video
        
        Returns:
            dict: Comprehensive analysis results
        """
        logger.info(f"🎬 Starting hybrid analysis: {video_path}")
        
        # 1. Temporal Analysis
        logger.info("📊 Running temporal consistency analysis...")
        temporal_results = self.temporal_analyzer.analyze_video(video_path)
        
        if "error" in temporal_results:
            return temporal_results
        
        # 2. Extract keyframes
        logger.info(f"🖼️  Extracting {self.num_keyframes} keyframes...")
        keyframes = self._extract_keyframes(video_path)
        
        if not keyframes:
            return {
                "error": "No keyframes could be extracted",
                "temporal_results": temporal_results
            }
        
        # 3. CNN Analysis on keyframes
        logger.info("🤖 Running CNN models on keyframes...")
        cnn_results = self._analyze_keyframes_with_cnns(keyframes)
        
        # 4. Fusion
        logger.info("⚡ Fusing temporal + CNN predictions...")
        final_results = self._fuse_predictions(temporal_results, cnn_results)
        
        return final_results
    
    def _extract_keyframes(self, video_path):
        """Extract evenly-spaced keyframes from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        # Select evenly spaced frames
        frame_indices = np.linspace(0, total_frames - 1, self.num_keyframes, dtype=int)
        
        keyframes = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Crop face
                face = self._crop_face(frame)
                if face is not None:
                    keyframes.append({
                        "frame_index": int(idx),
                        "face": face,
                        "frame": frame
                    })
        
        cap.release()
        logger.info(f"   Extracted {len(keyframes)}/{self.num_keyframes} frames with faces")
        
        return keyframes
    
    def _crop_face(self, frame):
        """Crop face from frame using MTCNN or OpenCV fallback"""
        if self.use_mtcnn:
            return self._crop_face_mtcnn(frame)
        else:
            return self._crop_face_opencv(frame)
    
    def _crop_face_mtcnn(self, frame):
        """Crop face using MTCNN"""
        try:
            # Convert frame to RGB (ensure it's the right type)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Ensure it's a contiguous array with correct dtype
            img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
            
            # Convert to PIL Image for MTCNN
            img_pil = Image.fromarray(img_rgb)
            
            # Detect face
            boxes, _ = self.mtcnn.detect(img_pil)
            
            if boxes is None:
                return None
            
            # Crop face
            x1, y1, x2, y2 = boxes[0].astype(int)
            
            # Add some padding
            h, w = img_rgb.shape[:2]
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            face = img_rgb[y1:y2, x1:x2]
            
            return face
        except Exception as e:
            logger.warning(f"MTCNN face crop failed: {e}")
            return None
    
    def _crop_face_opencv(self, frame):
        """Crop face using OpenCV Haar Cascade (fallback)"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return None
            
            # Get largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Add padding
            img_h, img_w = frame.shape[:2]
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img_w, x + w + padding)
            y2 = min(img_h, y + h + padding)
            
            face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = face[y1:y2, x1:x2]
            
            return face
        except Exception as e:
            logger.warning(f"OpenCV face crop failed: {e}")
            return None
    
    def _preprocess_face(self, face, size):
        """Preprocess face for CNN model"""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        return transform(face).unsqueeze(0).to(DEVICE)
    
    def _analyze_keyframes_with_cnns(self, keyframes):
        """Run all CNN models on keyframes"""
        frame_predictions = []
        
        for kf in keyframes:
            frame_preds = {
                "frame_index": kf["frame_index"],
                "model_predictions": {}
            }
            
            # Run each CNN model
            for name, model_info in self.models.items():
                try:
                    model = model_info["model"]
                    config = model_info["config"]
                    
                    # Preprocess
                    x = self._preprocess_face(kf["face"], config["input_size"])
                    
                    # Predict
                    with torch.no_grad():
                        output = model(x)
                        probs = F.softmax(output, dim=1)[0]
                        fake_prob = probs[1].item()
                    
                    frame_preds["model_predictions"][name] = float(fake_prob)
                    
                except Exception as e:
                    logger.warning(f"Model {name} failed on frame {kf['frame_index']}: {e}")
            
            frame_predictions.append(frame_preds)
        
        # Calculate ensemble scores
        cnn_ensemble = self._calculate_cnn_ensemble(frame_predictions)
        
        return {
            "frame_predictions": frame_predictions,
            "ensemble": cnn_ensemble
        }
    
    def _calculate_cnn_ensemble(self, frame_predictions):
        """Calculate weighted ensemble across all frames and models"""
        if not frame_predictions:
            return {
                "avg_fake_probability": 0.5,
                "per_model_avg": {}
            }
        
        # Collect all predictions per model
        model_scores = {name: [] for name in self.models.keys()}
        
        for frame_pred in frame_predictions:
            for model_name, fake_prob in frame_pred["model_predictions"].items():
                model_scores[model_name].append(fake_prob)
        
        # Average per model
        per_model_avg = {}
        for model_name, scores in model_scores.items():
            if scores:
                per_model_avg[model_name] = float(np.mean(scores))
        
        # Weighted ensemble (same weights as your original pipeline)
        if len(per_model_avg) >= 4:
            cnn_ensemble = (
                0.25 * per_model_avg.get("efficientnet_b3", 0.5) +
                0.25 * per_model_avg.get("xception", 0.5) +
                0.30 * per_model_avg.get("convnext", 0.5) +
                0.20 * per_model_avg.get("vit", 0.5)
            )
        else:
            # If some models failed, simple average
            cnn_ensemble = np.mean(list(per_model_avg.values())) if per_model_avg else 0.5
        
        return {
            "avg_fake_probability": float(cnn_ensemble),
            "per_model_avg": per_model_avg,
            "frames_analyzed": len(frame_predictions)
        }
    
    def _fuse_predictions(self, temporal_results, cnn_results):
        """
        Fuse temporal and CNN predictions with intelligent weighting
        """
        temporal_score = temporal_results["temporal_consistency_score"]
        cnn_score = cnn_results["ensemble"]["avg_fake_probability"]
        
        # Get component scores for intelligent fusion
        landmark_score = temporal_results["landmark_stability"]["score"]
        blink_score = temporal_results["blink_analysis"]["score"]
        optical_flow_score = temporal_results["optical_flow_analysis"]["score"]
        
        # Base fusion: 50/50 split
        base_fusion = 0.50 * temporal_score + 0.50 * cnn_score
        
        # Decision logic with multiple pathways
        decision_source = "hybrid_ensemble"
        final_score = base_fusion
        
        # NEW: Check for natural motion (fast movement scenario)
        # High landmarks + LOW optical flow = likely natural fast movement, not fake
        natural_fast_motion = (landmark_score >= 0.65 and 
                               optical_flow_score < 0.40 and 
                               blink_score < 0.70)
        
        # PATHWAY 1: Strong agreement (both high) - BUT check for false positive
        if temporal_score >= 0.50 and cnn_score >= 0.50:
            # Check if this might be natural fast motion being misclassified
            if natural_fast_motion:
                # Reduce confidence, don't automatically boost
                final_score = base_fusion * 0.85  # 15% penalty
                decision_source = "borderline_natural_motion"
                logger.warning("⚠️ Possible false positive: High landmarks but natural optical flow")
            else:
                # Both systems agree it's fake - high confidence
                final_score = max(base_fusion, 0.60)
                decision_source = "strong_agreement_fake"
        
        # PATHWAY 2: Strong agreement (both low)
        elif temporal_score < 0.35 and cnn_score < 0.35:
            # Both systems agree it's real - high confidence
            final_score = min(base_fusion, 0.30)
            decision_source = "strong_agreement_real"
        
        # PATHWAY 3: Temporal suspicious, CNN confirms - BUT with caveats
        elif landmark_score >= 0.65 and cnn_score >= 0.45:
            # Check if optical flow is natural
            if optical_flow_score < 0.40:
                # Landmarks unstable but motion is natural - reduce confidence
                final_score = 0.40 * temporal_score + 0.60 * cnn_score
                decision_source = "uncertain_natural_motion"
                logger.warning("⚠️ High landmarks + natural flow: possible fast movement")
            else:
                # Landmarks unstable + CNN suspicious + optical flow bad = likely fake
                final_score = max(base_fusion, 0.55)
                decision_source = "temporal_landmark_cnn_confirm"
        
        # PATHWAY 4: CNN very confident, temporal borderline
        elif cnn_score >= 0.70 and temporal_score < 0.50:
            # CNN caught something temporal missed
            final_score = 0.60 * cnn_score + 0.40 * temporal_score
            decision_source = "cnn_override"
        
        # PATHWAY 5: Temporal very confident, CNN borderline
        elif temporal_score >= 0.65 and cnn_score < 0.50:
            # Temporal caught something CNN missed
            final_score = 0.60 * temporal_score + 0.40 * cnn_score
            decision_source = "temporal_override"
        
        # PATHWAY 6: Disagreement (one high, one low)
        elif abs(temporal_score - cnn_score) > 0.30:
            # Systems disagree - be cautious, use average with penalty
            final_score = base_fusion * 0.90  # 10% penalty for disagreement
            decision_source = "disagreement_uncertain"
        
        # Clip to valid range
        final_score = float(np.clip(final_score, 0.0, 1.0))
        
        # Determine label with adjusted threshold for borderline cases
        # If decision source indicates uncertainty, raise threshold
        if decision_source in ["borderline_natural_motion", "uncertain_natural_motion"]:
            THRESHOLD = 0.55  # Higher threshold for uncertain cases
        else:
            THRESHOLD = 0.50
        
        if final_score >= THRESHOLD:
            label = "Fake"
            confidence = final_score
        else:
            label = "Real"
            confidence = 1 - final_score
        
        return {
            "label": label,
            "confidence": float(round(confidence, 4)),
            "final_score": float(round(final_score, 4)),
            "decision_source": decision_source,
            
            # Component scores
            "temporal_score": float(round(temporal_score, 4)),
            "cnn_score": float(round(cnn_score, 4)),
            
            # Detailed breakdowns
            "temporal_analysis": temporal_results,
            "cnn_analysis": cnn_results,
            
            # Fusion details
            "fusion_weights": {
                "temporal": 0.50,
                "cnn": 0.50
            }
        }


# Convenience function for easy usage
def analyze_video_hybrid(video_path, num_keyframes=5):
    """
    Easy-to-use function for hybrid video analysis
    
    Args:
        video_path: Path to video file
        num_keyframes: Number of frames to analyze with CNNs (default: 5)
    
    Returns:
        dict: Complete analysis results
    """
    analyzer = HybridVideoAnalyzer(num_keyframes=num_keyframes)
    return analyzer.analyze_video(video_path)