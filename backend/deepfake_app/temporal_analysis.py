"""
temporal_analysis.py (No dlib required)
Complete implementation of temporal consistency analysis for deepfake detection
Works with just OpenCV and basic face detection
"""

import cv2
import numpy as np
import logging
from collections import defaultdict
from tqdm import tqdm
logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Analyzes temporal consistency in videos to detect deepfakes
    
    Components:
    1. Blink detection (deepfakes often have abnormal blinking)
    2. Optical flow analysis (motion inconsistencies)
    3. Facial landmark stability (using OpenCV instead of dlib)
    """
    
    def __init__(self, max_frames=30, skip_frames=2):
        """
        Args:
            max_frames: Maximum number of frames to analyze
            skip_frames: Skip every N frames for efficiency
        """
        self.max_frames = max_frames
        self.skip_frames = skip_frames
        
        # Initialize face detector
        self._init_face_detector()
    
    def _init_face_detector(self):
        """Initialize face detector"""
        try:
            from facenet_pytorch import MTCNN
            import torch
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            self.face_detector = MTCNN(keep_all=False, device=DEVICE)
            self.face_detector_type = "mtcnn"
            logger.info("✅ MTCNN initialized")
        except:
            # Fallback to Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            self.face_detector_type = "haar"
            logger.info("✅ Haar Cascade initialized")
    
    def analyze_video(self, video_path):
        """
        Main analysis function
        
        Args:
            video_path: Path to video file
        
        Returns:
            dict: Complete temporal analysis results
        """
        logger.info(f"Starting temporal analysis on: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"error": "Cannot open video file"}
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {total_frames} frames, {fps:.1f} fps, {duration:.2f}s")
        
        # Collect frames for analysis
        frames_data = self._extract_frames_for_analysis(cap, total_frames)
        
        cap.release()
        
        if not frames_data:
            return {"error": "No faces detected in video"}
        
        # Run analyses
        blink_result = self._analyze_blinks(frames_data)
        optical_flow_result = self._analyze_optical_flow(frames_data)
        landmark_result = self._analyze_landmark_stability(frames_data)
        
        # Calculate overall temporal consistency score
        temporal_score, decision_source = self._calculate_temporal_score(
            blink_result, optical_flow_result, landmark_result
        )
        
        # Interpret results
        interpretation = self._interpret_temporal_score(temporal_score)
        
        return {
            "temporal_consistency_score": float(temporal_score),
            "interpretation": interpretation,
            "decision_source": decision_source,
            "blink_analysis": blink_result,
            "optical_flow_analysis": optical_flow_result,
            "landmark_stability": landmark_result,
            "video_info": {
                "duration": float(duration),
                "fps": float(fps),
                "total_frames": int(total_frames),
                "processed_frames": len(frames_data),
                "faces_detected": len(frames_data)
            }
        }
    
    def _extract_frames_for_analysis(self, cap, total_frames):
        """Extract frames with faces for temporal analysis"""
        frames_data = []
        
        # Calculate frame indices to process
        num_frames = min(self.max_frames, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for idx in tqdm(frame_indices, desc="⏱ Temporal Frames", leave=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Detect face
            face_box = self._detect_face(frame)
            
            if face_box is not None:
                # Get face region
                x1, y1, x2, y2 = face_box
                face = frame[y1:y2, x1:x2]
                
                # Get simple landmarks (eye regions) using OpenCV
                eye_regions = self._detect_eye_regions(frame, face_box)
                
                frames_data.append({
                    "frame_index": int(idx),
                    "frame": frame,
                    "face": face,
                    "face_box": face_box,
                    "eye_regions": eye_regions
                })
        
        logger.info(f"Processed {len(frames_data)} frames")
        
        return frames_data
    
    def _detect_face(self, frame):
        """Detect face in frame"""

        # ---------------- MTCNN PATH ----------------
        if self.face_detector_type == "mtcnn":
            boxes, _ = self.face_detector.detect(frame)

            if boxes is None or len(boxes) == 0:
                return None

            x1, y1, x2, y2 = map(int, boxes[0])

            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                return (x1, y1, x2, y2)

            return None

        # ---------------- HAAR PATH ----------------
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                return (x, y, x + w, y + h)

            return None

    
    def _detect_eye_regions(self, frame, face_box):
        """
        Detect eye regions using Haar cascade (no dlib needed)
        Returns approximate eye positions
        """
        x1, y1, x2, y2 = face_box
        face_region = frame[y1:y2, x1:x2]
        
        # Load eye cascade
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 5)
        
        eye_regions = []
        for (ex, ey, ew, eh) in eyes[:2]:  # Take first 2 eyes
            # Convert to frame coordinates
            abs_x = x1 + ex
            abs_y = y1 + ey
            eye_regions.append({
                "x": abs_x,
                "y": abs_y,
                "w": ew,
                "h": eh,
                "center": (abs_x + ew//2, abs_y + eh//2)
            })
        
        return eye_regions
    
    def _analyze_blinks(self, frames_data):
        """
        Analyze blinking patterns using brightness changes in eye regions
        Deepfakes often have abnormal or missing blinks
        """
        if len(frames_data) < 3:
            return {
                "score": 0.5,
                "blink_count": 0,
                "interpretation": "Too few frames for blink analysis"
            }
        
        # Detect blinks based on eye region brightness changes
        blink_count = self._detect_blinks_from_eye_regions(frames_data)
        
        # Calculate expected blinks (average 15-20 per minute)
        duration_seconds = len(frames_data) * 2.0  # Rough estimate
        expected_blinks = (duration_seconds / 60.0) * 17.5
        
        # Score based on deviation from expected
        if expected_blinks > 0:
            blink_ratio = blink_count / expected_blinks
        else:
            blink_ratio = 0
        
        # Deepfakes typically have fewer blinks
        if blink_ratio < 0.3:
            score = 0.8  # Very suspicious
            interpretation = "Very few blinks detected - suspicious"
        elif blink_ratio < 0.6:
            score = 0.5
            interpretation = "Below normal blink rate"
        elif blink_ratio > 1.5:
            score = 0.4
            interpretation = "Excessive blinking - unusual"
        else:
            score = 0.2
            interpretation = "Normal blink rate"
        
        return {
            "score": float(score),
            "blink_count": int(blink_count),
            "expected_blinks": float(expected_blinks),
            "interpretation": interpretation
        }
    
    def _detect_blinks_from_eye_regions(self, frames_data):
        """Detect blinks using eye region brightness changes"""
        blink_count = 0
        
        eye_brightness = []
        for frame_data in frames_data:
            eye_regions = frame_data.get("eye_regions", [])
            
            if not eye_regions:
                # Fallback: use upper face region
                face = frame_data["face"]
                h = face.shape[0]
                upper_face = face[:h//3, :]
                gray = cv2.cvtColor(upper_face, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
            else:
                # Use actual eye regions
                frame = frame_data["frame"]
                brightness_values = []
                
                for eye in eye_regions:
                    x, y, w, h = eye["x"], eye["y"], eye["w"], eye["h"]
                    eye_img = frame[y:y+h, x:x+w]
                    if eye_img.size > 0:
                        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
                        brightness_values.append(np.mean(gray))
                
                brightness = np.mean(brightness_values) if brightness_values else 128
            
            eye_brightness.append(brightness)
        
        # Detect sudden drops (potential blinks)
        threshold = 0.92  # 8% drop
        for i in range(1, len(eye_brightness) - 1):
            if (eye_brightness[i] < eye_brightness[i-1] * threshold and 
                eye_brightness[i] < eye_brightness[i+1] * threshold):
                blink_count += 1
        
        return blink_count
    
    def _analyze_optical_flow(self, frames_data):
        """
        Analyze optical flow consistency
        Manipulated videos often have inconsistent motion
        """
        if len(frames_data) < 2:
            return {
                "score": 0.5,
                "interpretation": "Too few frames for optical flow"
            }
        
        flow_magnitudes = []
        flow_inconsistencies = []
        
        for i in range(len(frames_data) - 1):
            frame1 = frames_data[i]["face"]
            frame2 = frames_data[i + 1]["face"]
            
            # Ensure same size
            if frame1.shape != frame2.shape:
                frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate flow magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_magnitude = np.mean(magnitude)
            flow_magnitudes.append(avg_magnitude)
            
            # Check for inconsistencies (sudden changes)
            if i > 0:
                magnitude_change = abs(avg_magnitude - flow_magnitudes[i-1])
                flow_inconsistencies.append(magnitude_change)
        
        # Calculate score based on inconsistencies
        if len(flow_inconsistencies) > 0:
            avg_inconsistency = np.mean(flow_inconsistencies)
            std_inconsistency = np.std(flow_inconsistencies)
            
            # High inconsistency = likely manipulated
            # But also check if it's just high overall motion (natural)
            avg_motion = np.mean(flow_magnitudes)
            
            # Normalize inconsistency by motion
            if avg_motion > 1.0:
                normalized_inconsistency = avg_inconsistency / avg_motion
            else:
                normalized_inconsistency = avg_inconsistency
            
            # Score calculation
            if normalized_inconsistency > 0.5:
                score = 0.8
                interpretation = "Significant motion inconsistencies - likely manipulated"
            elif normalized_inconsistency > 0.3:
                score = 0.6
                interpretation = "Moderate motion inconsistencies"
            else:
                score = 0.3
                interpretation = "Consistent motion - likely authentic"
        else:
            score = 0.5
            interpretation = "Insufficient data"
        
        return {
            "score": float(score),
            "avg_flow_magnitude": float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0,
            "flow_inconsistency": float(np.mean(flow_inconsistencies)) if flow_inconsistencies else 0.0,
            "interpretation": interpretation
        }
    
    def _analyze_landmark_stability(self, frames_data):
        """
        Analyze facial tracking stability using eye positions.
        Landmark jitter is NORMALIZED by face size.

        FIX: left and right eye series each carry their own face_sizes array
        so the shapes always match after np.diff — previously right_eye_positions
        could have fewer entries than face_sizes (only added when 2 eyes detected)
        causing a shape mismatch when dividing rj / face_sizes[2:].
        """
        left_eye_positions  = []
        left_face_sizes     = []
        right_eye_positions = []
        right_face_sizes    = []

        for frame_data in frames_data:
            eye_regions = frame_data.get("eye_regions", [])
            x1, y1, x2, y2 = frame_data["face_box"]

            face_width  = x2 - x1
            face_height = y2 - y1
            face_diag   = np.sqrt(face_width**2 + face_height**2)

            if face_diag <= 0:
                continue

            if len(eye_regions) >= 2:
                eyes_sorted = sorted(eye_regions, key=lambda e: e["center"][0])
                left_eye_positions.append(eyes_sorted[0]["center"])
                left_face_sizes.append(face_diag)
                right_eye_positions.append(eyes_sorted[1]["center"])
                right_face_sizes.append(face_diag)
            elif len(eye_regions) == 1:
                left_eye_positions.append(eye_regions[0]["center"])
                left_face_sizes.append(face_diag)

        if len(left_eye_positions) < 3:
            return {
                "score": 0.5,
                "avg_normalized_jitter": 0.0,
                "frames_analyzed": len(frames_data),
                "interpretation": "Insufficient eye tracking data"
            }

        # Left eye jitter normalized by left-eye-series face sizes
        left_positions = np.array(left_eye_positions)
        left_sizes     = np.array(left_face_sizes)

        lv = np.diff(left_positions, axis=0)
        la = np.diff(lv, axis=0)
        lj = np.sqrt(np.sum(la**2, axis=1))
        lj_norm    = lj / left_sizes[2:]   # left_sizes[2:] always same length as lj
        avg_jitter = float(np.mean(lj_norm))

        # Right eye jitter — uses its OWN face_sizes so shapes always match
        if len(right_eye_positions) >= 3:
            right_positions = np.array(right_eye_positions)
            right_sizes     = np.array(right_face_sizes)

            rv = np.diff(right_positions, axis=0)
            ra = np.diff(rv, axis=0)
            rj = np.sqrt(np.sum(ra**2, axis=1))
            rj_norm    = rj / right_sizes[2:]   # right_sizes[2:] always same length as rj
            avg_jitter = (avg_jitter + float(np.mean(rj_norm))) / 2

        # Scoring
        if avg_jitter > 0.025:
            score          = 0.8
            interpretation = "High instability - possible face swap artifacts"
        elif avg_jitter > 0.015:
            score          = 0.5
            interpretation = "Moderate landmark instability"
        else:
            score          = 0.2
            interpretation = "Stable normalized landmarks - likely authentic"

        return {
            "score":                 float(score),
            "avg_normalized_jitter": float(avg_jitter),
            "frames_analyzed":       len(frames_data),
            "interpretation":        interpretation
        }
    
    def _calculate_temporal_score(self, blink_result, optical_flow_result, landmark_result):
        """
        Calculate overall temporal consistency score
        
        Returns:
            (score, decision_source): Overall score and decision method
        """
        blink_score = blink_result["score"]
        flow_score = optical_flow_result["score"]
        landmark_score = landmark_result["score"]
        
        # Weighted ensemble
        # Optical flow is most reliable
        # Landmarks are good for face swaps
        # Blinks are supplementary
        
        base_score = (
            0.40 * flow_score +
            0.40 * landmark_score +
            0.20 * blink_score
        )
        
        decision_source = "weighted_ensemble"
        
        # Special cases
        
        # Strong agreement (all high)
        if blink_score > 0.6 and flow_score > 0.6 and landmark_score > 0.6:
            final_score = max(base_score, 0.7)
            decision_source = "strong_agreement_fake"
        
        # Strong agreement (all low)
        elif blink_score < 0.4 and flow_score < 0.4 and landmark_score < 0.4:
            final_score = min(base_score, 0.3)
            decision_source = "strong_agreement_real"
        
        # Landmark + flow agree (face swap signature)
        elif landmark_score > 0.65 and flow_score > 0.55:
            final_score = max(base_score, 0.65)
            decision_source = "landmark_flow_agree"
        
        # Disagreement
        elif max(blink_score, flow_score, landmark_score) - min(blink_score, flow_score, landmark_score) > 0.5:
            final_score = base_score * 0.9  # Reduce confidence
            decision_source = "component_disagreement"
        
        else:
            final_score = base_score
        
        return float(np.clip(final_score, 0, 1)), decision_source
    
    def _interpret_temporal_score(self, score):
        """Interpret temporal consistency score"""
        if score < 0.30:
            return "🟢 AUTHENTIC - Strong temporal consistency"
        elif score < 0.45:
            return "🟡 LIKELY AUTHENTIC - Good temporal consistency"
        elif score < 0.60:
            return "🟠 SUSPICIOUS - Multiple temporal inconsistencies found"
        elif score < 0.75:
            return "🔴 LIKELY FAKE - Significant temporal anomalies"
        else:
            return "🔴 HIGHLY SUSPICIOUS - Severe temporal manipulation detected"


# Convenience function
def analyze_video_temporal(video_path, max_frames=30, skip_frames=2):
    """
    Quick analysis function
    
    Args:
        video_path: Path to video
        max_frames: Max frames to analyze
        skip_frames: Skip rate
    
    Returns:
        dict: Analysis results
    """
    analyzer = TemporalAnalyzer(max_frames=max_frames, skip_frames=skip_frames)
    return analyzer.analyze_video(video_path)