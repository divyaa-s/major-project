"""
Standalone Test Script for Temporal Consistency Analysis
No Django required - just test the temporal analysis on videos

Usage:
    python test_temporal_analysis.py path/to/video.mp4
    python test_temporal_analysis.py path/to/video.mp4 --frames 30 --skip 2
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
import sys
import argparse
from scipy.spatial import distance
from collections import deque
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Eye landmark indices (MediaPipe 468-point model)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


class TemporalAnalyzer:
    """Analyzes video temporal consistency for deepfake detection"""
    
    def __init__(self, max_frames=30, skip_frames=2):
        self.max_frames = max_frames
        self.skip_frames = skip_frames
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def analyze_video(self, video_path):
        """Main analysis pipeline"""
        logger.info(f"🎬 Starting temporal analysis on: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error("❌ Failed to open video")
            return {"error": "Failed to open video"}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"📊 Video Info: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s")
        
        # Storage for analysis
        blink_intervals = []
        last_blink_frame = None
        ear_history = deque(maxlen=10)
        
        optical_flows = []
        prev_gray = None
        
        landmark_positions = []
        frame_count = 0
        processed_count = 0
        faces_detected = 0
        
        print("\n⏳ Processing frames...")
        
        while cap.isOpened() and processed_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for performance
            if frame_count % (self.skip_frames + 1) != 0:
                frame_count += 1
                continue
            
            # Progress indicator
            if processed_count % 5 == 0:
                print(f"   Frame {processed_count}/{self.max_frames}...", end='\r')
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Process facial landmarks
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                faces_detected += 1
                landmarks = results.multi_face_landmarks[0]
                
                # 1. Blink Detection
                ear = self._calculate_ear(landmarks, frame.shape)
                ear_history.append(ear)
                
                if len(ear_history) >= 3:
                    avg_ear = np.mean(list(ear_history))
                    if ear < 0.18 and avg_ear > 0.22:
                        if last_blink_frame is not None:
                            interval = frame_count - last_blink_frame
                            blink_intervals.append(interval)
                        last_blink_frame = frame_count
                
                # 2. Store landmark positions
                landmark_positions.append(
                    self._extract_key_landmarks(landmarks, frame.shape)
                )
            
            # 3. Optical Flow Analysis
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray_frame, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                optical_flows.append(np.mean(mag))
            
            prev_gray = gray_frame
            frame_count += 1
            processed_count += 1
        
        cap.release()
        print(f"\n✅ Processed {processed_count} frames ({faces_detected} with detected faces)")
        
        # Calculate scores
        logger.info("\n📈 Calculating scores...")
        blink_score = self._analyze_blink_pattern(blink_intervals, fps, duration)
        optical_flow_score = self._analyze_optical_flow(optical_flows)
        landmark_score = self._analyze_landmark_stability(landmark_positions)
        
        # Combined temporal consistency score
        # Adjusted weights: landmarks more reliable for modern deepfakes
        temporal_score = (
            0.20 * blink_score +      # Reduced (deepfakes getting better at blinks)
            0.30 * optical_flow_score + # Reduced slightly
            0.50 * landmark_score      # Increased (most reliable indicator)
        )
        
        # Override logic: High landmark instability is a strong signal
        decision_source = "weighted_ensemble"
        if landmark_score >= 0.65 and temporal_score < 0.50:
            temporal_score = max(temporal_score, 0.45)
            decision_source = "landmark_override"
            logger.warning(f"⚠️ Landmark override triggered! Landmark score: {landmark_score:.4f}")
        
        return {
            "temporal_consistency_score": float(round(temporal_score, 4)),
            "interpretation": self._interpret_score(temporal_score),
            "decision_source": decision_source,
            "weights_used": {
                "blink": 0.20,
                "optical_flow": 0.30,
                "landmark": 0.50
            },
            "blink_analysis": {
                "score": float(round(blink_score, 4)),
                "blink_count": int(len(blink_intervals)),
                "avg_interval": float(round(np.mean(blink_intervals), 2)) if blink_intervals else 0.0,
                "interpretation": self._interpret_blink(blink_score, len(blink_intervals), duration)
            },
            "optical_flow_analysis": {
                "score": float(round(optical_flow_score, 4)),
                "variance": float(round(np.var(optical_flows), 4)) if optical_flows else 0.0,
                "interpretation": self._interpret_optical_flow(optical_flow_score)
            },
            "landmark_stability": {
                "score": float(round(landmark_score, 4)),
                "frames_analyzed": int(len(landmark_positions)),
                "interpretation": self._interpret_landmark(landmark_score)
            },
            "video_info": {
                "total_frames": int(total_frames),
                "processed_frames": int(processed_count),
                "faces_detected": int(faces_detected),
                "fps": float(round(fps, 2)),
                "duration": float(round(duration, 2))
            }
        }
    
    def _calculate_ear(self, landmarks, frame_shape):
        """Calculate Eye Aspect Ratio for blink detection"""
        h, w = frame_shape[:2]
        
        def get_coords(indices):
            coords = []
            for idx in indices:
                lm = landmarks.landmark[idx]
                coords.append([lm.x * w, lm.y * h])
            return np.array(coords)
        
        left_eye = get_coords(LEFT_EYE)
        right_eye = get_coords(RIGHT_EYE)
        
        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        
        return (left_ear + right_ear) / 2.0
    
    def _eye_aspect_ratio(self, eye):
        """Calculate aspect ratio of eye"""
        v1 = distance.euclidean(eye[1], eye[5])
        v2 = distance.euclidean(eye[2], eye[4])
        h = distance.euclidean(eye[0], eye[3])
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def _extract_key_landmarks(self, landmarks, frame_shape):
        """Extract key facial landmarks for stability tracking"""
        h, w = frame_shape[:2]
        
        # Key points: nose tip, left eye, right eye, mouth corners
        key_indices = [1, 33, 263, 61, 291]
        
        coords = []
        for idx in key_indices:
            lm = landmarks.landmark[idx]
            coords.append([lm.x * w, lm.y * h])
        
        return np.array(coords)
    
    def _analyze_blink_pattern(self, intervals, fps, duration):
        """Analyze blink pattern authenticity"""
        if not intervals or duration < 1:
            return 0.5
        
        blink_count = len(intervals)
        blinks_per_minute = (blink_count / duration) * 60
        
        if 10 <= blinks_per_minute <= 25:
            rate_score = 0.0
        elif blinks_per_minute < 5:
            rate_score = 0.8
        elif blinks_per_minute > 35:
            rate_score = 0.7
        else:
            rate_score = 0.3
        
        if len(intervals) > 2:
            interval_variance = np.var(intervals)
            if interval_variance < 10:
                consistency_score = 0.6
            else:
                consistency_score = 0.1
        else:
            consistency_score = 0.5
        
        return (rate_score + consistency_score) / 2.0
    
    def _analyze_optical_flow(self, flows):
        """Analyze optical flow for unnatural motion patterns"""
        if len(flows) < 5:
            return 0.5
        
        flow_variance = np.var(flows)
        
        # UPDATED: More nuanced thresholds
        if flow_variance < 0.3:
            score = 0.8  # Very smooth = GAN-generated
        elif flow_variance < 0.5:
            score = 0.6  # Smooth but acceptable
        elif flow_variance > 15:
            score = 0.7  # Very jerky
        elif flow_variance > 10:
            score = 0.5  # Moderately jerky
        else:
            score = 0.2  # Natural motion (0.5-10)
        
        # Check for sudden spikes (warping artifacts)
        flows_array = np.array(flows)
        diff = np.abs(np.diff(flows_array))
        
        # More lenient: 3x std instead of 2x
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        spike_threshold = mean_diff + 3 * std_diff
        spike_count = np.sum(diff > spike_threshold)
        
        # Only penalize if >15% have spikes
        if spike_count > len(flows) * 0.15:
            score += 0.3
        
        return min(score, 1.0)
    
    def _analyze_landmark_stability(self, positions):
        """Analyze facial landmark stability across frames"""
        if len(positions) < 5:
            return 0.5
        
        positions_array = np.array(positions)
        
        displacements = []
        for i in range(1, len(positions_array)):
            diff = positions_array[i] - positions_array[i-1]
            displacement = np.mean(np.linalg.norm(diff, axis=1))
            displacements.append(displacement)
        
        displacement_std = np.std(displacements)
        
        if displacement_std > 15:
            score = 0.7
        elif displacement_std < 2:
            score = 0.5
        else:
            score = 0.2
        
        return score
    
    def _interpret_score(self, score):
        """Interpret temporal consistency score"""
        if score < 0.3:
            return "🟢 LIKELY REAL - Natural temporal patterns detected"
        elif score < 0.5:
            return "🟡 BORDERLINE - Some anomalies detected, further analysis recommended"
        elif score < 0.7:
            return "🟠 SUSPICIOUS - Multiple temporal inconsistencies found"
        else:
            return "🔴 LIKELY FAKE - Strong evidence of manipulation"
    
    def _interpret_blink(self, score, count, duration):
        """Interpret blink analysis"""
        rate = (count / duration * 60) if duration > 0 else 0
        
        if score < 0.3:
            return f"Natural blink pattern ({rate:.1f} blinks/min)"
        elif count < 2:
            return "Very few blinks detected - suspicious"
        else:
            return f"Abnormal blink pattern ({rate:.1f} blinks/min)"
    
    def _interpret_optical_flow(self, score):
        """Interpret optical flow score"""
        if score < 0.3:
            return "Natural motion detected"
        elif score < 0.6:
            return "Some motion artifacts detected"
        else:
            return "Significant motion inconsistencies - likely manipulated"
    
    def _interpret_landmark(self, score):
        """Interpret landmark stability"""
        if score < 0.3:
            return "Stable facial tracking"
        elif score < 0.6:
            return "Moderate tracking instability"
        else:
            return "High instability - possible face swap artifacts"
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


def print_results(results):
    """Pretty print results"""
    print("\n" + "="*70)
    print("🎯 TEMPORAL CONSISTENCY ANALYSIS RESULTS")
    print("="*70)
    
    if "error" in results:
        print(f"\n❌ Error: {results['error']}")
        return
    
    # Main score
    score = results['temporal_consistency_score']
    print(f"\n📊 OVERALL TEMPORAL SCORE: {score:.4f}")
    print(f"   {results['interpretation']}")
    print(f"   Decision Source: {results.get('decision_source', 'N/A')}")
    
    # Show weights used
    weights = results.get('weights_used', {})
    if weights:
        print(f"\n⚖️  WEIGHTS USED:")
        print(f"   Blink: {weights.get('blink', 0):.0%}")
        print(f"   Optical Flow: {weights.get('optical_flow', 0):.0%}")
        print(f"   Landmark: {weights.get('landmark', 0):.0%}")
    
    # Blink analysis
    print(f"\n👁️  BLINK ANALYSIS:")
    blink = results['blink_analysis']
    print(f"   Score: {blink['score']:.4f}")
    print(f"   Blinks detected: {blink['blink_count']}")
    print(f"   Average interval: {blink['avg_interval']:.2f} frames")
    print(f"   {blink['interpretation']}")
    
    # Optical flow
    print(f"\n🌊 OPTICAL FLOW ANALYSIS:")
    flow = results['optical_flow_analysis']
    print(f"   Score: {flow['score']:.4f}")
    print(f"   Variance: {flow['variance']:.4f}")
    print(f"   {flow['interpretation']}")
    
    # Landmark stability
    print(f"\n📍 LANDMARK STABILITY:")
    landmark = results['landmark_stability']
    print(f"   Score: {landmark['score']:.4f}")
    print(f"   Frames analyzed: {landmark['frames_analyzed']}")
    print(f"   {landmark['interpretation']}")
    
    # Video info
    print(f"\n📹 VIDEO INFORMATION:")
    info = results['video_info']
    print(f"   Total frames: {info['total_frames']}")
    print(f"   Processed frames: {info['processed_frames']}")
    print(f"   Faces detected: {info['faces_detected']}")
    print(f"   FPS: {info['fps']:.2f}")
    print(f"   Duration: {info['duration']:.2f}s")
    
    print("\n" + "="*70)
    
    # Final recommendation
    print("\n💡 RECOMMENDATION:")
    if score < 0.3:
        print("   ✅ Video appears authentic based on temporal analysis")
    elif score < 0.5:
        print("   ⚠️  Video shows some anomalies - recommend CNN analysis")
    elif score < 0.7:
        print("   ⚠️  Video is suspicious - likely manipulated")
    else:
        print("   ❌ High confidence this video is manipulated")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Test temporal consistency analysis on a video file'
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to video file (mp4, avi, mov, etc.)'
    )
    parser.add_argument(
        '--frames',
        type=int,
        default=30,
        help='Maximum frames to analyze (default: 30)'
    )
    parser.add_argument(
        '--skip',
        type=int,
        default=2,
        help='Process every Nth frame (default: 2, means every 3rd frame)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Run analysis
    print(f"\n🚀 Starting temporal analysis...")
    print(f"   Video: {video_path.name}")
    print(f"   Max frames: {args.frames}")
    print(f"   Skip frames: {args.skip}")
    
    analyzer = TemporalAnalyzer(max_frames=args.frames, skip_frames=args.skip)
    results = analyzer.analyze_video(str(video_path))
    
    # Display results
    print_results(results)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n📖 Usage: python test_temporal_analysis.py path/to/video.mp4")
        print("\nExample:")
        print("   python test_temporal_analysis.py sample_video.mp4")
        print("   python test_temporal_analysis.py sample_video.mp4 --frames 50 --skip 1")
        sys.exit(1)
    
    main()