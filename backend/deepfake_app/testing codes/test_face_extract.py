"""
extract_face_for_test.py
Extract a face from a video to test ConvNeXt model
"""

import cv2
import sys
from pathlib import Path

def extract_face_from_video(video_path, output_path="test_face.jpg"):
    """Extract a face from the first valid frame of a video"""
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    
    # Try multiple frames to find one with a face
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for frame_num in range(0, min(100, total_frames), 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Crop to face
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            
            # Resize to 224x224 (ConvNeXt input size)
            face_resized = cv2.resize(face, (224, 224))
            
            # Save
            cv2.imwrite(output_path, face_resized)
            print(f"✅ Extracted face from frame {frame_num}")
            print(f"✅ Saved to: {output_path}")
            
            cap.release()
            return True
    
    cap.release()
    print(f"❌ No face found in video")
    return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default video
        video_path = "D:/Projects/Major Project/Deepfake Detection/datasets/dfd/manipulated/01_02__hugging_happy__YVGY8LOK.mp4"
    
    output = "test_face_fake.jpg"
    
    if extract_face_from_video(video_path, output):
        print(f"\n🎯 Now run:")
        print(f'python debug_convnext.py "D:/Projects/Major Project/Deepfake Detection/models/convnext_tiny_deepfake.pth" {output}')
    else:
        print(f"\n❌ Failed to extract face. Try a different video.")