"""
debug_convnext.py
Debug script to test ConvNeXt model and diagnose the 0.0000 score issue
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from timm import create_model
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test_convnext_model(model_path, test_image_path=None):
    """
    Test ConvNeXt model to diagnose issues
    
    Args:
        model_path: Path to ConvNeXt model .pth file
        test_image_path: Optional path to a test face image
    """
    
    print("="*80)
    print("🔍 CONVNEXT MODEL DIAGNOSTIC TEST")
    print("="*80)
    
    print(f"\n1. Loading model from: {model_path}")
    print(f"   Device: {DEVICE}")
    
    # Check file exists and size
    import os
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found!")
        return
    
    file_size_mb = os.path.getsize(model_path) / (1024**2)
    print(f"   File size: {file_size_mb:.2f} MB")
    
    if file_size_mb < 50:
        print(f"   ⚠️  WARNING: File seems small for ConvNeXt (expected ~100-120 MB)")
    
    # Load model
    try:
        model = create_model("convnext_tiny", pretrained=False, num_classes=2)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        print(f"   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ ERROR loading model: {e}")
        return
    
    # Check model architecture
    print(f"\n2. Model architecture check:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Expected ~28M for ConvNeXt-tiny")
    
    # Check output layer
    if hasattr(model, 'head'):
        if hasattr(model.head, 'fc'):
            output_features = model.head.fc.out_features
        else:
            output_features = model.head.out_features
        print(f"   Output features: {output_features}")
        
        if output_features != 2:
            print(f"   ⚠️  WARNING: Expected 2 output features (real/fake), got {output_features}")
    
    # Check first layer weights
    print(f"\n3. Weight statistics (first layer):")
    first_layer = None
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            first_layer = param
            print(f"   Layer: {name}")
            print(f"   Shape: {param.shape}")
            print(f"   Mean: {param.mean():.6f}")
            print(f"   Std: {param.std():.6f}")
            print(f"   Min: {param.min():.6f}")
            print(f"   Max: {param.max():.6f}")
            break
    
    if first_layer is not None:
        if abs(first_layer.mean()) < 1e-6 and first_layer.std() < 1e-6:
            print(f"   ⚠️  WARNING: Weights seem uninitialized (all near zero)")
    
    # Test on random noise
    print(f"\n4. Testing on random noise:")
    with torch.no_grad():
        noise = torch.randn(1, 3, 224, 224).to(DEVICE)
        logits = model(noise)
        probs = F.softmax(logits, dim=1)[0]
        
        print(f"   Input: Random noise [1, 3, 224, 224]")
        print(f"   Logits: {logits[0].cpu().numpy()}")
        print(f"   Probabilities: real={probs[0]:.6f}, fake={probs[1]:.6f}")
        
        if abs(probs[0] - 0.5) < 0.01 and abs(probs[1] - 0.5) < 0.01:
            print(f"   ✅ Good: Model gives ~50-50 for random noise")
        elif probs[1] < 0.01 or probs[1] > 0.99:
            print(f"   ⚠️  WARNING: Model very confident on random noise!")
    
    # Test on multiple random inputs
    print(f"\n5. Testing on 10 random inputs:")
    fake_probs = []
    with torch.no_grad():
        for i in range(10):
            noise = torch.randn(1, 3, 224, 224).to(DEVICE)
            logits = model(noise)
            probs = F.softmax(logits, dim=1)[0]
            fake_probs.append(float(probs[1]))
    
    print(f"   Fake probabilities: {[f'{p:.4f}' for p in fake_probs]}")
    print(f"   Mean: {np.mean(fake_probs):.4f}")
    print(f"   Std: {np.std(fake_probs):.4f}")
    
    if np.std(fake_probs) < 0.01:
        print(f"   ⚠️  WARNING: Model gives same prediction for all inputs!")
        print(f"   This suggests model might be stuck or not properly trained.")
    
    # Test on actual image if provided
    if test_image_path and os.path.exists(test_image_path):
        print(f"\n6. Testing on actual image: {test_image_path}")
        
        try:
            # Load image
            img = cv2.imread(test_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            
            # Transform
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(Image.fromarray(img)).unsqueeze(0).to(DEVICE)
            
            # Predict
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1)[0]
                
                print(f"   Logits: {logits[0].cpu().numpy()}")
                print(f"   Probabilities: real={probs[0]:.6f}, fake={probs[1]:.6f}")
                
                if probs[1] < 0.05:
                    print(f"   🔴 Model says VERY REAL (fake prob < 5%)")
                elif probs[1] > 0.95:
                    print(f"   🔴 Model says VERY FAKE (fake prob > 95%)")
                else:
                    print(f"   ✅ Model gives reasonable confidence")
            
            # Test with different normalizations
            print(f"\n7. Testing different normalizations:")
            
            # Try alternative normalization
            transform_alt = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                   std=[0.5, 0.5, 0.5])
            ])
            
            input_tensor_alt = transform_alt(Image.fromarray(img)).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits_alt = model(input_tensor_alt)
                probs_alt = F.softmax(logits_alt, dim=1)[0]
                
                print(f"   Standard normalization: fake={probs[1]:.6f}")
                print(f"   Alternative normalization: fake={probs_alt[1]:.6f}")
                
                diff = abs(float(probs[1]) - float(probs_alt[1]))
                if diff > 0.1:
                    print(f"   ⚠️  Large difference ({diff:.4f}) - normalization might be wrong!")
        
        except Exception as e:
            print(f"   ❌ Error testing image: {e}")
    
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC SUMMARY")
    print(f"{'='*80}")
    
    # Provide recommendations
    if np.std(fake_probs) < 0.01:
        print(f"\n❌ CRITICAL ISSUE: Model outputs same prediction for all inputs")
        print(f"   Possible causes:")
        print(f"   1. Model file corrupted during training/saving")
        print(f"   2. Wrong model architecture loaded")
        print(f"   3. Model not actually trained (still at initialization)")
        print(f"\n   Recommendations:")
        print(f"   - Verify training completed successfully")
        print(f"   - Check training logs for final accuracy")
        print(f"   - Try loading the model checkpoint from an earlier epoch")
        print(f"   - Reduce ConvNeXt weight in ensemble until fixed")
    
    elif file_size_mb < 50:
        print(f"\n⚠️  WARNING: Model file seems incomplete")
        print(f"   Recommendation: Re-download or re-save the model")
    
    else:
        print(f"\n✅ Model structure appears OK")
        print(f"   If still getting 0.0000 in production:")
        print(f"   1. Check preprocessing pipeline matches training")
        print(f"   2. Verify input image quality and size")
        print(f"   3. Test with known fake samples to validate")
    
    print(f"\n{'='*80}\n")


def main():
    """Main function"""
    
    # Default model path
    model_path = "D:/Projects/Major Project/Deepfake Detection/models/convnext_tiny_deepfake.pth"
    test_image = None
    
    # Check command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        test_image = sys.argv[2]
    
    print(f"\n💡 Usage: python debug_convnext.py [model_path] [test_image_path]")
    print(f"   Running with: {model_path}\n")
    
    test_convnext_model(model_path, test_image)


if __name__ == "__main__":
    main()