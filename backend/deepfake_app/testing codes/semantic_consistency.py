# semantic_consistency_analysis.py

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class SemanticConsistencyAnalyzer:
    """
    Semantic Segmentation Consistency Analysis
    
    Academic Innovation:
    - Uses semantic segmentation to detect object-level inconsistencies
    - Real images: Coherent semantic boundaries
    - Fake images: Segmentation artifacts at synthetic boundaries
    - Novel forensic approach using high-level semantics
    """
    
    def __init__(self):
        # Load pre-trained DeepLabV3
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 
                                    'deeplabv3_resnet50', 
                                    pretrained=True)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
    
    def analyze_semantic_consistency(self, image_path):
        """
        Analyze semantic segmentation for inconsistencies
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(img_tensor)['out'][0]
            
            segmentation = output.argmax(0).cpu().numpy()
            
            # Analyze segmentation properties
            
            # 1. Boundary smoothness
            edges = cv2.Canny((segmentation * 12).astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Real images: Smoother semantic boundaries
            # Fake images: More jagged segmentation edges
            
            # 2. Region coherence
            unique_labels = np.unique(segmentation)
            region_coherences = []
            
            for label in unique_labels:
                mask = (segmentation == label).astype(np.uint8)
                num_components, _ = cv2.connectedComponents(mask)
                
                # Few components = coherent region
                coherence = 1.0 / (num_components + 1)
                region_coherences.append(coherence)
            
            avg_coherence = np.mean(region_coherences)
            
            # 3. Segmentation confidence
            softmax_output = torch.softmax(output, dim=0)
            max_confidence = softmax_output.max(dim=0)[0]
            avg_confidence = max_confidence.mean().item()
            confidence_std = max_confidence.std().item()
            
            # Low confidence + high variation = suspicious
            confidence_suspicion = 1 - avg_confidence
            variation_suspicion = confidence_std * 2  # Scale up
            variation_suspicion = np.clip(variation_suspicion, 0, 1)
            
            # 4. Boundary irregularity
            # High edge density = irregular boundaries = suspicious
            boundary_suspicion = np.clip(edge_density / 0.05, 0, 1)
            
            # Combined suspicion
            suspicion_score = (
                0.30 * confidence_suspicion +
                0.30 * variation_suspicion +
                0.25 * boundary_suspicion +
                0.15 * (1 - avg_coherence)
            )
            
            return {
                "suspicion_score": float(np.clip(suspicion_score, 0, 1)),
                "avg_segmentation_confidence": float(avg_confidence),
                "confidence_variation": float(confidence_std),
                "edge_density": float(edge_density),
                "region_coherence": float(avg_coherence),
                "num_semantic_classes": int(len(unique_labels)),
                "interpretation": self.interpret_results(suspicion_score, avg_confidence)
            }
        
        except Exception as e:
            print(f"Semantic analysis failed: {e}")
            return None
    
    def interpret_results(self, suspicion, confidence):
        """Interpret semantic analysis"""
        if suspicion > 0.6:
            return f"High suspicion: Low segmentation confidence ({confidence:.3f}) and irregular boundaries suggest synthetic generation"
        elif suspicion > 0.4:
            return f"Moderate suspicion: Some semantic inconsistencies detected"
        else:
            return f"Low suspicion: Coherent semantic structure with high confidence ({confidence:.3f})"


# Standalone function
def analyze_semantic_consistency(image_path):
    """Wrapper for easy use"""
    analyzer = SemanticConsistencyAnalyzer()
    return analyzer.analyze_semantic_consistency(image_path)
