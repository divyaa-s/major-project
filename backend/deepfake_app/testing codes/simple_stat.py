"""
Simple statistical classifier using RAW features (no normalization)
Based on diagnostic findings: brightness, edge density, variance
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# ============================================================================
# SIMPLE FEATURE EXTRACTION (RAW VALUES)
# ============================================================================
def extract_simple_features(img_path):
    """
    Extract simple, raw statistical features
    NO normalization, NO suspicion scores - just raw measurements
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    features = {}
    
    # 1. Brightness statistics (diagnostic showed 22.56 difference!)
    features['mean_brightness'] = np.mean(gray)
    features['std_brightness'] = np.std(gray)
    features['median_brightness'] = np.median(gray)
    
    # 2. Color channel statistics
    features['mean_red'] = np.mean(img_rgb[:,:,0])
    features['mean_green'] = np.mean(img_rgb[:,:,1])
    features['mean_blue'] = np.mean(img_rgb[:,:,2])
    
    features['std_red'] = np.std(img_rgb[:,:,0])
    features['std_green'] = np.std(img_rgb[:,:,1])
    features['std_blue'] = np.std(img_rgb[:,:,2])
    
    # 3. Edge density (diagnostic showed 0.0525 difference!)
    edges = cv2.Canny(gray, 50, 150)
    features['edge_density'] = np.sum(edges > 0) / edges.size
    
    # 4. Gradient statistics
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    features['mean_gradient'] = np.mean(gradient_magnitude)
    features['std_gradient'] = np.std(gradient_magnitude)
    
    # 5. Texture energy
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features['texture_energy'] = np.var(laplacian)
    
    # 6. Saturation (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features['mean_saturation'] = np.mean(hsv[:,:,1])
    features['std_saturation'] = np.std(hsv[:,:,1])
    
    # 7. High frequency content
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2
    
    # Create high-freq mask (outer region)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    high_freq_mask = r > (min(h, w) // 4)
    
    features['high_freq_energy'] = np.sum(magnitude[high_freq_mask]) / np.sum(high_freq_mask)
    
    # 8. Color variance ratio
    features['rgb_variance_ratio'] = features['std_red'] / (features['std_blue'] + 1e-10)
    
    return features

# ============================================================================
# DATA LOADING
# ============================================================================
def load_dataset(real_folder, fake_folder, max_samples=None):
    """Load images and extract features"""
    
    print("\nLoading dataset and extracting features...")
    
    X = []
    y = []
    feature_names = None
    
    # Get image paths
    real_imgs = [os.path.join(real_folder, f) for f in os.listdir(real_folder) 
                 if f.endswith(('.jpg', '.png', '.jpeg'))]
    fake_imgs = [os.path.join(fake_folder, f) for f in os.listdir(fake_folder) 
                 if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if max_samples:
        real_imgs = real_imgs[:max_samples]
        fake_imgs = fake_imgs[:max_samples]
    
    print(f"Found {len(real_imgs)} real images")
    print(f"Found {len(fake_imgs)} fake images")
    
    # Process real images
    for img_path in tqdm(real_imgs, desc="REAL"):
        features = extract_simple_features(img_path)
        if features:
            if feature_names is None:
                feature_names = list(features.keys())
            X.append([features[k] for k in feature_names])
            y.append(0)  # 0 = real
    
    # Process fake images
    for img_path in tqdm(fake_imgs, desc="FAKE"):
        features = extract_simple_features(img_path)
        if features:
            X.append([features[k] for k in feature_names])
            y.append(1)  # 1 = fake
    
    return np.array(X), np.array(y), feature_names

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================
def train_and_evaluate(X, y, feature_names):
    """Train classifier and evaluate"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split:")
    print(f"  Training:   {len(X_train)} samples")
    print(f"  Testing:    {len(X_test)} samples")
    
    # Try multiple classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', probability=True),
        'SVM (Linear)': SVC(kernel='linear', C=1, probability=True)
    }
    
    best_acc = 0
    best_clf = None
    best_name = None
    
    print("\n" + "="*70)
    print("TRAINING CLASSIFIERS")
    print("="*70)
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        
        # Evaluate
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"  Training accuracy:   {train_acc:.4f}")
        print(f"  Testing accuracy:    {test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_clf = clf
            best_name = name
    
    # Detailed evaluation of best classifier
    print("\n" + "="*70)
    print(f"BEST CLASSIFIER: {best_name}")
    print("="*70)
    
    y_pred = best_clf.predict(X_test)
    
    print(f"\nTest Accuracy: {best_acc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Real', 'Fake'],
                                digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Real    Fake")
    print(f"Actual Real   {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Fake   {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    # Feature importance (if available)
    if hasattr(best_clf, 'feature_importances_'):
        print("\nTop 10 Most Important Features:")
        importances = best_clf.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importances)), importances[np.argsort(importances)[::-1]])
        plt.xticks(range(len(importances)), 
                   [feature_names[i] for i in np.argsort(importances)[::-1]], 
                   rotation=45, ha='right')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150)
        print(f"\n📊 Feature importance saved to 'feature_importance.png'")
    
    return best_clf, best_name

# ============================================================================
# PREDICTION
# ============================================================================
def predict_image(img_path, classifier, feature_names):
    """Predict if image is real or fake"""
    
    features = extract_simple_features(img_path)
    if features is None:
        return None
    
    X = np.array([[features[k] for k in feature_names]])
    
    prediction = classifier.predict(X)[0]
    
    if hasattr(classifier, 'predict_proba'):
        proba = classifier.predict_proba(X)[0]
        confidence = proba[prediction]
    else:
        confidence = None
    
    return {
        'prediction': 'FAKE' if prediction == 1 else 'REAL',
        'confidence': confidence,
        'features': features
    }

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    
    REAL_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/test/real_and_fake_face/training_real"
    FAKE_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/test/real_and_fake_face/training_fake"
    
    print("="*70)
    print("SIMPLE STATISTICAL CLASSIFIER")
    print("="*70)
    
    # Load data
    X, y, feature_names = load_dataset(REAL_FOLDER, FAKE_FOLDER, max_samples=500)
    
    print(f"\nLoaded {len(X)} images with {len(feature_names)} features")
    print(f"Features: {', '.join(feature_names)}")
    
    # Train and evaluate
    classifier, classifier_name = train_and_evaluate(X, y, feature_names)
    
    # Save model
    model_path = 'simple_classifier.pkl'
    joblib.dump({
        'classifier': classifier,
        'classifier_name': classifier_name,
        'feature_names': feature_names
    }, model_path)
    
    print(f"\n✅ Model saved to '{model_path}'")
    
    # Test on sample images
    print("\n" + "="*70)
    print("TESTING ON SAMPLE IMAGES")
    print("="*70)
    
    real_test = os.path.join(REAL_FOLDER, os.listdir(REAL_FOLDER)[0])
    result = predict_image(real_test, classifier, feature_names)
    if result:
        print(f"\nReal image: {os.path.basename(real_test)}")
        print(f"  Prediction: {result['prediction']}")
        if result['confidence']:
            print(f"  Confidence: {result['confidence']:.2%}")
    
    fake_test = os.path.join(FAKE_FOLDER, os.listdir(FAKE_FOLDER)[0])
    result = predict_image(fake_test, classifier, feature_names)
    if result:
        print(f"\nFake image: {os.path.basename(fake_test)}")
        print(f"  Prediction: {result['prediction']}")
        if result['confidence']:
            print(f"  Confidence: {result['confidence']:.2%}")
    
    print("\n✅ Complete!")