import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 1. Load your training CSV (Point this to your ensemble_20k.csv or your combined 1k test CSV)
csv_path = "ensemble_20k_results.csv" 
df = pd.read_csv(csv_path)

print(f"Loaded dataset shape: {df.shape}")

# 2. Extract Base Models
base_models = ['convnext_fake', 'xception_fake', 'efficientnet_fake', 'vit_fake']

# 3. Calculate Smart Features (The Meta-Learner's secret weapon)
df['disagreement'] = df[base_models].max(axis=1) - df[base_models].min(axis=1)
df['max_confidence'] = df[base_models].max(axis=1)

# Ensure Watermark column exists (if not in old CSV, default to 0)
if 'watermark_prob' not in df.columns:
    df['watermark_prob'] = 0.0

# 4. Prepare Features Array (Order must match the main app exactly)
features = [
    'convnext_fake', 
    'xception_fake', 
    'efficientnet_fake', 
    'vit_fake', 
    'disagreement', 
    'max_confidence', 
    'watermark_prob'    # Replacing PRNU!
]

X = df[features]
# Map Labels to 1 (Fake) and 0 (Real)
y = df['true_label'].apply(lambda x: 1 if str(x).lower() == 'fake' else 0)

# 5. Train the V5 Meta-Learner
print("🧠 Training V5 Meta-Learner (CNN Fusion + Watermark)...")
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X, y)

# 6. Save the Model
save_path = "models/final/smart_meta_learner_v5.pkl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
joblib.dump(rf, save_path)
print(f"✅ Saved smart_meta_learner_v5.pkl to {save_path}!")