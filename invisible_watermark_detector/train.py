import os
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import joblib

from features.dct_features import extract_dct_features, extract_frequency_statistics

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────
DATA_FOLDERS = ["real", "ai_watermarked"]
RANDOM_STATE = 42
N_FOLDS = 5

# ────────────────────────────────────────────────
# Load & extract features
# ────────────────────────────────────────────────
X_dct, X_freq, y = [], [], []

print("Extracting features...")
for label, folder in enumerate(DATA_FOLDERS):
    path = f"data/{folder}"
    images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg','.jpeg','.png'))]

    for img_name in tqdm(images, desc=folder):
        img_path = os.path.join(path, img_name)

        dct_feats  = extract_dct_features(img_path)
        freq_stats = extract_frequency_statistics(img_path)

        if dct_feats is None or freq_stats is None:
            continue

        X_dct.append(dct_feats)
        X_freq.append(freq_stats)
        y.append(label)

if not X_dct:
    raise ValueError("No valid images found.")

X_dct  = np.array(X_dct)
X_freq = np.array(X_freq)
y      = np.array(y)

# Combine features
X = np.hstack([X_dct, X_freq])

print(f"Total valid samples: {len(X)}  (real={sum(y==0)}, watermarked={sum(y==1)})")

# ────────────────────────────────────────────────
# Normalize
# ────────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ────────────────────────────────────────────────
# Train + calibrate with cross-validation
# ────────────────────────────────────────────────
print("\nTraining & calibrating model with stratified k-fold...")

base_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=18,
    min_samples_split=8,
    min_samples_leaf=4,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

calibrated = CalibratedClassifierCV(
    base_model,
    method='isotonic',          # often better than sigmoid on imbalanced data
    cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
)

calibrated.fit(X, y)

# Quick final evaluation on hold-out set
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
calibrated.fit(X_tr, y_tr)  # refit on almost all data
probas = calibrated.predict_proba(X_te)[:, 1]
preds   = (probas > 0.5).astype(int)
acc     = accuracy_score(y_te, preds)
logloss = log_loss(y_te, probas)

print(f"Hold-out accuracy : {acc:.4f}")
print(f"Hold-out log-loss  : {logloss:.4f}")

# ────────────────────────────────────────────────
# Hold-out evaluation - already have this part
# X_tr, X_te, y_tr, y_te = train_test_split(...)
# calibrated.fit(X_tr, y_tr)
# probas = calibrated.predict_proba(X_te)[:, 1]
# preds   = (probas > 0.5).astype(int)
# acc     = accuracy_score(y_te, preds)
# logloss = log_loss(y_te, probas)

# ────────────────────────────────────────────────
# NEW: Detailed classification report & confusion matrix
# ────────────────────────────────────────────────
from sklearn.metrics import classification_report, confusion_matrix

y_pred = calibrated.predict(X_te)   # binary predictions (0 or 1)

print("\n" + "="*60)
print("CLASSIFICATION REPORT (hold-out set)")
print("="*60)
print(classification_report(y_te, y_pred, 
                          target_names=["REAL", "WATERMARKED/AI"],
                          digits=4))

print("\nCONFUSION MATRIX")
print("Rows: Actual  |  Columns: Predicted")
print("          REAL    WATERMARKED/AI")
print(confusion_matrix(y_te, y_pred))
print("="*60)

# ────────────────────────────────────────────────
# NEW: Probability distribution plot
# ────────────────────────────────────────────────
import matplotlib.pyplot as plt

prob_watermarked = calibrated.predict_proba(X_te)[:, 1]

plt.figure(figsize=(10, 6))
plt.hist(prob_watermarked[y_te == 0], bins=50, alpha=0.65, label="Real images", color="#2ecc71", density=True, edgecolor="white")
plt.hist(prob_watermarked[y_te == 1], bins=50, alpha=0.65, label="Watermarked / AI images", color="#e74c3c", density=True, edgecolor="white")
plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label="Default threshold 0.5")
plt.xlabel("Predicted probability of being WATERMARKED / AI", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title("Probability Distributions – Hold-out Set", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("prob_distribution_holdout.png", dpi=180)
plt.close()

print("Saved probability distribution plot → prob_distribution_holdout.png")
print("Look at the plot: good separation = two clear peaks, bad = strong overlap")

# ────────────────────────────────────────────────
# Threshold tuning experiment
# ────────────────────────────────────────────────
print("\nThreshold tuning on hold-out set:")
thresholds = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
print("Thresh | Acc   | Prec(AI) | Rec(AI) | F1(AI)  | FP rate | FN rate")
print("-" * 70)

prob_ai = calibrated.predict_proba(X_te)[:, 1]

for t in thresholds:
    pred_t = (prob_ai >= t).astype(int)
    acc_t = accuracy_score(y_te, pred_t)
    report = classification_report(y_te, pred_t, target_names=["REAL", "AI"], output_dict=True, digits=4, zero_division=0)
    prec_ai = report["AI"]["precision"]
    rec_ai  = report["AI"]["recall"]
    f1_ai   = report["AI"]["f1-score"]
    fp_rate = (pred_t[y_te==0] == 1).mean()  # false positive rate
    fn_rate = (pred_t[y_te==1] == 0).mean()  # false negative rate
    
    print(f"{t:.2f}   | {acc_t:.4f} | {prec_ai:.4f}   | {rec_ai:.4f}  | {f1_ai:.4f} | {fp_rate:.4f}  | {fn_rate:.4f}")


# ────────────────────────────────────────────────
# Save artifacts
# ────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(calibrated,   "models/new_watermark_classifier.pkl")
joblib.dump(scaler,       "models/new_feature_scaler.pkl")

print("\nModel & scaler saved.")