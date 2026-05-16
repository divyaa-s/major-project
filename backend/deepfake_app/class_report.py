import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, classification_report
)

# ── 1. Load the CSV ───────────────────────────────────────────────
df = pd.read_csv("checking_final_batch.csv")

# Clean labels and handle binary conversion
df["gt_label"] = df["gt_label"].str.strip().str.capitalize()
df = df[df["gt_label"].isin(["Real", "Fake"])]
df["true_binary"] = df["gt_label"].map({"Real": 0, "Fake": 1})

# ── 2. Select the Score to Evaluate ───────────────────────────────
score_col = "final_score"

# 🔥 THE FIX: Force the column to be numeric. If there's text, turn it to NaN
df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

# Drop any rows that couldn't be converted to numbers or missing labels
df = df.dropna(subset=[score_col, "true_binary"])

# ── 3. Find AUC ───────────────────────────────────────────────────
auc = roc_auc_score(df["true_binary"], df[score_col])
print(f"\n🚀 ROC-AUC Score: {auc:.4f}")

# ── 4. Find the Best Threshold (Optimizing for F1 Score) ──────────
best_f1, best_t = 0, 0
for t in np.arange(0.10, 0.95, 0.01): 
    preds = (df[score_col] >= t).astype(int)
    score = f1_score(df["true_binary"], preds, zero_division=0)
    if score > best_f1:
        best_f1, best_t = score, t

print(f"⭐ Optimal Threshold Found: {best_t:.2f} (F1 = {best_f1:.4f})")

# ── 5. Full Classification Report at Best Threshold ───────────────
df["pred_binary"] = (df[score_col] >= best_t).astype(int)

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(df["true_binary"], df["pred_binary"], target_names=["Real (0)", "Fake (1)"]))

# ── 6. Threshold Sweep Table ──────────────────────────────────────
print("\n========== THRESHOLD SWEEP ==========")
print(f"{'Threshold':>10} {'Acc':>8} {'F1':>8} {'FPR':>8} {'FNR':>8} {'TP':>5} {'FP':>5}")
print("-" * 65)

# Sweeping through a wider range to show performance around the optimal threshold
for t in np.arange(0.40, 0.90, 0.05):
    preds = (df[score_col] >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(df["true_binary"], preds).ravel()
    
    acc = accuracy_score(df['true_binary'], preds)
    f1 = f1_score(df['true_binary'], preds, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Highlight the optimal threshold row
    marker = " <--" if abs(t - best_t) < 0.025 else ""
    
    print(
        f"{t:>10.2f} "
        f"{acc:>8.4f} "
        f"{f1:>8.4f} "
        f"{fpr:>8.4f} "
        f"{fnr:>8.4f} "
        f"{tp:>5} "
        f"{fp:>5}"
        f"{marker}"
    )