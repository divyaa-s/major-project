import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import os


os.makedirs("evaluation_results", exist_ok=True)
# -------------------------
# LOAD CSV
# -------------------------
df = pd.read_csv("ensemble_20k_results.csv")

# Convert labels to 0/1
df["true_binary"] = df["true_label"].map({"Fake": 1, "Real": 0})
df["pred_binary"] = df["predicted_label"].map({"Fake": 1, "Real": 0})

y_true = df["true_binary"].values
y_pred = df["pred_binary"].values
y_scores = df["ensemble_fake_prob"].values  # probability for Fake
'''
# ==========================================================
# 1️⃣ CLASSIFICATION REPORT
# ==========================================================
print("\nCLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_true, y_pred, target_names=["Real","Fake"]))

# ==========================================================
# 2️⃣ CONFUSION MATRIX
# ==========================================================
cm = confusion_matrix(y_true, y_pred)
print("\nCONFUSION MATRIX")
print("="*60)
print(cm)
plt.figure()
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], ["Real","Fake"])
plt.yticks([0,1], ["Real","Fake"])
plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")

plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
'''
# ==========================================================
# 3️⃣ ROC CURVE + AUC
# ==========================================================
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

print("\nROC-AUC Score:", roc_auc)
'''
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()
'''

# Find optimal threshold (Youden’s J statistic)
j_scores = tpr - fpr
best_index = np.argmax(j_scores)
best_threshold = thresholds[best_index]

print("\nBest Threshold (Youden J):", best_threshold)



# Find optimal threshold (Youden’s J statistic)
j_scores = tpr - fpr
best_index = np.argmax(j_scores)
best_threshold = thresholds[best_index]

print("\nBest Threshold (Youden J):", best_threshold)

# Apply best threshold
new_preds = (y_scores >= 0.686).astype(int)

from sklearn.metrics import accuracy_score, confusion_matrix

new_acc = accuracy_score(y_true, new_preds)
new_cm = confusion_matrix(y_true, new_preds)

print("\nAccuracy at Optimal Threshold:", new_acc)
print("\nConfusion Matrix at Optimal Threshold:")
print(new_cm)

'''
# ==========================================================
# 4️⃣ PRECISION-RECALL CURVE
# ==========================================================
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
ap_score = average_precision_score(y_true, y_scores)

print("\nAverage Precision (AP):", ap_score)

plt.figure()
plt.plot(recall, precision, label=f"PR Curve (AP = {ap_score:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("precision_recall_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# 5️⃣ EXTRA METRICS
# ==========================================================
print("\nAdditional Metrics")
print("="*60)
print("ROC-AUC:", roc_auc_score(y_true, y_scores))'''