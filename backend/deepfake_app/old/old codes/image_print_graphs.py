import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = r"backend\deepfake_app\FINAL_Results_Threshold_0.50.csv"   # adjust if needed
OUT_DIR = "hmmmm_analysis_plots"
FAKE_THRESHOLD = 0.55

os.makedirs(OUT_DIR, exist_ok=True)
sns.set(style="whitegrid")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Normalize labels
df["true_label"] = df["true_label"].str.capitalize()
df["predicted_label"] = df["predicted_label"].str.capitalize()

real_df = df[df["true_label"] == "Real"]
fake_df = df[df["true_label"] == "Fake"]


# -----------------------------
# 1️⃣ ENSEMBLE SCORE DISTRIBUTION
# -----------------------------
plt.figure(figsize=(8, 5))
plt.hist(fake_df["ensemble_score"], bins=30, alpha=0.6, label="Fake", color="red")
plt.hist(real_df["ensemble_score"], bins=30, alpha=0.6, label="Real", color="green")
plt.axvline(FAKE_THRESHOLD, linestyle="--", color="black", label="Threshold")
plt.xlabel("Final Ensemble Score")
plt.ylabel("Count")
plt.title("Final Ensemble Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ensemble_distribution.png", dpi=120)
plt.close()

# -----------------------------
# 2️⃣ WATERMARK PROBABILITY DISTRIBUTION
# -----------------------------
plt.figure(figsize=(8, 5))
plt.hist(fake_df["watermark_prob"], bins=30, alpha=0.6, label="Fake", color="red")
plt.hist(real_df["watermark_prob"], bins=30, alpha=0.6, label="Real", color="green")
plt.axvline(0.30, linestyle="--", color="orange", label="Detect threshold")
plt.axvline(0.90, linestyle="--", color="black", label="Override threshold")
plt.xlabel("Watermark Probability")
plt.ylabel("Count")
plt.title("Watermark Probability Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/watermark_distribution.png", dpi=120)
plt.close()

# -----------------------------
# 3️⃣ CNN vs FORENSIC SCATTER
# -----------------------------
plt.figure(figsize=(7, 7))
sns.scatterplot(
    data=df,
    x="cnn_ensemble",
    y="forensic_score",
    hue="predicted_label",
    style="decision_source",
    alpha=0.75
)
plt.axvline(FAKE_THRESHOLD, linestyle="--", color="gray")
plt.axhline(0.45, linestyle="--", color="gray")
plt.xlabel("CNN Ensemble Score")
plt.ylabel("Forensic Score")
plt.title("CNN vs Forensic Decision Map")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cnn_vs_forensic.png", dpi=120)
plt.close()

# -----------------------------
# 4️⃣ CNN ENSEMBLE DISTRIBUTION
# -----------------------------
plt.figure(figsize=(8, 5))
sns.kdeplot(fake_df["cnn_ensemble"], label="Fake", fill=True)
sns.kdeplot(real_df["cnn_ensemble"], label="Real", fill=True)
plt.xlabel("CNN Ensemble Score")
plt.title("CNN Ensemble Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cnn_distribution.png", dpi=120)
plt.close()

# -----------------------------
# 5️⃣ FORENSIC SCORE DISTRIBUTION
# -----------------------------
plt.figure(figsize=(8, 5))
sns.kdeplot(fake_df["forensic_score"], label="Fake", fill=True)
sns.kdeplot(real_df["forensic_score"], label="Real", fill=True)
plt.xlabel("Forensic Score")
plt.title("Forensic Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/forensic_distribution.png", dpi=120)
plt.close()

# -----------------------------
# 6️⃣ DECISION SOURCE COUNTS
# -----------------------------
plt.figure(figsize=(9, 5))
df["decision_source"].value_counts().plot(kind="bar", color="steelblue")
plt.ylabel("Count")
plt.title("Decision Source Breakdown")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/decision_source_counts.png", dpi=120)
plt.close()

# -----------------------------
# 7️⃣ CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(
    df["true_label"],
    df["predicted_label"],
    labels=["Real", "Fake"]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Real", "Fake"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=120)
plt.close()
'''
# -----------------------------
# 8️⃣ WATERMARK OVERRIDE IMPACT
# -----------------------------
override_df = df[df["decision_source"].str.contains("watermark", na=False)]

plt.figure(figsize=(6, 4))
sns.countplot(
    data=override_df,
    x="true_label",
    palette={"Real": "green", "Fake": "red"}
)
plt.title("Watermark Override Impact (True Labels)")
plt.xlabel("True Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/watermark_override_impact.png", dpi=120)
plt.close()
'''
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# -----------------------------
# 9️⃣ CLASSIFICATION METRICS
# -----------------------------
y_true = df["true_label"]
y_pred = df["predicted_label"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label="Fake")
recall = recall_score(y_true, y_pred, pos_label="Fake")
f1 = f1_score(y_true, y_pred, pos_label="Fake")

print("\n📊 FINAL CLASSIFICATION METRICS")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred))



print(f"\n✅ All plots generated successfully in: {OUT_DIR}\n")
