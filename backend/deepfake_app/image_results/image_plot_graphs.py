import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv("clean_image_inference_results (1).csv")

df['true_binary'] = df['true_label'].map({"Real": 0, "Fake": 1})
df['correct'] = (df['true_label'] == df['predicted_label']).astype(int)

def save(name):
    plt.savefig(name, dpi=300, bbox_inches="tight")
    plt.close()

# =========================================================
# 1. ROC CURVE COMPARISON 🔥
# =========================================================
plt.figure(figsize=(8,6))

models = {
    "EfficientNet": "efficientnet_b3_score",
    "Xception": "xception_score",
    "ViT": "vit_score",
    "ConvNeXt": "convnext_score",
    "Final Pipeline": "final_score"
}

for name, col in models.items():
    if col in df.columns:
        fpr, tpr, _ = roc_curve(df['true_binary'], df[col])
        auc_score = auc(fpr, tpr)
        lw = 3 if name == "Final Pipeline" else 1.5
        plt.plot(fpr, tpr, lw=lw, label=f"{name} (AUC={auc_score:.3f})")

plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
save("roc_curves.png")

# =========================================================
# 2. PRECISION-RECALL CURVE
# =========================================================
precision, recall, _ = precision_recall_curve(df['true_binary'], df['final_score'])

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
save("pr_curve.png")

# =========================================================
# 3. CONFUSION MATRIX
# =========================================================
cm = confusion_matrix(df['true_label'], df['predicted_label'])
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6,5))
sns.heatmap(cm_norm, annot=True, fmt=".2f",
            xticklabels=['Pred Real','Pred Fake'],
            yticklabels=['True Real','True Fake'])
plt.title("Normalized Confusion Matrix")
save("confusion_matrix.png")

# =========================================================
# 4. THRESHOLD vs METRICS 🔥
# =========================================================
thresholds = np.linspace(0,1,100)
acc, prec, rec, f1 = [],[],[],[]

for t in thresholds:
    preds = (df['final_score'] >= t).astype(int)
    acc.append(accuracy_score(df['true_binary'], preds))
    prec.append(precision_score(df['true_binary'], preds, zero_division=0))
    rec.append(recall_score(df['true_binary'], preds))
    f1.append(f1_score(df['true_binary'], preds))

plt.figure()
plt.plot(thresholds, acc, label="Accuracy")
plt.plot(thresholds, prec, label="Precision")
plt.plot(thresholds, rec, label="Recall")
plt.plot(thresholds, f1, label="F1")
plt.axvline(0.5, linestyle='--')
plt.legend()
plt.title("Threshold vs Metrics")
save("threshold_metrics.png")

# =========================================================
# 5. SCORE DISTRIBUTION 🔥
# =========================================================
plt.figure(figsize=(8,6))

sns.kdeplot(df[df['true_label']=="Real"]['final_score'],
            fill=True, label="Real", alpha=0.5)

sns.kdeplot(df[df['true_label']=="Fake"]['final_score'],
            fill=True, label="Fake", alpha=0.5)

plt.axvline(0.5, linestyle='--')
plt.title("Final Score Distribution")
plt.legend()
save("score_distribution.png")

# =========================================================
# 6. ERROR BREAKDOWN
# =========================================================
fp = ((df.true_label=='Real') & (df.predicted_label=='Fake')).sum()
fn = ((df.true_label=='Fake') & (df.predicted_label=='Real')).sum()
tp = ((df.true_label=='Fake') & (df.predicted_label=='Fake')).sum()
tn = ((df.true_label=='Real') & (df.predicted_label=='Real')).sum()

plt.figure()
sns.barplot(x=['TP','TN','FP','FN'], y=[tp,tn,fp,fn])
plt.title("Error Breakdown")
save("error_analysis.png")

# =========================================================
# 7. MODEL DISAGREEMENT 🔥
# =========================================================
model_cols = [
    'efficientnet_b3_score',
    'xception_score',
    'vit_score',
    'convnext_score'
]

available = [c for c in model_cols if c in df.columns]

if len(available) >= 2:
    df['model_disagreement'] = df[available].max(axis=1) - df[available].min(axis=1)

    plt.figure(figsize=(8,6))
    sns.boxplot(x='correct', y='model_disagreement', data=df)
    plt.xticks([0,1], ["Incorrect", "Correct"])
    plt.title("Model Disagreement vs Error")
    save("model_disagreement.png")

# =========================================================
# 8. CORRELATION HEATMAP
# =========================================================
num = df.select_dtypes(include=np.number)

if len(num.columns) > 3:
    plt.figure(figsize=(10,8))
    sns.heatmap(num.corr(), cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    save("correlation_heatmap.png")

# =========================================================
# 9. DECISION SOURCE vs ACCURACY 🔥
# =========================================================
if {'decision_source','correct'}.issubset(df.columns):
    acc = df.groupby('decision_source')['correct'].mean()

    plt.figure()
    acc.plot(kind='bar')
    plt.title("Decision Source vs Accuracy")
    save("decision_source_accuracy.png")

# =========================================================
# 10. WATERMARK IMPACT 🔥
# =========================================================
if {'watermark_score','final_score'}.issubset(df.columns):
    plt.figure()
    sns.scatterplot(x='watermark_score', y='final_score', data=df)
    plt.title("Watermark Impact")
    save("watermark_impact.png")

# =========================================================
# 11. CONFIDENCE vs CORRECTNESS 🔥
# =========================================================
if 'confidence' in df.columns:
    plt.figure()
    sns.boxplot(x='correct', y='confidence', data=df)
    plt.xticks([0,1], ["Incorrect","Correct"])
    plt.title("Confidence vs Correctness")
    save("confidence_analysis.png")

# =========================================================
# 12. ERROR vs SCORE BINS 🔥
# =========================================================
df['score_bin'] = pd.cut(df['final_score'], bins=10)
error_rate = df.groupby('score_bin')['correct'].apply(lambda x: 1 - x.mean())

plt.figure(figsize=(10,6))
error_rate.plot(kind='bar')
plt.title("Error Rate across Score Ranges")
plt.ylabel("Error Rate")
save("error_vs_score_bins.png")

print("✅ 12 IMPORTANT GRAPHS GENERATED")