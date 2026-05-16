import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_curve, auc
import os

# ==========================================
# 1. CONFIGURATION (Update your CSV names here if needed)
# ==========================================
CLEAN_CSV = "clean_image_inference_results (1).csv" # Change this if your clean CSV has a different name!
DEGRADED_CSV = "degraded_inference_results.csv"
EXTREME_CSV = "extreme_degraded_inference_results.csv"

# Styling for academic paper
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)
COLORS = {"CNN": "#ff7675", "Meta": "#0984e3"}

def load_and_prep(csv_file, dataset_name):
    if not os.path.exists(csv_file):
        print(f"⚠️ Warning: Could not find {csv_file}. Skipping {dataset_name}.")
        return None
        
    df = pd.read_csv(csv_file)
    df['true_binary'] = df['true_label'].map({"Real": 0, "Fake": 1})
    
    # Calculate Base CNN Accuracy
    cnn_preds = (df['cnn_ensemble_score'] > 0.5).astype(int)
    cnn_acc = accuracy_score(df['true_binary'], cnn_preds) * 100
    
    # Calculate Meta-Learner Accuracy
    meta_preds = (df['final_score'] > 0.5).astype(int)
    meta_acc = accuracy_score(df['true_binary'], meta_preds) * 100
    
    return {
        "Dataset": dataset_name,
        "CNN_Acc": cnn_acc,
        "Meta_Acc": meta_acc,
        "y_true": df['true_binary'],
        "y_score_cnn": df['cnn_ensemble_score'],
        "y_score_meta": df['final_score']
    }

# ==========================================
# 2. LOAD ALL DATA
# ==========================================
print("Loading datasets...")
data = []
for file, name in [(CLEAN_CSV, "1. Clean (100% HQ)"), 
                   (DEGRADED_CSV, "2. Degraded (30% JPEG)"), 
                   (EXTREME_CSV, "3. Extreme (10% + Blur)")]:
    res = load_and_prep(file, name)
    if res:
        data.append(res)

if len(data) == 0:
    print("❌ No CSV files found. Please check your filenames.")
    exit()

# ==========================================
# GRAPH 1: ACCURACY RESILIENCE (BAR CHART)
# ==========================================
print("Generating Resilience Bar Chart...")
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(data))
width = 0.35

cnn_accs = [d["CNN_Acc"] for d in data]
meta_accs = [d["Meta_Acc"] for d in data]
labels = [d["Dataset"] for d in data]

rects1 = ax.bar(x - width/2, cnn_accs, width, label='Base CNN Ensemble', color=COLORS["CNN"], edgecolor="black")
rects2 = ax.bar(x + width/2, meta_accs, width, label='Meta-Learner (Stacking)', color=COLORS["Meta"], edgecolor="black")

ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_title('Pipeline Resilience Under Extreme Degradation', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.legend(loc='lower right')
ax.set_ylim(85, 100) # Zoom in on the top 15% to show the gap!

# Add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig("thesis_resilience_chart.png", dpi=300)
plt.close()

# ==========================================
# GRAPH 2: ROC CURVE DEGRADATION
# ==========================================
print("Generating ROC Curve Overlay...")
plt.figure(figsize=(8, 8))

# We will plot the Meta-Learner ROC for all 3 datasets
line_styles = ['-', '--', ':']
colors = ['#27ae60', '#f39c12', '#c0392b']

for i, d in enumerate(data):
    fpr, tpr, _ = roc_curve(d["y_true"], d["y_score_meta"])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, linestyle=line_styles[i], color=colors[i], lw=2.5,
             label=f'{d["Dataset"]} (AUC = {roc_auc:.4f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.title('Meta-Learner ROC Curves Across Difficulty Levels', fontweight='bold', pad=20)
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig("thesis_roc_degradation.png", dpi=300)
plt.close()

print("=====================================================")
print("✅ SUCCESS! Generated 2 high-resolution thesis charts:")
print("   1. thesis_resilience_chart.png")
print("   2. thesis_roc_degradation.png")
print("=====================================================")