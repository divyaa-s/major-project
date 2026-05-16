import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
'''
# Load your existing data
df = pd.read_csv("clean_image_inference_results (1).csv")
df['true_binary'] = df['true_label'].map({"Real": 0, "Fake": 1})

sns.set_palette("muted")

# =========================================================
# 1. CALIBRATION CURVE (Reliability Diagram)
# =========================================================
plt.figure(figsize=(8, 6))

# Calculate the calibration curve (10 bins)
prob_true, prob_pred = calibration_curve(df['true_binary'], df['final_score'], n_bins=10)

# Plot perfectly calibrated line
plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')

# Plot your Meta-Learner's calibration
plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color='#5b8cff', label='Meta-Learner Fusion')

plt.xlabel('Mean Predicted Probability (Confidence)')
plt.ylabel('Fraction of Actual Fakes')
plt.title('Calibration Curve (Reliability Diagram)')
plt.legend()
plt.tight_layout()
plt.savefig("calibration_curve.png", dpi=300)
plt.close()

# =========================================================
# 2. ABLATION STUDY (Component Analysis)
# =========================================================
# We will calculate the AUC for different combinations of your models
plt.figure(figsize=(10, 6))

ablation_results = {
    "All 5 Signals (Full Meta-Learner)": roc_auc_score(df['true_binary'], df['final_score']),
    "ViT + ConvNeXt Only (Top 2)": roc_auc_score(df['true_binary'], df[['vit_score', 'convnext_score']].mean(axis=1)),
    "Older CNNs Only (Xception + EffNet)": roc_auc_score(df['true_binary'], df[['xception_score', 'efficientnet_b3_score']].mean(axis=1)),
    "Watermark Only": roc_auc_score(df['true_binary'], df['watermark_score']),
}

# Sort by performance
ablation_results = dict(sorted(ablation_results.items(), key=lambda item: item[1]))

sns.barplot(x=list(ablation_results.values()), y=list(ablation_results.keys()), palette="viridis")
plt.xlabel("ROC-AUC Score")
plt.title("Ablation Study: Contribution of Different Components")
plt.xlim(0.5, 1.0) # Start X-axis at 0.5 (random guessing)

# Add text labels to bars
for i, v in enumerate(ablation_results.values()):
    plt.text(v - 0.05, i, f"{v:.4f}", color='white', fontweight='bold', va='center')

plt.tight_layout()
plt.savefig("ablation_study.png", dpi=300)
plt.close()

print("✅ Calibration Curve and Ablation Study graphs generated!")


"""
generate_advanced_graphs.py
===========================
Generates advanced thesis graphs using the combined_results.csv.
Graphs generated:
1. Inter-Signal Correlation Heatmap (Orthogonality Proof)
2. Feature Space Scatter Plot (Decision Boundary)
3. Feature Importance Ranking (Spearman Correlation)
4. Optimal Threshold Trade-off Curve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score
from scipy.stats import spearmanr

# =========================================================
# 1. LOAD AND PREP DATA
# =========================================================
print("Loading combined_results.csv...")
try:
    df = pd.read_csv("combined_results.csv")
except FileNotFoundError:
    print("❌ Error: combined_results.csv not found!")
    exit()

# Auto-detect label column name (handles both 'gt_label' and 'true_label')
label_col = 'true_label' if 'true_label' in df.columns else 'gt_label'

# Map Real=0, Fake=1 for mathematical plotting
df['true_binary'] = df[label_col].map({"Real": 0, "Fake": 1})

# Clean numeric columns, drop columns that are entirely 0 from the Kaggle run
cols_to_drop = ['blink_count', 'blink_expected', 'flow_avg_magnitude', 'flow_inconsistency', 
                'landmark_jitter', 'landmark_frames_analyzed', 'quality_avg_mismatch', 
                'quality_max_mismatch', 'processed_frames', 'total_frames', 'fps', 'duration_s']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Auto-detect the right signals based on what is actually in your CSV
possible_signals = ['efficientnet_b3_score', 'efficientnet_b3', 'xception_score', 'xception', 
                    'vit_score', 'vit', 'convnext_score', 'convnext', 'cnn_ensemble_score', 
                    'temporal_score', 'temporal', 'quality_score', 'quality', 'final_score']
signal_cols = [c for c in possible_signals if c in df.columns]

# Styling for academic paper
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

# =========================================================
# GRAPH 1: Inter-Signal Correlation Heatmap
# =========================================================
print("Generating Correlation Heatmap...")
corr_matrix = df[signal_cols].corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Inter-Signal Correlation Heatmap (Spearman)', fontsize=16, pad=15)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('advanced_1_correlation_heatmap.png', dpi=300)
plt.close()

# =========================================================
# GRAPH 2: 2D Decision Boundary Scatter Plot
# =========================================================
print("Generating Feature Space Scatter Plot...")
# Detect CNN and Temporal columns for X and Y axis
cnn_col = 'cnn_ensemble_score' if 'cnn_ensemble_score' in df.columns else ('vit' if 'vit' in df.columns else 'vit_score')
temp_col = 'temporal_score' if 'temporal_score' in df.columns else 'temporal'

plt.figure(figsize=(8, 6))
sns.scatterplot(x=cnn_col, y=temp_col, hue=label_col, 
                data=df, palette={'Real':'#2ecc71', 'Fake':'#e74c3c'}, 
                alpha=0.7, s=80, edgecolor='k')

# Drawing a theoretical decision boundary line
plt.plot([-0.1, 1.1], [1.1, -0.1], color='black', linestyle='--', label='Theoretical Fusion Boundary')
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xlabel('Primary CNN Score', fontsize=12)
plt.ylabel('Temporal Flow/Blink Score', fontsize=12)
plt.title('Feature Space: CNNs vs. Temporal Sensors', fontsize=14, pad=15)
plt.legend()
plt.tight_layout()
plt.savefig('advanced_2_decision_boundary.png', dpi=300)
plt.close()

# =========================================================
# GRAPH 3: Feature Importance Ranking
# =========================================================
print("Generating Feature Importance Ranking...")
correlations = {}
for col in signal_cols:
    if col != 'final_score': # Compare against everything except the final score itself
        corr, _ = spearmanr(df[col], df['true_binary'])
        correlations[col] = abs(corr)

# Sort dictionary by correlation strength
correlations = dict(sorted(correlations.items(), key=lambda item: item[1]))

plt.figure(figsize=(10, 6))
sns.barplot(x=list(correlations.values()), y=list(correlations.keys()), palette='magma')
plt.xlabel('Absolute Correlation with True Label', fontsize=12)
plt.title('Feature Importance: Which signals drive the predictions?', fontsize=14, pad=15)
for index, value in enumerate(correlations.values()):
    plt.text(value + 0.01, index, f"{value:.3f}", va='center', fontweight='bold')
plt.xlim(0, 1.0)
plt.tight_layout()
plt.savefig('advanced_3_feature_importance.png', dpi=300)
plt.close()

# =========================================================
# GRAPH 4: Optimal Threshold Curve
# =========================================================
print("Generating Optimal Threshold Curve...")
thresholds = np.linspace(0.01, 0.99, 100)
accuracies, f1_scores, precisions, recalls = [], [], [], []

for t in thresholds:
    preds = (df['final_score'] >= t).astype(int)
    accuracies.append(accuracy_score(df['true_binary'], preds))
    f1_scores.append(f1_score(df['true_binary'], preds, zero_division=0))
    precisions.append(precision_score(df['true_binary'], preds, zero_division=0))
    recalls.append(recall_score(df['true_binary'], preds, zero_division=0))

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracies, label='Accuracy', color='#3498db', lw=2)
plt.plot(thresholds, f1_scores, label='F1-Score', color='#9b59b6', lw=2)
plt.plot(thresholds, precisions, label='Precision', color='#2ecc71', lw=2, linestyle=':')
plt.plot(thresholds, recalls, label='Recall', color='#e74c3c', lw=2, linestyle='-.')

plt.axvline(x=0.5, color='black', linestyle='--', label='Default Threshold (0.5)')
plt.axvline(x=optimal_threshold, color='gold', linestyle='--', label=f'Optimal F1 Threshold ({optimal_threshold:.2f})')

plt.xlabel('Decision Threshold', fontsize=12)
plt.ylabel('Metric Score', fontsize=12)
plt.title('Threshold Optimization Trade-offs', fontsize=14, pad=15)
plt.legend(loc='lower left', fontsize=10)
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig('advanced_4_threshold_curve.png', dpi=300)
plt.close()

print("\n✅ All 4 advanced graphs successfully generated!")
'''

"""
generate_video_graphs.py
========================
Generates the two missing video pipeline graphs:
  1. Per-model mean score grouped bar chart (Section 6.8)
  2. Threshold 0.27 vs 0.50 comparison bar chart (Section 6.7)

Usage:
    python generate_video_graphs.py --csv final_hybrid_detection_results.csv

Outputs (saved to ./graphs/):
    - video_per_model_mean_scores.png
    - video_threshold_comparison.png

Requirements:
    pip install pandas matplotlib seaborn scikit-learn numpy
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUTPUT_DIR = "./graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REAL_COLOR  = "#2ECC71"   # green
FAKE_COLOR  = "#E74C3C"   # red
BLUE        = "#2980B9"
ORANGE      = "#E67E22"
DARK_BLUE   = "#1A3A5C"
GRID_COLOR  = "#E8E8E8"

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    print(f"Columns: {list(df.columns)}")

    # Normalise true_label to binary 0/1
    label_col = None
    for c in ["true_label", "true_binary", "label"]:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError("Cannot find a label column in the CSV.")

    df["true_binary"] = (df[label_col].astype(str).str.lower()
                         .isin(["fake", "1", "true"]).astype(int))

    # Predicted binary from final_score >= threshold
    if "final_score" not in df.columns:
        raise ValueError("Cannot find 'final_score' column.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 1 — Per-model mean score grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_per_model_mean_scores(df: pd.DataFrame):
    """
    For each signal component, compute mean predicted fake-probability
    for Real videos and for Fake videos, then plot side-by-side bars.
    Includes standard error bars.
    """

    # Map column names → display labels
    SIGNAL_MAP = {
        "efficientnet_b3_score": "EfficientNet-B3",
        "xception_score":        "Xception",
        "vit_score":             "ViT-Small",
        "convnext_score":        "ConvNeXt-Small",
        "temporal_score":        "Temporal Analyzer",
        "quality_score":         "Quality Mismatch",
    }

    # Keep only columns that exist in the dataframe
    available = {k: v for k, v in SIGNAL_MAP.items() if k in df.columns}
    if not available:
        # Try alternate column names
        ALT_MAP = {
            "efficientnet_b3": "EfficientNet-B3",
            "xception":        "Xception",
            "vit":             "ViT-Small",
            "convnext":        "ConvNeXt-Small",
            "temporal_score":  "Temporal Analyzer",
            "quality_score":   "Quality Mismatch",
            "cnn_score":       "CNN Ensemble",
        }
        available = {k: v for k, v in ALT_MAP.items() if k in df.columns}

    real_df = df[df["true_binary"] == 0]
    fake_df = df[df["true_binary"] == 1]

    signals      = []
    real_means   = []
    fake_means   = []
    real_sems    = []
    fake_sems    = []

    for col, label in available.items():
        r_vals = real_df[col].dropna()
        f_vals = fake_df[col].dropna()
        signals.append(label)
        real_means.append(r_vals.mean())
        fake_means.append(f_vals.mean())
        real_sems.append(r_vals.sem())
        fake_sems.append(f_vals.sem())

    n      = len(signals)
    x      = np.arange(n)
    width  = 0.35

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars_real = ax.bar(
        x - width / 2, real_means, width,
        yerr=real_sems, capsize=4,
        color=REAL_COLOR, alpha=0.88, label="Real Videos",
        error_kw={"elinewidth": 1.5, "ecolor": "#1A7A3E"}
    )
    bars_fake = ax.bar(
        x + width / 2, fake_means, width,
        yerr=fake_sems, capsize=4,
        color=FAKE_COLOR, alpha=0.88, label="Fake Videos",
        error_kw={"elinewidth": 1.5, "ecolor": "#8B0000"}
    )

    # Annotate bar tops
    for bar in bars_real:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                f"{h:.3f}", ha="center", va="bottom",
                fontsize=9, color="#1A7A3E", fontweight="bold")

    for bar in bars_fake:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                f"{h:.3f}", ha="center", va="bottom",
                fontsize=9, color="#8B0000", fontweight="bold")

    # Separation arrows
    for i, (rm, fm) in enumerate(zip(real_means, fake_means)):
        sep = abs(fm - rm)
        mid = (rm + fm) / 2
        ax.annotate(
            f"Δ{sep:.3f}",
            xy=(x[i], mid),
            fontsize=8, ha="center", va="center",
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CCCCCC", alpha=0.8)
        )

    ax.axhline(0.50, color="#555555", linestyle="--", linewidth=1.2, alpha=0.6,
               label="Decision Boundary (0.50)")

    ax.set_xticks(x)
    ax.set_xticklabels(signals, fontsize=11, rotation=12)
    ax.set_ylabel("Mean Predicted Fake Probability", fontsize=12)
    ax.set_xlabel("Signal Component", fontsize=12)
    ax.set_title(
        "Mean Component Scores: Real vs Fake Videos\n"
        "(Error bars = ±1 Standard Error)",
        fontsize=14, fontweight="bold", pad=16
    )
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, loc="upper left")

    # Colour-code x-axis labels: red for weak signals
    WEAK_SIGNALS = {"Temporal Analyzer", "Quality Mismatch"}
    for tick, label in zip(ax.get_xticklabels(), signals):
        if label in WEAK_SIGNALS:
            tick.set_color("#CC4444")
        else:
            tick.set_color(DARK_BLUE)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "video_per_model_mean_scores.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 2 — Threshold 0.27 vs 0.50 comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_threshold_comparison(df: pd.DataFrame):
    """
    Side-by-side grouped bar chart comparing four metrics
    (Accuracy, Precision, Recall, F1) at two thresholds:
    the implemented default (0.50) and the empirically optimal F1 (0.27).
    Also prints the exact numbers for reference.
    """

    y_true = df["true_binary"].values
    scores = df["final_score"].values

    THRESHOLDS = {
        "Default\n(0.50)": 0.50,
        "Optimal F1\n(0.27)": 0.27,
    }

    metrics_store = {}

    for label, thresh in THRESHOLDS.items():
        y_pred = (scores >= thresh).astype(int)
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        metrics_store[label] = {
            "Accuracy":  acc,
            "Precision": prec,
            "Recall":    rec,
            "F1-Score":  f1,
        }
        print(f"\nThreshold = {thresh:.2f} ({label.strip()}):")
        print(f"  Accuracy  = {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  Precision = {prec:.4f}")
        print(f"  Recall    = {rec:.4f}")
        print(f"  F1-Score  = {f1:.4f}")

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    thresh_labels = list(THRESHOLDS.keys())

    COLORS = {
        "Default\n(0.50)":    "#2980B9",
        "Optimal F1\n(0.27)": "#E67E22",
    }

    x     = np.arange(len(metric_names))
    width = 0.32
    n_t   = len(thresh_labels)

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    offsets = np.linspace(-(n_t - 1) * width / 2,
                           (n_t - 1) * width / 2, n_t)

    for i, (t_label, offset) in enumerate(zip(thresh_labels, offsets)):
        vals  = [metrics_store[t_label][m] for m in metric_names]
        color = COLORS[t_label]
        bars  = ax.bar(x + offset, vals, width,
                       label=t_label.replace("\n", " "),
                       color=color, alpha=0.88)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val*100:.1f}%",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color=color
            )

    # Highlight F1 column with background
    ax.axvspan(x[-1] - 0.5, x[-1] + 0.5,
               alpha=0.07, color="#E67E22", label="_nolegend_")

    ax.axhline(0.80, color="#AAAAAA", linestyle=":", linewidth=1, alpha=0.7)
    ax.axhline(0.90, color="#AAAAAA", linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Performance Metric", fontsize=12)
    ax.set_title(
        "Threshold Comparison: Default (0.50) vs Optimal F1 (0.27)\n"
        "Video Pipeline — 196 Balanced Evaluation Videos",
        fontsize=14, fontweight="bold", pad=16
    )
    ax.set_ylim(0, 1.08)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=GRID_COLOR)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, title="Decision Threshold", title_fontsize=10)

    # Annotation box explaining the F1 threshold
    diff_f1 = (metrics_store["Optimal F1\n(0.27)"]["F1-Score"] -
               metrics_store["Default\n(0.50)"]["F1-Score"])
    diff_rec = (metrics_store["Optimal F1\n(0.27)"]["Recall"] -
                metrics_store["Default\n(0.50)"]["Recall"])
    note = (f"Optimal F1 threshold (0.27)\n"
            f"F1 gain:    +{diff_f1*100:.1f}%\n"
            f"Recall gain: +{diff_rec*100:.1f}%")
    ax.text(0.98, 0.03, note,
            transform=ax.transAxes,
            fontsize=9, va="bottom", ha="right",
            bbox=dict(boxstyle="round", fc="#FFF3E0", ec="#E67E22", alpha=0.9))

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "video_threshold_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate missing video pipeline graphs for thesis."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to final_hybrid_detection_results.csv"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("VIDEO PIPELINE GRAPH GENERATOR")
    print("=" * 60)

    df = load_csv(args.csv)

    print("\n[1/2] Generating per-model mean score bar chart...")
    plot_per_model_mean_scores(df)

    print("\n[2/2] Generating threshold comparison bar chart...")
    plot_threshold_comparison(df)

    print("\n" + "=" * 60)
    print("Done. Both graphs saved to ./graphs/")
    print("=" * 60)


if __name__ == "__main__":
    main()