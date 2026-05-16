import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ────────────────────────────────────────────────
# Your existing confusion matrix (from earlier)
# Replace these numbers with your actual values if running separately
# ────────────────────────────────────────────────
cm = np.array([
    [726, 192],
    [219, 710]
])

class_names = ["REAL", "WATERMARKED/AI"]

# ────────────────────────────────────────────────
# Plot confusion matrix as image
# ────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            annot_kws={"size": 16})

plt.title('Confusion Matrix (Hold-out Set)', fontsize=14, pad=15)
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11, rotation=0)

# Add some padding
plt.tight_layout()

# Save as high-quality image
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.savefig('confusion_matrix.pdf', format='pdf', bbox_inches='tight')  # good for LaTeX
plt.show()  # optional: show in notebook/jupyter
plt.close()