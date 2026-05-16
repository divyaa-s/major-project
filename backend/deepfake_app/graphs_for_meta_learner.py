import matplotlib.pyplot as plt
import numpy as np

# Data from your benchmark results
strategies = ['Baseline (CNN)', 'Confidence Fusion', 'Dempster-Shafer', 'Meta-Learner']
accuracy = [0.9602, 0.5447, 0.5449, 0.9765]
f1_scores = [0.9737, 0.5857, 0.5859, 0.9847]

x = np.arange(len(strategies))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', color='#3498db')
rects2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='#e74c3c')

ax.set_ylabel('Score')
ax.set_title('Image Pipeline: Fusion Strategy Benchmark')
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.set_ylim(0, 1.1)
ax.legend()

# Add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('fusion_strategy_comparison.png', dpi=300)
plt.show()

import matplotlib.pyplot as plt

# Data from your coefficients
models = ['ConvNeXt-Small', 'ViT-Small', 'Xception', 'EfficientNet-B3', 'Watermark Detector']
weights = [5.7381, 5.5065, 1.7488, 0.5740, 0.0793]

# Sorting for visual appeal (highest to lowest)
data = sorted(zip(weights, models))
weights, models = zip(*data)

plt.figure(figsize=(10, 6))
bars = plt.barh(models, weights, color='#2ecc71')

plt.xlabel('Learned Coefficient (Weight)')
plt.title('Image Meta-Learner: Feature Importance / Model Trust')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add values to bars
for bar in bars:
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
             f'{bar.get_width():.2f}', va='center')

plt.tight_layout()
plt.savefig('meta_learner_weights.png', dpi=300)
plt.show()


import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, color='#9b59b6', linewidth=3)
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)

# Highlight the 0.5 Decision Boundary
plt.scatter([0], [0.5], color='red', zorder=5)
plt.annotate('Decision Boundary (0.50)', xy=(0, 0.5), xytext=(2, 0.4),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('Logistic Meta-Learner Calibration Curve')
plt.xlabel('Aggregated Level-0 Model Signals')
plt.ylabel('Final Probability (P_Fake)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_calibration_curve.png', dpi=300)
plt.show()