import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
'''
# 1. Load the High-Quality Datasets
fake_df = pd.read_csv('fake_new_pipeline_results.csv')
celeb_df = pd.read_csv('celeb_pipeline_results.csv')

# Combine them into one evaluation dataframe
df = pd.concat([fake_df, celeb_df])

# 2. Set up the plot aesthetics
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# 3. Plot the KDE distributions
sns.kdeplot(data=df[df['true_label'] == 'Real']['meta_learner_prob'], 
            color='blue', fill=True, label='Authentic (Real) Images', alpha=0.5, linewidth=2)
sns.kdeplot(data=df[df['true_label'] == 'Fake']['meta_learner_prob'], 
            color='darkorange', fill=True, label='Synthetic (Fake) Images', alpha=0.5, linewidth=2)

# 4. Add the Decision Boundary line
plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary (0.50)')

# 5. Format the graph for an academic paper
plt.title('Meta-Learner Output Probability Distribution\n(High-Fidelity Evaluation Corpus)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Fake Probability', fontsize=12)
plt.ylabel('Kernel Density', fontsize=12)
plt.xlim(0, 1)
plt.legend(fontsize=11)
plt.tight_layout()

# Save the high-res graph
plt.savefig('score_distribution_kde.png', dpi=300)
print("Saved score_distribution_kde.png")
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
fake_df = pd.read_csv('fake_new_pipeline_results.csv')
celeb_df = pd.read_csv('celeb_pipeline_results.csv')
df = pd.concat([fake_df, celeb_df])

# 2. Calculate CNN Disagreement (Max CNN score - Min CNN score)
base_models = ['vit_score', 'convnext_score', 'xception_score', 'effnet_score']
df['disagreement'] = df[base_models].max(axis=1) - df[base_models].min(axis=1)

# 3. Determine if the prediction was Correct or Incorrect
df['prediction_status'] = df['meta_learner_label'] == df['true_label']
df['prediction_status'] = df['prediction_status'].map({True: 'Correct Prediction', False: 'Incorrect Prediction'})

# 4. Plot the Boxplot
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

sns.boxplot(x='prediction_status', y='disagreement', data=df, 
            palette={'Correct Prediction': '#2ecc71', 'Incorrect Prediction': '#e74c3c'},
            width=0.5)

# 5. Format for academic paper
plt.title('Inter-Model Disagreement vs. Prediction Correctness', fontsize=14, fontweight='bold')
plt.xlabel('Pipeline Outcome', fontsize=12)
plt.ylabel('CNN Disagreement Spread (Max - Min)', fontsize=12)
plt.tight_layout()

# Save the high-res graph
plt.savefig('disagreement_boxplot.png', dpi=300)
print("Saved disagreement_boxplot.png")

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Hardcode the diagnostic results from our analysis
categories = ['Standard Media\n(Celeb/DSLR Dataset)', 'Extreme Post-Processing\n(Beauty Filtered Dataset)']
accuracies = [94.2, 13.4]

# 2. Set up the plot
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

# 3. Create the Bar Chart
bars = plt.bar(categories, accuracies, color=['#3498db', '#e74c3c'], width=0.5)

# 4. Add the percentage labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}%', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# 5. Format for academic paper
plt.title('Pipeline Recall on Authentic Images:\nImpact of High-Frequency Detail Loss', fontsize=14, fontweight='bold')
plt.ylabel('Authentic Recall Accuracy (%)', fontsize=12)
plt.ylim(0, 110) # Give room for the text label
plt.tight_layout()

# Save the high-res graph
plt.savefig('boundary_condition_barchart.png', dpi=300)
print("Saved boundary_condition_barchart.png")