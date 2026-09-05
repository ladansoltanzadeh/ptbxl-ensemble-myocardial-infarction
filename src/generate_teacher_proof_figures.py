# generate_teacher_proof_figures.py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(r"C:\ptbxl")
RESULTS_DIR = BASE_DIR / "results"
PROOF_DIR = RESULTS_DIR / "teacher_proof"
PROOF_DIR.mkdir(parents=True, exist_ok=True)

trials_json = RESULTS_DIR / "auto_teacher" / "all_trials_results.json"
with open(trials_json, 'r') as f:
    trials_data = json.load(f)

trial_histories = {}
trial_params = {}
for t in trials_data:
    name = t['trial_name']
    hist = pd.DataFrame(t['history'])
    trial_histories[name] = hist
    config = t['config']
    trial_params[name] = {
        'hidden_dim': config.get('transformer_hidden', 0),
        'layers': config.get('transformer_layers', 0),
        'dropout': config.get('dropout', 0),
        'lr': config.get('learning_rate', 0),
        'batch_size': config.get('batch_size', 0),
        'epochs': config.get('epochs_per_trial', 0),
        'best_val_acc': t['best_val_accuracy'],
        'best_val_sens': max(hist['val_sensitivity']) if not hist.empty else 0,
        'best_val_spec': max(hist['val_specificity']) if not hist.empty else 0,
    }
df_params = pd.DataFrame(trial_params).T.reset_index()
df_params.rename(columns={'index': 'Trial'}, inplace=True)

# فیگور ۱: منحنی‌های یادگیری
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, (name, hist) in enumerate(trial_histories.items()):
    if idx >= 6:
        break
    ax = axes[idx]
    ax.plot(hist['epoch'], hist['train_loss'], 'b-o', label='Train Loss', markersize=3)
    ax2 = ax.twinx()
    ax2.plot(hist['epoch'], hist['val_accuracy'], 'r-s', label='Val Acc', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color='b')
    ax2.set_ylabel('Accuracy (%)', color='r')
    ax.set_title(f'{name}')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
for i in range(len(trial_histories), 6):
    fig.delaxes(axes[i])
plt.tight_layout()
plt.savefig(PROOF_DIR / 'fig_learning_curves_all_trials.png', dpi=300)
plt.close()

# فیگور ۲: موازی
from pandas.plotting import parallel_coordinates
plot_df = df_params[['Trial', 'hidden_dim', 'layers', 'dropout', 'lr', 'best_val_acc']].copy()
plot_df['lr'] = plot_df['lr'] * 1e4
plot_df.columns = ['Trial', 'Hidden Dim', 'Layers', 'Dropout', 'LR (×1e4)', 'Val Acc (%)']
plt.figure(figsize=(12, 6))
parallel_coordinates(plot_df, class_column='Trial', colormap='viridis', alpha=0.7)
plt.title('Parallel Coordinates: Hyperparameters vs Validation Accuracy')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PROOF_DIR / 'fig_parallel_coordinates.png', dpi=300)
plt.close()

# فیگور ۳: حبابی
teacher_results = {
    'V1 Alone': {'acc': 86.08, 'sens': 76.46, 'params': 139389474},
    'Ensemble V1+V2': {'acc': 87.40, 'sens': 76.18, 'params': 278778948},
    'Ensemble V1+V2+V3': {'acc': 87.12, 'sens': 77.03, 'params': 418168422},
    'Weighted Ensemble': {'acc': 86.94, 'sens': 76.60, 'params': 278778948},
}
df_teacher = pd.DataFrame(teacher_results).T.reset_index()
df_teacher.rename(columns={'index': 'Model'}, inplace=True)

fig, ax = plt.subplots(figsize=(10, 6))
models = df_teacher['Model'].tolist()
params = df_teacher['params'].tolist()
acc = df_teacher['acc'].tolist()
sens = df_teacher['sens'].tolist()
sizes = [s * 50 for s in sens]
sc = ax.scatter(params, acc, s=sizes, c=sens, cmap='coolwarm', alpha=0.7, edgecolors='black')
ax.set_xscale('log')
ax.set_xlabel('Number of Parameters (log scale)')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Model Size vs Accuracy vs Sensitivity (Bubble size = Sensitivity)')
for i, name in enumerate(models):
    ax.annotate(name, (params[i], acc[i]), ha='center', va='bottom', fontsize=8)
cbar = plt.colorbar(sc)
cbar.set_label('Sensitivity (%)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PROOF_DIR / 'fig_bubble_params_acc_sens.png', dpi=300)
plt.close()

# فیگور ۴: همبستگی
corr_data = df_params[['hidden_dim', 'layers', 'dropout', 'lr', 'best_val_acc']].copy()
corr_data.rename(columns={'lr': 'Learning Rate', 'best_val_acc': 'Val Acc'}, inplace=True)
corr = corr_data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation of Hyperparameters with Validation Accuracy')
plt.tight_layout()
plt.savefig(PROOF_DIR / 'fig_hyperparameter_correlation_heatmap.png', dpi=300)
plt.close()

# فیگور ۵: مقایسه همه مدل‌ها
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(df_teacher))
width = 0.35
bars1 = ax.bar(x - width/2, df_teacher['acc'], width, label='Accuracy', color='#2E86AB')
bars2 = ax.bar(x + width/2, df_teacher['sens'], width, label='Sensitivity', color='#F18F01')
ax.set_xticks(x)
ax.set_xticklabels(df_teacher['Model'], rotation=15, ha='right')
ax.set_ylabel('Percentage (%)')
ax.set_title('Comparison of Teacher Models and Ensembles (Test Set)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(70, 95)
for bar, acc in zip(bars1, df_teacher['acc']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{acc:.1f}%', ha='center', fontsize=8)
for bar, sens in zip(bars2, df_teacher['sens']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{sens:.1f}%', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(PROOF_DIR / 'fig_all_models_comparison.png', dpi=300)
plt.close()

# جداول
table_trials = df_params[['Trial', 'hidden_dim', 'layers', 'dropout', 'lr', 'batch_size', 'epochs', 'best_val_acc', 'best_val_sens', 'best_val_spec']]
table_trials.columns = ['Trial', 'Hidden Dim', 'Layers', 'Dropout', 'LR', 'Batch Size', 'Epochs', 'Val Acc (%)', 'Val Sens (%)', 'Val Spec (%)']
table_trials.to_csv(PROOF_DIR / 'table_all_trials_comparison.csv', index=False)

table_ensembles = df_teacher.copy()
table_ensembles = table_ensembles[['Model', 'acc', 'sens', 'spec', 'auc', 'params']]
table_ensembles.columns = ['Model', 'Test Acc (%)', 'Test Sens (%)', 'Test Spec (%)', 'Test AUC (%)', 'Params']
table_ensembles.to_csv(PROOF_DIR / 'table_all_ensembles_comparison.csv', index=False)

print("✅ Teacher proof figures and tables generated successfully.")
