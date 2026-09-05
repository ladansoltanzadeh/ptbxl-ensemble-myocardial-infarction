# =============================================================================
# ensemble_three_models.py - Ensemble of V1, V2, V3 (logit averaging)
# =============================================================================
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from auto_train_teacher import (
    TeacherBinary, ECGDataset, tune_threshold,
    ROOT, VAL_CSV, TEST_CSV, make_autocast
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

RESULTS_DIR = Path(r"C:\ptbxlesultsuto_teacher")
V1_CKPT = RESULTS_DIR / "best_teacher_v1.pt"
V2_DIR = RESULTS_DIR / "wider_transformer_v2"
V3_DIR = RESULTS_DIR / "wider_transformer_v3"

def find_best_checkpoint(directory):
    history_path = directory / "history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"history.csv not found in {directory}")
    df = pd.read_csv(history_path)
    best_row = df.loc[df["val_accuracy"].idxmax()]
    best_epoch = int(best_row["epoch"])
    ckpt_path = directory / f"checkpoint_epoch_{best_epoch}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"✅ Found best checkpoint: epoch {best_epoch} (Acc: {best_row['val_accuracy']:.2f}%)")
    return ckpt_path, best_row["val_accuracy"]

print("=" * 60)
print("ENSEMBLE: V1 + V2 + V3 (logit averaging)")
print("=" * 60)

ckpt_v1 = torch.load(V1_CKPT, map_location="cpu")
config = ckpt_v1["config"]
model_v1 = TeacherBinary(
    num_classes=2,
    transformer_layers=config["transformer_layers"],
    hidden_dim=config["transformer_hidden"],
    dropout=config["dropout"],
).to(device)
model_v1.load_state_dict(ckpt_v1["model_state_dict"])
model_v1.eval()
print(f"✅ V1 loaded (params: {sum(p.numel() for p in model_v1.parameters()):,})")

v2_ckpt_path, _ = find_best_checkpoint(V2_DIR)
ckpt_v2 = torch.load(v2_ckpt_path, map_location="cpu")
model_v2 = TeacherBinary(
    num_classes=2,
    transformer_layers=config["transformer_layers"],
    hidden_dim=config["transformer_hidden"],
    dropout=config["dropout"],
).to(device)
model_v2.load_state_dict(ckpt_v2["model_state_dict"])
model_v2.eval()
print(f"✅ V2 loaded (params: {sum(p.numel() for p in model_v2.parameters()):,})")

v3_ckpt_path, _ = find_best_checkpoint(V3_DIR)
ckpt_v3 = torch.load(v3_ckpt_path, map_location="cpu")
model_v3 = TeacherBinary(
    num_classes=2,
    transformer_layers=config["transformer_layers"],
    hidden_dim=config["transformer_hidden"],
    dropout=config["dropout"],
).to(device)
model_v3.load_state_dict(ckpt_v3["model_state_dict"])
model_v3.eval()
print(f"✅ V3 loaded (params: {sum(p.numel() for p in model_v3.parameters()):,})")

@torch.no_grad()
def evaluate_ensemble(loader, tta=4, threshold=0.5):
    probs_all, y_all = [], []
    autocast = make_autocast(device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast():
            logits1 = model_v1(x)
            logits2 = model_v2(x)
            logits3 = model_v3(x)
            avg_logits = (logits1 + logits2 + logits3) / 3
            probs = torch.softmax(avg_logits, dim=1)[:, 1].cpu().numpy()
        probs_all.append(probs)
        y_all.append(y.numpy())
    p = np.concatenate(probs_all)
    y = np.concatenate(y_all)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    acc = 100.0 * (pred == y).mean()
    sens = 100.0 * tp / max(1, tp + fn)
    spec = 100.0 * tn / max(1, tn + fp)
    f1 = 100.0 * (2 * tp) / max(1, 2 * tp + fp + fn)
    auc = 100.0 * roc_auc_score(y, p)
    return {"accuracy": float(acc), "sensitivity": float(sens), "specificity": float(spec),
            "f1": float(f1), "auc": float(auc), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "probabilities": p, "targets": y}

print("
📊 Tuning threshold on Validation Set...")
val_ds = ECGDataset(VAL_CSV, ROOT, augment=False)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
metrics_val = evaluate_ensemble(val_loader, tta=2, threshold=0.5)
best_thr = tune_threshold(metrics_val["probabilities"], metrics_val["targets"])
print(f"   ✅ Best threshold: {best_thr['threshold']:.2f} (Acc: {best_thr['accuracy']:.2f}%)")

print("
🎯 Final Evaluation on Test Set (ONCE)...")
test_ds = ECGDataset(TEST_CSV, ROOT, augment=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
test_metrics = evaluate_ensemble(test_loader, tta=4, threshold=best_thr["threshold"])

print("
" + "=" * 60)
print("🏆 ENSEMBLE (V1+V2+V3) FINAL RESULTS — TEST SET")
print("=" * 60)
print(f"Threshold used:          {best_thr['threshold']:.2f}")
print(f"Accuracy:                {test_metrics['accuracy']:.2f}%")
print(f"Sensitivity (MI/Isch):   {test_metrics['sensitivity']:.2f}%")
print(f"Specificity:             {test_metrics['specificity']:.2f}%")
print(f"F1-Score:                {test_metrics['f1']:.2f}%")
print(f"AUC:                     {test_metrics['auc']:.2f}%")
print(f"Confusion Matrix:        TP={test_metrics['tp']}  FN={test_metrics['fn']}  TN={test_metrics['tn']}  FP={test_metrics['fp']}")
print("=" * 60)

results_df = pd.DataFrame([{
    "model": "Ensemble (V1+V2+V3)",
    "threshold": best_thr["threshold"],
    "accuracy": test_metrics["accuracy"],
    "sensitivity": test_metrics["sensitivity"],
    "specificity": test_metrics["specificity"],
    "f1": test_metrics["f1"],
    "auc": test_metrics["auc"],
    "tp": test_metrics["tp"],
    "fn": test_metrics["fn"],
    "tn": test_metrics["tn"],
    "fp": test_metrics["fp"],
}])
results_df.to_csv(RESULTS_DIR / "ensemble_three_models_test_results.csv", index=False)
print(f"
💾 Results saved to: {RESULTS_DIR / 'ensemble_three_models_test_results.csv'}")
