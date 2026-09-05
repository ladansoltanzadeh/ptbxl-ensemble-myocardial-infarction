# tune_student_threshold.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from auto_train_teacher import ROOT, VAL_CSV, make_autocast
from train_student_kd import StudentLite, KDDataset, evaluate_student

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

CKPT_DIR = Path(r"C:\ptbxl\project\checkpoints\student")
CKPT_PATH = CKPT_DIR / "student_epoch_19.pt"

if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

ckpt = torch.load(CKPT_PATH, map_location="cpu")
student = StudentLite(num_classes=2, in_ch=2).to(device)
student.load_state_dict(ckpt["model_state_dict"])
student.eval()
print(f"✅ Student loaded from {CKPT_PATH.name}")

val_ds = KDDataset(VAL_CSV, ROOT, augment=False)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
print(f"✅ Validation set loaded: {len(val_ds)} samples")

autocast = make_autocast(device)

thresholds = np.arange(0.30, 0.75, 0.01)
results = []
for thr in thresholds:
    metrics = evaluate_student(student, val_loader, device, autocast, threshold=thr)
    results.append({'threshold': thr, 'accuracy': metrics['accuracy'], 'sensitivity': metrics['sensitivity'], 'specificity': metrics['specificity']})

df = pd.DataFrame(results)

best = df[df['sensitivity'] >= 80].sort_values('accuracy', ascending=False).iloc[0]
print(f"
✅ Best threshold with Sens≥80%: {best['threshold']:.2f} (Acc: {best['accuracy']:.2f}%, Sens: {best['sensitivity']:.2f}%)")

best_acc = df.loc[df['accuracy'].idxmax()]
print(f"✅ Threshold with max Accuracy: {best_acc['threshold']:.2f} (Acc: {best_acc['accuracy']:.2f}%, Sens: {best_acc['sensitivity']:.2f}%)")
