# =============================================================================
# train_student_kd.py - Student Training with Knowledge Distillation
# Teacher: Ensemble V1+V2 (12-lead), Student: 2-lead (II + aVF)
# با قابلیت Resume کامل
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import time
from copy import deepcopy

# ─── تعریف device (رفع خطای Import) ────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ─── import از auto_train_teacher ──────────────────────────────
from auto_train_teacher import (
    TeacherBinary,
    ECGDataset,
    tune_threshold,
    ROOT,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    make_autocast,
)

# ─── مسیرهای Teacher ────────────────────────────────────────────
RESULTS_DIR = Path(r"C:\ptbxlesultsuto_teacher")
V1_CKPT = RESULTS_DIR / "best_teacher_v1.pt"
V2_DIR = RESULTS_DIR / "wider_transformer_v2"

# ════════════════════════════════════════════════════════════════
# 1. Dataset برای KD (هم ۱۲ لید و هم ۲ لید)
# ════════════════════════════════════════════════════════════════
class KDDataset(Dataset):
    """
    دیتاستی که برای هر نمونه، هم ۱۲ لید کامل و هم ۲ لید (II و aVF) را برمی‌گرداند.
    Teacher از ۱۲ لید و Student از ۲ لید استفاده می‌کند.
    """
    def __init__(self, csv_path, root, augment=False, limit=None):
        self.full_ds = ECGDataset(csv_path, root, augment=augment, limit=limit)
        # لیدهای II و aVF به ترتیب indexهای 1 و 5 هستند (طبق مستندات PTB-XL)
        self.lead_indices = [1, 5]

    def __len__(self):
        return len(self.full_ds)

    def __getitem__(self, idx):
        full_x, y = self.full_ds[idx]          # full_x: (12, 5000)
        student_x = full_x[self.lead_indices, :]  # (2, 5000)
        return full_x, student_x, y

# ════════════════════════════════════════════════════════════════
# 2. معماری Student (سبک‌وزن، ۲ لید)
# ════════════════════════════════════════════════════════════════
class StudentLite(nn.Module):
    def __init__(self, num_classes=2, in_ch=2):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, 32, 7, stride=2, padding=3)
        self.b1 = nn.BatchNorm1d(32)
        self.c2 = nn.Conv1d(32, 64, 9, stride=2, padding=4)
        self.b2 = nn.BatchNorm1d(64)
        self.c3 = nn.Conv1d(64, 128, 11, stride=2, padding=5)
        self.b3 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(50)
        # Transformer Encoder سبک
        enc = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        # Projection برای Feature Distillation (اختیاری)
        self.feat_proj = nn.Conv1d(128, 512, 1)
        # کلاسیفایر
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, 2, 5000)
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = F.relu(self.b3(self.c3(x)))
        f = self.pool(x)                     # (B, 128, 50)
        h = f.permute(0, 2, 1)               # (B, 50, 128)
        h = self.transformer(h)              # (B, 50, 128)
        f_proj = self.feat_proj(f)           # (B, 512, 50) برای Feature Distillation
        pooled = h.mean(dim=1)               # (B, 128)
        logits = self.classifier(pooled)
        return logits, f_proj

# ════════════════════════════════════════════════════════════════
# 3. بارگذاری Teacher Ensemble (V1+V2)
# ════════════════════════════════════════════════════════════════
def load_teacher_ensemble():
    # بارگذاری V1
    ckpt_v1 = torch.load(V1_CKPT, map_location="cpu")
    config_v1 = ckpt_v1["config"]
    model_v1 = TeacherBinary(
        num_classes=2,
        transformer_layers=config_v1["transformer_layers"],
        hidden_dim=config_v1["transformer_hidden"],
        dropout=config_v1["dropout"],
    ).to(device)
    model_v1.load_state_dict(ckpt_v1["model_state_dict"])
    model_v1.eval()
    for p in model_v1.parameters():
        p.requires_grad_(False)

    # بارگذاری V2 (بهترین چک‌پوینت)
    history_v2 = pd.read_csv(V2_DIR / "history.csv")
    best_row = history_v2.loc[history_v2["val_accuracy"].idxmax()]
    best_epoch = int(best_row["epoch"])
    ckpt_v2 = torch.load(V2_DIR / f"checkpoint_epoch_{best_epoch}.pt", map_location="cpu")
    model_v2 = TeacherBinary(
        num_classes=2,
        transformer_layers=config_v1["transformer_layers"],
        hidden_dim=config_v1["transformer_hidden"],
        dropout=config_v1["dropout"],
    ).to(device)
    model_v2.load_state_dict(ckpt_v2["model_state_dict"])
    model_v2.eval()
    for p in model_v2.parameters():
        p.requires_grad_(False)

    print("✅ Teacher Ensemble (V1+V2) loaded successfully")
    return model_v1, model_v2

# ════════════════════════════════════════════════════════════════
# 4. تابع ارزیابی Student
# ════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_student(model, loader, device, autocast, threshold=0.5):
    model.eval()
    probs_all, y_all = [], []
    for full_x, student_x, y in loader:
        student_x = student_x.to(device, non_blocking=True)
        with autocast():
            logits, _ = model(student_x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
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
    return {
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1),
        "auc": float(auc),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "probabilities": p,
        "targets": y,
    }

# ════════════════════════════════════════════════════════════════
# 5. تابع اصلی آموزش
# ════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("STUDENT TRAINING WITH KNOWLEDGE DISTILLATION")
    print("Teacher: Ensemble V1+V2 (12-lead)")
    print("Student: 2-lead (II + aVF)")
    print("=" * 60)

    autocast = make_autocast(device)

    # ─── بارگذاری Teacher ──────────────────────────────────────────
    teacher1, teacher2 = load_teacher_ensemble()

    # ─── داده‌ها ────────────────────────────────────────────────────
    train_ds = KDDataset(TRAIN_CSV, ROOT, augment=True)
    val_ds = KDDataset(VAL_CSV, ROOT, augment=False)
    test_ds = KDDataset(TEST_CSV, ROOT, augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    # ─── Student ────────────────────────────────────────────────────
    student = StudentLite(num_classes=2, in_ch=2).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"🎓 Student parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # ─── وزن‌های کلاس ──────────────────────────────────────────────
    train_df = train_ds.full_ds.df
    n_pos = int(train_df["class_id"].sum())
    n_neg = len(train_df) - n_pos
    class_weight = torch.tensor(
        [len(train_df) / (2 * max(1, n_neg)), len(train_df) / (2 * max(1, n_pos))],
        dtype=torch.float32,
        device=device,
    )
    print(f"⚖️ Class weights: neg={class_weight[0]:.3f}, pos={class_weight[1]:.3f}")

    # ─── Optimizer, Scheduler ──────────────────────────────────────
    optimizer = optim.AdamW(student.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-6)

    # ─── Hyperparameters KD ──────────────────────────────────────
    T = 2.0
    ALPHA = 0.7   # وزن CE hard
    BETA = 0.3    # وزن KD soft

    # ─── مسیر ذخیره ──────────────────────────────────────────────────
    CKPT_DIR = ROOT / "project" / "checkpoints" / "student"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR = ROOT / "results" / "student"
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Resume ────────────────────────────────────────────────────
    start_epoch = 1
    best_val_acc = -1.0
    best_state = None
    no_improve = 0
    patience = 5
    epochs = 20
    history = []

    ckpt_files = sorted(CKPT_DIR.glob("student_epoch_*.pt"))
    if ckpt_files:
        latest = ckpt_files[-1]
        print(f"🔄 Resuming from {latest.name}")
        ckpt = torch.load(latest, map_location=device)
        student.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", -1)
        best_state = ckpt.get("best_state")
        no_improve = ckpt.get("no_improve", 0)
        history = ckpt.get("history", [])
        print(f"   Resuming from epoch {start_epoch}/{epochs}")
    else:
        print("   Starting fresh training")

    # ─── حلقه آموزش ──────────────────────────────────────────────────
    print("
🔥 Starting training...")
    for epoch in range(start_epoch, epochs + 1):
        student.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for full_x, student_x, y in pbar:
            full_x = full_x.to(device, non_blocking=True)
            student_x = student_x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                # ── Teacher Ensemble ──
                with torch.no_grad():
                    logits_t1 = teacher1(full_x)
                    logits_t2 = teacher2(full_x)
                    logits_t_avg = (logits_t1 + logits_t2) / 2

                # ── Student ──
                logits_s, _ = student(student_x)

                # ── Loss ──
                loss_ce = F.cross_entropy(logits_s, y, weight=class_weight, label_smoothing=0.05)
                loss_kd = T**2 * F.kl_div(
                    F.log_softmax(logits_s / T, dim=1),
                    F.softmax(logits_t_avg / T, dim=1),
                    reduction='batchmean'
                )
                loss = ALPHA * loss_ce + BETA * loss_kd

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()

        # ── ارزیابی روی Validation ──
        metrics = evaluate_student(student, val_loader, device, autocast, threshold=0.5)
        avg_loss = total_loss / len(train_loader)
        row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "val_accuracy": metrics["accuracy"],
            "val_sensitivity": metrics["sensitivity"],
            "val_specificity": metrics["specificity"],
            "val_f1": metrics["f1"],
            "val_auc": metrics["auc"],
        }
        history.append(row)
        print(f"  Epoch {epoch:02d} | loss={avg_loss:.4f} | Val Acc={metrics['accuracy']:.2f}% | Sens={metrics['sensitivity']:.2f}% | Spec={metrics['specificity']:.2f}%")

        # ── ذخیره بهترین ──
        if metrics["accuracy"] > best_val_acc:
            best_val_acc = metrics["accuracy"]
            best_state = deepcopy(student.state_dict())
            no_improve = 0
            print("   ↳ New best!")
        else:
            no_improve += 1

        # ── Checkpoint ──
        torch.save({
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "best_state": best_state,
            "no_improve": no_improve,
            "history": history,
        }, CKPT_DIR / f"student_epoch_{epoch}.pt")

        # ── ذخیره تاریخچه ──
        pd.DataFrame(history).to_csv(HISTORY_DIR / "student_history.csv", index=False)

        if no_improve >= patience:
            print(f"⏹ Early stopping at epoch {epoch}")
            break

    # ─── بارگذاری بهترین Student ──────────────────────────────────
    if best_state is not None:
        student.load_state_dict(best_state)

    # ─── تنظیم آستانه روی Validation ──────────────────────────────
    val_metrics = evaluate_student(student, val_loader, device, autocast, threshold=0.5)
    best_thr = tune_threshold(val_metrics["probabilities"], val_metrics["targets"])
    print(f"
✅ Best threshold on Val: {best_thr['threshold']:.2f} (Acc: {best_thr['accuracy']:.2f}%)")

    # ─── ارزیابی نهایی روی Test ──────────────────────────────────
    test_metrics = evaluate_student(student, test_loader, device, autocast, threshold=best_thr["threshold"])

    print("
" + "=" * 60)
    print("🏆 STUDENT FINAL RESULTS — TEST SET")
    print("=" * 60)
    print(f"Threshold used:          {best_thr['threshold']:.2f}")
    print(f"Accuracy:                {test_metrics['accuracy']:.2f}%")
    print(f"Sensitivity (MI/Isch):   {test_metrics['sensitivity']:.2f}%")
    print(f"Specificity:             {test_metrics['specificity']:.2f}%")
    print(f"F1-Score:                {test_metrics['f1']:.2f}%")
    print(f"AUC:                     {test_metrics['auc']:.2f}%")
    print(f"Confusion Matrix:        TP={test_metrics['tp']}  FN={test_metrics['fn']}  TN={test_metrics['tn']}  FP={test_metrics['fp']}")
    print("=" * 60)

    # ─── ذخیره نتایج ──────────────────────────────────────────────
    results_df = pd.DataFrame([{
        "model": "Student (KD from Ensemble V1+V2)",
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
    results_df.to_csv(HISTORY_DIR / "student_test_results.csv", index=False)
    print(f"
💾 Results saved to: {HISTORY_DIR / 'student_test_results.csv'}")
    print(f"💾 Checkpoints saved in: {CKPT_DIR}")

if __name__ == "__main__":
    main()
