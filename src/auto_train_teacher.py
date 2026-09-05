#!/usr/bin/env python
# =============================================================================
# auto_train_teacher.py - Hyperparameter search + autonomous training
# Based on PTB-XL Teacher (MI/Ischemia vs Others)
# Fully corrected for dimension mismatch and CPU execution
# =============================================================================

import argparse
import json
import math
import os
import random
import time
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wfdb
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================
# 1. تنظیمات عمومی و ثابت
# ============================================================
SEED = 42
ROOT = Path(r"C:\ptbxl")
SPLIT_DIR = ROOT / "project" / "data" / "splits"
TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# 2. Dataset و Augmentation
# ============================================================
class ECGDataset(Dataset):
    def __init__(self, csv_path: Path, root: Path, augment=False, limit=None):
        self.root = root
        self.augment = augment
        df = pd.read_csv(csv_path).dropna(subset=["class_id", "filename_hr"]).copy()
        df["class_id"] = df["class_id"].astype(int)

        valid_rows = []
        missing = 0
        for _, r in df.iterrows():
            base = self.root / str(r["filename_hr"])
            if base.with_suffix(".dat").exists() and base.with_suffix(".hea").exists():
                valid_rows.append(r)
            else:
                missing += 1

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        if limit is not None:
            self.df = self.df.sample(n=min(limit, len(self.df)), random_state=SEED).reset_index(drop=True)

        pos = int(self.df["class_id"].sum())
        neg = len(self.df) - pos
        print(f"  {csv_path.name}: {len(self.df):,} usable | pos={pos:,} | neg={neg:,} | missing={missing:,} | aug={augment}")

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _augment(sig):
        # Gaussian noise
        if np.random.rand() < 0.6:
            snr = np.random.uniform(12, 30)
            p = np.mean(sig ** 2) + 1e-8
            noise_std = np.sqrt(p / (10 ** (snr / 10)))
            sig = sig + np.random.normal(0, noise_std, sig.shape)

        # Baseline wander
        if np.random.rand() < 0.5:
            t = np.arange(sig.shape[0]) / 500.0
            amp = np.random.uniform(0.05, 0.25)
            freq = np.random.uniform(0.1, 0.6)
            sig = sig + (amp * np.sin(2 * np.pi * freq * t))[:, None]

        # Amplitude scaling
        if np.random.rand() < 0.5:
            sig = sig * np.random.uniform(0.8, 1.2)

        # Random lead dropout
        if np.random.rand() < 0.3:
            n_drop = np.random.randint(1, 3)
            idx = np.random.choice(12, n_drop, replace=False)
            sig[:, idx] = 0.0

        # Time shift
        if np.random.rand() < 0.5:
            sig = np.roll(sig, np.random.randint(-500, 501), axis=0)

        # Cutout
        if np.random.rand() < 0.3:
            length = np.random.randint(250, 1001)
            start = np.random.randint(0, max(1, sig.shape[0] - length + 1))
            sig[start:start + length] = 0.0

        return sig

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        base = self.root / str(r["filename_hr"])

        try:
            sig, _ = wfdb.rdsamp(str(base))
        except Exception as e:
            raise RuntimeError(f"Failed reading {base}: {e}") from e

        if sig.shape != (5000, 12):
            raise RuntimeError(f"Unexpected shape {sig.shape} at {base}")

        sig = sig.astype(np.float32, copy=False)

        if self.augment:
            sig = self._augment(sig)

        # Z-score per lead
        mean = sig.mean(axis=0, keepdims=True)
        std = sig.std(axis=0, keepdims=True)
        sig = (sig - mean) / (std + 1e-8)

        x = torch.from_numpy(sig.T.copy()).float()
        y = torch.tensor(int(r["class_id"]), dtype=torch.long)
        return x, y

# ============================================================
# 3. معماری Teacher (قابل تنظیم با پارامترها) - اصلاح شده
# ============================================================
class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        hidden = max(1, c // r)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c, hidden),
            nn.ReLU(),
            nn.Linear(hidden, c),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, k=7, stride=1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=k // 2)
        self.b2 = nn.BatchNorm1d(cout)
        self.se = SE(cout)
        self.short = (
            nn.Sequential(
                nn.Conv1d(cin, cout, 1, stride=stride),
                nn.BatchNorm1d(cout),
            )
            if (cin != cout or stride != 1)
            else nn.Identity()
        )

    def forward(self, x):
        r = self.short(x)
        x = F.relu(self.b1(self.c1(x)))
        x = self.se(self.b2(self.c2(x)))
        return F.relu(x + r)


class TeacherBinary(nn.Module):
    """
    معماری Teacher با قابلیت تنظیم:
    - transformer_layers: تعداد لایه‌های Transformer
    - hidden_dim: ابعاد پنهان Transformer (می‌تواند با 512 متفاوت باشد)
    - dropout: نرخ dropout
    یک لایه Projection برای تطبیق خروجی CNN (512) به hidden_dim اضافه شده است.
    """
    def __init__(self, num_classes=2, transformer_layers=2, hidden_dim=512, dropout=0.2):
        super().__init__()
        # بخش CNN ثابت (خروجی 512 کانال)
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, 15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(ResBlock(64, 128, stride=2), ResBlock(128, 128))
        self.layer2 = nn.Sequential(ResBlock(128, 256, stride=2), ResBlock(256, 256))
        self.layer3 = nn.Sequential(ResBlock(256, 512, stride=2), ResBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool1d(50)  # خروجی: (B, 512, 50)

        # لایه Projection برای تطبیق 512 → hidden_dim
        self.projection = nn.Linear(512, hidden_dim)

        # Transformer با hidden_dim
        nhead = max(4, hidden_dim // 64)
        if hidden_dim % nhead != 0:
            for h in [8, 12, 16, 20, 24]:
                if hidden_dim % h == 0:
                    nhead = h
                    break
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # کلاسیفایر بر اساس hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 50, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)                     # (B, 512, 50)
        x = x.permute(0, 2, 1)               # (B, 50, 512)
        x = self.projection(x)               # (B, 50, hidden_dim)
        x = self.transformer(x)              # (B, 50, hidden_dim)
        x = x.reshape(x.size(0), -1)         # (B, hidden_dim * 50)
        return self.classifier(x)

# ============================================================
# 4. ابزارهای کمکی (EMA، TTA، ارزیابی)
# ============================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.state[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.state[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.state, strict=True)


def make_autocast(device):
    if device.type != "cuda":
        from contextlib import nullcontext
        return lambda: nullcontext()
    try:
        return lambda: torch.amp.autocast("cuda")
    except Exception:
        return lambda: torch.cuda.amp.autocast()


def make_scaler(device):
    if device.type != "cuda":
        try:
            return torch.amp.GradScaler("cuda", enabled=False)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=False)
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        return torch.cuda.amp.GradScaler()


def tta_aug(x):
    x = x.clone()
    b = x.size(0)
    x = x * torch.empty(b, 1, 1, device=x.device).uniform_(0.95, 1.05)
    x = x + torch.empty(b, 1, 1, device=x.device).uniform_(0.005, 0.02) * torch.randn_like(x)
    for i in range(b):
        if torch.rand(1).item() < 0.5:
            shift = int(torch.randint(-150, 151, (1,), device=x.device).item())
            x[i] = torch.roll(x[i], shift, dims=1)
    return x


@torch.no_grad()
def evaluate(model, loader, device, autocast, tta=2, threshold=0.5):
    model.eval()
    probs_all, y_all = [], []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device, non_blocking=True)
        with autocast():
            probs = [torch.softmax(model(x), dim=1)[:, 1]]
            for _ in range(tta):
                probs.append(torch.softmax(model(tta_aug(x)), dim=1)[:, 1])
        p = torch.stack(probs, dim=0).mean(dim=0)
        probs_all.append(p.cpu().numpy())
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

def tune_threshold(probabilities, targets):
    best = {"threshold": 0.5, "accuracy": -1.0}
    for thr in np.arange(0.20, 0.801, 0.01):
        pred = (probabilities >= thr).astype(np.int64)
        acc = 100.0 * (pred == targets).mean()
        if acc > best["accuracy"]:
            best = {"threshold": float(round(thr, 2)), "accuracy": float(acc)}
    return best

# ============================================================
# 5. تابع آموزش یک Trial
# ============================================================
def train_trial(
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    global_config: Dict,
    output_dir: Path,
) -> Dict:
    """
    Train a single hyperparameter configuration.
    Returns best validation accuracy and model state.
    """
    # استخراج پارامترها
    lr = config["learning_rate"]
    wd = config["weight_decay"]
    trans_layers = config["transformer_layers"]
    hidden_dim = config["transformer_hidden"]
    dropout = config["dropout"]
    batch_size = config.get("batch_size", 16)
    epochs_per_trial = config.get("epochs_per_trial", 6)
    patience = config.get("patience", 3)
    trial_name = config.get("name", "trial")

    # ایجاد مدل
    model = TeacherBinary(
        num_classes=2,
        transformer_layers=trans_layers,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"
🧪 Trial: {trial_name}")
    print(f"   Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # ─── وزن‌های کلاس با تاکید بر Sensitivity ──────────────────────
    train_df = train_loader.dataset.df
    n_pos = int(train_df["class_id"].sum())
    n_neg = len(train_df) - n_pos

    base_neg = len(train_df) / (2 * max(1, n_neg))
    base_pos = len(train_df) / (2 * max(1, n_pos))

    POS_WEIGHT_FACTOR = 4.0
    class_weight = torch.tensor(
        [base_neg, base_pos * POS_WEIGHT_FACTOR],
        dtype=torch.float32,
        device=device,
    )
    print(f"⚖️ Class weights (boosted positive): neg={class_weight[0]:.3f}, pos={class_weight[1]:.3f}")

    # Optimizer, Scheduler, EMA
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = global_config.get("warmup_epochs", 2)

    # Scheduler جدید: CosineAnnealingWarmRestarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=3,
        T_mult=2,
        eta_min=1e-6
    )

    scaler = make_scaler(device)
    autocast = make_autocast(device)
    ema = EMA(model, decay=global_config.get("ema_decay", 0.999))

    best_val_acc = -1.0
    best_ema_state = None
    no_improve = 0
    history = []

    label_smoothing = global_config.get("label_smoothing", 0.05)
    mixup_prob = global_config.get("mixup_prob", 0.5)
    mixup_alpha = global_config.get("mixup_alpha", 0.2)
    grad_clip = global_config.get("grad_clip", 1.0)

    start_time = time.time()

    # ─── پوشه Trial ──────────────────────────────────────────────────
    trial_dir = output_dir / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    # ─── Resume: پیدا کردن آخرین checkpoint ──────────────────────────
    start_epoch = 1
    checkpoint_files = sorted(trial_dir.glob("checkpoint_epoch_*.pt"))
    if checkpoint_files:
        latest_ckpt = checkpoint_files[-1]
        print(f"   🔄 Found existing checkpoint: {latest_ckpt.name}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        ema.state = {k: v.to(device) for k, v in ckpt["ema_state_dict"].items()}
        best_val_acc = ckpt["best_val_acc"]
        best_ema_state = ckpt.get("best_ema_state")
        no_improve = ckpt.get("no_improve", 0)
        history = ckpt.get("history", [])
        start_epoch = ckpt["epoch"] + 1
        print(f"   ▶️ Resuming from epoch {start_epoch}/{epochs_per_trial}")
    else:
        print(f"   ▶️ Starting fresh trial (epoch 1/{epochs_per_trial})")

    # ─── حلقه آموزش ──────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs_per_trial + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs_per_trial}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                if np.random.rand() < mixup_prob:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    idx = torch.randperm(x.size(0), device=device)
                    logits = model(lam * x + (1 - lam) * x[idx])
                    loss = lam * F.cross_entropy(logits, y, weight=class_weight, label_smoothing=label_smoothing)                            + (1 - lam) * F.cross_entropy(logits, y[idx], weight=class_weight, label_smoothing=label_smoothing)
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits, y, weight=class_weight, label_smoothing=label_smoothing)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # ارزیابی با EMA (بدون تخریب مدل)
        raw_state = deepcopy(model.state_dict())
        ema.copy_to(model)
        metrics = evaluate(model, val_loader, device, autocast, tta=2, threshold=0.5)
        model.load_state_dict(raw_state)

        avg_loss = total_loss / len(train_loader)
        row = {
            "epoch": epoch,
            "trial": trial_name,
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

        if metrics["accuracy"] > best_val_acc:
            best_val_acc = metrics["accuracy"]
            best_ema_state = {k: v.detach().cpu().clone() for k, v in ema.state.items()}
            no_improve = 0
            print("   ↳ New best!")
        else:
            no_improve += 1

        # ذخیره checkpoint هر epoch
        trial_dir = output_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_state_dict": {k: v.detach().cpu() for k, v in ema.state.items()},
                "best_ema_state": best_ema_state,
                "best_val_acc": best_val_acc,
                "no_improve": no_improve,
                "history": history,
            },
            trial_dir / f"checkpoint_epoch_{epoch}.pt",
        )
        pd.DataFrame(history).to_csv(trial_dir / "history.csv", index=False)

        if no_improve >= patience:
            print(f"⏹ Early stopping in trial {trial_name} at epoch {epoch}")
            break

    elapsed = time.time() - start_time

    # پس از پایان trial، بهترین مدل را با Threshold تنظیم شده روی validation ذخیره می‌کنیم
    if best_ema_state is None:
        best_ema_state = {k: v.detach().cpu().clone() for k, v in ema.state.items()}

    model.load_state_dict(best_ema_state)
    val_final = evaluate(model, val_loader, device, autocast, tta=4, threshold=0.5)
    tuned = tune_threshold(val_final["probabilities"], val_final["targets"])

    result = {
        "trial_name": trial_name,
        "best_val_accuracy": best_val_acc,
        "best_threshold": tuned["threshold"],
        "val_accuracy_at_threshold": tuned["accuracy"],
        "elapsed_seconds": elapsed,
        "best_ema_state": best_ema_state,
        "history": history,
        "config": config,
        "n_params": n_params,
    }
    return result

# ============================================================
# 6. تابع اصلی
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("hyperparameters.json"), help="Path to JSON config")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "auto_teacher", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed trial")
    args = parser.parse_args()

    # بارگذاری config
    with open(args.config, "r") as f:
        full_config = json.load(f)

    trials = full_config["trials"]
    global_config = full_config["global"]

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(global_config.get("seed", SEED))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("AUTOMATIC HYPERPARAMETER SEARCH FOR TEACHER")
    print("=" * 72)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Trials: {len(trials)}")
    print(f"Output: {output_dir}")
    print("=" * 72)

    # آماده‌سازی DataLoader
    train_ds = ECGDataset(TRAIN_CSV, ROOT, augment=True)
    val_ds = ECGDataset(VAL_CSV, ROOT, augment=False)

    results = []
    best_overall_acc = -1.0
    best_overall_state = None
    best_overall_trial = None

    # بارگذاری نتایج قبلی برای resume
    results_file = output_dir / "results_summary.csv"
    if args.resume and results_file.exists():
        existing = pd.read_csv(results_file)
        completed_trials = set(existing["trial_name"].tolist())
        print(f"Resuming: {len(completed_trials)} trials already completed.")
    else:
        completed_trials = set()

    for trial_idx, trial_config in enumerate(trials):
        trial_name = trial_config.get("name", f"trial_{trial_idx:02d}")
        if trial_name in completed_trials:
            print(f"Skipping already completed trial: {trial_name}")
            continue

        batch_size = trial_config.get("batch_size", 32)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        trial_result = train_trial(
            config=trial_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            global_config=global_config,
            output_dir=output_dir,
        )

        results.append(trial_result)

        if trial_result["best_val_accuracy"] > best_overall_acc:
            best_overall_acc = trial_result["best_val_accuracy"]
            best_overall_state = trial_result["best_ema_state"]
            best_overall_trial = trial_result["trial_name"]
            torch.save(
                {
                    "model_state_dict": best_overall_state,
                    "threshold": trial_result["best_threshold"],
                    "val_accuracy": trial_result["val_accuracy_at_threshold"],
                    "trial": trial_result["trial_name"],
                    "config": trial_result["config"],
                },
                output_dir / "best_teacher_overall.pt",
            )

        summary = pd.DataFrame([{
            "trial_name": r["trial_name"],
            "best_val_accuracy": r["best_val_accuracy"],
            "best_threshold": r["best_threshold"],
            "val_accuracy_at_threshold": r["val_accuracy_at_threshold"],
            "n_params": r["n_params"],
            "elapsed_seconds": r["elapsed_seconds"],
        } for r in results])
        summary.to_csv(output_dir / "results_summary.csv", index=False)

        with open(output_dir / "all_trials_results.json", "w") as f:
            serializable_results = []
            for r in results:
                r_copy = {k: v for k, v in r.items() if k not in ["best_ema_state", "history"]}
                r_copy["best_ema_state"] = None
                r_copy["history"] = r.get("history", [])
                serializable_results.append(r_copy)
            json.dump(serializable_results, f, indent=2)

        if best_overall_acc >= 90.0:
            print(f"
🎯 TARGET 90% REACHED with trial '{best_overall_trial}'! Stopping search.")
            break

    print("
" + "=" * 72)
    print("AUTO SEARCH COMPLETE")
    print("=" * 72)
    print(f"Best validation accuracy: {best_overall_acc:.2f}%")
    print(f"Best trial: {best_overall_trial}")
    print(f"Best model saved at: {output_dir / 'best_teacher_overall.pt'}")
    print(f"Results summary: {output_dir / 'results_summary.csv'}")
    print("=" * 72)

if __name__ == "__main__":
    main()
