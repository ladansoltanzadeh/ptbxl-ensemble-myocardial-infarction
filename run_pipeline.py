#!/usr/bin/env python
# =============================================================================
# run_pipeline.py - Master Script for Full Reproducibility
# This script runs the entire pipeline from data preparation to final results.
# =============================================================================

import subprocess
import sys
import os
from pathlib import Path

# ─── تنظیمات ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"C:\ptbxl")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

def run_command(cmd, description):
    """اجرای یک دستور و ذخیره خروجی در لاگ"""
    print(f"\n{'='*60}")
    print(f"▶️  {description}")
    print(f"📝 Command: {cmd}")
    print(f"{'='*60}")
    
    log_file = LOG_DIR / f"{description.replace(' ', '_')}.log"
    
    with open(log_file, 'w') as f:
        process = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, text=True)
    
    if process.returncode == 0:
        print(f"✅ {description} completed successfully. Log: {log_file.name}")
    else:
        print(f"❌ {description} failed. Check log: {log_file.name}")
        sys.exit(1)

def main():
    print("=" * 60)
    print("🚀 STARTING FULL REPRODUCIBLE PIPELINE")
    print("=" * 60)
    
    # ─── مرحله ۱: نصب وابستگی‌ها ──────────────────────────────
    run_command(
        "pip install -r requirements.txt",
        "Install dependencies"
    )
    
    # ─── مرحله ۲: آماده‌سازی داده ──────────────────────────────
    # (اگر داده از قبل دانلود شده، این مرحله را می‌توان غیرفعال کرد)
    # run_command(
    #     "python src/download_data.py",
    #     "Download PTB-XL dataset"
    # )
    
    # ─── مرحله ۳: آموزش Teacher (Hyperparameter Search) ──────
    run_command(
        "python src/auto_train_teacher.py --config hyperparameters.json --output results/auto_teacher --resume",
        "Teacher Training with Hyperparameter Search"
    )
    
    # ─── مرحله ۴: ارزیابی و Ensemble Teacher ──────────────────
    run_command(
        "python src/ensemble_teacher.py",
        "Teacher Ensemble (V1+V2)"
    )
    
    # ─── مرحله ۵: آموزش Student با Knowledge Distillation ────
    run_command(
        "python src/train_student_kd.py",
        "Student Training with KD"
    )
    
    # ─── مرحله ۶: تنظیم آستانه Student ──────────────────────────
    run_command(
        "python src/tune_student_threshold.py",
        "Student Threshold Tuning"
    )
    
    # ─── مرحله ۷: تولید جداول و نمودارهای مقاله اول ────────────
    run_command(
        "python src/generate_paper1_figures_tables.py",
        "Generate Paper 1 Tables & Figures"
    )
    
    # ─── مرحله ۸: تولید جداول و نمودارهای مقاله دوم ────────────
    run_command(
        "python src/generate_paper2_figures_tables.py",
        "Generate Paper 2 Tables & Figures"
    )
    
    # ─── مرحله ۹: تولید خروجی‌های اثبات فرآیند ──────────────────
    run_command(
        "python src/generate_teacher_proof_figures.py",
        "Generate Teacher Proof Figures"
    )
    
    # ─── مرحله ۱۰: تبدیل مدل به ONNX (اختیاری) ──────────────────
    run_command(
        "python src/export_to_onnx.py",
        "Export to ONNX"
    )
    
    # ─── مرحله ۱۱: تبدیل مدل به TorchScript ──────────────────────
    run_command(
        "python src/export_to_torchscript.py",
        "Export to TorchScript"
    )
    
    # ─── گزارش نهایی ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("📁 All results are available in:")
    print("   - results/paper1/  (Article 1 tables & figures)")
    print("   - results/paper2/  (Article 2 tables & figures)")
    print("   - results/teacher_proof/  (Proof figures)")
    print("   - models/  (Exported models)")
    print("📋 Logs are available in:")
    print(f"   - {LOG_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()