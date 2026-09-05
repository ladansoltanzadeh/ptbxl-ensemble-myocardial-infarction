# generate_synthetic_data.py
import numpy as np
import pandas as pd
from pathlib import Path
import wfdb
import os

def generate_synthetic_ecg(num_samples=100, num_leads=12, length=5000, sampling_rate=500):
    """تولید سیگنال ECG مصنوعی"""
    t = np.linspace(0, length/sampling_rate, length)
    sig = np.zeros((length, num_leads))
    
    for lead in range(num_leads):
        # ترکیبی از سینوس‌ها با فرکانس‌های مختلف (شبیه‌سازی ECG)
        sig[:, lead] = (
            0.5 * np.sin(2 * np.pi * 1.2 * t + lead * 0.3) +
            0.3 * np.sin(2 * np.pi * 2.4 * t + lead * 0.7) +
            0.2 * np.sin(2 * np.pi * 0.8 * t + lead * 1.1) +
            np.random.normal(0, 0.05, length)  # نویز
        )
        # اضافه کردن QRS کمپلکس (پیک‌های تیز)
        for i in range(4, length-4, 100):
            sig[i-2:i+3, lead] += np.random.uniform(0.5, 1.2) * np.exp(-((t[i-2:i+3] - t[i])**2) / 0.002)
    
    return sig

def generate_synthetic_dataset(output_dir="data/synthetic_ptbxl", num_samples=100):
    """تولید یک دیتاست کامل مصنوعی با ساختار PTB-XL"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # تولید رکوردها
    records = []
    for i in range(num_samples):
        record_name = f"synthetic_{i:04d}"
        sig = generate_synthetic_ecg()
        
        # ذخیره به فرمت WFDB
        wfdb.wrsamp(record_name, fs=500, units=['mV']*12, sig_name=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], 
                    p_signal=sig, fmt=['16']*12, write_dir=str(output_dir))
        
        # برچسب تصادفی (0=Normal, 1=MI/Ischemia)
        class_id = 1 if np.random.rand() > 0.7 else 0
        
        records.append({
            'ecg_id': i,
            'patient_id': i,
            'age': np.random.randint(30, 90),
            'sex': np.random.choice([0, 1]),
            'filename_hr': record_name,
            'class_id': class_id
        })
    
    # ایجاد فایل‌های CSV تقسیم‌بندی
    df = pd.DataFrame(records)
    
    # تقسیم‌بندی 80-10-10
    n = len(df)
    train = df.iloc[:int(0.8*n)]
    val = df.iloc[int(0.8*n):int(0.9*n)]
    test = df.iloc[int(0.9*n):]
    
    split_dir = Path("data/splits")
    split_dir.mkdir(parents=True, exist_ok=True)
    
    train.to_csv(split_dir / "train.csv", index=False)
    val.to_csv(split_dir / "val.csv", index=False)
    test.to_csv(split_dir / "test.csv", index=False)
    
    print(f"✅ Synthetic dataset created with {num_samples} samples")
    print(f"   Train: {len(train)} samples")
    print(f"   Val:   {len(val)} samples")
    print(f"   Test:  {len(test)} samples")
    print(f"📁 Location: {output_dir}")
    
    return df

if __name__ == "__main__":
    generate_synthetic_dataset(num_samples=100)