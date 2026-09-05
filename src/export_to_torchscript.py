# export_to_torchscript.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from train_student_kd import StudentLite

CKPT_PATH = Path(r"C:\ptbxl\project\checkpoints\student\student_epoch_19.pt")
MODEL_DIR = Path(r"C:\ptbxl\models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
model = StudentLite(num_classes=2, in_ch=2)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

script_model = torch.jit.script(model)
script_model.save(MODEL_DIR / "student_model.pt")
print(f"✅ TorchScript model saved to: {MODEL_DIR / 'student_model.pt'}")
print(f"📦 Size: {(MODEL_DIR / 'student_model.pt').stat().st_size / 1024:.1f} KB")
