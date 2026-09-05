# export_to_onnx.py
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

dummy_input = torch.randn(1, 2, 5000)

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        MODEL_DIR / "student_model.onnx",
        export_params=True,
        opset_version=14,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits", "features"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "features": {0: "batch_size"}
        },
        verbose=False
    )

print(f"✅ ONNX model saved to: {MODEL_DIR / 'student_model.onnx'}")
print(f"📦 Size: {(MODEL_DIR / 'student_model.onnx').stat().st_size / 1024:.1f} KB")
