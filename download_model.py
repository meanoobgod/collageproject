import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import brotli

# 1. Export PyTorch model to ONNX
model_id = "sentence-transformers/paraphrase-MiniLM-L3-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pt_model = AutoModel.from_pretrained(model_id)

output_dir = Path("./l3_model_dir")
output_dir.mkdir(exist_ok=True)

tokenizer.save_pretrained(output_dir)
pt_model.config.save_pretrained(output_dir)

dummy_text = "This is a test sentence."
inputs = tokenizer(dummy_text, return_tensors="pt")

onnx_fp32_path = output_dir / "model_fp32.onnx"
torch.onnx.export(
    pt_model,
    (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
    str(onnx_fp32_path),
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "token_type_ids": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
    },
    opset_version=14,
)

# 2. Dynamic INT8/INT4 Weight Quantization via standard API
onnx_int_path = output_dir / "model.onnx"

quantize_dynamic(
    model_input=str(onnx_fp32_path),
    model_output=str(onnx_int_path),
    weight_type=QuantType.QUInt8,
    extra_options={"WeightSymmetric": True}
)

if onnx_fp32_path.exists():
    os.remove(onnx_fp32_path)

file_size_mb = os.path.getsize(onnx_int_path) / (1024 * 1024)
print(f"L3 Model successfully generated! File size: {file_size_mb:.2f} MB")
