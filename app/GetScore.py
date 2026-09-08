import os

# 1. Disable ONNX Runtime Telemetry and C++ warning logs
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# 2. Resolve absolute path to the 'model' directory inside 'app/'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "model")

tokenizer_path = os.path.join(model_dir, "tokenizer.json")
onnx_path = os.path.join(model_dir, "model.onnx")

# Safety check for Vercel build log visibility
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Missing tokenizer at: {tokenizer_path}")

tokenizer = Tokenizer.from_file(tokenizer_path)

# 3. Disable telemetry natively via SessionOptions
opts = ort.SessionOptions()
opts.log_severity_level = 3
opts.enable_cpu_mem_arena = False

session = ort.InferenceSession(
    onnx_path,
    sess_options=opts,
    providers=["CPUExecutionProvider"]
)


def get_embedding(text: str) -> np.ndarray:
    encoded = tokenizer.encode(text)

    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    input_names = [i.name for i in session.get_inputs()]
    if "token_type_ids" in input_names:
        ort_inputs["token_type_ids"] = np.array([encoded.type_ids], dtype=np.int64)

    outputs = session.run(None, ort_inputs)
    token_embeddings = outputs[0]

    input_mask_expanded = np.expand_dims(attention_mask, -1)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

    sentence_embedding = sum_embeddings / sum_mask
    embedding = np.squeeze(sentence_embedding)

    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def getScore(a: str, b: str) -> float:
    vec_a = get_embedding(a.lower())
    vec_b = get_embedding(b.lower())
    return float(np.dot(vec_a, vec_b))