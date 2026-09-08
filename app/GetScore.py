import os

# 1. Suppress ONNX C++ level warnings before ONNX Runtime loads


import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# 2. Get absolute path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "model")

# 3. Load Tokenizer directly from tokenizers library (No torch required)
tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
opts = ort.SessionOptions()
opts.log_severity_level = 0
# 4. Load ONNX Session directly
session = ort.InferenceSession(
    os.path.join(model_dir, "model.onnx"),
    providers=["CPUExecutionProvider"]
)


def get_embedding(text: str) -> np.ndarray:
    # Tokenize input string
    encoded = tokenizer.encode(text)

    # Convert directly to standard NumPy arrays (int64)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    # Dynamically pass token_type_ids if required by the model architecture
    input_names = [i.name for i in session.get_inputs()]
    if "token_type_ids" in input_names:
        ort_inputs["token_type_ids"] = np.array([encoded.type_ids], dtype=np.int64)

    # Run inference directly with ONNX Runtime
    outputs = session.run(None, ort_inputs)
    token_embeddings = outputs[0]

    # Perform Mean Pooling via NumPy
    input_mask_expanded = np.expand_dims(attention_mask, -1)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

    sentence_embedding = sum_embeddings / sum_mask
    embedding = np.squeeze(sentence_embedding)

    # Normalize vector length
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def getScore(a: str, b: str) -> float:
    # Calculate cosine similarity using pure NumPy dot product
    vec_a = get_embedding(a.lower())
    vec_b = get_embedding(b.lower())

    return float(np.dot(vec_a, vec_b))