#get_embedding - function to get the vector of sentences
import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# 1. Load Tokenizer and ONNX Session directly
model_dir = r"collageproject/app/model"

tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))

session = ort.InferenceSession(
    os.path.join(model_dir, "model.onnx"),
    providers=["CPUExecutionProvider"]
)


def get_embedding(text):
    # Tokenize input
    encoded = tokenizer.encode(text)

    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

    # Prepare ONNX inputs
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    # Handle token_type_ids if present in model inputs
    input_names = [i.name for i in session.get_inputs()]

    if "token_type_ids" in input_names:
        token_type_ids = np.array([encoded.type_ids], dtype=np.int64)
        ort_inputs["token_type_ids"] = token_type_ids

    # Run ONNX model inference
    outputs = session.run(None, ort_inputs)

    token_embeddings = outputs[0]

    # Perform Mean Pooling over attention mask
    input_mask_expanded = np.expand_dims(attention_mask, -1)

    sum_embeddings = np.sum(
        token_embeddings * input_mask_expanded,
        axis=1
    )

    sum_mask = np.clip(
        input_mask_expanded.sum(axis=1),
        a_min=1e-9,
        a_max=None
    )

    sentence_embedding = sum_embeddings / sum_mask

    return np.squeeze(sentence_embedding)


# 2. Compute embeddings and Cosine Similarity

#final result
def getScore(a: str, b: str):
    a = a.lower()
    b = b.lower()
    a = get_embedding(a)
    b = get_embedding(b)
    
    cosine_sim = np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    return cosine_sim
