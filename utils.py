import numpy as np

def emb_to_blob(emb: np.ndarray) -> bytes:
    arr = np.asarray(emb, dtype=np.float32)
    return arr.tobytes()

def blob_to_emb(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape((dim,))
