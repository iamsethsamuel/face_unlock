import numpy as np
import kagglehub

def emb_to_blob(emb: np.ndarray) -> bytes:
    arr = np.asarray(emb, dtype=np.float32)
    return arr.tobytes()

def blob_to_emb(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape((dim,))

dataset_path = kagglehub.dataset_download("vishesh1412/celebrity-face-image-dataset")
dataset_path = dataset_path+"/Celebrity Faces Dataset"