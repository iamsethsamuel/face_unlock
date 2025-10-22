import os

import numpy as np
import cv2
import tensorflow as tf
from scipy.spatial.distance import cosine
from tensorflow import keras
from triplet_loss import triplet_loss

from img_extractor import generate_images_from_video
from utils import dataset_path



# generate_images_from_video("../Downloads/IMG_3211.MOV", num_frames=100)


keras.config.enable_unsafe_deserialization()

model = keras.models.load_model("./models/face_embeddings.keras",
                                compile=False,
                                # custom_objects={"=keras_tensor_500": l2_normalize},
                                )


def preprocess(img_bg, target_size=(160, 160)):
    img = cv2.cvtColor(cv2.imread(os.path.join(img_bg)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    arr = img.astype("float32")
    # arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def get_embeddings(face_bgr):
    x = preprocess(face_bgr)
    emb = model.predict(x, verbose=0)
    return emb[0]

print("Enrolling face")

def enroll_face(list_of_bgr_images):
    embs = [get_embeddings(img) for img in list_of_bgr_images]
    center = np.mean(embs, axis=0)
    center /= np.linalg.norm(center)
    return center

# change it to a folder where your pictures are located
user_face = [f"./train/user/{face}" for face in os.listdir("./train/user") if not face.startswith(".")]
user_embs = enroll_face(user_face)

def verify(face_bgr, db, threshold=0.7):
    embs = get_embeddings(face_bgr)
    best_user, best_score = None, -1.0

    for user, center in db.items():
        sim = 1 - cosine(embs, center)
        if sim > best_score:
            best_score, best_user = sim, user

    if best_score >= threshold:
        return True, best_user, best_score

    return False, None, best_score

print("Verifying face")
print(
    verify("./train/user/frame_31.jpg", {"user": user_embs}, 0.5),
)

print("Wrong Face")
print(
    verify(f"{dataset_path}/Brad Pitt/001_c04300ef.jpg", {"user": user_embs}, 0.5),
)

