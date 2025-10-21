import os
from data_pipeline import TripletSequence
from triplet_loss import triplet_model, triplet_loss
from embedding_model import embedding_model
import cv2
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cosine



def get_imgs(paths):
    imgs = {}
    for path in paths:

        imgs[path] = list(map(lambda p:f"./faces/{path}/{p}",os.listdir(f"./faces/{path}")))
    return imgs



train_seq = TripletSequence(get_imgs([p for p in os.listdir("./faces") if not p.startswith(".") ]), batch_size=16)
triplet_model.fit(train_seq, epochs=40, steps_per_epoch=40)

print("Training completed")
embedding_model.save("face_embeddings.keras")
triplet_model.save("triplet_model.keras")
print("Model saved")