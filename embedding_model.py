# The embedding model takes an image of a face and outputs a compact numerical vector (e.g., 128 numbers).
# Faces of the same person → vectors close together.
# Faces of different people → vectors far apart.
# This vector space is called the embedding space.
# Once you have this, you can do simple math:
# Measure cosine similarity between two embeddings.
# If similarity > threshold → same person (unlock).
# Else → reject.

import tensorflow as tf
from tensorflow import keras
from keras import Model
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from functools import partial

print("Initializing model...")


def embedding_model(shape=(160, 160, 3), embedding_dims=128):
    print("Defining the Embedding model...")
    input_layer = layers.Input(shape)
    aug = layers.RandomRotation(factor=0.05)(input_layer)
    flip = layers.RandomFlip()(aug)
    cont = layers.RandomContrast(factor=.5)(flip)

    # Preprocessing (MobileNetV2 expects inputs in [-1,1] if using preprocess_input)
    # x = layers.Resizing(shape[0], shape[1])(cont)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(cont)

    print("Defining MobileNetV2 model...")
    # Base model
    base_model = MobileNetV2(include_top=False, input_tensor=x, weights="imagenet", pooling='avg', input_shape=shape)

    x = base_model.output
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(embedding_dims, activation=None)(x)
    normalize_fn = partial(tf.nn.l2_normalize, axis=1)

    x = layers.UnitNormalization(axis=1, name='l2_normalization')(x)
    print("Model built successfully.")
    return Model(input_layer, x, name="embedding_model")


embedding_model = embedding_model()

# print(embedding_model.summary())
