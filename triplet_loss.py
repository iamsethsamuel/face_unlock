import tensorflow as tf
from tensorflow.keras import layers, Model
from embedding_model import embedding_model

margin = 0.3


def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))

    return loss


def build_triplet_model(embedding_model):
    print("Initializing Triplet Loss Model...")
    a = layers.Input(shape=embedding_model.input_shape[1:])
    p = layers.Input(shape=embedding_model.input_shape[1:])
    n = layers.Input(shape=embedding_model.input_shape[1:])

    ea = embedding_model(a)
    ep = embedding_model(p)
    en = embedding_model(n)

    embedding_dim = embedding_model.output_shape[-1]


    # stack embeddings so y_pred is (batch, 3, emb_dim)
    # out = layers.Stacked()(tf.stack([ea, ep, en], axis=1)) if False else tf.stack([ea, ep, en], axis=1)
    out = layers.Lambda(
        lambda tensors: tf.stack(tensors, axis=1),
        output_shape=(3, embedding_dim),
        name="stack_embeddings"
    )([ea, ep, en])
    # out = layers.Lambda(lambda x: x)(tf.stack([ea, ep, en], axis=1))
    print("Triplet Loss Model built successfully.")
    return Model(inputs=[a, p, n], outputs=out, name="triplet_model")


triplet_model = build_triplet_model(embedding_model)
triplet_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=triplet_loss)
