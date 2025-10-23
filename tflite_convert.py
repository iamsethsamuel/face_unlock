import tensorflow as tf
from tensorflow import keras

# Load the model safely with the new Keras API
model = keras.models.load_model('./models/face_embeddings.keras', compile=False)

# Rebuild a concrete function for TFLite
input_shape = model.inputs[0].shape
input_signature = [tf.TensorSpec(input_shape, tf.float32)]

@tf.function(input_signature=input_signature)
def model_func(inputs):
    return model(inputs, training=False)

concrete_func = model_func.get_concrete_function()

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], trackable_obj=model_func)
tflite_model = converter.convert()

with open('./models/face_embeddings.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Model successfully converted to face_embeddings.tflite")
