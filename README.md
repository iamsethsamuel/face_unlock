# Face Unlock

This project implements a face recognition system using a deep learning model. The system can be used to enroll users by their face and then verify their identity.

## How it Works

The core of the system is a convolutional neural network (CNN) that generates embeddings for face images. These embeddings are then used to compare faces and determine if they belong to the same person.

1.  **Data Preparation:** The `utils.py` script downloads a celebrity face image dataset from Kaggle. The `img_extractor.py` script can be used to extract frames from a video file to create a dataset of a user's face.
2.  **Model Training:** The `train.py` script trains the embedding model using the triplet loss function. The script uses the `TripletSequence` class to generate batches of triplets from the training data.
3.  **Face Enrollment:** The `inference.py` script can be used to enroll a new user. The script takes a list of images of the user's face, generates embeddings for each image, and then calculates the average embedding for the user. This average embedding is then stored in the database.
4.  **Face Verification:** The `inference.py` script can also be used to verify a user's identity. The script takes an image of the user's face, generates an embedding for the image, and then compares the embedding to the user's stored embedding in the database. If the distance between the two embeddings is below a certain threshold, the user is verified.

## Project Structure

*   **`embedding_model.py`**: This file defines the deep learning model used to generate face embeddings. It uses a pre-trained MobileNetV2 model as a base and adds a few custom layers to produce a 128-dimensional embedding vector.
*   **`triplet_loss.py`**: This file implements the triplet loss function, which is used to train the embedding model. The triplet loss function encourages the model to generate embeddings that are close for faces of the same person and far apart for faces of different people.
*   **`train.py`**: This file contains the code for training the embedding model. It uses a `TripletSequence` to generate batches of triplets (anchor, positive, negative) and then trains the model using the triplet loss function.
*   **`data_pipeline.py`**: This file defines the `TripletSequence` class, which is a Keras `Sequence` that generates batches of triplets for training.
*   **`inference.py`**: This file contains the code for using the trained model to enroll and verify faces.
*   **`db.py`**: This file handles the database operations, including storing and retrieving user information and face embeddings.
*   **`img_extractor.py`**: This file contains functions for extracting frames from a video file.
*   **`tflite_convert.py`**: This file contains the code for converting the trained model to a TensorFlow Lite model, which is optimized for mobile and embedded devices.
*   **`main.py`**: This file is empty, but it could be used to create a command-line interface or a web application for the face recognition system.

## How to Use

1.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the model:**

    ```bash
    python train.py
    ```

3.  **Enroll a user:**

    *   First, you need to have a folder with images of the user's face. You can use the `img_extractor.py` script to extract frames from a video file.
    *   Then, you can use the `inference.py` script to enroll the user.

4.  **Verify a user:**

    *   You can use the `inference.py` script to verify a user's identity. The script takes an image of the user's face and compares it to the user's stored embedding in the database.

