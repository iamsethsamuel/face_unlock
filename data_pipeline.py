import numpy as np
import random
import os
from tensorflow.keras.utils import Sequence
from PIL import Image


class TripletSequence(Sequence):
    def __init__(self, img_path,p_val, k_val, embedding_model, shape=(160, 160)):
        super().__init__()
        self.people = list(img_path.keys())
        self.paths = img_path
        self.batch_size = p_val * k_val
        self.shape = shape
        self.embedding_model = embedding_model
        self.all_images = []
        self.p_val = p_val
        self.k_val = k_val

        for person_id, paths in self.paths.items():
            for path in paths:
                self.all_images.append((person_id, path))

    def __len__(self):
        # number of triplets per epochs
        return 100

    def __getitem__(self, idx):

        selected_people = np.random.choice(self.people,size=self.p_val,replace=False)

        batch_images = []
        batch_labels = []

        for person in selected_people:
            image_path = np.random.choice(self.paths[person], size=self.k_val, replace=True)
            for path in image_path:
                batch_images.append(self._load_image(path))
                batch_labels.append(person)

        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)


        embeddings = self.embedding_model.predict(batch_images, verbose=0)

        anchors, positives, negatives = [], [], []

        for i in range(self.batch_size):
            anchor_emb = embeddings[i]
            anchor_label = batch_labels[i]

            # Calculate distances to all other embeddings
            distance = np.sum(np.square(anchor_emb-embeddings), axis=1)

            # Find hard positive
            pos_mask = (batch_labels == anchor_label) & (np.arange(self.batch_size) != i)
            if not np.any(pos_mask): continue # Skip if no positives in batch

            hardest_positive_idx = np.argmax(distance[pos_mask])
            positive_index = np.where(pos_mask)[0][hardest_positive_idx]

            neg_mask = (batch_labels != anchor_label)
            if not np.any(neg_mask): continue # Skip if no negative in batch

            hardest_negative_idx = np.argmin(distance[neg_mask])
            negative_index = np.where(neg_mask)[0][hardest_negative_idx]

            anchors.append(batch_images[i])
            positives.append(batch_images[positive_index])
            negatives.append(batch_images[negative_index])

        if not anchors:
            return (np.array([]), np.array([]), np.array([])), np.array([])

        return (np.array(anchors), np.array(positives), np.array(negatives)), np.zeros((len(anchors)))




    def _load_image(self, path):
        img = Image.open(path).convert('RGB').resize(self.shape)
        to_np_array = np.asarray(img).astype('float32')

        return to_np_array


