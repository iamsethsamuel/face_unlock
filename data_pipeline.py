import numpy as np
import random
import os
from tensorflow.keras.utils import Sequence
from PIL import Image


class TripletSequence(Sequence):
    def __init__(self, img_path, batch_size, embedding_model, shape=(160, 160)):
        super().__init__()
        self.people = list(img_path.keys())
        self.paths = img_path
        self.batch_size = batch_size
        self.shape = shape
        # self.embedding_model = embedding_model
        # self.all_images = []
        #
        # for person_id, path

    def __len__(self):
        # number of triplets per epochs
        return 1000

    def __getitem__(self, idx):
        anchors, positives, negatives = [], [], []
        for _ in range(self.batch_size):
            person = random.choice(self.people)
            imgs = random.sample(self.paths[person], 2)
            anchor_path, positive_path = imgs[0], imgs[1]
            negative_person = random.choice([p for p in self.people if p != person])
            negative_path = random.choice(self.paths[negative_person])

            anchors.append(self._load_image(anchor_path))
            positives.append(self._load_image(positive_path))
            negatives.append(self._load_image(negative_path))

        # Returns a tuple for the first element by changing square brackets [] to parentheses ()
        return (np.array(anchors), np.array(positives), np.array(negatives)), np.zeros((self.batch_size,))

    def _load_image(self, path):
        img = Image.open(path).convert('RGB').resize(self.shape)
        to_np_array = np.asarray(img).astype('float32')

        return to_np_array


