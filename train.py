import os
from data_pipeline import TripletSequence
from triplet_loss import triplet_model
from embedding_model import embedding_model
from utils import dataset_path


def get_imgs(base_folder):
    imgs = {}
    people_folders = [p for p in os.listdir(base_folder) if not p.startswith(".")]
    for person in people_folders:
        person_path = os.path.join(base_folder, person)
        imgs[person] = [os.path.join(person_path, f) for f in os.listdir(person_path)]
    return imgs


train_seq = TripletSequence(get_imgs(dataset_path), p_val=8, k_val=4,
                            embedding_model=embedding_model)
triplet_model.fit(train_seq, epochs=40)

print("Training completed")
embedding_model.save("./models/face_embeddings.keras")
triplet_model.save("triplet_model.keras")
print("Model saved")
