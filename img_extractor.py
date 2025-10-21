from math import e
import cv2
import random
import os

def extract_frames(video_path, num_frames=10):
    print(f"Extracting frames from {video_path} video to ...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        print("Error opening video file")
        raise IOError("Could not open video file.")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rand_frames = random.sample(range(total_frames), num_frames)
    for frame in rand_frames:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def save_frames(frames, output_dir="train/user"):
    dirs = []
    print(f"Saving pictures to directory {output_dir}...")
    if not os.path.exists("train"):
        print("Creating train directory...")
        os.mkdir("train")

    if output_dir.startswith("train"):
        if not os.path.exists(output_dir):
            print("Creating output directory...")
            os.mkdir(output_dir)
    else:
        if not os.path.exists(os.path.join("train", output_dir)):
            print("Creating output directory...")
            os.mkdir(os.path.join("train", output_dir))

    for i, frame in enumerate(frames):
        if output_dir.startswith("train"):
            cv2.imwrite(os.path.join(output_dir, f"frame_{i}.jpg"), frame)
            dirs.append(os.path.join(output_dir, f"frame_{i}.jpg"))
        else:
            cv2.imwrite(os.path.join("train", output_dir, f"frame_{i}.png"), frame)
            dirs.append(os.path.join("train", output_dir, f"frame_{i}.png"))

    return dirs


def generate_images_from_video(video_path, output_dir="train/user", num_frames=10):
    frames = extract_frames(video_path, num_frames)
    return save_frames(frames, output_dir)