import os
from pathlib import Path
import numpy as np
from src.config.paths import TRAIN_IMG_DIR, SYNTEHTIC_TEST_IMG_DIR, REAL_TEST_IMG_DIR, TRAIN_LANDMARKS_DIR, SYNTHETIC_TEST_LANDMARKS_DIR, REAL_TEST_LANDMARKS_DIR
from src.utils.landmarks import get_img_hand_landmarks, get_landmark_coordinates

os.makedirs(TRAIN_LANDMARKS_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_TEST_LANDMARKS_DIR, exist_ok=True)
os.makedirs(REAL_TEST_LANDMARKS_DIR, exist_ok=True)

def save_landmarks(data_dir, landmarks_dir):
    """
    Saves the hand landmarks for all images in a given directory to a target directory.

    The directory structure of the target directory will be the same as the source directory, with each image file
    replaced by a numpy file (.npy) containing the extracted hand landmarks.

    Args:
        data_dir (Path): The directory to extract landmarks from.
        landmarks_dir (Path): The target directory to save the landmarks to.
    """
    for class_name in os.listdir(data_dir):
        class_dir = data_dir / class_name
        target_class_dir = landmarks_dir / class_name
        
        os.makedirs(target_class_dir, exist_ok=True)
        
        for image_file in os.listdir(class_dir):
            image_path = class_dir / image_file
            
            img_landmarks = get_img_hand_landmarks(image_path)
            landmarks = get_landmark_coordinates(img_landmarks)
            
            assert landmarks.shape == (21, 3)
            
            filename = Path(image_file).stem + ".npy"
            np.save(target_class_dir / filename, landmarks)
            print(f"Landmarks successfully extracted for {image_file} in {class_name} class.")
            
            
save_landmarks(TRAIN_IMG_DIR, TRAIN_LANDMARKS_DIR)
save_landmarks(SYNTEHTIC_TEST_IMG_DIR, SYNTHETIC_TEST_LANDMARKS_DIR)
save_landmarks(REAL_TEST_IMG_DIR, REAL_TEST_LANDMARKS_DIR)