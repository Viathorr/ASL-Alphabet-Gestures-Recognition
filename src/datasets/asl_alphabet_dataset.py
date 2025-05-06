import os
import numpy as np
from torch.utils.data import Dataset
from src.config.hyperparameters import data_hyperparameters
from src.utils.io import read_image
from src.utils.transform_utils import transform_image_and_landmarks


class ASLAlphabetDataset(Dataset):
    def __init__(self, data_dir: str, landmarks_dir: str, class_to_idx: dict, transforms, rotate_flip=False):
        self.transforms = transforms
        self.rotate_flip = rotate_flip
        self.class_to_idx = class_to_idx
        self.samples = []

        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            class_landmarks_dir = os.path.join(landmarks_dir, class_name)
        
            for img_name in os.listdir(class_dir):
                filename = os.path.splitext(img_name)[0]
                img_path = os.path.join(class_dir, img_name)
                landmarks_path = os.path.join(class_landmarks_dir, filename + ".npy")
            
                self.samples.append((img_path, landmarks_path, class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, landmarks_path, class_name = self.samples[idx]

        img = read_image(img_path)  # PIL Image object of shape (H, W, C)
        landmarks = np.load(landmarks_path)  # (21, 3) numpy array

        if self.transforms:
            img, landmarks = transform_image_and_landmarks(img, landmarks, transforms=self.transforms, rotate_flip=self.rotate_flip if class_name != "nothing" else False, rotation_range=data_hyperparameters["rotation_range"], hflip_prob=data_hyperparameters["hflip_prob"])
            
            return img, landmarks, self.class_to_idx.get(class_name)
        else:
            raise ValueError("Please provide a valid 'transforms' argument when initializing the dataset.")