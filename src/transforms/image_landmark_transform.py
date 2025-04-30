import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF


class RandomRotateFlip:
    def __init__(self, rotation_range=12, horizontal_flip_prob=0.5, return_tensor=False):
        self.rotation_range = rotation_range
        self.horizontal_flip_prob = horizontal_flip_prob
        self.return_tensor = return_tensor

    def __call__(self, image: Image, landmarks: np.ndarray):
        """
        Randomly rotate and flip the given image and landmarks.

        Args:
            image (Image): The image to be transformed.
            landmarks (np.ndarray): The landmarks associated with the image, shape (n_landmarks, 3).

        Returns:
            tuple: The transformed image and landmarks.
        """
        if isinstance(landmarks, torch.Tensor):
            landmarks = landmarks.numpy()

        # 1. Apply Random Rotation to Image and Landmarks
        rotation_deg = random.uniform(-self.rotation_range, self.rotation_range)
        image, landmarks = self.rotate(image, landmarks, rotation_deg)

        # 2. Apply Random Horizontal Flip
        if random.random() < self.horizontal_flip_prob:
            image, landmarks = self.horizontal_flip(image, landmarks)

        if self.return_tensor:
            landmarks = torch.from_numpy(landmarks)

        return image, landmarks

    def rotate(self, image, landmarks, angle):
        # Rotate image
        """
        Rotate the image and associated landmarks by a given angle.

        Args:
            image (Image): The image to be rotated.
            landmarks (np.ndarray): The landmarks associated with the image, shape (n_landmarks, 3).
            angle (float): The angle in degrees to rotate the image and landmarks.

        Returns:
            tuple: The rotated image and landmarks.
        """

        image = TF.rotate(image, angle)

        # Rotate landmarks
        w, h = image.size
        
        landmarks_pixel = landmarks[:, :2] * np.array([w, h])  # Only x and y

        center_3d = np.array([w / 2, h / 2, 0])  # Center coordinates in 3D (z=0)

        # Shift landmarks to center, rotate, shift back
        centered_landmarks = landmarks_pixel - center_3d
        
        angle_rad = np.radians(angle)  
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
        
        rotated_landmarks = np.dot(centered_landmarks, rotation_matrix)
        rotated_landmarks = np.column_stack((rotated_landmarks, landmarks[:, 2]))  # Keep z-coordinates unchanged
        landmarks = rotated_landmarks + center_3d 
        
        landmarks[:, :2] = landmarks[:, :2] / np.array([w, h])
        
        return image, landmarks

    def horizontal_flip(self, image, landmarks):
        """Flip image and landmarks horizontally.

        Args:
            image (Image): The image to be flipped.
            landmarks (np.ndarray): The landmarks to be flipped, shape (n_landmarks, 3).

        Returns:
            tuple: Flipped image and landmarks.
        """
        image = TF.hflip(image)
        landmarks[:, 0] = 1 - landmarks[:, 0]  # Flip x-coordinates

        return image, landmarks