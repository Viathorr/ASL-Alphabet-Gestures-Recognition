import torch
from torchvision import transforms
import numpy as np
from src.utils.io import read_image
from src.utils.landmarks import get_img_hand_landmarks, get_landmarks_coordinates, normalize_landmarks
from src.transforms.image_landmark_transform import RandomRotateFlip


RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]


def denormalize(tensor: torch.Tensor, mean=RGB_MEAN, std=RGB_STD):
    """
    Denormalize a tensor by multiplying each channel by a standard deviation value, 
    and then adding a mean value to each channel. This is the inverse of the 
    standard normalization process. This function assumes that the tensor is
    in the range [0,1].

    Args:
        tensor: A tensor of values to be denormalized, shape (C, H, W)
        mean: A list of mean values to add to each channel of the tensor.
        std: A list of standard deviation values to multiply each channel of the tensor by.

    Returns:
        tensor: The denormalized tensor.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m) 
        
    return tensor


def augment_image_and_landmarks(image_path, landmarks, size=(256, 256), crop_size=(224, 224)):
    """
    Augment an image and associated landmarks.

    Args:
        image_path (str): The path to the image file.
        landmarks (np.ndarray): The landmarks associated with the image, shape (n_landmarks, 3).
        size (tuple, optional): The desired size of the output image, in (H, W) format. Defaults to (256, 256).
        crop_size (tuple, optional): The desired size of the output image after center cropping, in (H, W) format. Defaults to (224, 224).

    Returns:
        tuple: A tuple containing the augmented image tensor and the normalized and scaled landmarks, shape (C, H, W) and (n_landmarks, 3), respectively.
    """
    image = read_image(image_path, size=size)
    landmarks = get_img_hand_landmarks(image_path)
    
    if landmarks is None:
        return
    
    landmarks = get_landmarks_coordinates(landmarks)  # (21, 3)
    
    transform = RandomRotateFlip(rotation_range=15, horizontal_flip_prob=0.5)
    image, landmarks = transform(image, landmarks)  # Landmarks are transformed, but still in range [0, 1], so we need to normalize and scale them
    landmarks = normalize_landmarks(landmarks)
    
    image_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(crop_size),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ])
    
    return image_transforms(image), landmarks