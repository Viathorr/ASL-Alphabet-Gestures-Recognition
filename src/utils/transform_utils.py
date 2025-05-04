import torch
from src.utils.landmarks import normalize_landmarks
from src.config.hyperparameters import data_hyperparameters
from src.transforms.image_landmark_transform import RandomRotateFlip


def denormalize(tensor: torch.Tensor, mean=data_hyperparameters["mean"], std=data_hyperparameters["std"]):
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


def transform_image_and_landmarks(image, landmarks, transforms, rotate_flip=True, rotation_range=data_hyperparameters["rotation_range"], hflip_prob=data_hyperparameters["hflip_prob"]):
    """
    Apply a given transformation to an image and its associated landmarks.

    Args:
        image: The image to be transformed, a PIL Image.
        landmarks: The associated landmarks, shape (n_landmarks, 3) in range [0, 1].
        transforms: A torchvision transform to apply to the image.
        rotate_flip: If True, apply a random rotation and horizontal flip to the image and landmarks.
        rotation_range: The range of angles to randomly rotate the image and landmarks.
        hflip_prob: The probability of horizontally flipping the image and landmarks.

    Returns:
        tuple: The transformed image and landmarks. The image is a tensor in shape (3, height, width),
        and the landmarks are a tensor in shape (n_landmarks, 3) in range [-1, 1].
    """
    if rotate_flip:
        rotate_flip_transform = RandomRotateFlip(rotation_range=rotation_range, horizontal_flip_prob=hflip_prob, return_tensor=False)
        image, landmarks = rotate_flip_transform(image, landmarks)  # landmarks are transformed, but still in range [0, 1], so we need to normalize and scale them
        
    landmarks = normalize_landmarks(landmarks)
    
    return transforms(image), torch.from_numpy(landmarks)  # `image tensor` in shape (3, height, width); `landmarks tensor` in shape (n_landmarks, 3)  (n_landmarks = 21 in our case)