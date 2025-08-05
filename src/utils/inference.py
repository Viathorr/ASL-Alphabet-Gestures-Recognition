import torch
import string
import numpy as np
from PIL import Image
import mediapipe as mp
from src.utils.landmarks import get_landmark_coordinates, get_img_hand_landmarks
from src.utils.transform_utils import transform_image_and_landmarks
from src.transforms.transforms import get_grayscale_test_transforms


test_transforms = get_grayscale_test_transforms()


def predict_sign(model, image, device):
    """
    Predict the sign from the image and landmarks using the model.

    Args:
        model: The trained model for sign classification.
        image: The input image to be classified.
        device: The device to run the model on (CPU or GPU).

    Returns:
        str: The predicted sign.
    """
    signs = list(string.ascii_uppercase)
    signs.remove("Z") 
    signs.append("nothing")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    landmarks = get_landmark_coordinates(get_img_hand_landmarks(image))  # Get a numpy array of coordinates for landmarks from the image
    
    img, landmarks = transform_image_and_landmarks(image, landmarks, transforms=test_transforms, rotate_flip=False, normalize=True)
    
    with torch.inference_mode():
        img = img.float().unsqueeze(0).to(device)
        landmarks = landmarks.float().unsqueeze(0).to(device)
        
        logits = model(img, landmarks)
        pred = logits.argmax(dim=1).item()  # Get the index of the maximum logit value
        
    return signs[pred]
    
    