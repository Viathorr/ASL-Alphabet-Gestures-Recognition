import numpy as np
import mediapipe as mp

from src.utils.io import read_image


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)


def get_img_hand_landmarks(image_path):
    """
    Given an image path, this function reads the image, and uses MediaPipe
    to detect hand landmarks in the image. If the image contains a hand, the
    function returns the detected hand landmarks. Otherwise, it returns None.

    Args:
        image_path (str): The path to the image to be processed.

    Returns:
        hand_landmarks (mediapipe.solutions.hands.HandLandmarkList or None):
            The detected hand landmarks in the image. If the image does not
            contain a hand, then None is returned.
    """
    image = read_image(image_path)
    image = np.array(image)  # Convert to numpy array for MediaPipe processing
    
    results = hands.process(image)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        return hand_landmarks
    else:
        return None


def get_img_hand_landmarks(image):
    """
    Detects hand landmarks in a given image using MediaPipe.

    Args:
        image: An image in a format compatible with MediaPipe, typically a PIL Image or numpy array.

    Returns:
        mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList or None:
            If a hand is detected in the image, returns the landmarks for the first detected hand.
            If no hand is detected, returns None.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)  # Ensure the image is a numpy array
    
    results = hands.process(image)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        return hand_landmarks
    else:
        return None
    
    
def get_landmark_coordinates(landmarks):
    """
    Extracts the x, y, and z coordinates from a MediaPipe hand landmarks object.

    Args:
        landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList or None):
            The hand landmarks detected by MediaPipe. If no hand is detected, 
            this should be None.

    Returns:
        np.ndarray: A numpy array of shape (21, 3) containing the 
            x, y, and z coordinates of each of the 21 hand landmarks. If 
            landmarks is None, returns a numpy array of zeros.
    """
    if landmarks is None:
        return np.zeros((21, 3))
    
    coordinates = []
    for landmark in landmarks.landmark:
        coordinates.append([landmark.x, landmark.y, landmark.z])
    
    return np.array(coordinates)  # (21, 3) shape


def normalize_landmarks(landmarks):
    """
    Normalize hand landmarks:
    - Center relative to the wrist (landmark 0).
    - Scale to have coordinates approximately in [-1, 1].
    
    Args:
        landmarks (np.ndarray): (N_landmarks, 3) array of (x, y, z) coordinates.
        
    Returns:
        np.ndarray: Normalized landmarks, same shape.
    """
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
        assert landmarks.shape[1] == 3
    
    wrist = landmarks[0]
    centered = landmarks - wrist
    
    max_value = np.max(np.abs(centered))
    
    if max_value > 0:
        normalized = centered / max_value
    else:
        normalized = centered  # If hand is a point (degenerate case), skip scaling

    return normalized


def get_bbox_from_hand_landmarks(landmarks, img_width, img_height):
    """
    Given a set of hand landmarks and an image size, compute the bounding box
    containing the hand. The bounding box is computed by taking the minimum and
    maximum x and y coordinates of the landmarks, and then adding a 15% padding
    relative to the bounding box size. The bounding box is returned as two tuples of
    floats and integers: the first tuple contains the normalized coordinates
    (x_min, y_min, x_max, y_max) of the bounding box, and the second tuple
    contains the pixel coordinates (x_min_px, y_min_px, x_max_px, y_max_px) of
    the bounding box.

    Args:
        landmarks (list of tuples): The hand landmarks, where each landmark is
            a tuple of three floats containing the x, y, and z coordinates of
            the landmark.
        img_width (int): The width of the image in pixels.
        img_height (int): The height of the image in pixels.

    Returns:
        tuple: A tuple of two tuples of floats and integers, containing the
            normalized and pixel coordinates of the bounding box, respectively.
    """
    xs = [landmark[0] for landmark in landmarks]
    ys = [landmark[1] for landmark in landmarks]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    padding = 0.15  # 15% of image size relative range
    x_range = x_max - x_min
    y_range = y_max - y_min

    x_min = x_min - padding * x_range
    x_max = x_max + padding * x_range
    y_min = y_min - padding * y_range
    y_max = y_max + padding * y_range

    x_min_px = int(x_min * img_width)
    x_max_px = int(x_max * img_width)
    y_min_px = int(y_min * img_height)
    y_max_px = int(y_max * img_height)

    return (x_min, y_min, x_max, y_max), (x_min_px, y_min_px, x_max_px, y_max_px)