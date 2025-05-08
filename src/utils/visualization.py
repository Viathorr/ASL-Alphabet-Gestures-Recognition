import numpy as np
from torchvision.utils import make_grid
import seaborn as sns
import matplotlib.pyplot as plt
import mediapipe as mp
from sklearn.metrics import confusion_matrix
from src.utils.io import read_image
from src.utils.landmarks import get_img_hand_landmarks, mp_hands
from src.utils.transform_utils import denormalize


def show_image(image, title=None):
    """
    Display an image in a matplotlib window.

    Args:
        image (Image or np.ndarray): The image to display.
        title (str, optional): An optional title to display at the top of the window.

    Returns:
        matplotlib.figure.Figure: The figure object containing the displayed image.
    """
    plt.imshow(image)
    
    if title:
        plt.title(title)
        
    plt.axis('off')
    fig = plt.gcf()
    plt.show()
    
    return fig


def show_image_grid(images, ncols=4, title=None):
    """
    Display a grid of images in a matplotlib window.

    Args:
        images: A list or tensor of images to display in a grid.
        ncols (int, optional): The number of columns in the grid. Defaults to 4.
        title (str, optional): An optional title to display at the top of the window.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object containing the image grid.
    """
    grid = make_grid(images, nrow=ncols)
    plt.imshow(grid.permute(1, 2, 0))
    
    if title:
        plt.title(title)
        
    plt.axis('off')
    fig = plt.gcf()
    plt.show()
    
    return fig


def display_img_hand_landmarks(image_path, title=None):
    mp_drawing = mp.solutions.drawing_utils
        
    image = read_image(image_path)
    hand_landmarks = get_img_hand_landmarks(image_path)
    
    if hand_landmarks is not None:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,  
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3), 
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)  
        )
        
    if title:
        plt.title(title)
        
    plt.imshow(image)
    plt.axis('off')
    fig = plt.gcf()
    plt.show()
    
    return fig


def display_transformed_image(image_tensor, landmarks=None, title=None):
    """
    Display a transformed image tensor in a matplotlib window.

    Args:
        image_tensor (torch.Tensor): The image tensor to be displayed, shape (C, H, W).
        landmarks (np.ndarray or None, optional): The landmarks associated with the image, shape (n_landmarks, 3).
            If None, then no landmarks are displayed. Defaults to None.
        title (str, optional): An optional title to display at the top of the window. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The figure object containing the displayed image.
    """
    image_tensor = denormalize(image_tensor)
    image_tensor = image_tensor.numpy() 
    image_tensor = np.transpose(image_tensor, (1, 2, 0))  # Convert to HWC format
    h, w, c = image_tensor.shape
    
    plt.imshow(image_tensor)
    
    if landmarks is not None:  # Landmarks are in range [0,1] (not normalized and not scaled)
        landmarks_rescaled = landmarks * np.array([w, h, 1])
        
        plt.scatter(landmarks_rescaled[:, 0], landmarks_rescaled[:, 1], c='red', s=10, marker='o')
    
    if title:
        plt.title(title)
    
    plt.axis("off")
    fig = plt.gcf()
    plt.show()
    
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot a confusion matrix for a given set of true labels and predicted labels.

    Args:
        y_true (np.ndarray): The true labels, shape (n_samples,).
        y_pred (np.ndarray): The predicted labels, shape (n_samples,).
        class_names (list): The list of class names, shape (n_classes,).

    Returns:
        matplotlib.figure.Figure: The figure object containing the displayed confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 10}, linewidths=1, square=True, cmap="coolwarm", xticklabels=class_names, yticklabels=class_names)

    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)
    plt.title("Confusion Matrix", fontsize=10)

    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    
    return fig


def save_figure(fig, filename):
    """
    Save a matplotlib figure to a file.

    Args:
        fig (matplotlib.figure.Figure): The figure to be saved.
        filename (str): The path where the figure should be saved.
    """
    fig.savefig(filename)
    print(f"Figure saved ✅")