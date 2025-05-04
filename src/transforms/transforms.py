import torchvision.transforms as transforms
from src.config.hyperparameters import data_hyperparameters


def get_train_transforms():
    """
    Returns a transformation pipeline to be used for training images. This pipeline consists of:

    1. Resizing the image to `img_size`.
    2. Cropping the image to `img_crop_size` centered at the middle of the image.
    3. Random color jittering.
    4. Random Gaussian blurring.
    5. Converting the image to a tensor.
    6. Normalizing the tensor with the mean and standard deviation from the training dataset.

    Returns:
        train_transforms (torchvision.transforms.Compose): The transformation pipeline for training images.
    """    
    train_transforms = transforms.Compose([
        transforms.Resize(data_hyperparameters["img_size"]),
        transforms.CenterCrop(data_hyperparameters["img_crop_size"]),
        transforms.ColorJitter(brightness=data_hyperparameters["brightness"], contrast=data_hyperparameters["contrast"], saturation=data_hyperparameters["saturation"], hue=data_hyperparameters["hue"]),
        transforms.GaussianBlur(kernel_size=3, sigma=data_hyperparameters["gauss_blur_sigma"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_hyperparameters["mean"], std=data_hyperparameters["std"])
    ])
    
    return train_transforms


def get_test_transforms():
    """
    Returns a transformation pipeline to be used for test images. This pipeline consists of:

    1. Resizing the image to `img_crop_size`.
    2. Converting the image to a tensor.
    3. Normalizing the tensor with the mean and standard deviation from the training dataset.

    Args:

    Returns:
        test_transforms (torchvision.transforms.Compose): The transformation pipeline for test images.
    """
    test_transforms = transforms.Compose([
        # Using `img_crop_size` instead of `img_size` because we aren't using cropping here, just resizing
        transforms.Resize(data_hyperparameters["img_crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_hyperparameters["mean"], std=data_hyperparameters["std"])
    ])
    
    return test_transforms