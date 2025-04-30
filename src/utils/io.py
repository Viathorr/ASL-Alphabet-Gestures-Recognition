import os
from PIL import Image


IMG_SIZE = (256, 256)


def get_dir_filenames(data_dir, class_name=None) -> list:
    """
    Retrieve a list of image file paths from a specified directory.

    Args:
        data_dir (str): The directory to search for image files.
        class_name (str, optional): If provided, only files from directories 
            with this name will be included.

    Returns:
        list: A list of file paths for images with '.png', '.jpg', or '.jpeg' extensions.
    """
    filenames = []
    
    for root, _, files in os.walk(data_dir):
        basename = os.path.basename(root)
        
        if class_name is not None and basename == class_name:
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    filenames.append(os.path.join(root, file))
                
    return filenames


def read_image(image_path, size= IMG_SIZE):
    """
    Read an image from a given file path and resize it to a specified size.

    Args:
        image_path (str): The path to the image file.
        size (tuple, optional): The desired size of the output image, in (H, W)
            format. Defaults to (256, 256).

    Returns:
        PIL.Image: The resized image in RGB format.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(size) 
    
    return image  # (H, W, C) format