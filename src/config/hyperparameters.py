import torch

data_hyperparameters = {
    "img_size": (256, 256),
    "img_crop_size": (224, 224),
    "batch_size": 16,
    "hflip_prob": 0.5,
    "rotation_range": 15,
    "gauss_blur_sigma": (0.5, 1.5),
    "brightness": 0.1,
    "contrast": 0.2,
    "saturation": 0.1,
    "hue": 0.1,
    # RGB mean and std need to be computed from the dataset
    "rgb_mean": [0.485, 0.456, 0.406],
    "rgb_std": [0.229, 0.224, 0.225],
    "rgb_test_mean": [0.485, 0.456, 0.406],  
    "rgb_test_std": [0.229, 0.224, 0.225],
    "grayscale_mean": [0.473],  # 0.4734218597757503
    "grayscale_std": [0.243],  # 0.24328415710831242
    "grayscale_test_mean": [0.545],  # 0.5454909532256121
    "grayscale_test_std": [0.212],  # 0.21164453086096316
    "train_shuffle": True,
    "test_shuffle": False,
    "num_workers": 0
}

training_hyperparameters = {
    "num_classes": 26,
    "num_landmarks": 21,  # 21 hand landmarks
    "num_coordinates": 3,  # x, y, z
    "num_epochs": 10,
    "learning_rate": 3e-3,
    "weight_decay": 1e-4,  # for L2 regularization
    "lr_scheduler_mode": "min",  # reduce LR on loss rise
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 3
}