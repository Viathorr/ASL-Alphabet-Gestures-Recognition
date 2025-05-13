import torch

data_hyperparameters = {
    "img_size": (256, 256),
    "img_crop_size": (224, 224),
    "batch_size": 16,
    "hflip_prob": 0.5,
    "rotation_range": 15,
    "gauss_blur_sigma": (0.1, 2.0),
    "brightness": 0.2,
    "contrast": 0.3,
    "saturation": 0.2,
    "hue": 0.1,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "train_shuffle": True,
    "test_shuffle": False,
    "num_workers": torch.cuda.device_count() if torch.cuda.is_available() else 0
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