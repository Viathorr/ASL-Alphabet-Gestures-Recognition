from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src.config.hyperparameters import data_hyperparameters
from src.config.paths import TRAIN_IMG_DIR, REAL_TEST_IMG_DIR
from src.utils.transform_utils import compute_data_mean_std


transform = transforms.Compose([
    transforms.Resize(data_hyperparameters["img_size"]),
    transforms.Grayscale(1),
    transforms.ToTensor()
])

train_dataset = ImageFolder(root=TRAIN_IMG_DIR, transform=transform)
test_dataset = ImageFolder(root=REAL_TEST_IMG_DIR, transform=transform)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False, num_workers=0)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=0)


# Training dataset mean and std computation
TRAIN_MEAN, TRAIN_STD = compute_data_mean_std(train_dataloader)

# Real test dataset mean and std computation
TEST_MEAN, TEST_STD = compute_data_mean_std(test_dataloader)

print(f"Training dataset mean: {TRAIN_MEAN}, std: {TRAIN_STD}")
print(f"Test dataset mean: {TEST_MEAN}, std: {TEST_STD}")