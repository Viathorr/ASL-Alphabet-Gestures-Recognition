import string
import time
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from src.config.paths import TRAIN_IMG_DIR, TRAIN_LANDMARKS_DIR, SYNTEHTIC_TEST_IMG_DIR, SYNTHETIC_TEST_LANDMARKS_DIR, MODEL_CHECKPOINTS_DIR
from src.config.hyperparameters import training_hyperparameters
from src.config.hyperparameters import data_hyperparameters
from src.transforms.transforms import get_train_transforms, get_test_transforms
from src.models.alphabet_gesture_classification_model import ASLAlphabetClassificationModel
from src.datasets.asl_alphabet_dataset import ASLAlphabetDataset


# Hyperparameters
batch_size = data_hyperparameters["batch_size"]
learning_rate = training_hyperparameters["learning_rate"]
weight_decay = training_hyperparameters["weight_decay"]
num_epochs = training_hyperparameters["num_epochs"]
num_workers = data_hyperparameters["num_workers"]
train_shuffle = data_hyperparameters["train_shuffle"]
test_shuffle = data_hyperparameters["test_shuffle"]
lr_mode = training_hyperparameters["lr_scheduler_mode"]
lr_factor = training_hyperparameters["lr_scheduler_factor"]
lr_patience = training_hyperparameters["lr_scheduler_patience"]


signs = list(string.ascii_uppercase)
signs.remove("Z")  # Remove 'Z' as it requires movement.
# Keep 'J' despite its movement, since a static 'J' gesture 
# remains valid and is not easily confused with other gestures.
signs.append("nothing")  # Add a class for 'nothing' (background without a gesture)

# Number of classes
num_classes = len(signs)
print(f"Number of classes: {num_classes}")

class_to_idx = {signs[i]: i for i in range(num_classes)}


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}\n")


train_transforms = get_train_transforms()
val_transforms = get_test_transforms()

train_dataset = ASLAlphabetDataset(data_dir=TRAIN_IMG_DIR, landmarks_dir=TRAIN_LANDMARKS_DIR, class_to_idx=class_to_idx, transforms=train_transforms, rotate_flip=True)
val_dataset = ASLAlphabetDataset(data_dir=SYNTEHTIC_TEST_IMG_DIR, landmarks_dir=SYNTHETIC_TEST_LANDMARKS_DIR, class_to_idx=class_to_idx, transforms=val_transforms, rotate_flip=False)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=test_shuffle, num_workers=num_workers)

model = ASLAlphabetClassificationModel(num_classes=num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=lr_mode, factor=lr_factor, patience=lr_patience)


# Training and validation steps
def train_step(model, train_dataloader, criterion, optimizer, device=device):
    model = model.to(device)
    model.train()

    all_preds = []
    all_labels = []

    train_loss = 0.

    for images, landmarks, labels in train_dataloader:
        images, landmarks, labels = images.float().to(device), landmarks.float().to(device), labels.to(device)

        y_pred_logits = model(images, landmarks)
        y_preds = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)

        loss = criterion(y_pred_logits, labels)
        train_loss += loss.item()

        all_preds.extend(y_preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_dataloader)
    train_acc = accuracy_score(all_labels, all_preds) * 100

    return train_loss, train_acc


def val_step(model: nn.Module, val_dataloader, criterion, device=device):
    val_loss = 0.
    all_preds = []
    all_labels = []

    model.eval()

    with torch.inference_mode():
        for images, landmarks, labels in val_dataloader:
            images, landmarks, labels = images.float().to(device), landmarks.float().to(device), labels.to(device)

            y_pred_logits = model(images, landmarks)
            y_preds = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)
    
            loss = criterion(y_pred_logits, labels)
            val_loss += loss.item()
    
            all_preds.extend(y_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = val_loss / len(val_dataloader)
    val_acc = accuracy_score(all_labels, all_preds) * 100

    return val_loss, val_acc

start_time = time.time()

results = {"train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []}

for epoch in tqdm(range(num_epochs)):
    print("Epoch: ", epoch + 1)
    train_loss, train_acc = train_step(model, train_dataloader, criterion, optimizer)
    val_loss, val_acc = val_step(model, val_dataloader, criterion)

    if lr_scheduler:
        lr_scheduler.step(val_acc)

    print(f"Epoch {epoch + 1}: | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["val_loss"].append(val_loss)
    results["val_acc"].append(val_acc)

    torch.cuda.empty_cache()
    
end_time = time.time()

print(f"Training time: {end_time - start_time:.2f} seconds")

# Save model's state dict
# torch.save(model.state_dict(), MODEL_CHECKPOINTS_DIR / "model_state_dict.pth")