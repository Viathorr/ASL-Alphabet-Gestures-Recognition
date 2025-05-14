import string
import torch
from torch.utils.data import DataLoader

from src.utils.model_testing_utils import evaluate_model
from src.transforms.transforms import get_test_transforms, get_grayscale_test_transforms
from src.datasets.asl_alphabet_dataset import ASLAlphabetDataset
from src.models.alphabet_gesture_classification_model import ASLAlphabetClassificationModel
from src.config.paths import MODEL_CHECKPOINTS_DIR, REAL_TEST_IMG_DIR, REAL_TEST_LANDMARKS_DIR

model_name = "grayscale_inception_model_state_dict.pth"

signs = list(string.ascii_uppercase)
signs.remove("Z") 
signs.append("nothing") 

class_to_idx = {signs[i]: i for i in range(len(signs))}


model = ASLAlphabetClassificationModel(len(signs), 128, 128)
model.load_state_dict(torch.load(MODEL_CHECKPOINTS_DIR / model_name, map_location=torch.device("cpu")))
model.eval()


test_transforms = get_grayscale_test_transforms()

test_dataset = ASLAlphabetDataset(
    data_dir=REAL_TEST_IMG_DIR,
    landmarks_dir=REAL_TEST_LANDMARKS_DIR,
    transforms=test_transforms,
    class_to_idx=class_to_idx,
    rotate_flip=False
)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


_ = evaluate_model(model, test_dataloader, signs)