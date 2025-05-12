import torch
import string
import numpy as np
from PIL import Image
import cv2
import time
import mediapipe as mp
from src.config.paths import MODEL_CHECKPOINTS_DIR
from src.utils.landmarks import get_landmark_coordinates
from src.utils.transform_utils import transform_image_and_landmarks
from src.transforms.transforms import get_test_transforms
from src.models.alphabet_gesture_classification_model import ASLAlphabetClassificationModel

msg = ""

signs = list(string.ascii_uppercase)
signs.remove("Z") 
signs.append("nothing") 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
test_transforms = get_test_transforms()

model = ASLAlphabetClassificationModel(26, 128, 128)

# # Loading model's state dict
model.load_state_dict(torch.load(MODEL_CHECKPOINTS_DIR / "model_state_dict (2).pth", map_location=device))
model.to(device)
model.eval()
# Model takes as an input a tensor of normalized images of shape (batch_dim, 3, 224, 224) and normalized landmarks tensor of shape (batch_dim, 21, 3)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

cap = cv2.VideoCapture(0)  # using 0 for default camera

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_img = Image.fromarray(rgb_frame)
        landmarks = []
        
        result = hands.process(rgb_frame)
        
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            landmarks = get_landmark_coordinates(hand_landmarks)
        else:
            landmarks = np.zeros((21, 3))
            
        img, landmarks = transform_image_and_landmarks(rgb_img, landmarks, transforms=test_transforms, rotate_flip=False, normalize=True)
        # print(img.shape, landmarks.shape)
        # print(landmarks)
        
        with torch.inference_mode():
            img = img.float().unsqueeze(0).to(device)
            landmarks = landmarks.float().unsqueeze(0).to(device)
            
            pred_logits = model(img, landmarks)
            pred = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1)
            
            predicted_class = signs[pred.item()]
            
        # msg += predicted_class if predicted_class != "nothing" else " "
        
        cv2.putText(frame, predicted_class, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Webcam', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # time.sleep(2)  # wait for 2 seconds before next prediction
        
finally:
    cap.release()
    cv2.destroyAllWindows()