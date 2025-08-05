from PIL import Image
import cv2
import time
import torch
from src.utils.inference import predict_sign
from src.config.paths import MODEL_CHECKPOINTS_DIR
from src.models.alphabet_gesture_classification_model import ASLAlphabetClassificationModel

model_name = "final_model_state_dict.pth"
msg = ""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = ASLAlphabetClassificationModel(26, 128, 128)

# Loading model's state dict
model.load_state_dict(torch.load(MODEL_CHECKPOINTS_DIR / model_name, map_location=device))
model.to(device)
model.eval()

# Model takes as an input a tensor of normalized images of shape (batch_dim, 3, 224, 224) and normalized landmarks tensor of shape (batch_dim, 21, 3)

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
        
        predicted_class, _ = predict_sign(model, rgb_img, device)
            
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