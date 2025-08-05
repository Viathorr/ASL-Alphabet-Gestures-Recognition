import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
from src.config.paths import MODEL_CHECKPOINTS_DIR
from src.models.alphabet_gesture_classification_model import ASLAlphabetClassificationModel
from src.utils.inference import predict_sign

app = FastAPI()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = ASLAlphabetClassificationModel(26, 128, 128)

# Loading model's state dict
model_name = "final_model_state_dict.pth"
model.load_state_dict(torch.load(MODEL_CHECKPOINTS_DIR / model_name, map_location=device))
model.to(device)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        return {"error": "Invalid file format. Please upload an image."}
    
    content = await file.read()
    image = Image.open(BytesIO(content)).convert("RGB")
    predicted_letter, confidence = predict_sign(model, image, device)
    
    return {"prediction": predicted_letter, "confidence": np.round(confidence, 2)}  # Return prediction and confidence