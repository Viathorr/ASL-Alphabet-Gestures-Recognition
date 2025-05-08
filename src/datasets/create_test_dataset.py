import os
import time
import string
import cv2

from src.config.paths import REAL_TEST_IMG_DIR

signs = list(string.ascii_uppercase)
signs.remove("Z")
signs.append("nothing")

cap = cv2.VideoCapture(0)

os.makedirs(REAL_TEST_IMG_DIR, exist_ok=True)

for sign in signs:
    os.makedirs(REAL_TEST_IMG_DIR / sign, exist_ok=True)

    print(f"Collecting images for {sign}...")
    for i in range(50):
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to capture image")
            break

        # Save the image         
        img_path = REAL_TEST_IMG_DIR / sign / f"{sign}_{i}.jpg"
        cv2.imwrite(str(img_path), frame)
        cv2.imshow("Webcam", frame)
        
        time.sleep(1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        
    print(f"Collected 50 images for {sign}.")