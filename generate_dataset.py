import cv2
import numpy as np
import os

clean_path = "C:/Users/prate/Downloads/train (1)/train/train_cleaned/"
noisy_path = "C:/Users/prate/OneDrive/Desktop/ImageDenoisingApp/train/"

os.makedirs(noisy_path, exist_ok=True)

for file in os.listdir(clean_path):
    img = cv2.imread(os.path.join(clean_path, file), 0)

    if img is None:
        continue

    # --- Blur ---
    noisy = cv2.GaussianBlur(img, (5,5), 0)

    # --- Gaussian Noise ---
    noise = np.random.normal(0, 15, img.shape)
    noisy = noisy + noise

    # --- Random brightness variation ---
    alpha = np.random.uniform(0.8, 1.2)
    noisy = noisy * alpha

    # --- Clip ---
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(noisy_path, file), noisy)

print("✅ Dataset generated successfully")