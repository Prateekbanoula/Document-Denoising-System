"""
Test script to verify the denoising model works correctly.
Run this to test the model before using the Streamlit app.
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Configuration
IMG_SIZE = 128
MODEL_PATH = "denoising_model.h5"
# Auto pick image from train folder
train_folder = "train/"
files = os.listdir(train_folder)

if len(files) == 0:
    print("❌ No images found in train/")
    exit()

TEST_IMAGE_PATH = os.path.join(train_folder, files[0])


def preprocess_image(image_path, img_size=IMG_SIZE):
    """Preprocess image for the model."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = img.reshape(1, img_size, img_size, 1)

    return img


def main():
    print("=" * 50)
    print("Document Denoising Model Test")
    print("=" * 50)

    # Check model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    print("\n[1] Loading model...")
    model = load_model(MODEL_PATH, compile=False)
    print("   ✅ Model loaded")

    # Check test image
    if not os.path.exists(TEST_IMAGE_PATH):
        print("❌ Test image not found")
        return

    print(f"\n[2] Loading image: {TEST_IMAGE_PATH}")
    img = preprocess_image(TEST_IMAGE_PATH)

    print("   Shape:", img.shape)
    print("   Range:", img.min(), img.max())

    # Predict
    print("\n[3] Running prediction...")
    prediction = model.predict(img, verbose=0)
    print("   ✅ Prediction done")

    # 🔥 RAW OUTPUT (NO PROCESSING)
    raw = prediction[0, :, :, 0]

    print("\n[4] RAW OUTPUT CHECK")
    print("   Min:", raw.min())
    print("   Max:", raw.max())
    print("   Mean:", raw.mean())
    print("   Std:", raw.std())

    # Save RAW output
    raw_img = (raw * 255).astype(np.uint8)
    cv2.imwrite("raw_output.png", raw_img)
    print("   ✅ raw_output.png saved")

    # Save simple output (no fancy processing)
    output = np.clip(raw, 0, 1)
    output = (output * 255).astype(np.uint8)

    cv2.imwrite("test_output.png", output)
    print("   ✅ test_output.png saved")

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)


if __name__ == "__main__":
    main()