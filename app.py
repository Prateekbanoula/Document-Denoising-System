import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# =========================
# CONFIG
# =========================
IMG_SIZE = 128
MODEL_PATH = "denoising_model.h5"

# =========================
# ENHANCEMENT (SCANNER MODE)
# =========================
def enhance_document(img):
    denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)

    thresh = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    final = cv2.medianBlur(thresh, 3)

    return final

# =========================
# PREPROCESS
# =========================
def preprocess_image(image):
    if image.mode != 'L':
        image = image.convert('L')

    img = np.array(image)

    h, w = img.shape
    scale = IMG_SIZE / max(h, w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    img_resized = cv2.resize(img, (new_w, new_h))

    padded = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
    padded[:new_h, :new_w] = img_resized

    img_normalized = padded.astype(np.float32) / 255.0
    img_reshaped = img_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return img_reshaped, padded

# =========================
# POSTPROCESS (ML OUTPUT)
# =========================
def postprocess_image(prediction):
    pred = prediction[0, :, :, 0]

    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    pred = (pred * 255).astype(np.uint8)

    blur = cv2.GaussianBlur(pred, (0,0), 1)
    pred = cv2.addWeighted(pred, 1.4, blur, -0.4, 0)

    return pred

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model_cached():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH, compile=False)
    return None

# =========================
# UI
# =========================
st.set_page_config(page_title="Document Denoising", layout="wide")
st.title("📄 Document Image Denoising System")

model = load_model_cached()

if model is None:
    st.error("❌ Model not found. Run: python train_model.py")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Load image
        input_image = Image.open(uploaded_file)
        img = np.array(input_image.convert("L"))

        # Preprocess
        preprocessed, _ = preprocess_image(input_image)

        # Prediction
        with st.spinner("Processing..."):
            prediction = model.predict(preprocessed)

        # ML Output
        denoised_image = postprocess_image(prediction)

        # 🔥 Resize ML output to match input size
        denoised_resized = cv2.resize(
            denoised_image, 
            (img.shape[1], img.shape[0])
        )

        # Enhanced Output (scanner-like)
        enhanced = enhance_document(img)

        # Display
        st.success("✅ Done!")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Input")
            st.image(img)

        with col2:
            st.subheader("ML Output")
            st.image(denoised_resized)

        with col3:
            st.subheader("Enhanced Output")
            st.image(enhanced)