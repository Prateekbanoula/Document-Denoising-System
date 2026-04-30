# 📄 Document Image Denoising System

## 🔍 Overview
This project focuses on removing noise from document images using a deep learning approach. A CNN-based U-Net autoencoder is used to transform noisy images into clean and readable documents.

## 🚀 Features
- Deep learning-based denoising (U-Net)
- Image enhancement using OpenCV
- Streamlit web app for real-time usage
- Supports noisy scanned documents

## 🧠 Methodology
1. Input noisy document image
2. Preprocessing (resize, normalize)
3. U-Net model prediction
4. Postprocessing
5. Enhancement (thresholding, contrast)

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit

## 📁 Dataset

The dataset consists of paired noisy and clean document images.

Due to size limitations, the dataset is not included in this repository.

Users can generate or use their own dataset following the same structure:
- train/ (noisy images)
- train_cleaned/ (clean images)
## 📊 Output
Input → ML Output → Enhanced Output

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
