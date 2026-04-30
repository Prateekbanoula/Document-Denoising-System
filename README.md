# Document Image Denoising Project

A CNN-based autoencoder for denoising document images.

## Project Structure

```
ImageDenoisingApp/
├── train/              # Noisy document images (144 images)
├── train_cleaned/      # Clean document images (144 images)
├── train_model.py      # Training script
├── app.py              # Streamlit web app
├── denoising_model.h5  # Trained model
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Train the Model (if needed)

```bash
python train_model.py
```

This will:
- Load 144 image pairs from `train/` and `train_cleaned/`
- Train a CNN autoencoder for 15 epochs
- Save the model to `denoising_model.h5`

### Option 2: Run the Streamlit App

```bash
streamlit run app.py
```

Then open http://localhost:8502 in your browser.

### Using the App

1. Upload a noisy or blurred document image (PNG, JPG, or JPEG)
2. The model will automatically denoise the image
3. View the input and output side-by-side
4. Optionally toggle post-processing (sharpen/contrast)
5. Download the denoised image

## Model Details

| Property | Value |
|----------|-------|
| Architecture | CNN Autoencoder (Encoder-Decoder) |
| Input Size | 128×128 grayscale |
| Parameters | ~723K |
| Loss Function | Mean Squared Error (MSE) |
| Training Data | 144 image pairs |

## Requirements

- Python 3.8+
- TensorFlow
- Streamlit
- OpenCV
- NumPy
- Pillow

## Troubleshooting

### Model not found error
If you see "Model not found", run the training script first:
```bash
python train_model.py
```

### Keras compatibility issues
The app handles Keras version compatibility automatically. If issues persist, try:
```bash
pip install keras==3.0.0 tensorflow==2.15.0
```

## License

MIT License