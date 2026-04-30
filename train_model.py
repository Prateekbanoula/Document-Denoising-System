
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2DTranspose



IMG_SIZE = 128  
BATCH_SIZE = 16 
EPOCHS = 40     
LEARNING_RATE = 0.0001


TRAIN_PATH = "train/"
CLEAN_PATH = "C:/Users/prate/Downloads/train (1)/train/train_cleaned/"
MODEL_PATH = "denoising_model.h5"

print("=" * 60)
print("Document Image Denoising - CNN Autoencoder Training")
print("=" * 60)


def load_images(noisy_dir, clean_dir, img_size=IMG_SIZE):
    """Load and preprocess image pairs."""
    images = []
    masks = []
    
    
    noisy_files = sorted(os.listdir(noisy_dir))
    
    for filename in noisy_files:
        
        noisy_path = os.path.join(noisy_dir, filename)
        clean_path = os.path.join(clean_dir, filename)
        
        
        if not os.path.exists(clean_path):
            print(f"Warning: No clean version for {filename}, skipping...")
            continue
        
        
        noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
        
        if noisy_img is None or clean_img is None:
            print(f"Warning: Could not read {filename}, skipping...")
            continue
        
        #
        noisy_img = cv2.resize(noisy_img, (img_size, img_size))
        clean_img = cv2.resize(clean_img, (img_size, img_size))
        
        
        noisy_img = noisy_img.astype(np.float32) / 255.0
        clean_img = clean_img.astype(np.float32) / 255.0
        
        images.append(noisy_img)
        masks.append(clean_img)
    
    
    images = np.array(images).reshape(-1, img_size, img_size, 1)
    masks = np.array(masks).reshape(-1, img_size, img_size, 1)
    
    return images, masks


print("\n[1] Loading dataset...")
X_train, y_train = load_images(TRAIN_PATH, CLEAN_PATH)
print(f"    Dataset shape: {X_train.shape}")
print(f"    Value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print(f"    Clean images range: [{y_train.min():.3f}, {y_train.max():.3f}]")

def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    
    c1 = Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
   
    c2 = Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    
    c3 = Conv2D(128, 3, activation='relu', padding='same')(c3)

    # Decoder
    u1 = Conv2DTranspose(64, 3, strides=2, padding='same')(c3)
    u1 = tf.keras.layers.concatenate([u1, c2])
    c4 = Conv2D(64, 3, activation='relu', padding='same')(u1)
    c4 = Conv2D(64, 3, activation='relu', padding='same')(c4)

    u2 = Conv2DTranspose(32, 3, strides=2, padding='same')(c4)
    u2 = tf.keras.layers.concatenate([u2, c1])
    c5 = Conv2D(32, 3, activation='relu', padding='same')(u2)
    c5 = Conv2D(32, 3, activation='relu', padding='same')(c5)

    outputs = Conv2D(1, 1, activation='sigmoid')(c5)

    return Model(inputs, outputs)

print("\n[2] Building model...")
autoencoder = build_unet()
autoencoder.summary()

autoencoder.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='mse',
    metrics=['mae']
)


callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]


print("\n[3] Training model...")
print(f"    Epochs: {EPOCHS}")
print(f"    Batch size: {BATCH_SIZE}")
print(f"    Learning rate: {LEARNING_RATE}")


history = autoencoder.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15,  # 15% for validation
    callbacks=callbacks,
    verbose=1
)

# Evaluate 
print("\n[4] Evaluating model...")
train_loss = autoencoder.evaluate(X_train, y_train, verbose=0)

loss = train_loss[0] if isinstance(train_loss, list) else train_loss
print(f"Final Training Loss (MSE): {loss:.6f}")

# Test prediction to check for model collapse
test_pred = autoencoder.predict(X_train[:1], verbose=0)
print(f"    Test prediction range: [{test_pred.min():.3f}, {test_pred.max():.3f}]")
print(f"    Test prediction mean: {test_pred.mean():.3f}")

# Check for model collapse (constant output)
if test_pred.std() < 0.01:
    print("\n⚠️  WARNING: Model may be collapsing (very low output variance)")
    print("    Try reducing model complexity or adjusting learning rate")
else:
    print("\n✅ Model output looks healthy (good variance)")

# Save the model
print("\n[5] Saving model...")
autoencoder.save(MODEL_PATH)
print(f"    Model saved to: {MODEL_PATH}")

# Save training history info
print("\n[6] Training Summary:")
print(f"    Total epochs trained: {len(history.history['loss'])}")
print(f"    Best val_loss: {min(history.history['val_loss']):.6f}")

print("\n" + "=" * 60)
print("✅ Training complete!")
print("=" * 60)
print(f"\nTo run the Streamlit app:")
print(f"    streamlit run app.py")

# Test prediction to check for model collapse
test_pred = autoencoder.predict(X_train[:1], verbose=0)

print(f"Test prediction range: [{test_pred.min():.3f}, {test_pred.max():.3f}]")
print(f"Test prediction mean: {test_pred.mean():.3f}")

# 🔥 ADD THIS (important)
if test_pred.std() < 0.01:
    print("⚠️ WARNING: Model collapsing (output nearly constant)")
else:
    print("✅ Model output looks healthy")

print("\n[5] Saving model...")
autoencoder.save("denoising_model.h5")
print("✅ Model saved as denoising_model.h5")