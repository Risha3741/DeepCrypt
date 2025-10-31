import numpy as np
from tensorflow.keras.models import load_model
from utils_crypto import generate_key, aes_encrypt
from model import LATENT_CHANNELS
from data_loader import TARGET_SIZE # Import target size for consistency
import cv2
import os

# --- Constants ---
PASSWORD = "DeepCryptSecure"
ENCRYPTED_FILENAME = "encrypted_image.bin"
SAMPLE_IMAGE_FILENAME = "sample_image.jpg"


# --- Load the trained Encoder Model (FCAE) ---
try:
    # Model architecture is dynamic (None, None, 3)
    encoder = load_model("models/encoder_trained.keras", compile=False)
except FileNotFoundError:
    print("Error: 'models/encoder_trained.keras' not found. Please train the model first.")
    exit(1)


# --- Load and Preprocess Sample Image ---
original_img = cv2.imread(SAMPLE_IMAGE_FILENAME)
if original_img is None:
    raise FileNotFoundError(f"{SAMPLE_IMAGE_FILENAME} not found in project folder.")

# CRITICAL FIX: Resize the image to the exact training size (384x384)
# This satisfies the fixed dimension constraint imposed by Keras during saving.
img_resized = cv2.resize(original_img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

# Normalize and add batch dimension (1, 384, 384, 3)
img = img_resized.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)


# --- CNN Encoding (Deep Learning Obfuscation) ---
# Output shape will be (1, 384, 384, 512)
encoded = encoder.predict(img)

# Dynamic latent shape metadata (now fixed for testing)
latent_height, latent_width, _ = encoded.shape[1:]
# Note: latent_size calculation here is unnecessary but harmless: 
# latent_size = latent_height * latent_width * LATENT_CHANNELS


# Flatten encoded output and convert to raw bytes (required for AES)
encoded_bytes = encoded.tobytes()

# --- AES Encryption (Cryptographic Confidentiality) ---
key, salt = generate_key(PASSWORD)
encrypted_data = aes_encrypt(encoded_bytes, key)

# Save the dynamic latent shape metadata along with Salt/IV/Ciphertext
metadata = np.array([latent_height, latent_width], dtype=np.int32).tobytes()

# Final output format: [Salt] + [H, W Metadata] + [IV + Ciphertext]
with open(ENCRYPTED_FILENAME, "wb") as f:
    f.write(salt + metadata + encrypted_data)


output_path = os.path.abspath(ENCRYPTED_FILENAME)
print("\n=== HIGH-FIDELITY ENCRYPTION COMPLETE ===")
print(f"INFO: Encrypting image at fixed resolution: {latent_height}x{latent_width}")
print(f"Encrypted file saved at: {output_path}")