import numpy as np
from tensorflow.keras.models import load_model
from utils_crypto import generate_key, aes_decrypt
from model import LATENT_CHANNELS
import cv2
import os
from math import log10, sqrt

# --- Constants ---
PASSWORD = "DeepCryptSecure"
ENCRYPTED_FILENAME = "encrypted_image.bin"
RECONSTRUCTED_FILENAME = "reconstructed_image.jpg"
SAMPLE_IMAGE_FILENAME = "sample_image.jpg"

# --- Metadata Sizes ---
SALT_SIZE = 16
METADATA_SIZE = 8 # 2 integers (H, W) * 4 bytes/int32


def calculate_psnr(original, reconstructed):
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in dB."""
    # Ensure all data is float32 for stable calculation
    mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0, 0.0
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr, mse


# --- Step 1: Load trained autoencoder decoder ---
try:
    decoder = load_model("models/decoder_trained.keras", compile=False)
except FileNotFoundError:
    print("Error: 'models/decoder_trained.keras' not found. Please train the model first.")
    exit(1)


# --- Step 2: Load encrypted file and metadata ---
try:
    with open(ENCRYPTED_FILENAME, "rb") as f:
        file_data = f.read()
except FileNotFoundError:
    print(f"\nError: {ENCRYPTED_FILENAME} not found. Run encrypt_image.py first.")
    exit(1)

# Extract salt, metadata, and encrypted data
salt = file_data[:SALT_SIZE]
metadata_bytes = file_data[SALT_SIZE : SALT_SIZE + METADATA_SIZE]
encrypted_data = file_data[SALT_SIZE + METADATA_SIZE :]


# --- Extract Dynamic Shape ---
metadata = np.frombuffer(metadata_bytes, dtype=np.int32)
latent_height = metadata[0]
latent_width = metadata[1]
print(f"INFO: Decrypted Metadata - H: {latent_height}, W: {latent_width}")


# --- Step 3 & 4: Regenerate AES key and Decrypt Data ---
key, _ = generate_key(PASSWORD, salt=salt)
decrypted_bytes = aes_decrypt(encrypted_data, key)


# --- Step 5: Convert bytes back to latent tensor ---
# CRITICAL: Define latent_shape using the recovered metadata
latent_size_total = latent_height * latent_width * LATENT_CHANNELS
latent_shape = (1, latent_height, latent_width, LATENT_CHANNELS) # <--- FIX IS HERE

expected_byte_size = latent_size_total * 4 # 4 bytes per float32

if len(decrypted_bytes) != expected_byte_size:
    print(f"FATAL ERROR: Data size mismatch. Expected {expected_byte_size} bytes, got {len(decrypted_bytes)}.")
    exit(1)
    
decoded_latent = np.frombuffer(decrypted_bytes, dtype=np.float32)
decoded_latent = decoded_latent.reshape(latent_shape)


# --- Step 6 & 7: Reconstruct image and save ---
reconstructed_img_normalized = decoder.predict(decoded_latent)[0]
reconstructed_img_255 = (reconstructed_img_normalized * 255).astype(np.uint8)

# Save the reconstructed image at its training resolution
cv2.imwrite(RECONSTRUCTED_FILENAME, reconstructed_img_255)


# --- Step 8: Output Verification (PSNR/MSE) ---
original_img = cv2.imread(SAMPLE_IMAGE_FILENAME)
if original_img is None:
    print(f"Warning: '{SAMPLE_IMAGE_FILENAME}' not found for PSNR calculation. Cannot verify quality.")
    exit(1)

# CRITICAL: Resize ORIGINAL image to MATCH the reconstructed image size (384x384 in this test)
original_img_match_size = cv2.resize(original_img, (latent_width, latent_height), interpolation=cv2.INTER_AREA)

# Perform PSNR calculation
psnr, mse = calculate_psnr(original_img_match_size, reconstructed_img_255)


print("\nImage decrypted and reconstructed successfully!")
print("\n=== VERIFICATION RESULTS (High-Fidelity Metrics) ===")
print(f"Resolution of Comparison: {latent_width}x{latent_height} pixels")
print(f"Latent Channels: {LATENT_CHANNELS} | Loss Function: MAE (L1)")
print(f"Mean Squared Error (MSE): {mse:.4f} (Lower is better)")
print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB (Goal: > 50 dB for numerical fidelity)")
print("=====================================================")
print(f"Reconstructed image saved at: {os.path.abspath(RECONSTRUCTED_FILENAME)}")
