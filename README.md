Deepcrypt: High-Fidelity Image Encryption
Deepcrypt is an innovative project that combines a high-capacity Deep Learning model (a Fully Convolutional Autoencoder) with strong military-grade cryptography (AES-256) to secure images.

This approach offers two layers of protection: Obfuscation (making the data look like random noise using the neural network) and Confidentiality (making the data mathematically impossible to recover without the correct password and key).

Features:

Dual-Layer Security: Combines Deep Learning encoding with AES-256-CFB encryption.
High Fidelity: Uses a Fully Convolutional Autoencoder (FCAE) trained with MAE (L1) loss to ensure the reconstructed image is nearly identical to the original (high PSNR).
Fixed Architecture: Uses a specialized $384 \times 384$ latent tensor for consistent and efficient encryption/decryption of images.
Strong Key Derivation: Uses PBKDF2 with 100,000 iterations to derive the encryption key from a simple password.

Architecture Overview:

The system works in three main phases:
1. Encoding (encoder.predict): The image (resized to $384 \times 384$) is passed through the Encoder part of the Neural Network.
This turns the human-readable image into a massive, highly complex data tensor  (384 x 384 x 512 floating-point numbers), which looks like noise.

2.Encryption (aes_encrypt): This data tensor is converted into raw bytes and cryptographically encrypted using AES-256 with a password-derived key.
The final encrypted file (encrypted_image.bin) contains the necessary salt, metadata, and ciphertext.

3.Decryption & Reconstruction: The encrypted data is decrypted back into the raw tensor, which is then fed through the Decoder part of the 
Neural Network to reconstruct the original image.

Getting Started:

Prerequisites:
You need Python 3 and a few essential libraries.

1. Install Dependencies
Make sure you have a requirements.txt file (if you don't, create one with the content below).

tensorflow==2.20.0
numpy
opencv-python
scikit-image
cryptography
tensorflowjs

2.Install everything using the -r flag:

pip install -r requirements.txt


2. Prepare Data
Create a folder named training_data/ and fill it with several images (JPG or PNG) of any size.
Place one image you wish to encrypt (e.g., test.jpg) in the main project folder and rename it to sample_image.jpg.

Usage

Step 1: Train the Autoencoder
This step teaches the network how to encode and decode images efficiently.
python train_autoencoder.py
This will take time. After it finishes, you will find encoder_trained.keras and decoder_trained.keras in the models/ folder.

Step 2: Encrypt an Image
This script uses the trained encoder and your password to create the secure binary file.
python encypt_image.py
This creates the encrypted_image.bin file.

Step 3: Decrypt and Reconstruct
This script uses the same password to decrypt the data and uses the decoder to turn the latent tensor back into the original image. It also reports the PSNR (quality score).
python decrypt_reconstruct.py

This creates the final image, reconstructed_image.jpg.

Project Status
This is an educational project demonstrating the fusion of Cryptography and Deep Learning. It is currently under active development and open for contributions.
