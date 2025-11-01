import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import os
import math
from model import build_autoencoder, LATENT_CHANNELS
from data_loader import load_training_images, TARGET_SIZE # TARGET_SIZE is now 384

# --- Hyperparameters ---
BATCH_SIZE = 4 # Reduced batch size due to high 384x384 resolution (Crucial for GPU memory)
EPOCHS = 10 # Practice run
# Initial learning rate schedule
INITIAL_LR = 1e-4

# --- Learning Rate Scheduler Function ---
def lr_scheduler(epoch, lr):
    """Decays the learning rate by a factor of 0.5 every 5 epochs."""
    if epoch > 0 and epoch % 5 == 0:
        return lr * 0.5
    return lr

# --- Step 1: Load High-Resolution Data ---
print(f"Loading and processing images from 'training_data/' at {TARGET_SIZE}x{TARGET_SIZE}...")
x_train = load_training_images(target_size=TARGET_SIZE)

if x_train is None:
    print("\nFATAL ERROR: Training data is empty or failed to load. Cannot proceed.")
    exit()

# --- Step 2: Build Fully Convolutional Autoencoder (FCAE) ---
# The input shape is now fixed to 384x384 for batch compatibility
autoencoder, encoder, decoder = build_autoencoder(
    input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
    latent_channels=LATENT_CHANNELS
)

# CRITICAL: Use MAE (L1 Loss) for numerical exactness
optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR)
autoencoder.compile(optimizer=optimizer, loss="mae")


# --- Step 3: Prepare checkpoint directory and callbacks ---
os.makedirs("checkpoints", exist_ok=True)
checkpoint_path = os.path.join("checkpoints", "autoencoder_best.keras")
checkpoint = ModelCheckpoint(
    checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
)
lr_callback = LearningRateScheduler(lr_scheduler)


# --- Step 4: Train autoencoder ---
# Note: Splitting the loaded data for validation (90% train, 10% val)
validation_split_index = int(len(x_train) * 0.9)
x_val = x_train[validation_split_index:]
x_train = x_train[:validation_split_index]


print("\n" + "="*80)
print(f"STARTING TRAINING: Dynamic FCAE Model at {TARGET_SIZE}x{TARGET_SIZE} Resolution")
print(f"Loss Function: MAE (L1) for Numerical Precision | Epochs: {EPOCHS}")
print(f"Training Samples: {len(x_train)} | Validation Samples: {len(x_val)}")
print("="*80 + "\n")


history = autoencoder.fit(
    x_train,
    x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_val, x_val),
    callbacks=[checkpoint, lr_callback],
)


# --- Step 5: Save final models ---
os.makedirs("models", exist_ok=True)
encoder.save(os.path.join("models", "encoder_trained.keras"))
decoder.save(os.path.join('models', 'decoder_trained.keras'))
autoencoder.save(os.path.join("models", "autoencoder_trained.keras"))

print("\n Training complete! Models saved in the 'models' folder.")
