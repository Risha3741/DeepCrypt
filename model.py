import os
import tensorflow as tf
from tensorflow.keras import layers, Model

# --- CONSTANTS: HIGH FIDELITY, ARBITRARY INPUT ---
# Using 512 channels for maximum descriptive power to ensure numerical fidelity.
LATENT_CHANNELS = 512

def build_encoder(input_shape=(None, None, 3), latent_channels=LATENT_CHANNELS):
    """
    Builds the Fully Convolutional Encoder with lighter filter structure (64, 128, 256 filters).
    This reduces computational cost for faster execution.
    """
    inp = layers.Input(shape=input_shape, name='encoder_input')

    # Feature extraction layers (using strides=1 to maintain 1:1 spatial mapping)
    # Reverting to the less dense structure
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='enc_conv1')(inp)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='enc_conv2')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='enc_conv3')(x)
    
    # Latent representation (output size W x H x 512)
    latent = layers.Conv2D(latent_channels, (3, 3), padding='same', activation='relu', name='enc_latent')(x)
    return Model(inputs=inp, outputs=latent, name='encoder')


def build_decoder(latent_shape=(None, None, LATENT_CHANNELS)):
    """
    Builds the Fully Convolutional Decoder, mirroring the Lighter Encoder structure.
    """
    latent_in = layers.Input(shape=latent_shape, name='decoder_input')

    # Feature reconstruction layers (mirroring the encoder: 256, 128, 64 filters)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='dec_conv1')(latent_in)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='dec_conv2')(x)
    
    # Final layer before output is 64 filters
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='dec_conv3')(x)

    # Final output layer: 3 color channels, normalized [0, 1]
    x = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid', name='dec_out')(x)

    return Model(inputs=latent_in, outputs=x, name='decoder')


def build_autoencoder(input_shape=(None, None, 3), latent_channels=LATENT_CHANNELS):
    """Connects Encoder and Decoder to form the Autoencoder."""
    encoder = build_encoder(input_shape=input_shape, latent_channels=latent_channels)
    
    # The latent shape must use (None, None) for height and width
    latent_shape_dynamic = (None, None, latent_channels)
    decoder = build_decoder(latent_shape=latent_shape_dynamic)

    inp = encoder.input
    out = decoder(encoder.output)
    autoencoder = Model(inputs=inp, outputs=out, name='autoencoder')

    return autoencoder, encoder, decoder


if __name__ == '__main__':
    # Sanity check uses fixed size input (e.g., 512x512) just for summary purposes
    ae, enc, dec = build_autoencoder(input_shape=(512, 512, 3))
    ae.compile(optimizer='adam', loss='mae')
    print("\n=== FULLY CONVOLUTIONAL AUTOENCODER SUMMARY ===")
    ae.summary()
    
    os.makedirs('models', exist_ok=True)
    enc.save(os.path.join('models', 'encoder_test.keras'))
    dec.save(os.path.join('models', 'decoder_test.keras'))
    ae.save(os.path.join('models', 'autoencoder_test.keras'))
    print(f"\nSaved test models for dynamic input with latent channels: {LATENT_CHANNELS}.")

