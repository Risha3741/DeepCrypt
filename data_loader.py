import os
import cv2
import numpy as np
from skimage.io import imread

# --- CONSTANTS ---
# Target size reduced to 384x384 for stable GPU training on Colab T4
TARGET_SIZE = 384
# Base directory name
DEFAULT_DATA_DIR = 'training_data'


def load_training_images(target_size=TARGET_SIZE):
    """
    Loads images from the training_data folder, resizes them to the target size,
    and handles potential Colab path issues.
    
    Args:
        target_size (int): The side length (W=H) for training patches.

    Returns:
        numpy.ndarray: Array of processed images (float32, normalized).
    """
    image_list = []
    current_data_dir = DEFAULT_DATA_DIR
    
    # 1. Check if the default directory exists (e.g., if files were unzipped in /content/)
    if not os.path.exists(current_data_dir):
        # 2. If not, check the nested structure that often happens when unzipping in Colab
        colab_data_dir = 'DeepCryption/training_data'
        if os.path.exists(colab_data_dir):
            current_data_dir = colab_data_dir
            print(f"INFO: Switching data directory to nested path: {current_data_dir}")
        else:
            print(f"FATAL ERROR: Directory '{DEFAULT_DATA_DIR}' not found. Check file structure.")
            return None

    print(f"INFO: Loading images from {current_data_dir} and resizing to {target_size}x{target_size}...")

    # Iterate through all files in the directory
    for filename in os.listdir(current_data_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(current_data_dir, filename)
            
            try:
                # Use imread from skimage as a robust alternative
                img_rgb = imread(filepath)
                
                if img_rgb is not None and img_rgb.ndim == 3:
                    # Resize the image to the fixed target size
                    resized_img = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
                    
                    # Convert to float32 and normalize [0, 1]
                    normalized_img = resized_img.astype("float32") / 255.0
                    image_list.append(normalized_img)
                # Note: No 'else' block for unread images to avoid slowing down Colab output
            except Exception as e:
                # This catches files that might be corrupted or malformed images
                pass # Simply skip the file
                
    if not image_list:
        print("FATAL ERROR: No images loaded. Check file paths and formats.")
        return None

    # Stack all images into a single NumPy array (N, H, W, C)
    return np.array(image_list)


# If this file is run directly, it will print basic info
if __name__ == '__main__':
    data = load_training_images(TARGET_SIZE)
    if data is not None:
        print(f"Successfully loaded {len(data)} images.")
        print(f"Shape of output array: {data.shape}")
