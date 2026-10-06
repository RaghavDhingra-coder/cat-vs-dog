"""
Shared image preprocessing.

This single function is used by training, evaluation, predict.py AND the
Flask backend, so every image goes through exactly the same steps:

    1. Resize to 128 x 128 (bilinear)
    2. Convert to float32
    3. Normalize pixel values from [0, 255] to [0, 1]
"""
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

IMG_SIZE = 128
CLASS_NAMES = ["cat", "dog"]  # label 0 = cat, label 1 = dog


def preprocess_image(image):
    """Resize + normalize one RGB image tensor of shape (H, W, 3)."""
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))  # bilinear, returns float32
    image = image / 255.0
    return image


def load_rgb_array(file_or_path):
    """Open an image file and return it as an upright RGB uint8 array."""
    with Image.open(file_or_path) as img:
        # Phone photos store their rotation in EXIF metadata. Browsers apply it
        # automatically, so we must too, otherwise the model sees a sideways image.
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")  # handles PNG with alpha, grayscale, etc.
        return np.asarray(img, dtype=np.uint8)


def prepare_for_cnn(rgb_array):
    """Preprocess an RGB array into a batch of one: shape (1, 128, 128, 3)."""
    return tf.expand_dims(preprocess_image(rgb_array), axis=0)


def interpret_prediction(probability):
    """Convert the sigmoid output into a label and a confidence percentage.

    The sigmoid output is P(dog). If it is >= 0.5 the image is a dog,
    otherwise a cat with confidence 1 - P(dog).
    """
    probability = float(probability)
    if probability >= 0.5:
        return "dog", probability * 100
    return "cat", (1 - probability) * 100
