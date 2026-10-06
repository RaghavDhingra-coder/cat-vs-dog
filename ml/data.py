"""
Dataset loading.

Uses TensorFlow Datasets (TFDS) to automatically download the standard
Microsoft "Cats vs Dogs" dataset (~23,000 images, ~800 MB). It is downloaded
once to ~/tensorflow_datasets and reused afterwards. Corrupted images in the
original dataset are skipped by TFDS automatically.

Split (from the single "train" split TFDS provides):
    80% training | 10% validation | 10% test
"""
import tensorflow as tf
import tensorflow_datasets as tfds

from preprocess import preprocess_image

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

SPLITS = {
    "train": "train[:80%]",
    "validation": "train[80%:90%]",
    "test": "train[90%:]",
}


def load_split(name):
    """Load one raw split as (image, label) pairs. Downloads on first use."""
    return tfds.load("cats_vs_dogs", split=SPLITS[name], as_supervised=True)


def _preprocess(image, label):
    return preprocess_image(image), label


def _augment(image, label):
    # Simple augmentation for training only: random left-right flip.
    return tf.image.random_flip_left_right(image), label


def get_dataset(name, shuffle=False, augment=False):
    ds = load_split(name)
    ds = ds.map(_preprocess, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2000, seed=42)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
