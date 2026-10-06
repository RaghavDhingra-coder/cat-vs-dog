"""
"Is this a cat or a dog at all?" check (gatekeeper).

Our CNN only knows two classes, so it MUST answer "cat" or "dog" for every
image - even a pig or a person. To reject such images we first run a
pretrained EfficientNetB0 (trained on ImageNet, 1000 everyday object classes;
weights are downloaded automatically by Keras the first time, ~21 MB).

ImageNet contains 118 dog-breed classes (indices 151-268) and 5 domestic cat
classes (indices 281-285). We add up the probability EfficientNetB0 gives to
all of those classes. If that total is below GATE_THRESHOLD, the image is
rejected as "not a cat or dog". Otherwise our own CNN decides cat vs dog.

EfficientNetB0 is ONLY used for this yes/no check. The cat vs dog answer
always comes from our own trained CNN.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import efficientnet

GATE_IMG_SIZE = 224
DOG_CLASSES = range(151, 269)  # ImageNet dog breeds
CAT_CLASSES = range(281, 286)  # tabby, tiger cat, Persian, Siamese, Egyptian cat
PET_CLASSES = np.array(list(DOG_CLASSES) + list(CAT_CLASSES))

# Minimum total cat+dog probability for an image to be accepted.
GATE_THRESHOLD = 0.20


class CatDogGate:
    def __init__(self):
        self.model = efficientnet.EfficientNetB0(weights="imagenet")

    def pet_scores(self, rgb_dataset):
        """Cat+dog scores for a tf.data dataset of single RGB images (used in evaluation)."""
        ds = rgb_dataset.map(
            lambda img: efficientnet.preprocess_input(
                tf.image.resize(img, (GATE_IMG_SIZE, GATE_IMG_SIZE))
            )
        ).batch(32)
        probs = self.model.predict(ds, verbose=0)
        return probs[:, PET_CLASSES].sum(axis=1)

    def check(self, rgb_array):
        """Return (is_cat_or_dog, pet_score, best_guess_label)."""
        image = tf.image.resize(rgb_array, (GATE_IMG_SIZE, GATE_IMG_SIZE))
        image = efficientnet.preprocess_input(image)
        probs = self.model.predict(tf.expand_dims(image, 0), verbose=0)[0]

        pet_score = float(probs[PET_CLASSES].sum())
        top = efficientnet.decode_predictions(probs[np.newaxis], top=1)[0][0]
        best_guess = top[1].replace("_", " ")
        return pet_score >= GATE_THRESHOLD, pet_score, best_guess
