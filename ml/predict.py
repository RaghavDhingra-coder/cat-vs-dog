"""
Predict whether a single image is a cat or a dog.

Usage:
    python predict.py path/to/image.jpg

Step 1: the gate (pretrained EfficientNetB0) checks the image contains a cat or dog.
Step 2: our own trained CNN decides CAT vs DOG.
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorflow import keras

from gate import CatDogGate
from preprocess import interpret_prediction, load_rgb_array, prepare_for_cnn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cat_dog_model.keras")


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: model not found at {MODEL_PATH}. Run train.py first.")
        sys.exit(1)

    try:
        rgb = load_rgb_array(image_path)
    except Exception as exc:
        print(f"Error: could not read image ({exc})")
        sys.exit(1)

    is_pet, pet_score, best_guess = CatDogGate().check(rgb)
    if not is_pet:
        print("Prediction: NOT A CAT OR DOG")
        print(f"Looks like: {best_guess}  (cat/dog score {pet_score * 100:.2f}%)")
        return

    model = keras.models.load_model(MODEL_PATH)
    probability = model.predict(prepare_for_cnn(rgb), verbose=0)[0][0]
    label, confidence = interpret_prediction(probability)

    print(f"Prediction: {label.upper()}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()
