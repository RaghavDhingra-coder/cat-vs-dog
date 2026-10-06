"""
Flask backend for the Cat vs Dog classifier.

Endpoints:
    GET  /health   -> {"status": "ok"}
    POST /predict  -> multipart/form-data with field "image"
                      returns {"prediction": "cat"|"dog", "confidence": 97.42}
                      or, if the image has no cat/dog in it:
                      {"prediction": "not_cat_or_dog", "confidence": 95.9,
                       "detected": "hog"}

Prediction is a two-step process:
    1. Gate: pretrained EfficientNetB0 checks the image contains a cat or dog.
    2. Our own trained CNN decides cat vs dog.

Both models are loaded ONCE at startup (nothing is retrained here).
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import UnidentifiedImageError
from tensorflow import keras

# Reuse the exact same preprocessing code that was used for training.
ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml")
sys.path.insert(0, os.path.abspath(ML_DIR))
from gate import CatDogGate  # noqa: E402
from preprocess import interpret_prediction, load_rgb_array, prepare_for_cnn  # noqa: E402

MODEL_PATH = os.path.abspath(os.path.join(ML_DIR, "models", "cat_dog_model.keras"))
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif", "webp"}

if not os.path.exists(MODEL_PATH):
    sys.exit(f"Model not found at {MODEL_PATH}. Run 'python train.py' in the ml folder first.")

print(f"Loading model from {MODEL_PATH} ...")
model = keras.models.load_model(MODEL_PATH)
print("Loading cat/dog gate (EfficientNetB0) ...")
gate = CatDogGate()
print("Models loaded.")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit
CORS(app)  # allow the React dev server to call this API


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Send a file in the 'image' field."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Invalid file type. Please upload a JPG, PNG, BMP, GIF or WEBP image."}), 400

    try:
        rgb = load_rgb_array(file.stream)
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "The uploaded file is not a valid image."}), 400

    try:
        # Step 1: reject images that do not contain a cat or a dog.
        is_pet, pet_score, best_guess = gate.check(rgb)
        if not is_pet:
            return jsonify(
                {
                    "prediction": "not_cat_or_dog",
                    "confidence": round((1 - pet_score) * 100, 2),
                    "detected": best_guess,
                }
            )

        # Step 2: our CNN decides cat vs dog.
        probability = model.predict(prepare_for_cnn(rgb), verbose=0)[0][0]
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    label, confidence = interpret_prediction(probability)
    return jsonify({"prediction": label, "confidence": round(confidence, 2)})


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "Image is too large. Maximum size is 10 MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
