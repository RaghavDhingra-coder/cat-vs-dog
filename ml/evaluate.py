"""
Evaluate the trained model on the held-out TEST split (10% of the data,
never seen during training).

Usage:
    python evaluate.py

Outputs:
    results/confusion_matrix.png
    results/classification_report.txt
    results/metrics.json
    sample_images/           - a few test images to try with predict.py / the UI
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tensorflow import keras

from data import get_dataset, load_split
from gate import GATE_THRESHOLD, CatDogGate
from preprocess import CLASS_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cat_dog_model.keras")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_images")


def save_sample_images(per_class=3):
    """Save a few raw test-split images as JPEGs for demos."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    counts = {0: 0, 1: 0}
    for image, label in load_split("test"):
        label = int(label)
        if counts[label] >= per_class:
            continue
        counts[label] += 1
        path = os.path.join(SAMPLES_DIR, f"{CLASS_NAMES[label]}_{counts[label]}.jpg")
        Image.fromarray(image.numpy()).save(path, quality=95)
        if all(c >= per_class for c in counts.values()):
            break


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run train.py first.")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = keras.models.load_model(MODEL_PATH)
    test_ds = get_dataset("test")

    loss, acc = model.evaluate(test_ds, verbose=1)

    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    probs = model.predict(test_ds, verbose=1).ravel()
    y_pred = (probs >= 0.5).astype(int)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print("\n===== Test set evaluation =====")
    print(f"Test images:    {len(y_true)}")
    print(f"Test loss:      {loss:.4f}")
    print(f"Test accuracy:  {acc * 100:.2f}%")
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(cm)
    print("\nClassification report:")
    print(report)

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Test images:   {len(y_true)}\n")
        f.write(f"Test loss:     {loss:.4f}\n")
        f.write(f"Test accuracy: {acc * 100:.2f}%\n\n")
        f.write("Confusion matrix (rows = actual, cols = predicted):\n")
        f.write(f"{cm}\n\n")
        f.write(report)

    metrics = {
        "test_images": int(len(y_true)),
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "sklearn_accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
    }
    history_path = os.path.join(RESULTS_DIR, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            hist = json.load(f)
        best = int(np.argmin(hist["val_loss"]))
        metrics.update(
            {
                "best_epoch": best + 1,
                "train_accuracy": hist["accuracy"][best],
                "val_accuracy": hist["val_accuracy"][best],
                "train_loss": hist["loss"][best],
                "val_loss": hist["val_loss"][best],
            }
        )
    # How often does the cat/dog gate accept the (genuine) cat & dog test images?
    print("\nChecking the cat/dog gate on the test images...")
    scores = CatDogGate().pet_scores(load_split("test").map(lambda img, _: img))
    gate_pass = float((scores >= GATE_THRESHOLD).mean())
    print(f"Gate accepted {gate_pass * 100:.2f}% of test cat/dog images")
    metrics["gate_threshold"] = GATE_THRESHOLD
    metrics["gate_accept_rate_on_test"] = gate_pass
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "a") as f:
        f.write(f"\nCat/dog gate (EfficientNetB0, threshold {GATE_THRESHOLD}) ")
        f.write(f"accepted {gate_pass * 100:.2f}% of test cat/dog images\n")

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    disp = ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix (Test accuracy {acc * 100:.2f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()

    save_sample_images()
    print(f"Results saved to {RESULTS_DIR}")
    print(f"Sample test images saved to {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
