"""
Train the Cat vs Dog CNN.

Usage:
    python train.py

Outputs:
    models/cat_dog_model.keras   - best model (lowest validation loss)
    results/history.json         - per-epoch accuracy / loss
    results/accuracy.png         - training vs validation accuracy
    results/loss.png             - training vs validation loss
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info logs

import matplotlib

matplotlib.use("Agg")  # save plots to files, no display needed
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data import get_dataset
from preprocess import IMG_SIZE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(MODEL_DIR, "cat_dog_model.keras")

EPOCHS = 15


def build_model():
    """Simple 3-block CNN."""
    model = keras.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            # Block 3
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            # Classifier
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),  # output = P(dog)
        ],
        name="cat_dog_cnn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    epochs = range(1, len(history["accuracy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["accuracy"], "o-", label="Training accuracy")
    plt.plot(epochs, history["val_accuracy"], "o-", label="Validation accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["loss"], "o-", label="Training loss")
    plt.plot(epochs, history["val_loss"], "o-", label="Validation loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss.png"), dpi=120)
    plt.close()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tf.random.set_seed(42)

    print("Loading dataset (downloads automatically the first time)...")
    train_ds = get_dataset("train", shuffle=True, augment=True)
    val_ds = get_dataset("validation")

    model = build_model()
    model.summary()

    callbacks = [
        # Stop when validation loss stops improving and keep the best weights.
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        # Save the best model seen so far to disk.
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks
    )

    # Make sure the saved model has the best (restored) weights.
    model.save(MODEL_PATH)

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    plot_history(hist)

    best = min(range(len(hist["val_loss"])), key=lambda i: hist["val_loss"][i])
    print("\n===== Training finished =====")
    print(f"Epochs run:                {len(hist['loss'])}")
    print(f"Best epoch:                {best + 1}")
    print(f"Training accuracy:         {hist['accuracy'][best] * 100:.2f}%")
    print(f"Validation accuracy:       {hist['val_accuracy'][best] * 100:.2f}%")
    print(f"Training loss:             {hist['loss'][best]:.4f}")
    print(f"Validation loss:           {hist['val_loss'][best]:.4f}")
    print(f"Model saved to:            {MODEL_PATH}")
    print(f"Plots saved to:            {RESULTS_DIR}")


if __name__ == "__main__":
    main()
