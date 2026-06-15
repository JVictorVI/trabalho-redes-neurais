"""Rotinas de treinamento dos modelos Keras."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from tensorflow import keras


class EpochTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        self.epoch_seconds: list[float] = []

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self._start_time = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.epoch_seconds.append(time.perf_counter() - self._start_time)


def train_keras_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    patience: int,
) -> dict[str, Any]:
    """Treina um modelo Keras com early stopping baseado na acurácia de validação."""

    timer = EpochTimer()
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=patience,
        restore_best_weights=True,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[timer, early_stopping],
        verbose=2,
    )

    rows: list[dict[str, Any]] = []
    for i in range(len(history.history["loss"])):
        rows.append(
            {
                "epoch": i + 1,
                "train_loss": float(history.history["loss"][i]),
                "train_accuracy": float(history.history["accuracy"][i]),
                "val_loss": float(history.history["val_loss"][i]),
                "val_accuracy": float(history.history["val_accuracy"][i]),
                "epoch_seconds": float(timer.epoch_seconds[i]),
            }
        )

    val_accuracies = history.history["val_accuracy"]
    val_losses = history.history["val_loss"]
    best_accuracy_index = int(np.argmax(val_accuracies))
    min_loss_index = int(np.argmin(val_losses))
    return {
        "history": rows,
        "best_val_accuracy": float(history.history["val_accuracy"][best_accuracy_index]),
        "val_loss_at_best_val_accuracy": float(history.history["val_loss"][best_accuracy_index]),
        "best_val_loss": float(history.history["val_loss"][min_loss_index]),
        "epochs_ran": len(rows),
        "parameter_count": int(model.count_params()),
    }


def predict_classes(model: keras.Model, x: np.ndarray, batch_size: int) -> np.ndarray:
    probabilities = model.predict(x, batch_size=batch_size, verbose=0)
    return np.argmax(probabilities, axis=1)
