"""Construtores dos modelos Keras usados na classificação de fontes."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_mlp(
    input_dim: int,
    n_classes: int,
    *,
    hidden_units: tuple[int, int] = (256, 128),
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """MLP baseline: recebe a imagem vetorizada e ignora a estrutura espacial."""

    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,), name="pixels"),
            layers.Dense(hidden_units[0], activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(hidden_units[1], activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(n_classes, activation="softmax"),
        ],
        name="mlp_keras",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_batchnorm(
    input_shape: tuple[int, int, int],
    n_classes: int,
    *,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """CNN principal: preserva vizinhanca espacial e normaliza ativacoes por bloco."""

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape, name="image"),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(n_classes, activation="softmax"),
        ],
        name="cnn_batchnorm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_vgg16(
    input_shape: tuple[int, int, int],
    n_classes: int,
    *,
    learning_rate: float = 1e-4,
    weights: str | None = "imagenet",
) -> keras.Model:
    """VGG16 por transferência: base congelada e cabeçalho novo."""

    try:
        base_model = keras.applications.VGG16(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
        )
    except Exception as exc:
        if weights == "imagenet":
            raise RuntimeError(
                "Não foi possível carregar os pesos ImageNet da VGG16. "
                "Verifique a conexão com o host de pesos do Keras ou execute com "
                "--vgg-weights none. Também é possível informar o caminho "
                "local para um arquivo .h5 compatível com VGG16."
            ) from exc
        raise
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape, name="image_rgb")
    x = keras.applications.vgg16.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="vgg16_finetuning")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def enable_vgg_finetuning(
    model: keras.Model,
    *,
    trainable_layers: int = 4,
    learning_rate: float = 1e-5,
) -> None:
    """Descongela as últimas camadas convolucionais da VGG16 e recompila."""

    base_model = next(layer for layer in model.layers if layer.name == "vgg16")
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def set_global_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
