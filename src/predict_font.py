"""Predição de fonte para uma imagem individual usando modelos Keras salvos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from data_utils import load_image_array, load_image_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classifica uma imagem com o modelo Keras salvo.")
    parser.add_argument("image", type=Path, help="Caminho para uma imagem PNG.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs/models/cnn_batchnorm_fonts_classifier.keras"),
        help="Arquivo .keras gerado pelo treinamento.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def load_model_metadata(model_path: Path) -> dict[str, object]:
    metadata_path = model_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadados do modelo não encontrados: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def prepare_input(image_path: Path, image_size: tuple[int, int], input_shape: tuple[int | None, ...]) -> np.ndarray:
    width, height = image_size
    if len(input_shape) == 2:
        return load_image_vector(image_path, (width, height))[None, :]

    channels = input_shape[-1]
    image = load_image_array(image_path, (width, height))[None, :, :, None]
    if channels == 3:
        return np.repeat(image, 3, axis=-1)
    return image


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {args.image}")
    if not args.model.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {args.model}. Execute python src/font_classification.py primeiro."
        )

    metadata = load_model_metadata(args.model)
    classes = [str(item) for item in metadata["classes"]]
    width, height = [int(item) for item in metadata["image_size"]]

    model = keras.models.load_model(args.model)
    x = prepare_input(args.image, (width, height), model.input_shape)
    probs = model.predict(x, verbose=0)[0]
    order = np.argsort(probs)[::-1][: args.top_k]

    print(f"Imagem: {args.image}")
    for index in order:
        print(f"{classes[index]}: {probs[index]:.4f}")


if __name__ == "__main__":
    main()
