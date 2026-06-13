from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

from data_utils import load_image_vector, load_metadata

__all__ = [
    "load_image_vector",
    "load_metadata",
    "run_pipeline",
]


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    import pipeline

    pipeline = importlib.reload(pipeline)
    _run_pipeline = pipeline.run_pipeline

    return _run_pipeline(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline completo para classificação de fontes em imagens."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/20_fonts_classification"),
        help="Diretório do dataset contendo README.md, metadata.csv e files/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Diretório para métricas, figuras, splits e modelo treinado.",
    )
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mlp-epochs", type=int, default=24)
    parser.add_argument("--cnn-epochs", type=int, default=24)
    parser.add_argument("--vgg-head-epochs", type=int, default=8)
    parser.add_argument("--vgg-finetune-epochs", type=int, default=16)
    parser.add_argument("--mlp-lr", type=float, default=0.001)
    parser.add_argument("--cnn-lr", type=float, default=0.001)
    parser.add_argument("--vgg-head-lr", type=float, default=0.001)
    parser.add_argument("--vgg-finetune-lr", type=float, default=0.00001)
    parser.add_argument("--vgg-trainable-layers", type=int, default=4)
    parser.add_argument(
        "--vgg-weights",
        default="imagenet",
        help=(
            "Use imagenet para transferência pré-treinada, none para treinar sem pesos externos "
            "ou informe o caminho de um arquivo .h5 com pesos locais."
        ),
    )
    parser.add_argument("--patience", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
