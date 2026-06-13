"""Leitura, validação, split e pré-processamento do dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class SplitData:
    """Agrupa os arrays de treino/validação/teste e os metadados dos splits."""

    x_train: np.ndarray
    x_train_images: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    x_val_images: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    x_test_images: np.ndarray
    y_test: np.ndarray
    classes: list[str]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def load_metadata(dataset_dir: Path) -> pd.DataFrame:
    """Lê o metadata.csv e confirma se todas as imagens citadas existem."""

    metadata_path = dataset_dir / "metadata.csv"
    files_dir = dataset_dir / "files"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {metadata_path}")
    if not files_dir.exists():
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {files_dir}")

    # Mantém palavras como "None" como texto real, em vez de converter para NaN.
    df = pd.read_csv(metadata_path, keep_default_na=False)
    expected = {"FileName", "Font", "Text"}
    missing_columns = expected.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Colunas ausentes no metadata.csv: {sorted(missing_columns)}")

    df = df.copy()
    df["ImagePath"] = df["FileName"].map(lambda name: str(files_dir / name))
    missing_files = [path for path in df["ImagePath"] if not Path(path).exists()]
    if missing_files:
        preview = ", ".join(missing_files[:5])
        raise FileNotFoundError(f"{len(missing_files)} imagens ausentes. Exemplos: {preview}")
    return df


def make_stratified_split(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide os dados preservando a mesma quantidade relativa por classe."""

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby("Font", sort=True):
        # Cada fonte é embaralhada separadamente para manter o split balanceado.
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        split_values = np.array(["test"] * n, dtype=object)
        split_values[:n_train] = "train"
        split_values[n_train : n_train + n_val] = "val"
        split_df = df.loc[indices].copy()
        split_df["Split"] = split_values
        parts.append(split_df)

    split_all = pd.concat(parts, ignore_index=True)
    split_all = split_all.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return (
        split_all[split_all["Split"] == "train"].reset_index(drop=True),
        split_all[split_all["Split"] == "val"].reset_index(drop=True),
        split_all[split_all["Split"] == "test"].reset_index(drop=True),
    )


def save_splits(paths: dict[str, Path], split_data: tuple[pd.DataFrame, ...]) -> None:
    for name, frame in zip(("train", "val", "test"), split_data):
        cols = ["FileName", "Font", "Text", "Split"]
        frame[cols].to_csv(paths["splits"] / f"{name}.csv", index=False)


def image_resample_filter() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def load_image_vector(path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    """Carrega uma imagem e devolve um vetor numérico pronto para o modelo."""

    return load_image_array(path, image_size).reshape(-1)


def load_image_array(path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    """Carrega uma imagem como tensor 2D normalizado, com tinta como sinal alto."""

    with Image.open(path) as image:
        gray = image.convert("L").resize(image_size, image_resample_filter())
    pixels = np.asarray(gray, dtype=np.float32) / 255.0
    # Como o fundo é claro e o texto é escuro, inverter deixa o traço como sinal alto.
    ink_as_signal = 1.0 - pixels
    return ink_as_signal


def load_images(frame: pd.DataFrame, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    n = len(frame)
    width, height = image_size
    images = np.empty((n, height, width, 1), dtype=np.float32)
    for i, path in enumerate(frame["ImagePath"]):
        images[i, :, :, 0] = load_image_array(path, image_size)
    vectors = images.reshape(n, width * height)
    return vectors, images


def encode_labels(labels: pd.Series, classes: list[str]) -> np.ndarray:
    class_to_idx = {name: i for i, name in enumerate(classes)}
    return labels.map(class_to_idx).to_numpy(dtype=np.int64)


def prepare_data(
    df: pd.DataFrame,
    image_size: tuple[int, int],
    seed: int,
    paths: dict[str, Path],
) -> SplitData:
    """Executa split, carregamento das imagens e codificação das classes."""

    train_df, val_df, test_df = make_stratified_split(df, seed=seed)
    save_splits(paths, (train_df, val_df, test_df))
    classes = sorted(df["Font"].unique().tolist())
    print("Carregando imagens de treino...")
    x_train, x_train_images = load_images(train_df, image_size)
    print("Carregando imagens de validação...")
    x_val, x_val_images = load_images(val_df, image_size)
    print("Carregando imagens de teste...")
    x_test, x_test_images = load_images(test_df, image_size)
    return SplitData(
        x_train=x_train,
        x_train_images=x_train_images,
        y_train=encode_labels(train_df["Font"], classes),
        x_val=x_val,
        x_val_images=x_val_images,
        y_val=encode_labels(val_df["Font"], classes),
        x_test=x_test,
        x_test_images=x_test_images,
        y_test=encode_labels(test_df["Font"], classes),
        classes=classes,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )
