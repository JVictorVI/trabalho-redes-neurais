"""Geração das figuras usadas na análise dos resultados."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def plot_samples(df: pd.DataFrame, classes: list[str], figure_path: Path) -> None:
    """Gera uma figura com uma amostra visual de cada fonte."""

    fig, axes = plt.subplots(4, 5, figsize=(13, 7))
    axes = axes.ravel()
    for ax, font in zip(axes, classes):
        row = df[df["Font"] == font].iloc[0]
        with Image.open(row["ImagePath"]) as image:
            ax.imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
        ax.set_title(font, fontsize=8)
        ax.axis("off")
    fig.suptitle("Amostras do dataset 20 Fonts Classification", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def plot_class_distribution(df: pd.DataFrame, figure_path: Path) -> None:
    counts = df["Font"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(counts.index, counts.values, color="#3c6e71")
    ax.set_title("Distribuição de imagens por classe")
    ax.set_ylabel("Quantidade de imagens")
    ax.set_xlabel("Fonte")
    ax.tick_params(axis="x", labelrotation=70)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def plot_training_curves(results: dict[str, Any], figure_path: Path) -> None:
    """Plota loss e accuracy para comparar os modelos ao longo das épocas."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {
        "mlp": "#2a9d8f",
        "cnn_batchnorm": "#3c6e71",
        "vgg16": "#e76f51",
    }
    labels = {
        "mlp": "MLP Keras",
        "cnn_batchnorm": "CNN BatchNorm",
        "vgg16": "VGG16 Fine-tuning",
    }
    fallback_colors = ["#264653", "#f4a261", "#457b9d", "#6d597a"]
    for i, key in enumerate(results):
        color = colors.get(key, fallback_colors[i % len(fallback_colors)])
        label = results[key].get("label", labels.get(key, key))
        hist = pd.DataFrame(results[key]["history"])
        axes[0].plot(hist["epoch"], hist["train_loss"], "--", color=color, alpha=0.65)
        axes[0].plot(hist["epoch"], hist["val_loss"], "-", color=color, label=label)
        axes[1].plot(hist["epoch"], hist["train_accuracy"], "--", color=color, alpha=0.65)
        axes[1].plot(hist["epoch"], hist["val_accuracy"], "-", color=color, label=label)
    axes[0].set_title("Loss por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Accuracy por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Linhas tracejadas: treino | linhas continuas: validação", fontsize=10)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def plot_confusion(
    cm: np.ndarray, classes: list[str], figure_path: Path, model_label: str = "MLP"
) -> None:
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"Matriz de confusão normalizada - {model_label}")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_xticks(range(len(classes)), labels=classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), labels=classes, fontsize=7)
    for i in range(len(classes)):
        for j in range(len(classes)):
            value = normalized[i, j]
            if value >= 0.20:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def plot_f1(report: pd.DataFrame, figure_path: Path, model_label: str = "MLP") -> None:
    sorted_report = report.sort_values("f1")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(sorted_report["class"], sorted_report["f1"], color="#e76f51")
    ax.set_title(f"F1-score por classe - {model_label}")
    ax.set_xlabel("F1-score")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
