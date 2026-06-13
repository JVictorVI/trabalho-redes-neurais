"""Orquestração do pipeline completo de classificação de fontes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tensorflow import keras

from config import FIGURE_NAMES
from data_utils import load_metadata, prepare_data
from evaluation import classification_report, top_confusions
from models import (
    build_cnn_batchnorm,
    build_mlp,
    build_vgg16,
    enable_vgg_finetuning,
    set_global_seed,
)
from training import predict_classes, train_keras_model
from visualization import (
    plot_class_distribution,
    plot_confusion,
    plot_f1,
    plot_samples,
    plot_training_curves,
)


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "output": output_dir,
        "figures": output_dir / "figures",
        "models": output_dir / "models",
        "splits": output_dir / "splits",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def to_rgb(images: np.ndarray) -> np.ndarray:
    """Repete o canal cinza em RGB para modelos pre-treinados em imagens coloridas."""

    return np.repeat(images, 3, axis=-1)


def to_pretrained_rgb(images: np.ndarray) -> np.ndarray:
    """Prepara RGB com contraste original para modelos pre-treinados em ImageNet."""

    return to_rgb(1.0 - images)


def save_keras_model(
    model: keras.Model,
    path: Path,
    classes: list[str],
    image_size: tuple[int, int],
    config: dict[str, Any],
) -> None:
    model.save(path)
    metadata = {
        "classes": classes,
        "image_size": list(image_size),
        "config": config,
    }
    path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def evaluate_and_save(
    key: str,
    label: str,
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    paths: dict[str, Path],
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
    pred = predict_classes(model, x_test, batch_size=batch_size)
    report, cm, summary = classification_report(y_test, pred, classes)
    report.to_csv(paths["output"] / f"classification_report_{key}.csv", index=False)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        paths["output"] / f"confusion_matrix_{key}.csv"
    )
    print(
        f"{label} | teste: accuracy={summary['accuracy']:.4f} "
        f"macro_f1={summary['macro_f1']:.4f}"
    )
    return report, cm, summary


def merge_training_phases(
    head_result: dict[str, Any],
    finetune_result: dict[str, Any],
) -> dict[str, Any]:
    """Combina historicos da cabeca congelada e do ajuste fino."""

    rows: list[dict[str, Any]] = []
    for phase_name, result in (("head", head_result), ("fine_tuning", finetune_result)):
        for row in result["history"]:
            rows.append({**row, "epoch": len(rows) + 1, "phase": phase_name})

    best_row = max(rows, key=lambda row: row["val_accuracy"])
    return {
        "history": rows,
        "best_val_accuracy": float(best_row["val_accuracy"]),
        "best_val_loss": float(best_row["val_loss"]),
        "epochs_ran": len(rows),
        "parameter_count": int(finetune_result["parameter_count"]),
        "phases": {
            "head": head_result,
            "fine_tuning": finetune_result,
        },
    }


def get_arg(args: Any, name: str) -> Any:
    if hasattr(args, name):
        return getattr(args, name)
    raise AttributeError(f"Argumento ausente: {name}")


def run_pipeline(args: Any) -> dict[str, Any]:
    """Orquestra dados, treino, avaliação, figuras e modelos salvos."""

    set_global_seed(args.seed)
    dataset_dir = args.dataset
    output_dir = args.output
    paths = ensure_dirs(output_dir)
    image_size = (args.image_width, args.image_height)
    vgg_head_epochs = get_arg(args, "vgg_head_epochs")
    vgg_finetune_epochs = get_arg(args, "vgg_finetune_epochs")
    vgg_head_lr = get_arg(args, "vgg_head_lr")
    vgg_finetune_lr = get_arg(args, "vgg_finetune_lr")
    vgg_trainable_layers = get_arg(args, "vgg_trainable_layers")
    vgg_weights_arg = get_arg(args, "vgg_weights")
    vgg_weights = None if vgg_weights_arg == "none" else vgg_weights_arg
    config = {
        "dataset": str(dataset_dir),
        "output": str(output_dir),
        "image_size": list(image_size),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "mlp_epochs": args.mlp_epochs,
        "cnn_epochs": args.cnn_epochs,
        "vgg_head_epochs": vgg_head_epochs,
        "vgg_finetune_epochs": vgg_finetune_epochs,
        "mlp_lr": args.mlp_lr,
        "cnn_lr": args.cnn_lr,
        "vgg_head_lr": vgg_head_lr,
        "vgg_finetune_lr": vgg_finetune_lr,
        "vgg_trainable_layers": vgg_trainable_layers,
        "patience": args.patience,
        "vgg_weights": vgg_weights_arg,
    }

    metadata = load_metadata(dataset_dir)
    data = prepare_data(metadata, image_size=image_size, seed=args.seed, paths=paths)
    input_dim = data.x_train.shape[1]
    image_shape = data.x_train_images.shape[1:]
    n_classes = len(data.classes)

    print("Treinando MLP Keras baseline...")
    mlp_model = build_mlp(
        input_dim=input_dim,
        n_classes=n_classes,
        learning_rate=args.mlp_lr,
    )
    mlp_result = train_keras_model(
        mlp_model,
        data.x_train,
        data.y_train,
        data.x_val,
        data.y_val,
        epochs=args.mlp_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    print("Treinando CNN Keras com BatchNormalization...")
    cnn_model = build_cnn_batchnorm(
        input_shape=image_shape,
        n_classes=n_classes,
        learning_rate=args.cnn_lr,
    )
    cnn_result = train_keras_model(
        cnn_model,
        data.x_train_images,
        data.y_train,
        data.x_val_images,
        data.y_val,
        epochs=args.cnn_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    print("Preparando tensores RGB para VGG16...")
    x_train_rgb = to_pretrained_rgb(data.x_train_images)
    x_val_rgb = to_pretrained_rgb(data.x_val_images)
    x_test_rgb = to_pretrained_rgb(data.x_test_images)

    print("Treinando VGG16 por transferência de aprendizado...")
    vgg_model = build_vgg16(
        input_shape=x_train_rgb.shape[1:],
        n_classes=n_classes,
        learning_rate=vgg_head_lr,
        weights=vgg_weights,
    )
    vgg_head_result = train_keras_model(
        vgg_model,
        x_train_rgb,
        data.y_train,
        x_val_rgb,
        data.y_val,
        epochs=vgg_head_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    print("Ajustando as últimas camadas da VGG16...")
    enable_vgg_finetuning(
        vgg_model,
        trainable_layers=vgg_trainable_layers,
        learning_rate=vgg_finetune_lr,
    )
    vgg_finetune_result = train_keras_model(
        vgg_model,
        x_train_rgb,
        data.y_train,
        x_val_rgb,
        data.y_val,
        epochs=vgg_finetune_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    vgg_result = merge_training_phases(
        vgg_head_result,
        vgg_finetune_result,
    )

    print("Avaliando modelos no teste...")
    mlp_report, mlp_cm, mlp_summary = evaluate_and_save(
        "mlp",
        "MLP Keras",
        mlp_model,
        data.x_test,
        data.y_test,
        data.classes,
        paths,
        args.batch_size,
    )
    cnn_report, cnn_cm, cnn_summary = evaluate_and_save(
        "cnn_batchnorm",
        "CNN BatchNorm",
        cnn_model,
        data.x_test_images,
        data.y_test,
        data.classes,
        paths,
        args.batch_size,
    )
    vgg_report, vgg_cm, vgg_summary = evaluate_and_save(
        "vgg16",
        "VGG16 Fine-tuning",
        vgg_model,
        x_test_rgb,
        data.y_test,
        data.classes,
        paths,
        args.batch_size,
    )

    results: dict[str, Any] = {
        "config": config,
        "dataset": {
            "total_images": int(len(metadata)),
            "class_count": int(n_classes),
            "image_original_size": [200, 50],
            "class_distribution": metadata["Font"].value_counts().sort_index().astype(int).to_dict(),
            "word_length_min": int(metadata["Text"].str.len().min()),
            "word_length_max": int(metadata["Text"].str.len().max()),
            "split_sizes": {
                "train": int(len(data.train_df)),
                "val": int(len(data.val_df)),
                "test": int(len(data.test_df)),
            },
        },
        "models": {
            "mlp": {
                **mlp_result,
                "label": "MLP Keras",
                "test_summary": mlp_summary,
                "top_confusions": top_confusions(mlp_cm, data.classes),
            },
            "cnn_batchnorm": {
                **cnn_result,
                "label": "CNN BatchNorm",
                "test_summary": cnn_summary,
                "top_confusions": top_confusions(cnn_cm, data.classes),
            },
            "vgg16": {
                **vgg_result,
                "label": "VGG16 Fine-tuning",
                "test_summary": vgg_summary,
                "top_confusions": top_confusions(vgg_cm, data.classes),
            },
        },
        "classes": data.classes,
    }
    (paths["output"] / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Gerando figuras...")
    plot_samples(metadata, data.classes, paths["figures"] / FIGURE_NAMES["samples"])
    plot_class_distribution(metadata, paths["figures"] / FIGURE_NAMES["distribution"])
    plot_training_curves(results["models"], paths["figures"] / FIGURE_NAMES["curves"])
    plot_confusion(cnn_cm, data.classes, paths["figures"] / FIGURE_NAMES["confusion"], "CNN BatchNorm")
    plot_f1(cnn_report, paths["figures"] / FIGURE_NAMES["f1"], "CNN BatchNorm")

    print("Salvando modelos Keras...")
    save_keras_model(mlp_model, paths["models"] / "mlp_fonts_classifier.keras", data.classes, image_size, config)
    save_keras_model(
        cnn_model,
        paths["models"] / "cnn_batchnorm_fonts_classifier.keras",
        data.classes,
        image_size,
        config,
    )
    save_keras_model(
        vgg_model,
        paths["models"] / "vgg16_fonts_classifier.keras",
        data.classes,
        image_size,
        config,
    )

    print("Pipeline concluído.")
    return results
