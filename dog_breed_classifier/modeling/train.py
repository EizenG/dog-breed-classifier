"""Entraînement des modèles avec optimisation Optuna et tracking MLflow."""
import json
import os
import subprocess
import time
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import optuna
import tensorflow as tf
import typer
import yaml
from loguru import logger

from dog_breed_classifier.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT, REPORTS_DIR

app = typer.Typer()
AUTOTUNE = tf.data.AUTOTUNE


# ── Callback ──────────────────────────────────────────────────────────────────

class _MLflowEpochLogger(tf.keras.callbacks.Callback):
    """Logue loss/accuracy dans le run MLflow actif à chaque epoch."""

    def __init__(self, step_offset: int = 0) -> None:
        super().__init__()
        self._offset = step_offset

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs:
            mlflow.log_metrics(
                {k: float(v) for k, v in logs.items()},
                step=self._offset + epoch,
            )


# ── Chargement config ─────────────────────────────────────────────────────────

def load_params() -> dict:
    with open(PROJ_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


# ── Données ───────────────────────────────────────────────────────────────────

def build_datasets(batch_size: int, img_size: tuple, seed: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DATA_DIR / "train",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=seed,
    ).prefetch(AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DATA_DIR / "val",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    ).prefetch(AUTOTUNE)

    return train_ds, val_ds


# ── Architectures ─────────────────────────────────────────────────────────────

def _augmentation_block(aug: dict) -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(aug["rotation"]),
        tf.keras.layers.RandomZoom(aug["zoom"]),
        tf.keras.layers.RandomTranslation(aug["translation"], aug["translation"]),
        tf.keras.layers.RandomBrightness(aug["brightness"]),
    ], name="augmentation")


def build_cnn_scratch(
    num_classes: int, img_size: tuple, aug: dict, dropout_rate: float, l2_reg: float
) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(l2_reg)
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*img_size, 3)),
        _augmentation_block(aug),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same", kernel_regularizer=reg),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", kernel_regularizer=reg),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same", kernel_regularizer=reg),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="cnn_scratch")


def build_efficientnet(
    num_classes: int, img_size: tuple, aug: dict, dropout_rate: float, l2_reg: float
) -> tf.keras.Model:
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(*img_size, 3)
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(*img_size, 3))
    x = _augmentation_block(aug)(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
    )(x)
    return tf.keras.Model(inputs, outputs, name="efficientnet_b0")


def build_inception(
    num_classes: int, img_size: tuple, aug: dict, dropout_rate: float, l2_reg: float
) -> tf.keras.Model:
    base = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet", input_shape=(*img_size, 3)
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(*img_size, 3))
    x = _augmentation_block(aug)(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
    )(x)
    return tf.keras.Model(inputs, outputs, name="inception_v3")


# ── Callbacks & unfreeze ──────────────────────────────────────────────────────

def _default_callbacks(training_cfg: dict, extra: list | None = None) -> list:
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=training_cfg["early_stopping"]["patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=training_cfg["reduce_lr"]["factor"],
            patience=training_cfg["reduce_lr"]["patience"],
            verbose=0,
        ),
    ]
    if extra:
        cbs.extend(extra)
    return cbs


def _unfreeze_efficientnet(model: tf.keras.Model, learning_rate: float, top_layers: int) -> None:
    base = model.get_layer("efficientnetb0")
    base.trainable = True
    for layer in base.layers[:-top_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def _unfreeze_inception(model: tf.keras.Model, learning_rate: float, from_layer: int) -> None:
    base = model.get_layer("inception_v3")
    base.trainable = True
    for layer in base.layers[:from_layer]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


# ── Entraînement ──────────────────────────────────────────────────────────────

def train_transfer(
    model: tf.keras.Model,
    model_name: str,
    train_ds,
    val_ds,
    learning_rate: float,
    training_cfg: dict,
    log_epochs: bool = False,
) -> float | tuple:
    """2 phases. Retourne val_accuracy, ou (val_accuracy, h_phase1, h_phase2) si log_epochs=True."""
    p1 = training_cfg["phase1"]
    p2 = training_cfg["phase2"]
    unfreeze_cfg = training_cfg["unfreeze"]

    p1_extra = [_MLflowEpochLogger(step_offset=0)] if log_epochs else None
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=p1["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info(f"[{model_name}] Phase 1 — head only")
    h1 = model.fit(
        train_ds, validation_data=val_ds, epochs=p1["epochs"],
        callbacks=_default_callbacks(training_cfg, extra=p1_extra), verbose=0,
    )

    p1_actual = len(h1.epoch)
    p2_extra = [_MLflowEpochLogger(step_offset=p1_actual)] if log_epochs else None

    logger.info(f"[{model_name}] Phase 2 — partial unfreeze, lr={learning_rate:.0e}")
    if "efficientnet" in model_name:
        _unfreeze_efficientnet(model, learning_rate, unfreeze_cfg["efficientnet_top_layers"])
    else:
        _unfreeze_inception(model, learning_rate, unfreeze_cfg["inception_from_layer"])

    h2 = model.fit(
        train_ds, validation_data=val_ds, epochs=p2["epochs"],
        callbacks=_default_callbacks(training_cfg, extra=p2_extra), verbose=0,
    )

    val_acc = max(h2.history["val_accuracy"])
    return (val_acc, h1, h2) if log_epochs else val_acc


# ── Visualisation ─────────────────────────────────────────────────────────────

def _save_training_curves(histories: dict, model_name: str) -> tuple[str, plt.Figure]:
    """Génère et sauvegarde la figure loss/accuracy. Retourne (chemin, figure)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if "cnn" in histories:
        h = histories["cnn"].history
        epochs = range(1, len(h["loss"]) + 1)
        ax1.plot(epochs, h["accuracy"], label="train")
        ax1.plot(epochs, h["val_accuracy"], label="val")
        ax2.plot(epochs, h["loss"], label="train")
        ax2.plot(epochs, h["val_loss"], label="val")
    else:
        h1 = histories["phase1"].history
        h2 = histories["phase2"].history
        n1, n2 = len(h1["loss"]), len(h2["loss"])
        e1 = range(1, n1 + 1)
        e2 = range(n1 + 1, n1 + n2 + 1)

        ax1.plot(e1, h1["accuracy"], "b-", label="train — phase 1")
        ax1.plot(e1, h1["val_accuracy"], "b--", label="val — phase 1")
        ax1.plot(e2, h2["accuracy"], "r-", label="train — phase 2")
        ax1.plot(e2, h2["val_accuracy"], "r--", label="val — phase 2")
        ax1.axvline(n1 + 0.5, color="gray", linestyle=":", label="début phase 2")

        ax2.plot(e1, h1["loss"], "b-", label="train — phase 1")
        ax2.plot(e1, h1["val_loss"], "b--", label="val — phase 1")
        ax2.plot(e2, h2["loss"], "r-", label="train — phase 2")
        ax2.plot(e2, h2["val_loss"], "r--", label="val — phase 2")
        ax2.axvline(n1 + 0.5, color="gray", linestyle=":")

    for ax, title in [(ax1, "Accuracy"), (ax2, "Loss")]:
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Courbes d'entraînement — {model_name}", fontsize=13)
    fig.tight_layout()
    path = FIGURES_DIR / f"training_curves_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return str(path), fig


# ── Optuna objective ──────────────────────────────────────────────────────────

def _make_objective(model_name: str, model_cfg: dict, training_cfg: dict, optuna_cfg: dict):
    num_classes = model_cfg["num_classes"]
    img_size = tuple(model_cfg["img_size"])
    aug = model_cfg["augmentation"]
    seed = model_cfg["seed"]
    ss = optuna_cfg["search_space"]

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", ss["learning_rate"]["min"], ss["learning_rate"]["max"], log=True)
        dropout = trial.suggest_float("dropout_rate", ss["dropout_rate"]["min"], ss["dropout_rate"]["max"])
        batch_size = trial.suggest_categorical("batch_size", ss["batch_size"])
        l2_reg = trial.suggest_float("l2_reg", ss["l2_reg"]["min"], ss["l2_reg"]["max"], log=True)

        train_ds, val_ds = build_datasets(batch_size, img_size, seed)

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params({
                "model": model_name, "learning_rate": lr,
                "dropout_rate": dropout, "batch_size": batch_size, "l2_reg": l2_reg,
            })

            if model_name == "cnn_scratch":
                model = build_cnn_scratch(num_classes, img_size, aug, dropout, l2_reg)
                model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                              loss="categorical_crossentropy", metrics=["accuracy"])
                h = model.fit(train_ds, validation_data=val_ds,
                              epochs=training_cfg["cnn_scratch"]["epochs"],
                              callbacks=_default_callbacks(training_cfg), verbose=0)
                val_acc = max(h.history["val_accuracy"])
            elif model_name == "efficientnet_b0":
                model = build_efficientnet(num_classes, img_size, aug, dropout, l2_reg)
                val_acc = train_transfer(model, model_name, train_ds, val_ds, lr, training_cfg)
            else:
                model = build_inception(num_classes, img_size, aug, dropout, l2_reg)
                val_acc = train_transfer(model, model_name, train_ds, val_ds, lr, training_cfg)

            mlflow.log_metric("val_accuracy", val_acc)

        return val_acc

    return objective


# ── Entraînement final ────────────────────────────────────────────────────────

def _train_best(
    model_name: str, best_params: dict, model_cfg: dict, training_cfg: dict
) -> tuple[tf.keras.Model, dict]:
    """Réentraîne avec les meilleurs hyperparamètres. Retourne (model, histories)."""
    num_classes = model_cfg["num_classes"]
    img_size = tuple(model_cfg["img_size"])
    aug = model_cfg["augmentation"]
    seed = model_cfg["seed"]

    dropout = best_params["dropout_rate"]
    l2_reg = best_params["l2_reg"]
    lr = best_params["learning_rate"]
    train_ds, val_ds = build_datasets(best_params["batch_size"], img_size, seed)

    if model_name == "cnn_scratch":
        model = build_cnn_scratch(num_classes, img_size, aug, dropout, l2_reg)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        h = model.fit(
            train_ds, validation_data=val_ds,
            epochs=training_cfg["cnn_scratch"]["epochs"],
            callbacks=_default_callbacks(training_cfg, extra=[_MLflowEpochLogger()]),
            verbose=1,
        )
        histories = {"cnn": h}
    elif model_name == "efficientnet_b0":
        model = build_efficientnet(num_classes, img_size, aug, dropout, l2_reg)
        _, h1, h2 = train_transfer(model, model_name, train_ds, val_ds, lr, training_cfg, log_epochs=True)
        histories = {"phase1": h1, "phase2": h2}
    else:
        model = build_inception(num_classes, img_size, aug, dropout, l2_reg)
        _, h1, h2 = train_transfer(model, model_name, train_ds, val_ds, lr, training_cfg, log_epochs=True)
        histories = {"phase1": h1, "phase2": h2}

    return model, histories


# ── Main ──────────────────────────────────────────────────────────────────────

@app.command()
def main(
    n_trials: Optional[int] = typer.Option(None, help="Override n_trials depuis params.yaml"),
    models: str = typer.Option(
        "all",
        help="Modèles à entraîner : all | cnn_scratch | efficientnet_b0 | inception_v3",
    ),
):
    """Entraînement avec recherche d'hyperparamètres Optuna et tracking MLflow."""
    params = load_params()
    model_cfg = params["model"]
    model_cfg["num_classes"] = len(params["data"]["races"])
    training_cfg = params["training"]
    optuna_cfg = params["optuna"]
    effective_trials = n_trials if n_trials is not None else optuna_cfg["n_trials"]
    img_size = tuple(model_cfg["img_size"])
    tf.random.set_seed(model_cfg["seed"])
    np.random.seed(model_cfg["seed"])

    model_names = (
        ["cnn_scratch", "efficientnet_b0", "inception_v3"]
        if models == "all"
        else [models]
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
    mlflow.set_experiment("dog-breed-classifier")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJ_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJ_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"
        git_branch = "unknown"

    for model_name in model_names:
        logger.info(f"=== {model_name} — {effective_trials} trials Optuna ===")

        with mlflow.start_run(run_name=f"{model_name}_{run_timestamp}") as run:
            mlflow.set_tags({
                "model_name": model_name,
                "stage": "training",
                "n_trials": str(effective_trials),
                "git_commit": git_commit,
                "git_branch": git_branch,
                "timestamp": run_timestamp,
            })

            # ── Recherche Optuna ──
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=model_cfg["seed"]),
                pruner=optuna.pruners.MedianPruner(),
            )
            study.optimize(
                _make_objective(model_name, model_cfg, training_cfg, optuna_cfg),
                n_trials=effective_trials,
            )

            best = study.best_params
            logger.info(f"[{model_name}] best={best}  val_acc={study.best_value:.4f}")
            mlflow.log_params({f"best_{k}": v for k, v in best.items()})
            mlflow.log_metric("best_val_accuracy", study.best_value)

            # ── Entraînement final ──
            t0 = time.time()
            final_model, histories = _train_best(model_name, best, model_cfg, training_cfg)
            training_duration = time.time() - t0

            # ── Métriques et artefacts ──
            mlflow.log_metrics({
                "training_duration_s": round(training_duration, 1),
                "total_params": final_model.count_params(),
            })

            curves_path, curves_fig = _save_training_curves(histories, model_name)
            mlflow.log_figure(curves_fig, f"figures/training_curves_{model_name}.png")
            plt.close(curves_fig)
            mlflow.log_artifact(str(PROJ_ROOT / "params.yaml"), artifact_path="config")

            save_path = MODELS_DIR / f"{model_name}.keras"
            final_model.save(save_path)
            sample = np.zeros((1, *img_size, 3), dtype=np.float32)
            prediction = final_model.predict(sample, verbose=0)
            signature = mlflow.models.infer_signature(sample, prediction)
            mlflow.tensorflow.log_model(
                final_model, name="model", signature=signature, input_example=sample
            )

            # Persiste le run_id pour evaluate.py
            run_ids_path = REPORTS_DIR / "run_ids.json"
            run_ids = json.loads(run_ids_path.read_text()) if run_ids_path.exists() else {}
            run_ids[model_name] = run.info.run_id
            run_ids_path.write_text(json.dumps(run_ids, indent=2))

            logger.success(
                f"[{model_name}] run_id={run.info.run_id} | "
                f"val_acc={study.best_value:.4f} | "
                f"durée={training_duration:.0f}s | "
                f"params={final_model.count_params():,}"
            )


if __name__ == "__main__":
    app()
