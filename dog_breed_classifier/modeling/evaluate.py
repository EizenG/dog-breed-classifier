"""Évaluation des modèles entraînés sur le test set — logue dans le run MLflow existant."""
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
import typer
import yaml
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix

from dog_breed_classifier.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT, REPORTS_DIR

app = typer.Typer()
AUTOTUNE = tf.data.AUTOTUNE


def load_params() -> dict:
    with open(PROJ_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def _load_model(model_name: str, run_id: str) -> tf.keras.Model:
    """Charge le modèle depuis le disque local ou le télécharge depuis MLflow."""
    model_path = MODELS_DIR / f"{model_name}.keras"
    if model_path.exists():
        logger.info(f"[{model_name}] Chargement local : {model_path}")
        return tf.keras.models.load_model(model_path)

    logger.info(f"[{model_name}] Modèle absent localement — téléchargement depuis MLflow (run={run_id})")
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="model", dst_path=str(MODELS_DIR)
    )
    return tf.keras.models.load_model(local_path)


def _save_confusion_matrix(cm: np.ndarray, class_names: list, model_name: str) -> tuple[str, plt.Figure]:
    """Génère et sauvegarde la figure de la matrice de confusion. Retourne (chemin, figure)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ticks = range(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)

    ax.set_ylabel("Vraie classe")
    ax.set_xlabel("Classe prédite")
    ax.set_title(f"Matrice de confusion — {model_name}")
    fig.tight_layout()

    path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return str(path), fig


def evaluate_model(model_name: str, run_id: str, params: dict) -> dict:
    """Évalue un modèle sur le test set et logue les résultats dans son run MLflow."""
    img_size = tuple(params["model"]["img_size"])
    races = params["data"]["races"]
    batch_size = 32

    model = _load_model(model_name, run_id)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DATA_DIR / "test",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    ).prefetch(AUTOTUNE)

    # Métriques globales
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    logger.info(f"[{model_name}] test_accuracy={test_accuracy:.4f}  test_loss={test_loss:.4f}")

    # Prédictions pour la matrice de confusion et le rapport par classe
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=races, output_dict=True)
    report_txt = classification_report(y_true, y_pred, target_names=races)

    # Sauvegarde du rapport texte (écrasé à chaque run — source de vérité : MLflow)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"classification_report_{model_name}.txt"
    report_path.write_text(report_txt)

    # Sauvegarde de la matrice de confusion
    cm_path, cm_fig = _save_confusion_matrix(cm, races, model_name)

    # Logue dans le run MLflow existant
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
        })
        # Métriques par classe
        for race in races:
            mlflow.log_metrics({
                f"{race}_precision": report[race]["precision"],
                f"{race}_recall": report[race]["recall"],
                f"{race}_f1": report[race]["f1-score"],
            })
        mlflow.log_figure(cm_fig, f"figures/confusion_matrix_{model_name}.png")
        plt.close(cm_fig)
        mlflow.log_artifact(str(report_path), artifact_path="reports")

    return {
        "model": model_name,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "macro_f1": report["macro avg"]["f1-score"],
    }


@app.command()
def main(
    models: str = typer.Option(
        "all",
        help="Modèles à évaluer : all | cnn_scratch | efficientnet_b0 | inception_v3",
    ),
    run_ids_path: Optional[Path] = typer.Option(
        None,
        help="Chemin vers run_ids.json (défaut : reports/run_ids.json)",
    ),
):
    """Évalue les modèles sur le test set et logue les résultats dans MLflow."""
    params = load_params()
    races = params["data"]["races"]
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))

    ids_path = run_ids_path or (REPORTS_DIR / "run_ids.json")
    if not ids_path.exists():
        logger.error(f"run_ids.json introuvable : {ids_path}. Lance d'abord train.py.")
        raise typer.Exit(1)

    run_ids: dict = json.loads(ids_path.read_text())

    model_names = (
        ["cnn_scratch", "efficientnet_b0", "inception_v3"]
        if models == "all"
        else [models]
    )

    results = []
    for model_name in model_names:
        if model_name not in run_ids:
            logger.warning(f"[{model_name}] absent de run_ids.json — ignoré.")
            continue
        result = evaluate_model(model_name, run_ids[model_name], params)
        results.append(result)

    # Tableau comparatif
    if len(results) > 1:
        comparison_path = REPORTS_DIR / "model_comparison.json"
        comparison_path.write_text(json.dumps(results, indent=2))
        logger.info(f"Comparaison sauvegardée → {comparison_path}")

    logger.success("Évaluation terminée.")
    for r in results:
        logger.info(
            f"  {r['model']:<20} test_acc={r['test_accuracy']:.4f}  "
            f"macro_f1={r['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    app()
