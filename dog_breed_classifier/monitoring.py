"""Détection de data drift avec Evidently AI.

Compare les features visuelles de production avec la distribution de référence
(calculée sur le dataset d'entraînement).
"""
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from dog_breed_classifier.config import REPORTS_DIR

REFERENCE_CSV = REPORTS_DIR / "reference_features.csv"
PRODUCTION_CSV = REPORTS_DIR / "production_features.csv"
DRIFT_REPORTS_DIR = REPORTS_DIR / "drift"

FEATURE_COLUMNS = ["brightness", "contrast", "saturation", "R", "G", "B"]


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not REFERENCE_CSV.exists():
        raise FileNotFoundError(
            f"Fichier de référence introuvable : {REFERENCE_CSV}\n"
            "Exécute la cellule 'Save reference_features' dans le notebook EDA."
        )
    if not PRODUCTION_CSV.exists():
        raise FileNotFoundError(
            f"Pas encore de données de production : {PRODUCTION_CSV}\n"
            "Envoie au moins une requête /predict à l'API."
        )

    ref = pd.read_csv(REFERENCE_CSV, usecols=FEATURE_COLUMNS)
    prod = pd.read_csv(PRODUCTION_CSV, usecols=FEATURE_COLUMNS)

    logger.info(f"Référence : {len(ref)} lignes | Production : {len(prod)} lignes")
    return ref, prod


def run_drift_report(
    min_production_rows: int = 50,
    log_to_mlflow: bool = False,
) -> Path:
    """Génère un rapport HTML de drift et retourne son chemin.

    Args:
        min_production_rows: Nombre minimum de requêtes avant d'analyser le drift.
        log_to_mlflow: Si True, logue le rapport dans le run MLflow actif.

    Returns:
        Chemin vers le rapport HTML généré.
    """
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    ref, prod = _load_data()

    if len(prod) < min_production_rows:
        logger.warning(
            f"Seulement {len(prod)} requêtes de production "
            f"(minimum recommandé : {min_production_rows}). "
            "Le rapport peut ne pas être représentatif."
        )

    column_mapping = ColumnMapping(
        numerical_features=FEATURE_COLUMNS,
        target=None,
        prediction=None,
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=prod, column_mapping=column_mapping)

    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = DRIFT_REPORTS_DIR / f"drift_report_{timestamp}.html"
    report.save_html(str(report_path))
    logger.success(f"Rapport de drift sauvegardé : {report_path}")

    summary = _extract_drift_summary(report)
    summary_path = DRIFT_REPORTS_DIR / f"drift_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Résumé JSON : {summary_path}")

    if log_to_mlflow:
        _log_to_mlflow(report_path, summary)

    return report_path


def _extract_drift_summary(report) -> dict:
    """Extrait les métriques clés du rapport Evidently sous forme de dict."""
    results = report.as_dict()
    metrics = results.get("metrics", [])

    drift_detected = False
    n_drifted = 0
    n_features = 0
    per_feature = {}

    for metric in metrics:
        result = metric.get("result", {})
        if "dataset_drift" in result:
            drift_detected = result["dataset_drift"]
            n_drifted = result.get("number_of_drifted_columns", 0)
            n_features = result.get("number_of_columns", len(FEATURE_COLUMNS))
        if "drift_by_columns" in result:
            for col, col_result in result["drift_by_columns"].items():
                per_feature[col] = {
                    "drift_detected": col_result.get("drift_detected", False),
                    "stattest": col_result.get("stattest_name", ""),
                    "p_value": round(col_result.get("p_value", 1.0), 4),
                }

    return {
        "timestamp": datetime.now().isoformat(),
        "production_rows": int(pd.read_csv(PRODUCTION_CSV).shape[0]),
        "dataset_drift_detected": drift_detected,
        "drifted_features": n_drifted,
        "total_features": n_features,
        "per_feature": per_feature,
    }


def _log_to_mlflow(report_path: Path, summary: dict) -> None:
    """Logue le rapport et les métriques de drift dans le run MLflow actif."""
    try:
        import mlflow

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)

        with mlflow.start_run(run_name="drift_monitoring", nested=True):
            mlflow.log_artifact(str(report_path), artifact_path="drift_reports")
            mlflow.log_metric("dataset_drift_detected", int(summary["dataset_drift_detected"]))
            mlflow.log_metric("drifted_features", summary["drifted_features"])
            mlflow.log_metric("production_rows", summary["production_rows"])
            for feature, stats in summary.get("per_feature", {}).items():
                mlflow.log_metric(f"drift_pvalue_{feature}", stats["p_value"])

        logger.info("Métriques de drift loguées dans MLflow.")
    except Exception as e:
        logger.warning(f"Échec du log MLflow ({e}) — rapport local disponible.")


if __name__ == "__main__":
    report_path = run_drift_report(min_production_rows=10, log_to_mlflow=False)
    print(f"\nRapport disponible : {report_path}")
