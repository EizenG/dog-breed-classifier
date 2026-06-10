"""Chargement du modèle depuis MLflow Model Registry et preprocessing pour l'inférence."""
import os

from loguru import logger
import numpy as np
from PIL import Image


def load_model():
    """Charge le modèle depuis MLflow Model Registry.

    Variables d'environnement requises :
        MLFLOW_TRACKING_URI   — URI du tracking server (DagsHub ou local)
        MLFLOW_MODEL_NAME     — nom du modèle enregistré (ex: RaceClassificationInception)
        MLFLOW_MODEL_ALIAS    — alias cible (défaut: production)
    """
    import mlflow
    import mlflow.keras

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI non défini. "
            "Ajoute-le dans ton fichier .env."
        )

    mlflow.set_tracking_uri(tracking_uri)
    model_name = os.getenv("MLFLOW_MODEL_NAME", "RaceClassificationInception")
    model_alias = os.getenv("MLFLOW_MODEL_ALIAS", "production")

    model_uri = f"models:/{model_name}@{model_alias}"
    logger.info(f"Chargement depuis MLflow Registry : {model_uri}")
    model = mlflow.keras.load_model(model_uri)
    logger.success(f"Modèle chargé : {model_name}@{model_alias}")
    return model


def preprocess(image: Image.Image, img_size: tuple[int, int]) -> np.ndarray:
    """Resize et convertit en float32.

    Pas de normalisation ici : les modèles embarquent leur propre couche Rescaling.
    """
    arr = np.array(image.convert("RGB").resize(img_size), dtype=np.float32)
    return np.expand_dims(arr, axis=0)
