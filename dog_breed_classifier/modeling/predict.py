"""Chargement du modèle depuis MLflow Model Registry et preprocessing pour l'inférence."""
import os

import numpy as np
from loguru import logger
from PIL import Image


def load_model():
    """Charge le modèle depuis MLflow Model Registry.

    Variables d'environnement requises :
        MLFLOW_TRACKING_URI  — URI du tracking server (DagsHub ou local)
        MLFLOW_MODEL_NAME    — nom du modèle enregistré (défaut : DogBreedClassifier)
        MLFLOW_MODEL_STAGE   — stage cible (défaut : Production)
    """
    import mlflow
    import mlflow.keras

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI non défini. "
            "Ajoute-le dans ton fichier .env ou dans les variables d'environnement."
        )

    mlflow.set_tracking_uri(mlflow_uri)
    model_name = os.getenv("MLFLOW_MODEL_NAME", "DogBreedClassifier")
    model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")

    logger.info(f"Chargement depuis MLflow Registry : {model_name} / {model_stage}")
    model = mlflow.keras.load_model(f"models:/{model_name}/{model_stage}")
    logger.success("Modèle chargé depuis MLflow Model Registry.")
    return model


def preprocess(image: Image.Image, img_size: tuple[int, int]) -> np.ndarray:
    """Resize et convertit en float32.

    Pas de normalisation ici : les modèles embarquent leur propre couche Rescaling.
    """
    arr = np.array(image.convert("RGB").resize(img_size), dtype=np.float32)
    return np.expand_dims(arr, axis=0)
