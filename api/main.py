"""API FastAPI — classification de races de chiens."""
import csv
import io
import os
from contextlib import asynccontextmanager

import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from loguru import logger
from PIL import Image, ImageStat
from pydantic import BaseModel

from dog_breed_classifier.config import MODELS_DIR, PROJ_ROOT, REPORTS_DIR

PARAMS_PATH = PROJ_ROOT / "params.yaml"
PRODUCTION_FEATURES_CSV = REPORTS_DIR / "production_features.csv"
IMG_SIZE = (224, 224)

with open(PARAMS_PATH) as f:
    _params = yaml.safe_load(f)
BREEDS: list[str] = _params["data"]["races"]

model = None


def _load_model():
    """Charge le modèle depuis MLflow Model Registry, fallback sur fichier local."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        try:
            import mlflow
            import mlflow.keras
            mlflow.set_tracking_uri(mlflow_uri)
            model_name = os.getenv("MLFLOW_MODEL_NAME", "DogBreedClassifier")
            model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
            loaded = mlflow.keras.load_model(f"models:/{model_name}/{model_stage}")
            logger.info("Modèle chargé depuis MLflow Model Registry.")
            return loaded
        except Exception as e:
            logger.warning(f"MLflow indisponible ({e}), bascule sur fichier local.")

    keras_files = sorted(MODELS_DIR.glob("*.keras"))
    if keras_files:
        import tensorflow as tf
        loaded = tf.keras.models.load_model(keras_files[-1])
        logger.info(f"Modèle chargé depuis fichier local : {keras_files[-1].name}")
        return loaded

    raise RuntimeError("Aucun modèle disponible (ni MLflow ni fichier .keras local).")


def _preprocess(image: Image.Image) -> np.ndarray:
    """Redimensionne et normalise une image PIL pour l'inférence."""
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _extract_features(image: Image.Image) -> dict:
    """Extrait les features visuelles utilisées par Evidently pour le drift."""
    rgb = image.convert("RGB").resize(IMG_SIZE)
    stat = ImageStat.Stat(rgb)
    r_mean, g_mean, b_mean = stat.mean
    brightness = sum(stat.mean) / 3
    contrast = sum(stat.stddev) / 3

    # Saturation calculée manuellement depuis les moyennes RGB normalisées
    r, g, b = r_mean / 255.0, g_mean / 255.0, b_mean / 255.0
    cmax, cmin = max(r, g, b), min(r, g, b)
    saturation = (cmax - cmin) / cmax if cmax > 0 else 0.0

    return {
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "saturation": round(saturation, 4),
        "r_mean": round(r_mean, 4),
        "g_mean": round(g_mean, 4),
        "b_mean": round(b_mean, 4),
    }


def _log_production_features(features: dict) -> None:
    """Ajoute les features de l'image reçue dans production_features.csv."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not PRODUCTION_FEATURES_CSV.exists()
    with open(PRODUCTION_FEATURES_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(features.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(features)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = _load_model()
    except RuntimeError as e:
        logger.warning(f"Modèle non chargé au démarrage : {e}")
    yield


app = FastAPI(
    title="Dog Breed Classifier API",
    description="API de classification de races de chiens — Stanford Dogs Dataset (10 races).",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    breed: str
    confidence: float
    top_3: list[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class BreedsResponse(BaseModel):
    breeds: list[str]
    count: int


@app.get("/health", response_model=HealthResponse, summary="Statut de l'API")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/breeds", response_model=BreedsResponse, summary="Liste des races reconnues")
def breeds():
    return {"breeds": BREEDS, "count": len(BREEDS)}


@app.post("/predict", response_model=PredictionResponse, summary="Prédiction de race")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Format non supporté. Envoyez un JPEG ou PNG.")

    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Image invalide ou corrompue.")

    arr = _preprocess(image)
    preds = model.predict(arr, verbose=0)[0]

    top_indices = np.argsort(preds)[::-1][:3]
    top_3 = [{"breed": BREEDS[i], "confidence": round(float(preds[i]), 4)} for i in top_indices]

    features = _extract_features(image)
    _log_production_features(features)

    return {
        "breed": top_3[0]["breed"],
        "confidence": top_3[0]["confidence"],
        "top_3": top_3,
    }
