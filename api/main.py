"""API FastAPI — classification de races de chiens."""
import csv
import io
import json
import os
from contextlib import asynccontextmanager

import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from dog_breed_classifier.config import PROJ_ROOT, REPORTS_DIR
from dog_breed_classifier.features import extract_visual_features
from dog_breed_classifier.modeling.predict import load_model, preprocess
from dog_breed_classifier.monitoring import DRIFT_REPORTS_DIR, run_drift_report

PRODUCTION_FEATURES_CSV = REPORTS_DIR / "production_features.csv"

with open(PROJ_ROOT / "params.yaml") as f:
    _params = yaml.safe_load(f)
BREEDS: list[str] = _params["data"]["races"]
IMG_SIZE: tuple[int, int] = tuple(_params["model"]["img_size"])

model = None


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
        model = load_model()
    except Exception as e:
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


class FeatureDriftStats(BaseModel):
    drift_detected: bool
    stattest: str
    p_value: float


class DriftSummaryResponse(BaseModel):
    timestamp: str
    production_rows: int
    dataset_drift_detected: bool
    drifted_features: int
    total_features: int
    per_feature: dict[str, FeatureDriftStats]


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

    arr = preprocess(image, IMG_SIZE)
    preds = model.predict(arr, verbose=0)[0]

    top_indices = np.argsort(preds)[::-1][:3]
    top_3 = [{"breed": BREEDS[i], "confidence": round(float(preds[i]), 4)} for i in top_indices]

    features = extract_visual_features(image)
    _log_production_features(features)

    return {
        "breed": top_3[0]["breed"],
        "confidence": top_3[0]["confidence"],
        "top_3": top_3,
    }


def _latest_drift_summary() -> dict | None:
    """Retourne le contenu du dernier fichier drift_summary_*.json, ou None."""
    if not DRIFT_REPORTS_DIR.exists():
        return None
    summaries = sorted(DRIFT_REPORTS_DIR.glob("drift_summary_*.json"))
    if not summaries:
        return None
    with open(summaries[-1]) as f:
        return json.load(f)


@app.get(
    "/drift/summary",
    response_model=DriftSummaryResponse,
    summary="Dernier résumé de drift (cache)",
)
def drift_summary():
    """Retourne le dernier rapport de drift calculé sans relancer l'analyse."""
    summary = _latest_drift_summary()
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail="Aucun rapport de drift disponible. Appelle POST /drift/run d'abord.",
        )
    return summary


@app.post(
    "/drift/run",
    response_model=DriftSummaryResponse,
    summary="Déclenche une analyse de drift",
)
def drift_run(min_rows: int = 10):
    """Lance une analyse Evidently et retourne le résumé.

    - **min_rows** : nombre minimum de requêtes de production requises (défaut : 10).
    """
    try:
        run_drift_report(min_production_rows=min_rows, log_to_mlflow=False)
    except FileNotFoundError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors du calcul du drift : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur analyse drift : {e}")

    summary = _latest_drift_summary()
    if summary is None:
        raise HTTPException(status_code=500, detail="Rapport généré mais introuvable.")
    return summary
