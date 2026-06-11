---
title: Dog Breed Classifier
emoji: 🐶
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---


# 🐕 Dog Breed Classifier — MLOps Pipeline

[![CI](https://github.com/EizenG/dog-breed-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/EizenG/dog-breed-classifier/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![DVC](https://img.shields.io/badge/DVC-pipeline-945DD6)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)](https://streamlit.io/)

Système de classification automatique de races de chiens basé sur le Stanford Dogs Dataset.  
Projet MLOps — cours Mously DIAW.

---

## LIEN

### Streamlit (UI)
https://ba-ameth-dog-breed-classifier-ui.hf.space/

### Documentation de l'API
https://ba-ameth-dog-breed-classifier.hf.space/docs

### MLflow Tracking (DagsHub)
https://dagshub.com/EizenG/dog-breed-classifier.mlflow

## Résultats

| Modèle | Test Accuracy | Test Loss | Macro F1 |
|---|---|---|---|
| InceptionV3 *(prod)* | **97.9 %** | 0.097 | 0.979 |
| EfficientNetB0 | 96.3 % | 0.178 | 0.962 |
| CNN scratch | 10.6 % | 2.361 | 0.031 |

10 races reconnues : Chihuahua · German Shepherd · Golden Retriever · Siberian Husky · Pug · Rottweiler · French Bulldog · Old English Sheepdog · Doberman · Standard Poodle

---

## Architecture

```
┌─────────────┐    HTTP    ┌──────────────────┐    MLflow    ┌──────────────────┐
│  Streamlit  │ ─────────► │    FastAPI        │ ──────────► │ DagsHub Registry │
│  HF Spaces  │            │  /predict         │             │ InceptionV3      │
└─────────────┘            │  /drift/summary   │             │ @production      │
                           │  /drift/run       │             └──────────────────┘
                           └──────────────────┘
                                    │
                            reports/
                            ├── reference_features.csv
                            ├── production_features.csv
                            └── drift/
                                ├── drift_report_*.html   (Evidently)
                                └── drift_summary_*.json
```

## Stack technique

| Couche | Technologie |
|---|---|
| Langage | Python 3.12 |
| Gestion dépendances | Poetry |
| Pipeline données | DVC + DagsHub |
| Tracking expériences | MLflow + DagsHub |
| Deep Learning | TensorFlow / Keras |
| API | FastAPI + Uvicorn |
| Interface | Streamlit |
| Monitoring | Evidently AI |
| Conteneurisation | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Tests | pytest (23 tests) |

---

## Structure du projet

```
dog-breed-classifier/
├── .github/workflows/
│   └── ci.yml                    
├── api/
│   └── main.py                   
├── app.py                        
├── pages/
│   ├── 1_Prediction.py           
│   └── 2_Monitoring.py           
├── dog_breed_classifier/
│   ├── config.py                 
│   ├── dataset.py                
│   ├── features.py               
│   ├── monitoring.py             
│   └── modeling/
│       ├── train.py              
│       ├── evaluate.py           
│       └── predict.py            
├── notebooks/
│   └── 01_exploration.ipynb      
├── tests/
│   ├── conftest.py
│   ├── test_api.py               
│   ├── test_features.py          
│   ├── test_predict.py           
│   └── test_data.py              
├── dvc.yaml                      
├── params.yaml                   
├── Dockerfile                    
├── docker-compose.yml            
└── requirements.txt              
```

---

## Installation

```bash
# Cloner le repo
git clone https://github.com/EizenG/dog-breed-classifier.git
cd dog-breed-classifier

# Installer les dépendances
poetry install

# Configurer les variables d'environnement dans le fichier .env

```

### Variables d'environnement (`.env`)

```env
MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<username>
MLFLOW_TRACKING_PASSWORD=<token>
MLFLOW_MODEL_NAME=RaceClassificationInception
MLFLOW_MODEL_ALIAS=production
API_URL=http://localhost:7860          # pour Streamlit local
```

---

## Pipeline DVC

```bash
# Récupérer les données trackées
poetry run dvc pull data/raw/selected.dvc

# Lancer le pipeline complet (features → train → evaluate)
poetry run dvc repro

```

**Stages :**

| Stage | Commande | Entrées | Sorties |
|---|---|---|---|
| `features` | `poetry run python -m dog_breed_classifier.features` | `data/raw/selected` | `data/processed/` |
| `train` | `poetry run python -m dog_breed_classifier.modeling.train` | `data/processed/` | `reports/run_ids.json` + modèles → MLflow |
| `evaluate` | `poetry run python -m dog_breed_classifier.modeling.evaluate` | `run_ids.json` + `data/processed/test` | `reports/model_comparison.json` |

---

## Lancement local

### API

```bash
poetry run uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload
```

### Streamlit

```powershell
$env:API_URL = "http://localhost:7860"
poetry run streamlit run app.py
```


L'API est disponible sur `http://localhost:7860`.  
Documentation interactive : `http://localhost:7860/docs`

---

## Endpoints API

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/health` | Statut de l'API et du modèle |
| `GET` | `/breeds` | Liste des 10 races reconnues |
| `POST` | `/predict` | Prédiction depuis une image JPEG/PNG |
| `GET` | `/drift/summary` | Dernier rapport de drift (cache JSON) |
| `POST` | `/drift/run` | Déclenche une analyse Evidently |

---

## Tests

```bash
poetry run pytest tests/ -v
# 23 passed
```

```bash
# Avec couverture
poetry run pytest tests/ --cov=dog_breed_classifier --cov=api --cov-report=term-missing
```

---

## Déploiement

- API — Docker deployer sur HuggingFace Spaces

- Streamlit deployer sur HuggingFace Spaces


---

## Monitoring

Le monitoring de data drift compare la distribution des features visuelles de production (images reçues par `/predict`) avec la distribution de référence calculée sur le dataset d'entraînement.

**Features surveillées :** `brightness`, `contrast`, `saturation`, `R`, `G`, `B`

**Workflow :**
1. Chaque appel à `/predict` ajoute une ligne dans `reports/production_features.csv`
2. Appel à `POST /drift/run` → Evidently compare production vs référence
3. Résultat disponible via `GET /drift/summary` (JSON) ou rapport HTML dans `reports/drift/`
