---
title: Dog Breed Classifier
emoji: 🐶
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---


# Dog Breed Classifier

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Poetry](https://img.shields.io/badge/Poetry-dependency%20manager-purple)
![DVC](https://img.shields.io/badge/DVC-data%20versioning-945DD6)
![MLflow](https://img.shields.io/badge/MLflow-experiment%20tracking-0194E2)

Système de classification automatique de races de chiens basé sur le Stanford Dogs Dataset.
Le modèle identifie 10 races parmi les 120 disponibles, exposé via une API FastAPI et un tableau de bord Streamlit.

---

## Stack technique

| Composant | Outil |
|-----------|-------|
| Langage | Python 3.12 |
| Dépendances | Poetry |
| Versioning données | DVC + DagsHub |
| Expériences | MLflow (hébergé sur DagsHub) |
| Deep Learning | TensorFlow / Keras |
| Optimisation hyperparamètres | Optuna |
| API | FastAPI |
| Dashboard monitoring | Streamlit + Evidently AI |
| CI/CD | GitHub Actions |
| Déploiement | Hugging Face Spaces |

---

## Structure du projet

```
dog-breed-classifier/
├── .dvc/                        # Configuration DVC
├── dvc.yaml                     # Pipeline DVC (stages reproductibles)
├── params.yaml                  # Paramètres du projet (races, split, etc.)
├── pyproject.toml               # Dépendances Poetry et config outils
│
├── data/
│   ├── raw/
│   │   ├── selected/            # 10 races sélectionnées (~1 600 images) — géré par DVC
│   │   └── selected.dvc         # Pointeur DVC vers DagsHub
│   └── processed/               # Images splitées train/val/test — généré par dvc repro
│
├── dog_breed_classifier/
│   ├── config.py                # Chemins et variables globales
│   ├── dataset.py               # Téléchargement et filtrage Stanford Dogs Dataset
│   ├── features.py              # Split stratifié 70/15/15
│   └── modeling/
│       ├── train.py             # Entraînement des modèles + MLflow
│       └── predict.py           # Inférence
│
├── notebooks/                   # Exploration et prototypage
├── models/                      # Modèles entraînés (.keras) — géré par DVC
├── reports/                     # Métriques, figures, rapport d'évaluation
└── docs/                        # Documentation technique
```

---

## Installation

### Prérequis

- Python 3.11 ou 3.12
- [Poetry](https://python-poetry.org/docs/#installation)
- [DVC](https://dvc.org/doc/install) (installé automatiquement via Poetry)

### Cloner et installer

```bash
git clone https://github.com/EizenG/dog-breed-classifier.git
cd dog-breed-classifier
poetry install
```

### Configuration

Créer un fichier `.env` à la racine (ne jamais le commiter) :

```
MLFLOW_TRACKING_URI=https://dagshub.com/EizenG/dog-breed-classifier.mlflow
MLFLOW_TRACKING_USERNAME=<votre_username_dagshub>
MLFLOW_TRACKING_PASSWORD=<votre_token_dagshub>
DAGSHUB_USER=EizenG
DAGSHUB_TOKEN=<votre_token_dagshub>
```

Le token DagsHub se génère depuis : **DagsHub → Settings → Tokens**.

---

## Pipeline de données

### 1. Récupérer les données brutes

Les données brutes (10 races sélectionnées, ~1 600 images) sont versionnées sur DagsHub via DVC.

```bash
dvc pull data/raw/selected.dvc
```

Cette commande télécharge uniquement `data/raw/selected/` depuis DagsHub. Les credentials `.env` doivent être configurés au préalable.

> `dvc pull` sans argument tente aussi de récupérer les outputs du pipeline (`data/processed/`), qui ne sont pas stockés sur DagsHub — utiliser la commande ciblée ci-dessus.

### 2. Reproduire le pipeline

```bash
poetry run dvc repro
```

Exécute les stages définis dans `dvc.yaml` dans l'ordre :

| Stage | Script | Entrée | Sortie |
|-------|--------|--------|--------|
| `features` | `features.py` | `data/raw/selected/` | `data/processed/` |

`data/processed/` contient les images organisées en trois splits :

```
data/processed/
├── train/    # 70% — 1 120 images
├── val/      # 15% — 240 images
└── test/     # 15% — 240 images
```

Chaque sous-dossier contient un répertoire par race. Le split est **stratifié** : chaque race est représentée proportionnellement dans les trois splits.

### Races sélectionnées

| Race | Race |
|------|------|
| Chihuahua | German Shepherd |
| Golden Retriever | Siberian Husky |
| Pug | Rottweiler |
| French Bulldog | Old English Sheepdog |
| Doberman | Standard Poodle |

---

## Paramètres (`params.yaml`)

Les paramètres du projet sont centralisés dans `params.yaml` et lus par DVC pour détecter les changements :

```yaml
data:
  races: [...]       # 10 races sélectionnées
  split:
    train: 0.70
    val: 0.15
    test: 0.15
```

Modifier un paramètre puis relancer `dvc repro` régénère automatiquement les étapes affectées.

---

## Workflow Git

```
main                  # production
└── develop           # intégration
    ├── feature/data-pipeline
    ├── feature/eda
    ├── feature/training
    └── feature/api
```

Toute nouvelle fonctionnalité passe par une branche `feature/` puis une Pull Request vers `develop`.

---

## Roadmap

- [x] Structure du projet (Cookiecutter Data Science)
- [x] Versioning des données (DVC + DagsHub)
- [x] Pipeline de données (dataset.py + features.py)
- [ ] EDA (notebooks/01_exploration.ipynb)
- [ ] Entraînement des modèles (train.py + MLflow + Optuna)
- [ ] Évaluation (evaluate.py)
- [ ] API FastAPI
- [ ] Tests automatisés
- [ ] Docker + CI/CD GitHub Actions
- [ ] Déploiement Hugging Face Spaces
- [ ] Dashboard Streamlit + Evidently AI
