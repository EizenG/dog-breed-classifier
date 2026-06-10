"""Tests de la couche données — chargement et split."""
from pathlib import Path

import pytest

from dog_breed_classifier.config import PROCESSED_DATA_DIR


def test_processed_splits_exist():
    """Les splits train/val/test doivent exister après le preprocessing."""
    for split in ("train", "val", "test"):
        assert (PROCESSED_DATA_DIR / split).exists(), (
            f"Split '{split}' introuvable dans {PROCESSED_DATA_DIR}. "
            "Lance d'abord : python -m dog_breed_classifier.features"
        )
