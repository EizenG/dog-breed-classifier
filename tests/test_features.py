"""Tests unitaires pour dog_breed_classifier/features.py."""
import numpy as np
import pytest
from PIL import Image

from dog_breed_classifier.features import extract_visual_features

EXPECTED_KEYS = {"brightness", "contrast", "saturation", "R", "G", "B"}


def test_extract_visual_features_keys(sample_image):
    result = extract_visual_features(sample_image)
    assert set(result.keys()) == EXPECTED_KEYS


def test_extract_visual_features_values_in_range(sample_image):
    result = extract_visual_features(sample_image)
    for key, value in result.items():
        assert 0.0 <= value <= 1.0, f"{key}={value} hors de [0, 1]"


def test_extract_visual_features_rounded(sample_image):
    result = extract_visual_features(sample_image)
    for key, value in result.items():
        assert value == round(value, 4), f"{key} n'est pas arrondi à 4 décimales"


def test_extract_visual_features_pure_red():
    """Image rouge pur → R élevé, G et B proches de 0."""
    red_image = Image.new("RGB", (50, 50), color=(255, 0, 0))
    result = extract_visual_features(red_image)
    assert result["R"] > 0.9
    assert result["G"] < 0.05
    assert result["B"] < 0.05


def test_extract_visual_features_pure_white():
    """Image blanche → brightness proche de 1, contrast proche de 0."""
    white_image = Image.new("RGB", (50, 50), color=(255, 255, 255))
    result = extract_visual_features(white_image)
    assert result["brightness"] > 0.95
    assert result["contrast"] < 0.05


def test_extract_visual_features_pure_black():
    """Image noire → brightness proche de 0."""
    black_image = Image.new("RGB", (50, 50), color=(0, 0, 0))
    result = extract_visual_features(black_image)
    assert result["brightness"] < 0.05


def test_extract_visual_features_accepts_non_rgb():
    """La fonction doit convertir les images non-RGB sans erreur."""
    rgba_image = Image.new("RGBA", (50, 50), color=(100, 150, 200, 128))
    result = extract_visual_features(rgba_image)
    assert set(result.keys()) == EXPECTED_KEYS
