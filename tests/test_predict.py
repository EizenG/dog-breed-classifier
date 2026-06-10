"""Tests unitaires pour dog_breed_classifier/modeling/predict.py."""
import numpy as np
import pytest
from PIL import Image

from dog_breed_classifier.modeling.predict import preprocess


def test_preprocess_output_shape(sample_image):
    result = preprocess(sample_image, img_size=(224, 224))
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_output_dtype(sample_image):
    result = preprocess(sample_image, img_size=(224, 224))
    assert result.dtype == np.float32


def test_preprocess_no_normalization():
    """Les valeurs doivent rester dans [0, 255] — pas de division par 255."""
    bright_image = Image.new("RGB", (50, 50), color=(200, 200, 200))
    result = preprocess(bright_image, img_size=(224, 224))
    assert result.max() > 1.0, "Les pixels ne doivent pas être normalisés dans preprocess"


def test_preprocess_resizes_correctly():
    """L'image doit être redimensionnée à la taille demandée."""
    small_image = Image.new("RGB", (32, 32), color=(100, 100, 100))
    result = preprocess(small_image, img_size=(299, 299))
    assert result.shape == (1, 299, 299, 3)


def test_preprocess_converts_to_rgb():
    """Les images RGBA ou L doivent être converties en RGB."""
    gray_image = Image.new("L", (50, 50), color=128)
    result = preprocess(gray_image, img_size=(224, 224))
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_batch_dimension(sample_image):
    """La première dimension doit toujours être 1 (batch size)."""
    result = preprocess(sample_image, img_size=(224, 224))
    assert result.ndim == 4
    assert result.shape[0] == 1
