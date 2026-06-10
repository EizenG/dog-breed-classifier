"""Fixtures partagées entre tous les tests."""
import io

import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock


@pytest.fixture
def sample_image() -> Image.Image:
    """Image PIL RGB 100x100 avec des pixels variés."""
    arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_image_bytes(sample_image) -> bytes:
    """Image PIL encodée en JPEG (pour simuler un upload)."""
    buf = io.BytesIO()
    sample_image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_model() -> MagicMock:
    """Modèle factice retournant des probabilités uniformes sur 10 races."""
    model = MagicMock()
    model.predict.return_value = np.array([[0.1] * 10])
    return model
