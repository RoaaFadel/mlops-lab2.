import pytest
from fastapi.testclient import TestClient

from src.app.main import app
from src.train.train import MODEL_PATH, train


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Make sure a trained model exists before any test runs."""
    if not MODEL_PATH.exists():
        train()
    return MODEL_PATH


@pytest.fixture()
def client():
    # The `with` block runs the app startup, which loads the model.
    with TestClient(app) as c:
        yield c
