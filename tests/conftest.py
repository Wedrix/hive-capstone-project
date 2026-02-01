"""
Pytest configuration and shared fixtures
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.model_service import ModelService


@pytest.fixture(scope="function")
def client():
    """Create a test client for the FastAPI app"""
    # Ensure models are loaded for testing
    if not ModelService.are_models_loaded():
        ModelService.load_models()

    with TestClient(app) as test_client:
        yield test_client

    # Cleanup after each test
    ModelService.unload_models()


@pytest.fixture(scope="function")
def sample_features():
    """Sample feature vector for testing (10 features: 7 numeric + 3 categorical)"""
    return [25.0, 60.0, 10.0, 0.0, 1013.25, 5.0, 10.0, "cloudy", "summer", "inland"]


@pytest.fixture(scope="function")
def sample_batch_features():
    """Sample batch feature vectors for testing (10 features each)"""
    return [
        [25.0, 60.0, 10.0, 0.0, 1013.25, 5.0, 10.0, "cloudy", "summer", "inland"],
        [15.0, 80.0, 25.0, 50.0, 1005.0, 2.0, 5.0, "overcast", "spring", "mountain"],
    ]
