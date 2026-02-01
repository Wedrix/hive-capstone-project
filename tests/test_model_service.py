"""
Tests for model service
"""

from app.services.model_service import ModelService


class TestModelService:
    """Test cases for ModelService"""

    def setup_method(self):
        """Setup before each test method"""
        ModelService.unload_models()
        ModelService.load_models()

    def teardown_method(self):
        """Cleanup after each test method"""
        ModelService.unload_models()

    def test_load_models(self):
        """Test that models can be loaded"""
        ModelService.unload_models()
        assert not ModelService.are_models_loaded()

        ModelService.load_models()
        assert ModelService.are_models_loaded()

    def test_unload_models(self):
        """Test that models can be unloaded"""
        ModelService.load_models()
        assert ModelService.are_models_loaded()

        ModelService.unload_models()
        assert not ModelService.are_models_loaded()

    def test_are_models_loaded(self):
        """Test model loading status check"""
        ModelService.unload_models()
        assert ModelService.are_models_loaded() is False

        ModelService.load_models()
        assert ModelService.are_models_loaded() is True

    def test_predict_single(self):
        """Test single prediction"""
        # 10 features: 7 numeric + 3 categorical (cloud_cover, season, location)
        features = [25.0, 60.0, 10.0, 0.0, 1013.25, 5.0, 10.0, "cloudy", "summer", "inland"]
        prediction, confidence, model_id = ModelService.predict(features)

        assert prediction is not None
        assert isinstance(prediction, (int, float, str, list))
        assert confidence is None or isinstance(confidence, float)
        assert model_id is not None and isinstance(model_id, str)

    def test_predict_batch(self):
        """Test batch prediction"""
        features_list = [
            [25.0, 60.0, 10.0, 0.0, 1013.25, 5.0, 10.0, "cloudy", "summer", "inland"],
            [15.0, 80.0, 25.0, 50.0, 1005.0, 2.0, 5.0, "overcast", "spring", "mountain"],
        ]
        predictions = ModelService.predict_batch(features_list)

        assert len(predictions) == len(features_list)
        for pred, conf, model_id in predictions:
            assert pred is not None
            assert isinstance(pred, (int, float, str, list))
            assert conf is None or isinstance(conf, float)
            assert model_id is not None and isinstance(model_id, str)
