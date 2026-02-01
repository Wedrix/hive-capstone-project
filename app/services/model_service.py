"""
Model service for loading and managing weather inference models
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Model paths
BASE_DIR = os.path.join(os.path.dirname(__file__), "../../models")
SCALER_FILENAME = "scaler.joblib"
SCALER_PATH = os.path.join(BASE_DIR, SCALER_FILENAME)


def _humanize_model_id(model_id: str) -> str:
    """Convert model_id (e.g. 'logistic_regression_model') to display name."""
    return model_id.replace("_", " ").strip().title() or model_id


def _discover_predictor_files() -> List[Tuple[str, str]]:
    """Scan models/ for .joblib files (exclude scaler). Return [(model_id, filename), ...]."""
    if not os.path.isdir(BASE_DIR):
        return []
    result = []
    for name in sorted(os.listdir(BASE_DIR)):
        if not name.endswith(".joblib"):
            continue
        if name == SCALER_FILENAME:
            continue
        stem = os.path.splitext(name)[0]
        if stem:
            result.append((stem, name))
    return result


class ModelService:
    """Service for managing weather inference models"""

    _models: Dict[str, Any] = {}  # model_id -> loaded predictor
    _model_names: Dict[str, str] = {}  # model_id -> display name
    _scaler: Any = None
    _models_loaded: bool = False
    _model_ready: bool = False  # At least one predictor + scaler ready
    _training_feature_cols: List[str] = None  # Columns used in training X (from first loaded model)

    # Feature column names matching training data
    NUMERICAL_COLS = [
        "Temperature",
        "Humidity",
        "Wind Speed",
        "Precipitation (%)",
        "Atmospheric Pressure",
        "UV Index",
        "Visibility (km)",
    ]
    CATEGORICAL_COLS = ["Cloud Cover", "Season", "Location"]

    # Valid categorical values
    VALID_CLOUD_COVER = {"cloudy", "clear", "partly cloudy", "overcast"}
    VALID_SEASON = {"spring", "summer", "autumn", "fall", "winter"}
    VALID_LOCATION = {"mountain", "coastal", "inland"}

    @classmethod
    def load_models(cls):
        """Load models and scaler from disk. Discovers *.joblib in models/ (skip scaler)."""
        try:
            cls._models = {}
            cls._model_names = {}
            cls._model_ready = False

            # Discover and load each predictor .joblib (exclude scaler.joblib)
            for model_id, filename in _discover_predictor_files():
                path = os.path.join(BASE_DIR, filename)
                try:
                    cls._models[model_id] = joblib.load(path)
                    cls._model_names[model_id] = _humanize_model_id(model_id)
                    cls._model_ready = True
                    logger.info(f"Model '{model_id}' loaded from {path}")
                except Exception as e:
                    logger.error(f"Error loading model {model_id} from {path}: {str(e)}")

            # Load scaler (shared)
            if os.path.exists(SCALER_PATH):
                try:
                    cls._scaler = joblib.load(SCALER_PATH)
                    logger.info(f"Scaler loaded from {SCALER_PATH}")
                except Exception as e:
                    logger.warning(f"Error loading scaler: {str(e)}. Scaler will be None.")
                    cls._scaler = None
            else:
                logger.info("Scaler file not found. Scaler will be None.")
                cls._scaler = None

            # Infer training feature columns from the first loaded model
            first_model = next(iter(cls._models.values()), None)
            if cls._model_ready and first_model is not None:
                try:
                    if hasattr(first_model, "n_features_in_"):
                        n_features = first_model.n_features_in_
                        logger.info(f"Model expects {n_features} features")
                        cls._training_feature_cols = cls.NUMERICAL_COLS.copy()
                        if n_features > 7:
                            encoded_cols_count = n_features - 7
                            for i in range(encoded_cols_count):
                                cls._training_feature_cols.append(f"encoded_{i}")
                            logger.info(
                                f"Inferred {len(cls._training_feature_cols)} total feature columns"
                            )
                except Exception as e:
                    logger.warning(f"Could not infer training columns: {str(e)}")
                    cls._training_feature_cols = None

            cls._models_loaded = True
            if cls._model_ready:
                logger.info(
                    f"Models and scaler loaded successfully. Available: {list(cls._models.keys())}"
                )
            else:
                logger.warning("No predictor model ready for predictions.")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            cls._model_ready = False
            cls._models_loaded = True

    @classmethod
    def get_available_models(cls) -> List[Dict[str, str]]:
        """Return list of available model ids and display names (only loaded ones)."""
        return [
            {"id": mid, "name": cls._model_names.get(mid, _humanize_model_id(mid))}
            for mid in cls._models
        ]

    @classmethod
    def get_model(cls, model_id: Optional[str] = None) -> Any:
        """Get predictor by id. If model_id is None or missing, return default (first available)."""
        if not cls._models:
            return None
        if model_id and model_id in cls._models:
            return cls._models[model_id]
        return next(iter(cls._models.values()))

    @classmethod
    def unload_models(cls):
        """Unload models from memory"""
        cls._models = {}
        cls._model_names = {}
        cls._scaler = None
        cls._models_loaded = False
        cls._model_ready = False
        logger.info("Models unloaded")

    @classmethod
    def are_models_loaded(cls) -> bool:
        """Check if real models are loaded and ready"""
        return cls._model_ready

    @classmethod
    def _preprocess_features(cls, features: List) -> np.ndarray:
        """
        Preprocess raw input features to match training preprocessing exactly

        Mirrors the training preprocessing:
        1. Create DataFrame with input features
        2. One-hot encode categorical columns (drop_first=True)
        3. Reindex to match training columns (fill missing with 0)
        4. Scale numerical features
        5. Ensure final column order matches training data

        Input: [temp, humidity, wind_speed, precipitation, pressure, uv_index,
        visibility, cloud_cover, season, location]

        Args:
            features: List of 10 features (7 numeric + 3 categorical)

        Returns:
            Preprocessed numpy array ready for model prediction

        Raises:
            ValueError: If features are invalid
        """
        if len(features) != 10:
            raise ValueError(f"Expected 10 features, got {len(features)}")

        # Extract numeric and categorical features
        numeric_values = []
        for i in range(7):
            try:
                val = float(features[i])
                if not np.isfinite(val):
                    raise ValueError(f"Feature {i} must be a finite number, got {features[i]}")
                numeric_values.append(val)
            except (ValueError, TypeError):
                raise ValueError(f"Feature {i} must be numeric, got {features[i]}")

        # Extract and validate categorical values
        cloud_cover = str(features[7]).strip().lower()
        season = str(features[8]).strip().lower()
        location = str(features[9]).strip().lower()

        # Validate categorical values
        if cloud_cover not in cls.VALID_CLOUD_COVER:
            raise ValueError(
                f"Invalid cloud_cover '{cloud_cover}'. Must be one of: {cls.VALID_CLOUD_COVER}"
            )
        if season not in cls.VALID_SEASON:
            raise ValueError(f"Invalid season '{season}'. Must be one of: {cls.VALID_SEASON}")
        if location not in cls.VALID_LOCATION:
            raise ValueError(f"Invalid location '{location}'. Must be one of: {cls.VALID_LOCATION}")

        # Step 1: Create DataFrame for preprocessing (matching training format)
        data_dict = {
            "Temperature": numeric_values[0],
            "Humidity": numeric_values[1],
            "Wind Speed": numeric_values[2],
            "Precipitation (%)": numeric_values[3],
            "Atmospheric Pressure": numeric_values[4],
            "UV Index": numeric_values[5],
            "Visibility (km)": numeric_values[6],
            "Cloud Cover": cloud_cover,
            "Season": season,
            "Location": location,
        }

        df = pd.DataFrame([data_dict])

        # Step 2: One-hot encode categorical features (drop_first=True to match training)
        df_encoded = pd.get_dummies(df[cls.CATEGORICAL_COLS], drop_first=True).astype(int)

        # Step 3: Reindex to match training columns
        # If scaler and model are loaded, we know the expected feature columns
        if cls._training_feature_cols is not None:
            # Get the one-hot encoded column names that should exist
            x_encoded_cols = [
                col
                for col in cls._training_feature_cols
                if col not in cls.NUMERICAL_COLS and col not in cls.CATEGORICAL_COLS
            ]

            # Fill missing one-hot columns with 0
            for col in x_encoded_cols:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0

            # Reorder to match training X columns
            df_encoded = df_encoded[x_encoded_cols]

        # Step 4: Scale numerical features (BEFORE concatenation, matching training)
        if cls._scaler is not None:
            df[cls.NUMERICAL_COLS] = cls._scaler.transform(df[cls.NUMERICAL_COLS])

        # Step 5: Concatenate numerical (scaled) and categorical (encoded) features
        df_processed = pd.concat(
            [df[cls.NUMERICAL_COLS].reset_index(drop=True), df_encoded.reset_index(drop=True)],
            axis=1,
        )

        # Final step: Ensure column order matches training data X
        if cls._training_feature_cols is not None:
            df_processed = df_processed[cls._training_feature_cols]

        return df_processed.values[0]  # Return 1D array

    @classmethod
    def predict(cls, features: List, model_id: Optional[str] = None) -> Tuple[str, float, str]:
        """
        Make a prediction for a single sample

        Args:
            features: List of 10 features (7 numeric + 3 categorical)
            model_id: Optional model id (e.g. 'model', 'logistic_regression_model').
                Default: first available.

        Returns:
            Tuple of (prediction, confidence, model_id_used)

        Raises:
            RuntimeError: If model is not ready for predictions
            ValueError: If features are invalid or model_id not found
        """
        if not cls.are_models_loaded():
            raise RuntimeError(
                "Model not available. Please add model files to the models/ directory."
            )

        model = cls.get_model(model_id)
        if model is None:
            raise RuntimeError("No predictor model loaded.")
        used_id = (
            model_id if (model_id and model_id in cls._models) else next(iter(cls._models.keys()))
        )

        # Preprocess features (includes validation, one-hot encoding, and scaling)
        preprocessed = cls._preprocess_features(features)

        # Make prediction
        predictions = model.predict([preprocessed])
        prediction = predictions[0] if isinstance(predictions, np.ndarray) else predictions

        # Get probabilities for confidence
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba([preprocessed])
            proba = probas[0] if isinstance(probas, np.ndarray) else probas
            confidence = float(np.max(proba))
        else:
            confidence = 1.0

        return str(prediction), confidence, used_id

    @classmethod
    def predict_batch(
        cls, features_list: List[List], model_id: Optional[str] = None
    ) -> List[Tuple[str, float, str]]:
        """
        Make predictions for multiple samples

        Args:
            features_list: List of feature lists
            model_id: Optional model id. Default: first available.

        Returns:
            List of tuples (prediction, confidence, model_id_used)

        Raises:
            RuntimeError: If model is not ready for predictions
            ValueError: If features are invalid
        """
        if not cls.are_models_loaded():
            raise RuntimeError(
                "Model not found or misconfigured. Please add model files to the models/ directory."
            )

        model = cls.get_model(model_id)
        if model is None:
            raise RuntimeError("No predictor model loaded.")
        used_id = (
            model_id if (model_id and model_id in cls._models) else next(iter(cls._models.keys()))
        )

        results = []
        preprocessed_list = []

        for features in features_list:
            preprocessed = cls._preprocess_features(features)
            preprocessed_list.append(preprocessed)

        predictions = model.predict(preprocessed_list)

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(preprocessed_list)
            confidences = [float(np.max(p)) for p in probas]
        else:
            confidences = [1.0] * len(predictions)

        for pred, conf in zip(predictions, confidences):
            results.append((str(pred), conf, used_id))

        return results
