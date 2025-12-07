"""
Model service for loading and managing weather inference models
"""

import logging
import os
import joblib
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Model paths
BASE_DIR = os.path.join(os.path.dirname(__file__), '../../models')
MODEL_PATH = os.path.join(BASE_DIR, 'model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.joblib')


class ModelService:
    """Service for managing weather inference models"""

    _model: Any = None
    _scaler: Any = None
    _models_loaded: bool = False
    _model_ready: bool = False  # Track if real model is loaded (not dummy)
    _training_feature_cols: List[str] = None  # Columns used in training X

    # Feature column names matching training data
    NUMERICAL_COLS = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 
                      'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
    CATEGORICAL_COLS = ['Cloud Cover', 'Season', 'Location']
    
    # Valid categorical values
    VALID_CLOUD_COVER = {'cloudy', 'clear', 'partly cloudy', 'overcast'}
    VALID_SEASON = {'spring', 'summer', 'autumn', 'fall', 'winter'}
    VALID_LOCATION = {'mountain', 'coastal', 'inland'}

    @classmethod
    def load_models(cls):
        """Load the weather inference model and scaler from disk"""
        try:
            # Load model
            if os.path.exists(MODEL_PATH):
                try:
                    cls._model = joblib.load(MODEL_PATH)
                    cls._model_ready = True
                    logger.info(f"Model loaded from {MODEL_PATH}")
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
                    cls._model_ready = False
            else:
                logger.warning(f"Model file not found at {MODEL_PATH}")
                cls._model_ready = False

            # Load scaler
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

            # Try to infer training feature columns from the model's expected input
            # This helps us align preprocessing output with what the model expects
            if cls._model_ready and cls._model is not None:
                try:
                    # For sklearn models, check n_features_in_
                    if hasattr(cls._model, 'n_features_in_'):
                        n_features = cls._model.n_features_in_
                        logger.info(f"Model expects {n_features} features")
                        
                        # Infer column names: 7 numerical + (n_features - 7) categorical encoded
                        # Build list of expected categorical encoded columns
                        # This is a placeholder - ideally saved with the model
                        cls._training_feature_cols = cls.NUMERICAL_COLS.copy()
                        
                        # Add one-hot encoded categorical columns
                        # Assuming: Cloud Cover (2), Season (4), Location (2) with drop_first=True
                        # Based on standard one-hot encoding
                        if n_features > 7:
                            # Add placeholder encoded column names
                            encoded_cols_count = n_features - 7
                            for i in range(encoded_cols_count):
                                cls._training_feature_cols.append(f"encoded_{i}")
                            logger.info(f"Inferred {len(cls._training_feature_cols)} total feature columns")
                except Exception as e:
                    logger.warning(f"Could not infer training columns: {str(e)}")
                    cls._training_feature_cols = None

            cls._models_loaded = True
            if cls._model_ready:
                logger.info("Model and scaler loaded successfully.")
            else:
                logger.warning("Model not ready for predictions.")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            cls._model_ready = False
            cls._models_loaded = True

    @classmethod
    def unload_models(cls):
        """Unload models from memory"""
        cls._model = None
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
        
        Input: [temp, humidity, wind_speed, precipitation, pressure, uv_index, visibility, cloud_cover, season, location]
        
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
            raise ValueError(f"Invalid cloud_cover '{cloud_cover}'. Must be one of: {cls.VALID_CLOUD_COVER}")
        if season not in cls.VALID_SEASON:
            raise ValueError(f"Invalid season '{season}'. Must be one of: {cls.VALID_SEASON}")
        if location not in cls.VALID_LOCATION:
            raise ValueError(f"Invalid location '{location}'. Must be one of: {cls.VALID_LOCATION}")
        
        # Step 1: Create DataFrame for preprocessing (matching training format)
        data_dict = {
            'Temperature': numeric_values[0],
            'Humidity': numeric_values[1],
            'Wind Speed': numeric_values[2],
            'Precipitation (%)': numeric_values[3],
            'Atmospheric Pressure': numeric_values[4],
            'UV Index': numeric_values[5],
            'Visibility (km)': numeric_values[6],
            'Cloud Cover': cloud_cover,
            'Season': season,
            'Location': location
        }
        
        df = pd.DataFrame([data_dict])
        
        # Step 2: One-hot encode categorical features (drop_first=True to match training)
        df_encoded = pd.get_dummies(df[cls.CATEGORICAL_COLS], drop_first=True).astype(int)
        
        # Step 3: Reindex to match training columns
        # If scaler and model are loaded, we know the expected feature columns
        if cls._training_feature_cols is not None:
            # Get the one-hot encoded column names that should exist
            x_encoded_cols = [col for col in cls._training_feature_cols 
                            if col not in cls.NUMERICAL_COLS and col not in cls.CATEGORICAL_COLS]
            
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
        df_processed = pd.concat([df[cls.NUMERICAL_COLS].reset_index(drop=True), 
                                 df_encoded.reset_index(drop=True)], axis=1)
        
        # Final step: Ensure column order matches training data X
        if cls._training_feature_cols is not None:
            df_processed = df_processed[cls._training_feature_cols]
        
        return df_processed.values[0]  # Return 1D array

    @classmethod
    def predict(cls, features: List) -> Tuple[str, float]:
        """
        Make a prediction for a single sample

        Args:
            features: List of 10 features (7 numeric + 3 categorical)

        Returns:
            Tuple of (prediction, confidence)

        Raises:
            RuntimeError: If model is not ready for predictions
            ValueError: If features are invalid
        """
        if not cls.are_models_loaded():
            raise RuntimeError("Model not available. Please upload model.joblib and scaler.joblib to the models/ directory.")

        # Preprocess features (includes validation, one-hot encoding, and scaling)
        preprocessed = cls._preprocess_features(features)
        
        # Make prediction
        predictions = cls._model.predict([preprocessed])
        # Extract scalar from array
        prediction = predictions[0] if isinstance(predictions, np.ndarray) else predictions
        
        # Get probabilities for confidence
        if hasattr(cls._model, 'predict_proba'):
            probas = cls._model.predict_proba([preprocessed])
            proba = probas[0] if isinstance(probas, np.ndarray) else probas
            confidence = float(np.max(proba))
        else:
            confidence = 1.0
        
        return str(prediction), confidence

    @classmethod
    def predict_batch(cls, features_list: List[List]) -> List[Tuple[str, float]]:
        """
        Make predictions for multiple samples

        Args:
            features_list: List of feature lists

        Returns:
            List of tuples (prediction, confidence)

        Raises:
            RuntimeError: If model is not ready for predictions
            ValueError: If features are invalid
        """
        if not cls.are_models_loaded():
            raise RuntimeError("Model not found or misconfigured. Please add model.joblib and scaler.joblib to the root directory.")

        results = []
        preprocessed_list = []
        
        # Preprocess all features
        for features in features_list:
            preprocessed = cls._preprocess_features(features)
            preprocessed_list.append(preprocessed)
        
        # Make predictions
        predictions = cls._model.predict(preprocessed_list)
        
        # Get probabilities for confidence
        if hasattr(cls._model, 'predict_proba'):
            probas = cls._model.predict_proba(preprocessed_list)
            confidences = [float(np.max(p)) for p in probas]
        else:
            confidences = [1.0] * len(predictions)
        
        for pred, conf in zip(predictions, confidences):
            results.append((str(pred), conf))
        
        return results

    @classmethod
    def list_models(cls) -> List[str]:
        """List all available model names"""
        return ["default"]
