"""
Weather Inference endpoints
"""

import logging
from typing import Any, List, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.model_service import ModelService

logger = logging.getLogger(__name__)

router = APIRouter()


class PredictionRequest(BaseModel):
    """Request model for predictions.

    Features: temp, humidity, wind_speed, precipitation, pressure, uv_index,
    visibility, cloud_cover, season, location (7 numeric + 3 categorical).
    """

    model_config = {"protected_namespaces": ()}

    features: List[Union[float, str]] = Field(
        ..., min_length=10, max_length=1000, description="Feature values: 7 numeric + 3 categorical"
    )
    model_id: Optional[str] = (
        None  # e.g. 'model', 'logistic_regression_model'. Default: first available.
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""

    model_config = {"protected_namespaces": ()}

    features: List[List[Union[float, str]]] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of feature vectors: 7 numeric + 3 categorical each",
    )
    model_id: Optional[str] = None  # Which model to use for inference.


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    prediction: Any
    confidence: Optional[float] = None
    model_id: Optional[str] = None  # Model used for this prediction


@router.get("/models")
async def list_models():
    """
    List available inference models (ids and display names).
    """
    try:
        available = ModelService.get_available_models()
        return {
            "available_models": available,
            "models_loaded": ModelService.are_models_loaded(),
        }
    except Exception as e:
        logger.error(f"List models failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list models.")


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Single prediction endpoint

    Args:
        request: PredictionRequest with 10 features (7 numeric + 3 categorical), optional model_id

    Returns:
        PredictionResponse with prediction result and model_id used
    """
    try:
        if not ModelService.are_models_loaded():
            raise HTTPException(
                status_code=503,
                detail="Models are not loaded. Please wait for initialization.",
            )

        prediction, confidence, model_id = ModelService.predict(
            features=request.features, model_id=request.model_id or None
        )

        return PredictionResponse(prediction=prediction, confidence=confidence, model_id=model_id)
    except ValueError as e:
        # Input validation errors - safe to expose
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Don't expose internal error details in production
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again later."
        )


@router.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction endpoint

    Args:
        request: BatchPredictionRequest with list of 10-feature vectors

    Returns:
        List of PredictionResponse objects
    """
    try:
        if not ModelService.are_models_loaded():
            raise HTTPException(
                status_code=503,
                detail="Models are not loaded. Please wait for initialization.",
            )

        predictions = ModelService.predict_batch(
            features_list=request.features, model_id=request.model_id or None
        )

        return [
            PredictionResponse(prediction=pred, confidence=conf, model_id=mid)
            for pred, conf, mid in predictions
        ]
    except ValueError as e:
        # Input validation errors - safe to expose
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Don't expose internal error details in production
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again later."
        )
