"""
Weather Inference endpoints
"""

import logging
import re
from typing import Any, List, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.services.model_service import ModelService

logger = logging.getLogger(__name__)

router = APIRouter()


class PredictionRequest(BaseModel):
    """Request model for predictions
    
    Accepts features as: [temp, humidity, wind_speed, precipitation, pressure, uv_index, visibility, cloud_cover, season, location]
    where cloud_cover, season, location are categorical string values
    """

    model_config = {"protected_namespaces": ()}

    features: List[Union[float, str]] = Field(..., min_length=10, max_length=1000, description="Feature values: 7 numeric + 3 categorical")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""

    model_config = {"protected_namespaces": ()}

    features: List[List[Union[float, str]]] = Field(
        ..., min_length=1, max_length=1000, description="List of feature vectors: 7 numeric + 3 categorical each"
    )

class PredictionResponse(BaseModel):
    """Response model for predictions"""

    prediction: Any
    confidence: float = None


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Single prediction endpoint

    Args:
        request: PredictionRequest with 10 features (7 numeric + 3 categorical)

    Returns:
        PredictionResponse with prediction result
    """
    try:
        if not ModelService.are_models_loaded():
            raise HTTPException(
                status_code=503,
                detail="Models are not loaded. Please wait for initialization.",
            )

        prediction, confidence = ModelService.predict(features=request.features)

        return PredictionResponse(prediction=prediction, confidence=confidence)
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

        predictions = ModelService.predict_batch(features_list=request.features)

        return [
            PredictionResponse(prediction=pred, confidence=conf)
            for pred, conf in predictions
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


@router.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": ModelService.list_models(),
        "models_loaded": ModelService.are_models_loaded(),
    }
