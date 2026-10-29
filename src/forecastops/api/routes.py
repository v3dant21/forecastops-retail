"""API routes for ForecastOps."""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from forecastops.api.schemas import (
    LoadModelRequest,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    EvaluateRequest,
    EvaluateResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse,
)
from forecastops.utils.logger import setup_logger
from forecastops.serving.predictor import ModelPredictor

logger = setup_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ForecastOps API"])

# Global predictor instance
predictor: Optional[ModelPredictor] = None


def ensure_active_forecast_source():
    """Never return an M5 forecast as a forecast for imported business invoices."""
    import json
    from pathlib import Path
    active = Path("reports/active_dataset.json")
    if active.exists() and json.loads(active.read_text()).get("source_type") == "retail_invoice":
        raise HTTPException(status_code=503, detail="Invoice sales are available for analysis; a validated invoice forecasting model is not yet available.")


def get_predictor() -> ModelPredictor:
    """Get the global predictor instance.
    
    Returns:
        ModelPredictor instance
        
    Raises:
        HTTPException: If predictor is not loaded
    """
    global predictor
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Call /load-model first."
        )
    return predictor


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint.
    
    Returns:
        Health status
    """
    global predictor
    
    return HealthResponse(
        status="healthy",
        timestamp=int(time.time()),
        model_loaded=predictor is not None,
        model_name=predictor.model_info.get("name") if predictor else None
    )


@router.post("/load-model", response_model=ModelInfo)
def load_model(
    request: Optional[LoadModelRequest] = None,
    model_path: Optional[str] = None
):
    """Load a model from path.
    
    Args:
        model_path: Path to model file
        feature_columns: Optional list of feature columns
        
    Returns:
        Model information
    """
    global predictor
    
    model_path = model_path or (request.model_path if request else "models/retail_model.pkl")
    feature_columns = request.feature_columns if request else None
    try:
        candidate = ModelPredictor(model_path)
        candidate.load()
        from forecastops.data.m5 import ARCHIVE_SHA256
        provenance = candidate.model_info.get("metadata", {}).get("provenance", {})
        if (provenance.get("source_type") != "real_m5"
                or provenance.get("archive_sha256") != ARCHIVE_SHA256):
            raise ValueError("Only models trained on verified M5 observations can be served")
        
        if feature_columns:
            if feature_columns != candidate.feature_columns:
                raise ValueError("Feature columns must match the trained model")
        
        info = candidate.get_model_info()
        logger.info(f"Model loaded: {info['name']}")
        
        response = ModelInfo(
            name=info["name"],
            version=info.get("metadata", {}).get("version", "1.0.0"),
            trained=info["trained"],
            feature_count=len(info.get("feature_columns", [])),
            feature_columns=info.get("feature_columns", []),
            target_column=info.get("target_column", "sales"),
            params=info.get("params", {})
        )
        predictor = candidate
        return response
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="Model artifact not found") from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Generate forecast prediction.
    
    Args:
        request: Prediction request
        
    Returns:
        Prediction response
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Call /load-model first."
        )
    
    ensure_active_forecast_source()
    start_time = time.time()
    
    try:
        # Combine features
        features = {**request.features}
        if request.product_id:
            features["product_id"] = request.product_id
        if request.store_id:
            features["store_id"] = request.store_id
        
        # Validate input
        if not predictor.validate_input(features):
            logger.warning(f"Invalid input features: {features}")
        
        # Make prediction
        features_df = pd.DataFrame([features])
        prediction = predictor.predict(features_df)[0]
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse.create(
            prediction=float(prediction),
            model_name=predictor.model_info.get("name", "unknown"),
            model_version=predictor.model_info.get("params", {}).get("version", "v1.0.0"),
            processing_time_ms=processing_time
        )
    
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest):
    """Generate batch predictions.
    
    Args:
        request: Batch prediction request
        
    Returns:
        Batch prediction response
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Call /load-model first."
        )
    
    ensure_active_forecast_source()
    start_time = time.time()
    
    try:
        # Make batch prediction
        features_df = pd.DataFrame(request.features_list)
        predictions = predictor.predict(features_df)
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            model_name=predictor.model_info.get("name", "unknown"),
            model_version=predictor.model_info.get("params", {}).get("version", "v1.0.0"),
            timestamp=int(time.time()),
            processing_time_ms=processing_time,
            total_predictions=len(predictions)
        )
    
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest):
    """Evaluate predictions against actual values.
    
    Args:
        request: Evaluation request
        
    Returns:
        Evaluation response
    """
    from forecastops.evaluation.metrics import calculate_all_metrics
    
    try:
        actual = request.actual
        predicted = request.predictions
        
        if not actual or not predicted:
            raise HTTPException(status_code=400, detail="Evaluation arrays cannot be empty")
        if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise HTTPException(status_code=400, detail="Evaluation values must be finite")
        if len(actual) != len(predicted):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Actual and predicted arrays must have the same length"
            )
        
        metrics = calculate_all_metrics(
            actual=np.array(actual),
            predicted=np.array(predicted)
        )
        
        return EvaluateResponse(
            metrics=metrics,
            model_name=request.model_name,
            model_version=request.model_version,
            timestamp=int(time.time())
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.get("/model/info", response_model=ModelInfo)
def get_model_info():
    """Get information about the loaded model.
    
    Returns:
        Model information
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Call /load-model first."
        )
    
    info = predictor.get_model_info()
    
    return ModelInfo(
        name=info["name"],
        version=info.get("params", {}).get("version", "v1.0.0"),
        trained=info["trained"],
        feature_count=len(info.get("feature_columns", [])),
        feature_columns=info.get("feature_columns", []),
        target_column=info.get("target_column", "sales"),
        params=info.get("params", {})
    )


@router.get("/model/list")
def list_models(models_dir: str = "models"):
    """List available models.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        List of model files
    """
    import os
    
    try:
        if not os.path.exists(models_dir):
            return {"models": []}
        
        model_files = []
        for f in os.listdir(models_dir):
            if f.endswith(".pkl"):
                model_files.append({
                    "name": f,
                    "path": os.path.join(models_dir, f),
                    "size_bytes": os.path.getsize(os.path.join(models_dir, f))
                })
        
        return {"models": model_files}
    
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )
