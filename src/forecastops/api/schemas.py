"""Pydantic schemas for ForecastOps API."""

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    
    features: Dict[str, Any] = Field(
        ..., 
        description="Feature values for prediction"
    )
    product_id: Optional[str] = Field(
        None,
        description="Product ID (if not in features)"
    )
    store_id: Optional[str] = Field(
        None,
        description="Store ID (if not in features)"
    )
    model_version: Optional[str] = Field(
        None,
        description="Specific model version to use"
    )


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    
    prediction: float = Field(
        ..., 
        description="Predicted sales value"
    )
    model_name: str = Field(
        ..., 
        description="Model used for prediction"
    )
    model_version: str = Field(
        ..., 
        description="Model version used"
    )
    timestamp: int = Field(
        ..., 
        description="Timestamp of prediction"
    )
    processing_time_ms: float = Field(
        ..., 
        description="Time taken for prediction in milliseconds"
    )
    
    @classmethod
    def create(
        cls,
        prediction: float,
        model_name: str,
        model_version: str,
        processing_time_ms: float
    ) -> "PredictionResponse":
        """Create a response instance.
        
        Args:
            prediction: Predicted value
            model_name: Model name
            model_version: Model version
            processing_time_ms: Processing time
            
        Returns:
            Response instance
        """
        return cls(
            prediction=prediction,
            model_name=model_name,
            model_version=model_version,
            timestamp=int(time.time()),
            processing_time_ms=processing_time_ms
        )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction endpoint."""
    
    features_list: List[Dict[str, Any]] = Field(
        ..., 
        description="List of feature dictionaries"
    )
    model_version: Optional[str] = Field(
        None,
        description="Specific model version to use"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction endpoint."""
    
    predictions: List[float] = Field(
        ..., 
        description="List of predicted values"
    )
    model_name: str = Field(
        ..., 
        description="Model used for prediction"
    )
    model_version: str = Field(
        ..., 
        description="Model version used"
    )
    timestamp: int = Field(
        ..., 
        description="Timestamp of prediction"
    )
    processing_time_ms: float = Field(
        ..., 
        description="Time taken for batch prediction"
    )
    total_predictions: int = Field(
        ..., 
        description="Number of predictions made"
    )


class EvaluateRequest(BaseModel):
    """Request schema for evaluation endpoint."""
    
    predictions: List[float] = Field(
        ..., 
        description="Predicted values"
    )
    actual: List[float] = Field(
        ..., 
        description="Actual values"
    )
    model_name: Optional[str] = Field(
        None,
        description="Model name"
    )
    model_version: Optional[str] = Field(
        None,
        description="Model version"
    )


class EvaluateResponse(BaseModel):
    """Response schema for evaluation endpoint."""
    
    metrics: Dict[str, float] = Field(
        ..., 
        description="Evaluation metrics"
    )
    model_name: Optional[str] = Field(
        None,
        description="Model name"
    )
    model_version: Optional[str] = Field(
        None,
        description="Model version"
    )
    timestamp: int = Field(
        ..., 
        description="Timestamp of evaluation"
    )


class ModelInfo(BaseModel):
    """Model information schema."""
    
    name: str = Field(
        ..., 
        description="Model name"
    )
    version: str = Field(
        ..., 
        description="Model version"
    )
    trained: bool = Field(
        ..., 
        description="Whether model is trained"
    )
    feature_count: int = Field(
        ..., 
        description="Number of features"
    )
    feature_columns: List[str] = Field(
        ..., 
        description="List of feature column names"
    )
    target_column: str = Field(
        ..., 
        description="Target column name"
    )
    params: Dict[str, Any] = Field(
        ..., 
        description="Model parameters"
    )


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(
        ..., 
        description="Health status"
    )
    timestamp: int = Field(
        ..., 
        description="Timestamp of health check"
    )
    model_loaded: bool = Field(
        ..., 
        description="Whether model is loaded"
    )
    model_name: Optional[str] = Field(
        None,
        description="Loaded model name"
    )


class ErrorDetails(BaseModel):
    """Error details schema."""
    
    error: str = Field(
        ..., 
        description="Error message"
    )
    code: str = Field(
        ..., 
        description="Error code"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: ErrorDetails = Field(
        ..., 
        description="Error details"
    )


class LoadModelRequest(BaseModel):
    model_path: str = "models/retail_model.pkl"
    feature_columns: Optional[List[str]] = None
