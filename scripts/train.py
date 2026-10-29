#!/usr/bin/env python3
"""Train ForecastOps models."""

import argparse
import logging
import os
import pickle
import sys

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from forecastops.utils.config import Config, load_config
from forecastops.utils.logger import setup_logger
from forecastops.utils.helpers import set_seed
from forecastops.evaluation.metrics import calculate_all_metrics
from forecastops.models.baseline import NaiveForecaster, SeasonalNaiveForecaster, LinearRegressor
from forecastops.models.tree import XGBoostForecaster, LightGBMForecaster, CatBoostForecaster
from forecastops.models.lstm import LSTMForecaster

logger = setup_logger(__name__)


def load_features(config, path="data/features/features.parquet"):
    """Fail clearly instead of inventing sales when verified data is absent."""
    from forecastops.data.m5 import require_real_features
    if not os.path.exists(path):
        raise FileNotFoundError("Actual M5 features are missing. Run make data first.")
    frame = pd.read_parquet(path)
    require_real_features(frame)
    return frame


def split_data(features_df, config):
    """Split data into train/val/test sets."""
    from forecastops.utils.helpers import split_features
    return split_features(features_df, config)


def train_model(model_name, X_train, y_train, X_val, y_val, config):
    """Train a model."""
    set_seed(config.model.random_seed)
    
    if model_name == "naive":
        model = NaiveForecaster()
    elif model_name == "seasonal_naive":
        model = SeasonalNaiveForecaster()
    elif model_name == "linear":
        model = LinearRegressor()
    elif model_name == "ridge":
        model = LinearRegressor(use_ridge=True)
    elif model_name == "xgboost":
        model = XGBoostForecaster()
    elif model_name == "lightgbm":
        model = LightGBMForecaster()
    elif model_name == "catboost":
        model = CatBoostForecaster()
    elif model_name == "lstm":
        model = LSTMForecaster({"lstm": {"epochs": config.model.epochs,
            "batch_size": config.model.batch_size, "learning_rate": config.model.learning_rate,
            "hidden_size": config.model.lstm_hidden_size, "num_layers": config.model.lstm_num_layers,
            "dropout": config.model.lstm_dropout}})
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Fit model
    from forecastops.tracking.tracker import MLflowTracker
    tracker = MLflowTracker(config)
    with tracker.start_run(run_name=model_name):
        tracker.log_parameters(model.get_params())
        model.fit(X_train, y_train, X_val, y_val)
        tracker.log_eval_metrics(calculate_all_metrics(y_val.values, model.predict(X_val)), split="val")
        tracker.log_model(model, "model")
    
    # Calculate metrics
    val_pred = model.predict(X_val)
    metrics = calculate_all_metrics(y_val.values, val_pred)
    
    return model, metrics


def save_model(model, model_name, metrics, provenance=None):
    """Save model to file."""
    os.makedirs("models", exist_ok=True)
    
    model.metadata = {"metrics": metrics, "version": "1.0.0",
                      "evaluation": "chronological one-step-ahead", "feature_pipeline_version": 3, "provenance": provenance}
    
    model_path = f"models/{model_name}.pkl"
    model.save(model_path)
    
    logger.info(f"Saved model to {model_path}")
    return model_path


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ForecastOps models")
    parser.add_argument("--model", "-m", required=True, help="Model to train")
    parser.add_argument("--config", "-c", default=None, help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    logger.info(f"Starting training for model: {args.model}")
    
    # Load or create features
    logger.info("Loading verified M5 features...")
    features_df = load_features(config)
    
    # Split data
    logger.info("Splitting data...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(features_df, config)
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train model
    logger.info(f"Training {args.model}...")
    model, metrics = train_model(args.model, X_train, y_train, X_val, y_val, config)
    
    # Save model
    logger.info("Saving model...")
    model_path = save_model(model, args.model, metrics, features_df.attrs["provenance"])
    
    # Print metrics
    logger.info(f"\nModel: {args.model}")
    logger.info(f"Validation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
