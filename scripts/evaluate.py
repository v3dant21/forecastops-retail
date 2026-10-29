#!/usr/bin/env python3
"""Evaluation script for ForecastOps."""

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
from forecastops.utils.helpers import set_seed, calculate_wape
from forecastops.evaluation.metrics import (
    mae, rmse, wape, smape, mape, rolling_origin_cv, expanding_window_cv
)
from forecastops.evaluation.validator import TimeSeriesValidator

logger = setup_logger(__name__)


def load_model(model_path: str):
    """Load a model from file."""
    from forecastops.models.io import load_model as restore_model
    model = restore_model(model_path)
    return {"model": model, "name": model.name, "params": model.get_params()}


def load_features(path: str) -> pd.DataFrame:
    """Load features from file."""
    from forecastops.data.m5 import require_real_features
    frame = pd.read_parquet(path)
    require_real_features(frame)
    return frame


def evaluate_model(
    model_data: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> dict:
    """Evaluate a trained model.
    
    Args:
        model_data: Model data dictionary
        X_test: Test features
        y_test: Test targets
        X_val: Validation features
        y_val: Validation targets
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Dictionary of evaluation metrics
    """
    model = model_data.get("model")
    model_name = model_data.get("name", "unknown")
    
    if model is None:
        raise ValueError("Model not found in saved data")
    
    # Get predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Calculate metrics for each split
    metrics = {}
    
    for split_name, y_true, y_pred in [
        ("train", y_train.values, train_pred),
        ("val", y_val.values, val_pred),
        ("test", y_test.values, test_pred)
    ]:
        metrics[f"{split_name}_mae"] = mae(y_true, y_pred)
        metrics[f"{split_name}_rmse"] = rmse(y_true, y_pred)
        metrics[f"{split_name}_wape"] = wape(y_true, y_pred)
        metrics[f"{split_name}_smape"] = smape(y_true, y_pred)
        metrics[f"{split_name}_mape"] = mape(y_true, y_pred)
    
    # Check for overfitting
    train_mae = metrics["train_mae"]
    val_mae = metrics["val_mae"]
    metrics["overfitting_ratio"] = (val_mae - train_mae) / train_mae if train_mae > 0 else 0
    
    # These are chronological holdout metrics. Cross-validation requires fresh
    # model fits, and is exposed separately by evaluation.metrics.
    # Low/high demand performance
    metrics["low_demand_mae"] = mae(
        y_test.values[y_test.values <= 0],
        test_pred[y_test.values <= 0]
    ) if np.any(y_test.values <= 0) else 0
    
    high = y_test.values > np.percentile(y_test.values, 75)
    metrics["high_demand_mae"] = float(mae(y_test.values[high], test_pred[high])) if high.any() else 0.0
    
    return {
        "model_name": model_name,
        "metrics": metrics,
        "predictions": test_pred.tolist(),
        "actual": y_test.values.tolist()
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate ForecastOps models")
    parser.add_argument("--model", "-m", help="Model file to evaluate (optional, evaluates all if not specified)")
    parser.add_argument("--config", "-c", default=None, help="Path to config file")
    parser.add_argument("--output", "-o", default="reports/evaluation_results.json", help="Output file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    logger.info("Loading features...")
    
    # Load features
    try:
        features_df = load_features("data/features/features.parquet")
    except FileNotFoundError:
        logger.error("Features not found. Run training first.")
        return
    
    # Load models
    models_dir = "models"
    if args.model:
        model_files = [args.model]
    else:
        model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    
    if not model_files:
        logger.error("No models found.")
        return
    
    # Split data
    from forecastops.utils.helpers import split_features
    X_train, y_train, X_val, y_val, X_test, y_test = split_features(features_df, config)

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Evaluate each model
    results = {}
    
    for model_file in model_files:
        try:
            model_path = model_file if os.path.isfile(model_file) else os.path.join(models_dir, model_file)
            logger.info(f"Evaluating {model_file}...")
            
            model_data = load_model(model_path)
            result = evaluate_model(model_data, X_test, y_test, X_val, y_val, X_train, y_train)
            
            results[model_file] = result
            
            # Print summary
            logger.info(f"\n{model_file}:")
            logger.info(f"  Test MAE: {result['metrics']['test_mae']:.4f}")
            logger.info(f"  Test RMSE: {result['metrics']['test_rmse']:.4f}")
            logger.info(f"  Test WAPE: {result['metrics']['test_wape']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_file}: {e}")
            results[model_file] = {"error": str(e)}
    
    # Save results
    os.makedirs("reports", exist_ok=True)
    import json
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")
    
    # Generate comparison report
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    
    for model_file, result in results.items():
        if "error" not in result:
            logger.info(f"\n{model_file}:")
            logger.info(f"  Test MAE: {result['metrics']['test_mae']:.4f}")
            logger.info(f"  Test RMSE: {result['metrics']['test_rmse']:.4f}")
            logger.info(f"  Test WAPE: {result['metrics']['test_wape']:.4f}")
            logger.info(f"  Overfitting Ratio: {result['metrics']['overfitting_ratio']:.4f}")


if __name__ == "__main__":
    main()
