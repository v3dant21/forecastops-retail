#!/usr/bin/env python3
"""Hyperparameter tuning script for ForecastOps."""

import argparse
import logging
import os
import sys

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from forecastops.utils.config import Config, load_config
from forecastops.utils.logger import setup_logger
from forecastops.utils.helpers import set_seed
from forecastops.evaluation.metrics import calculate_all_metrics

logger = setup_logger(__name__)

# Try to import Optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Install with: pip install optuna")


def load_features(path: str = "data/features/features.parquet") -> pd.DataFrame:
    """Load features from file."""
    from forecastops.data.m5 import require_real_features
    frame = pd.read_parquet(path)
    require_real_features(frame)
    return frame


def split_data(features_df: pd.DataFrame, config: Config):
    """Split data into train/val/test sets."""
    from forecastops.utils.helpers import split_features
    X_train, y_train, X_val, y_val, _, _ = split_features(features_df, config)
    return X_train, X_val, y_train, y_val


def create_model(model_type: str, params: dict):
    """Create a model with given parameters."""
    if model_type == "xgboost":
        from forecastops.models.tree import XGBoostForecaster
        model = XGBoostForecaster({"xgboost": params})
    elif model_type == "lightgbm":
        from forecastops.models.tree import LightGBMForecaster
        model = LightGBMForecaster({"lightgbm": params})
    elif model_type == "catboost":
        from forecastops.models.tree import CatBoostForecaster
        model = CatBoostForecaster({"catboost": params})
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def objective(trial: optuna.Trial, model_type: str, X_train, y_train, X_val, y_val):
    """Optuna objective function."""
    set_seed(42)
    
    # Define search space based on model type
    if model_type == "xgboost":
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "random_state": 42,
            "n_jobs": -1,
        }
    elif model_type == "lightgbm":
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    elif model_type == "catboost":
        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "iterations": trial.suggest_int("iterations", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "random_seed": 42,
            "verbose": 0,
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create and train model
    model = create_model(model_type, params)
    
    try:
        model.fit(X_train, y_train, X_val, y_val)
        
        # Calculate metrics
        val_pred = model.predict(X_val)
        metrics = calculate_all_metrics(y_val.values, val_pred)
        
        return metrics["rmse"]
    
    except Exception as e:
        logger.warning(f"Trial failed: {e}")
        return float("inf")


def main():
    """Main tuning function."""
    parser = argparse.ArgumentParser(description="Tune ForecastOps models")
    parser.add_argument("--model", "-m", required=True, help="Model to tune (xgboost, lightgbm, catboost)")
    parser.add_argument("--n-trials", "-n", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--timeout", "-t", type=int, default=3600, help="Timeout in seconds")
    parser.add_argument("--config", "-c", default=None, help="Path to config file")
    parser.add_argument("--study-name", default="forecastops_tuning", help="Optuna study name")
    args = parser.parse_args()
    
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna not installed. Install with: pip install optuna")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    logger.info(f"Starting hyperparameter tuning for {args.model}")
    logger.info(f"Trials: {args.n_trials}, Timeout: {args.timeout}s")
    
    # Load features
    logger.info("Loading features...")
    features_df = load_features()
    
    # Split data
    logger.info("Splitting data...")
    X_train, X_val, y_train, y_val = split_data(features_df, config)
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        load_if_exists=True,
        storage=f"sqlite:///{args.study_name}.db" if args.study_name else None
    )
    
    # Run optimization
    logger.info("Starting optimization...")
    study.optimize(
        lambda trial: objective(trial, args.model, X_train, y_train, X_val, y_val),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True
    )
    
    # Print results
    logger.info("=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    
    best_trial = study.best_trial
    if not np.isfinite(best_trial.value):
        raise RuntimeError("All tuning trials failed; no model will be saved")
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best value (RMSE): {best_trial.value:.4f}")
    logger.info(f"Best parameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    # Save best parameters
    import json
    
    results = {
        "model": args.model,
        "best_trial": best_trial.number,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "n_trials": len(study.trials),
    }
    
    os.makedirs("models", exist_ok=True)
    with open(f"models/{args.model}_best_params.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Best parameters saved to models/{args.model}_best_params.json")
    
    # Train final model with best parameters
    logger.info(f"Training final {args.model} with best parameters...")
    
    if args.model == "xgboost":
        from forecastops.models.tree import XGBoostForecaster
        model = XGBoostForecaster({"xgboost": best_trial.params})
    elif args.model == "lightgbm":
        from forecastops.models.tree import LightGBMForecaster
        model = LightGBMForecaster({"lightgbm": best_trial.params})
    elif args.model == "catboost":
        from forecastops.models.tree import CatBoostForecaster
        model = CatBoostForecaster({"catboost": best_trial.params})
    
    model.fit(X_train, y_train, X_val, y_val)
    
    # Save model
    model.metadata = {"best_params": best_trial.params}
    model.save(f"models/{args.model}_tuned.pkl")
    
    logger.info(f"Model saved to models/{args.model}_tuned.pkl")
    logger.info("Tuning complete!")


if __name__ == "__main__":
    main()
