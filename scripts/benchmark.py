#!/usr/bin/env python3
"""Benchmark script for ForecastOps."""

import argparse
import logging
import os
import pickle
import sys
import time

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from forecastops.utils.config import Config, load_config
from forecastops.utils.logger import setup_logger
from forecastops.utils.helpers import set_seed

logger = setup_logger(__name__)


def load_model(model_path: str):
    """Load a model from file."""
    from forecastops.models.io import load_model as restore_model
    model = restore_model(model_path)
    return {"model": model, "name": model.name, "params": model.get_params()}


def load_features(path: str = "data/features/features.parquet") -> pd.DataFrame:
    """Load features from file."""
    from forecastops.data.m5 import require_real_features
    frame = pd.read_parquet(path)
    require_real_features(frame)
    return frame


def benchmark_model(model_data: dict, X_test: pd.DataFrame, n_runs: int = 10) -> dict:
    """Benchmark a model's inference performance.
    
    Args:
        model_data: Model data dictionary
        X_test: Test features
        n_runs: Number of benchmark runs
        
    Returns:
        Dictionary of benchmark metrics
    """
    model = model_data.get("model")
    model_name = model_data.get("name", "unknown")
    
    if model is None:
        raise ValueError("Model not found in saved data")
    
    # Warmup
    for _ in range(5):
        _ = model.predict(X_test.iloc[:1])
    
    # Benchmark single prediction latency
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X_test.iloc[:1])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
    
    # Benchmark batch inference
    batch_sizes = [1, 8, 16, 32, 64, 128]
    throughput_results = {}
    
    for batch_size in batch_sizes:
        # Warmup
        _ = model.predict(X_test.iloc[:batch_size])
        
        times = []
        for _ in range(max(1, n_runs // 2)):
            start = time.perf_counter()
            _ = model.predict(X_test.iloc[:batch_size])
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        mean_time = np.mean(times)
        throughput_results[batch_size] = {
            "latency_ms": float(np.mean(times)),
            "throughput_per_sec": float(min(batch_size, len(X_test)) * 1000 / mean_time) if mean_time > 0 else 0,
        }
    
    return {
        "model_name": model_name,
        "single_prediction": {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
        },
        "batch_inference": throughput_results,
    }


def benchmark_model_size(model_path: str) -> dict:
    """Benchmark model artifact size.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Dictionary of size metrics
    """
    size_bytes = os.path.getsize(model_path)
    
    return {
        "size_bytes": size_bytes,
        "size_kb": size_bytes / 1024,
        "size_mb": size_bytes / (1024 * 1024),
    }


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark ForecastOps models")
    parser.add_argument("--model", "-m", help="Model file to benchmark (optional, benchmarks all if not specified)")
    parser.add_argument("--output", "-o", default="reports/benchmark_results.json", help="Output file")
    args = parser.parse_args()
    
    # Load features
    logger.info("Loading features...")
    try:
        features_df = load_features()
    except FileNotFoundError:
        logger.error("Features not found. Run training first.")
        return
    
    target_col = "sales"
    exclude_cols = ["date", "id", target_col, "product_id", "store_id"]
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    
    X_test = features_df.drop(columns=[target_col, "id"], errors="ignore").iloc[:1000].copy()  # Use subset for benchmarking
    X_test = X_test.fillna(0)
    
    # Load models
    models_dir = "models"
    if args.model:
        model_files = [args.model]
    else:
        model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    
    if not model_files:
        logger.error("No models found.")
        return
    
    # Benchmark each model
    results = {}
    
    for model_file in model_files:
        try:
            model_path = os.path.join(models_dir, model_file)
            logger.info(f"Benchmarking {model_file}...")
            
            model_data = load_model(model_path)
            
            # Inference benchmark
            inference_results = benchmark_model(model_data, X_test)
            
            # Size benchmark
            size_results = benchmark_model_size(model_path)
            
            results[model_file] = {
                "inference": inference_results,
                "size": size_results,
            }
            
            # Print summary
            logger.info(f"\n{model_file}:")
            logger.info(f"  Single prediction (ms): {inference_results['single_prediction']['mean_ms']:.2f} ± {inference_results['single_prediction']['std_ms']:.2f}")
            logger.info(f"  Size (MB): {size_results['size_mb']:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to benchmark {model_file}: {e}")
            results[model_file] = {"error": str(e)}
    
    # Save results
    os.makedirs("reports", exist_ok=True)
    import json
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")
    
    # Generate comparison report
    logger.info("\n" + "=" * 60)
    logger.info("MODEL BENCHMARK SUMMARY")
    logger.info("=" * 60)
    
    for model_file, result in results.items():
        if "error" not in result:
            inference = result["inference"]["single_prediction"]
            size = result["size"]
            
            logger.info(f"\n{model_file}:")
            logger.info(f"  Mean Latency (ms): {inference['mean_ms']:.2f}")
            logger.info(f"  P95 Latency (ms): {inference['p95_ms']:.2f}")
            logger.info(f"  Size (MB): {size['size_mb']:.2f}")


if __name__ == "__main__":
    main()
