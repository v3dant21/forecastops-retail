"""Model benchmarking for ForecastOps."""

import logging
import os
import pickle
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from forecastops.utils.logger import setup_logger

logger = setup_logger(__name__)


class InferenceBenchmark:
    """Benchmark model inference performance."""
    
    def __init__(self):
        """Initialize benchmark."""
        self.logger = setup_logger(__name__)
        self.results: Dict[str, Any] = {}
    
    def benchmark_latency(
        self,
        model: Any,
        X: pd.DataFrame,
        n_warmup: int = 10,
        n_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark inference latency.
        
        Args:
            model: Trained model
            X: Features
            n_warmup: Number of warmup runs
            n_runs: Number of benchmark runs
            
        Returns:
            Dictionary of latency metrics
        """
        # Warmup
        for _ in range(n_warmup):
            _ = model.predict(X.iloc[:1])
        
        # Benchmark
        latencies = []
        for i in range(n_runs):
            start = time.perf_counter()
            _ = model.predict(X.iloc[:1])
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        results = {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }
        
        self.results["latency"] = results
        return results
    
    def benchmark_throughput(
        self,
        model: Any,
        X: pd.DataFrame,
        batch_sizes: List[int] = [1, 8, 16, 32, 64, 128]
    ) -> Dict[int, Dict[str, float]]:
        """Benchmark throughput for different batch sizes.
        
        Args:
            model: Trained model
            X: Features
            batch_sizes: List of batch sizes
            
        Returns:
            Dictionary of batch size to throughput metrics
        """
        results = {}
        
        for batch_size in batch_sizes:
            # Warmup
            _ = model.predict(X.iloc[:batch_size])
            
            # Benchmark
            n_runs = 50
            times = []
            
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model.predict(X.iloc[:batch_size])
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            mean_time = np.mean(times)
            
            results[batch_size] = {
                "mean_ms": float(mean_time),
                "throughput_per_sec": float(batch_size * 1000 / mean_time) if mean_time > 0 else 0,
                "latency_per_sample_ms": float(mean_time / batch_size),
            }
        
        self.results["throughput"] = results
        return results
    
    def benchmark_model_size(
        self,
        model_path: str
    ) -> Dict[str, float]:
        """Benchmark model artifact size.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dictionary of size metrics
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        size_bytes = os.path.getsize(model_path)
        
        results = {
            "size_bytes": size_bytes,
            "size_kb": size_bytes / 1024,
            "size_mb": size_bytes / (1024 * 1024),
            "size_gb": size_bytes / (1024 * 1024 * 1024),
        }
        
        self.results["size"] = results
        return results
    
    def benchmark_memory(
        self,
        model: Any,
        X: pd.DataFrame
    ) -> Dict[str, float]:
        """Benchmark memory usage.
        
        Args:
            model: Trained model
            X: Features
            
        Returns:
            Dictionary of memory metrics
        """
        import tracemalloc
        
        # Start tracing
        tracemalloc.start()
        
        # Make prediction
        _ = model.predict(X)
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results = {
            "current_mb": current / (1024 * 1024),
            "peak_mb": peak / (1024 * 1024),
        }
        
        self.results["memory"] = results
        return results
    
    def benchmark_all(
        self,
        model: Any,
        X: pd.DataFrame,
        model_path: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Run all benchmarks.
        
        Args:
            model: Trained model
            X: Features
            model_path: Optional path to model file
            
        Returns:
            Dictionary of all benchmark results
        """
        results = {}
        
        results["latency"] = self.benchmark_latency(model, X)
        results["throughput"] = self.benchmark_throughput(model, X)
        results["memory"] = self.benchmark_memory(model, X)
        
        if model_path:
            results["size"] = self.benchmark_model_size(model_path)
        
        self.results = results
        return results
    
    def get_summary(self) -> str:
        """Get benchmark summary as string.
        
        Returns:
            Summary string
        """
        if not self.results:
            return "No benchmark results available"
        
        lines = ["=" * 60, "INFERENCE BENCHMARK SUMMARY", "=" * 60]
        
        for benchmark_name, results in self.results.items():
            lines.append(f"\n{benchmark_name.upper()}:")
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict):
                        lines.append(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            lines.append(f"    {sub_key}: {sub_value:.4f}")
                    else:
                        lines.append(f"  {key}: {value:.4f}")
        
        return "\n".join(lines)


class ModelOptimizer:
    """Optimize model for inference."""
    
    def __init__(self):
        """Initialize optimizer."""
        self.logger = setup_logger(__name__)
        self.optimizations_applied: List[str] = []
    
    def optimize_feature_computation(
        self,
        model: Any,
        X: pd.DataFrame,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """Optimize feature computation by only using required features.
        
        Args:
            model: Trained model
            X: Full features dataframe
            feature_columns: List of required feature columns
            
        Returns:
            Optimized features dataframe
        """
        # Filter to only required columns
        available_cols = [c for c in feature_columns if c in X.columns]
        X_optimized = X[available_cols].copy()
        
        self.optimizations_applied.append("feature_filtering")
        
        return X_optimized
    
    def optimize_model_loading(
        self,
        model_path: str,
        device: str = "cpu"
    ) -> Any:
        """Optimize model loading for specific device.
        
        Args:
            model_path: Path to model file
            device: Target device
            
        Returns:
            Optimized model
        """
        import torch
        import pickle
        
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        # Move model to device if applicable
        if device == "cpu" and hasattr(model_data["model"], "cpu"):
            model_data["model"] = model_data["model"].cpu()
        
        self.optimizations_applied.append(f"device_optimization_{device}")
        
        return model_data
    
    def optimize_batch_inference(
        self,
        model: Any,
        X: pd.DataFrame,
        batch_size: int = 128
    ) -> np.ndarray:
        """Perform batch inference for improved throughput.
        
        Args:
            model: Trained model
            X: Features
            batch_size: Batch size
            
        Returns:
            Predictions
        """
        n = len(X)
        predictions = []
        
        for i in range(0, n, batch_size):
            batch = X.iloc[i:i + batch_size]
            batch_pred = model.predict(batch)
            predictions.extend(batch_pred)
        
        return np.array(predictions)
    
    def get_optimizations(self) -> List[str]:
        """Get list of applied optimizations.
        
        Returns:
            List of optimization names
        """
        return self.optimizations_applied
