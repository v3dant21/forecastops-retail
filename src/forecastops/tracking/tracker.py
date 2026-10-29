"""MLflow tracking for ForecastOps."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import pandas as pd
import numpy as np

from forecastops.utils.logger import setup_logger
from forecastops.utils.config import Config

logger = setup_logger(__name__)


class MLflowTracker:
    """Tracker for experiments using MLflow."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize MLflow tracker.
        
        Args:
            config: Optional configuration
        """
        self.config = config or Config()
        self.mlflow_config = self.config.mlflow
        self.logger = setup_logger(__name__)
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
        mlflow.set_experiment(self.mlflow_config.experiment_name)
        
        self.logger.info(f"MLflow tracking initialized: {self.mlflow_config.tracking_uri}")
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run.
        
        Args:
            run_name: Optional run name
            tags: Optional tags
            
        Returns:
            Active run context
        """
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_parameters(
        self,
        params: Dict[str, Any],
        prefix: str = ""
    ) -> None:
        """Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters
            prefix: Optional prefix for parameter names
        """
        for key, value in params.items():
            full_key = f"{prefix}{key}" if prefix else key
            mlflow.log_param(full_key, value)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        prefix: str = ""
    ) -> None:
        """Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics
            prefix: Optional prefix for metric names
        """
        for key, value in metrics.items():
            full_key = f"{prefix}{key}" if prefix else key
            mlflow.log_metric(full_key, value)
    
    def log_artifact(
        self,
        path: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """Log a file as an artifact.
        
        Args:
            path: Path to the file
            artifact_path: Optional artifact path
        """
        mlflow.log_artifact(path, artifact_path)
    
    def log_artifacts(
        self,
        directory: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """Log all files in a directory as artifacts.
        
        Args:
            directory: Directory path
            artifact_path: Optional artifact path
        """
        mlflow.log_artifacts(directory, artifact_path)
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        input_example: Optional[pd.DataFrame] = None
    ) -> None:
        """Log a model as an artifact.
        
        Args:
            model: Model to log
            artifact_path: Artifact path
            input_example: Optional input example
        """
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            model.save(str(path))
            mlflow.log_artifact(str(path), artifact_path)
    
    def log_figure(
        self,
        fig,
        artifact_path: str
    ) -> None:
        """Log a matplotlib figure.
        
        Args:
            fig: Matplotlib figure
            artifact_path: Artifact path
        """
        import matplotlib.pyplot as plt
        
        plt.figure(fig.number)
        mlflow.log_figure(fig, artifact_path)
    
    def log_table(
        self,
        data: pd.DataFrame,
        artifact_path: str
    ) -> None:
        """Log a DataFrame as a table artifact.
        
        Args:
            data: DataFrame to log
            artifact_path: Artifact path
        """
        # Convert to CSV and log
        csv_path = "/tmp/table.csv"
        data.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path)
        os.remove(csv_path)
    
    def log_eval_metrics(
        self,
        metrics: Dict[str, float],
        split: str = "test"
    ) -> None:
        """Log evaluation metrics with split prefix.
        
        Args:
            metrics: Dictionary of metrics
            split: Split name (train, val, test)
        """
        self.log_metrics(metrics, prefix=f"{split}_")
    
    def log_model_config(
        self,
        model_name: str,
        params: Dict[str, Any]
    ) -> None:
        """Log model configuration.
        
        Args:
            model_name: Model name
            params: Model parameters
        """
        self.log_parameters({
            "model_name": model_name,
            **params
        })
    
    def log_training_info(
        self,
        model_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        dataset_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log complete training information.
        
        Args:
            model_name: Model name
            params: Model parameters
            metrics: Training metrics
            dataset_info: Optional dataset information
            
        Returns:
            Run ID
        """
        with self.start_run() as run:
            self.log_parameters({
                "model_name": model_name,
                **params
            })
            
            self.log_metrics(metrics)
            
            if dataset_info:
                self.log_parameters(dataset_info, prefix="dataset_")
            
            return run.info.run_id
    
    def get_run(self, run_id: str) -> mlflow.entities.Run:
        """Get a run by ID.
        
        Args:
            run_id: Run ID
            
        Returns:
            Run object
        """
        return mlflow.get_run(run_id)
    
    def search_runs(
        self,
        experiment_name: Optional[str] = None,
        filter_string: str = "",
        max_results: int = 100
    ) -> List[mlflow.entities.Run]:
        """Search for runs.
        
        Args:
            experiment_name: Optional experiment name
            filter_string: Filter string
            max_results: Maximum results
            
        Returns:
            List of runs
        """
        experiment = mlflow.get_experiment_by_name(experiment_name or self.mlflow_config.experiment_name)
        if experiment:
            return mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                max_results=max_results
            )
        return []
    
    def download_artifacts(
        self,
        run_id: str,
        path: str,
        dst_path: Optional[str] = None
    ) -> str:
        """Download artifacts from a run.
        
        Args:
            run_id: Run ID
            path: Artifact path
            dst_path: Optional destination path
            
        Returns:
            Local path to downloaded artifact
        """
        return mlflow.tracking.MlflowClient().download_artifacts(run_id, path, dst_path)
    
    def get_best_run(
        self,
        metric: str = "val_rmse",
        mode: str = "min"
    ) -> Optional[mlflow.entities.Run]:
        """Get the best run based on a metric.
        
        Args:
            metric: Metric to optimize
            mode: Optimization mode ("min" or "max")
            
        Returns:
            Best run or None
        """
        import pandas as pd
        
        runs_df = mlflow.search_runs(
            experiment_ids=[mlflow.get_experiment_by_name(self.mlflow_config.experiment_name).experiment_id]
        )
        
        if runs_df.empty:
            return None
        
        # Find best run
        if mode == "min":
            best_idx = runs_df[metric].idxmin()
        else:
            best_idx = runs_df[metric].idxmax()
        
        best_run_id = runs_df.loc[best_idx, "run_id"]
        return mlflow.get_run(best_run_id)
    
    def get_tracking_uri(self) -> str:
        """Get the tracking URI.
        
        Returns:
            Tracking URI
        """
        return mlflow.get_tracking_uri()
    
    def get_experiment_id(self) -> str:
        """Get the current experiment ID.
        
        Returns:
            Experiment ID
        """
        experiment = mlflow.get_experiment_by_name(self.mlflow_config.experiment_name)
        return experiment.experiment_id if experiment else ""
