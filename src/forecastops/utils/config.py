"""Configuration management using Pydantic settings."""

import os
from typing import List, Optional, Union

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseSettings):
    """Configuration for data processing."""

    model_config = SettingsConfigDict(
        env_prefix="DATA_",
        env_file=".env",
        extra="ignore"
    )

    dataset_path: str = Field(default="data/raw", description="Path to raw dataset")
    subset_products: int = Field(default=100, description="Number of products (use -1 for all)")
    subset_stores: int = Field(default=5, description="Number of stores (use -1 for all)")
    subset_days: int = Field(default=1000, description="Number of days (use -1 for all)")


class DatabaseConfig(BaseSettings):
    """Configuration for database connection."""

    model_config = SettingsConfigDict(
        env_prefix="DATABASE_",
        env_file=".env",
        extra="ignore"
    )

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="forecastops", description="Database name")
    user: str = Field(default="forecastops", description="Database user")
    password: str = Field(default="forecastops", description="Database password")

    connection_url: Optional[str] = Field(default=None, validation_alias="DATABASE_URL")

    @model_validator(mode="after")
    def apply_connection_url(self):
        if self.connection_url:
            from urllib.parse import urlsplit, unquote
            parsed = urlsplit(self.connection_url)
            if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
                raise ValueError("DATABASE_URL must be a PostgreSQL connection URL")
            self.host = parsed.hostname
            self.port = parsed.port or 5432
            self.name = parsed.path.lstrip("/") or self.name
            self.user = unquote(parsed.username) if parsed.username else self.user
            self.password = unquote(parsed.password) if parsed.password else self.password
        return self

    @property
    def url(self) -> str:
        """Return PostgreSQL connection URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class FeatureConfig(BaseSettings):
    """Configuration for feature engineering."""

    model_config = SettingsConfigDict(
        env_prefix="FEATURE_",
        env_file=".env",
        extra="ignore"
    )

    # Parse lag/rolling windows from comma-separated strings
    lags_str: str = Field(default="1,2,3,7,14,21,28", validation_alias=AliasChoices("FEATURE_LAGS", "FEATURE_LAGS_STR", "lags_str"), description="Comma-separated lag windows")
    rolling_windows_str: str = Field(default="7,14,28", validation_alias=AliasChoices("FEATURE_ROLLING_WINDOWS", "FEATURE_ROLLING_WINDOWS_STR", "rolling_windows_str"), description="Comma-separated rolling windows")
    rolling_functions_str: str = Field(default="mean,std", validation_alias=AliasChoices("FEATURE_ROLLING_FUNCTIONS", "FEATURE_ROLLING_FUNCTIONS_STR", "rolling_functions_str"), description="Comma-separated rolling functions")

    @staticmethod
    def _parse_list(value):
        import json
        if value.strip().startswith("["):
            return json.loads(value)
        return [x.strip() for x in value.split(",") if x.strip()]

    @property
    def lags(self) -> List[int]:
        return [int(x) for x in self._parse_list(self.lags_str)]

    @property
    def rolling_windows(self) -> List[int]:
        return [int(x) for x in self._parse_list(self.rolling_windows_str)]

    @property
    def rolling_functions(self) -> List[str]:
        return self._parse_list(self.rolling_functions_str)


class ModelConfig(BaseSettings):
    """Configuration for models."""

    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        env_file=".env",
        extra="ignore"
    )

    target_column: str = "sales"
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    batch_size: int = Field(default=128, description="Training batch size")
    epochs: int = Field(default=50, description="Number of training epochs")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    weight_decay: float = Field(default=0.01, description="Weight decay for optimizer")

    # LSTM specific
    lstm_hidden_size: int = Field(default=64, description="LSTM hidden size")
    lstm_num_layers: int = Field(default=2, description="Number of LSTM layers")
    lstm_dropout: float = Field(default=0.2, description="LSTM dropout rate")


class TrainingConfig(BaseSettings):
    """Configuration for training process."""

    model_config = SettingsConfigDict(
        env_prefix="TRAINING_",
        env_file=".env",
        extra="ignore"
    )

    validation_split: float = Field(default=0.2, description="Validation split ratio")
    test_split: float = Field(default=0.1, description="Test split ratio")
    time_aware_split: bool = Field(default=True, description="Use time-aware splitting")


class OptunaConfig(BaseSettings):
    """Configuration for Optuna hyperparameter tuning."""

    model_config = SettingsConfigDict(
        env_prefix="OPTUNA_",
        env_file=".env",
        extra="ignore"
    )

    n_trials: int = Field(default=100, description="Number of Optuna trials")
    timeout: int = Field(default=3600, description="Timeout in seconds")
    study_name: str = Field(default="forecastops_tuning", description="Study name")


class MLflowConfig(BaseSettings):
    """Configuration for MLflow tracking."""

    model_config = SettingsConfigDict(
        env_prefix="MLFLOW_",
        env_file=".env",
        extra="ignore"
    )

    tracking_uri: str = Field(default="sqlite:///mlruns/mlflow.db", description="MLflow tracking URI")
    experiment_name: str = Field(default="forecastops", description="Experiment name")


class APIConfig(BaseSettings):
    """Configuration for FastAPI service."""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        extra="ignore"
    )

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    reload: bool = Field(default=True, description="Auto-reload on code changes")


class Config(BaseSettings):
    """Global configuration combining all sub-configs."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = Field(default="development")
    app_debug: bool = Field(default=True)

    data: DataConfig = Field(default_factory=DataConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    api: APIConfig = Field(default_factory=APIConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from environment and optional config file."""
    if config_path:
        import yaml
        with open(config_path) as source:
            values = yaml.safe_load(source) or {}
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a YAML mapping")
        return Config(**values)
    return Config()


# Global config instance
config = load_config()
