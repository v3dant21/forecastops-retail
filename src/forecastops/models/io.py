"""Shared restoration of trusted local forecasting artifacts."""

import pickle

from forecastops.models.baseline import LinearRegressor, NaiveForecaster, SeasonalNaiveForecaster
from forecastops.models.tree import CatBoostForecaster, LightGBMForecaster, XGBoostForecaster


def model_from_artifact(data):
    factories = {
        "naive": NaiveForecaster,
        "seasonal_naive": SeasonalNaiveForecaster,
        "linear": LinearRegressor,
        "xgboost": XGBoostForecaster,
        "lightgbm": LightGBMForecaster,
        "catboost": CatBoostForecaster,
    }
    name = data.get("name")
    if name == "lstm":
        from forecastops.models.lstm import LSTMForecaster  # noqa: PLC0415

        factory = LSTMForecaster
    else:
        factory = factories.get(name)
    if factory is None:
        raise ValueError(f"Unsupported model type: {name}")
    model = factory()
    if "state" in data:
        model.__dict__.update(data["state"])
    else:
        # Older tree artifacts contain sufficient state. Other wrappers lost
        # preprocessing/history on save and must be retrained.
        if name not in {"xgboost", "lightgbm", "catboost"}:
            raise ValueError("Legacy artifact lacks preprocessing/history; retrain this model")
        model.model = data.get("model")
        model.feature_columns = data.get("feature_columns", [])
        model.target_column = data.get("target_column", "sales")
        model.metadata = data.get("metadata", {})
        model.trained = data.get("trained", False)
        model.set_params(data.get("params", {}))
    if not model.trained:
        raise ValueError("Artifact contains an untrained model")
    return model


def load_model(path):
    """Load only artifacts from a trusted source: pickle executes Python code."""
    with open(path, "rb") as source:
        return model_from_artifact(pickle.load(source))
