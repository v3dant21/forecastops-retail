"""Regression coverage for prediction, persistence and forecasting boundaries."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from forecastops.api import routes
from forecastops.api.main import app
from forecastops.features.engineer import FeatureEngineer
from forecastops.models.baseline import LinearRegressor, NaiveForecaster, SeasonalNaiveForecaster
from forecastops.models.io import load_model
from forecastops.models.tree import CatBoostForecaster, LightGBMForecaster, XGBoostForecaster
from forecastops.serving.predictor import ModelPredictor
from forecastops.utils.config import Config, DatabaseConfig, FeatureConfig, load_config
from forecastops.utils.helpers import train_val_test_split
from scripts.train import split_data


@pytest.fixture
def panel():
    return pd.DataFrame(
        [
            {
                "product_id": product,
                "store_id": "s",
                "date": date,
                "sales": float(i + offset),
                "x": float(i),
            }
            for product, offset in [("a", 2), ("b", 20)]
            for i, date in enumerate(pd.date_range("2024-01-01", periods=40))
        ]
    )


@pytest.mark.parametrize(
    "factory",
    [
        NaiveForecaster,
        SeasonalNaiveForecaster,
        LinearRegressor,
        XGBoostForecaster,
        LightGBMForecaster,
        CatBoostForecaster,
    ],
)
def test_model_round_trip(factory, panel, tmp_path):
    features = panel.drop(columns="sales")
    model = factory()
    if hasattr(model, "params"):
        model.params.update(
            {"n_jobs": 1}
        ) if factory != CatBoostForecaster else model.params.update({"thread_count": 1})
    model.fit(features, panel.sales, features, panel.sales)
    expected = model.predict(features)
    path = tmp_path / "model.pkl"
    model.save(path)
    np.testing.assert_allclose(load_model(path).predict(features), expected)
    predictor = ModelPredictor(str(path))
    predictor.load()
    np.testing.assert_allclose(predictor.predict(features), expected)


def test_api_prediction_validation_and_atomic_reload(panel, tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "predictor", None)
    monkeypatch.setattr(routes, "ensure_active_forecast_source", lambda: None)
    model = XGBoostForecaster({"xgboost": {"n_estimators": 3, "n_jobs": 1}})
    model.fit(panel[["x"]], panel.sales)
    from forecastops.data.m5 import ARCHIVE_SHA256
    # Provenance stub for this isolated API transport test; never a runtime dataset.
    model.metadata["provenance"] = {"source_type": "real_m5", "archive_sha256": ARCHIVE_SHA256}
    path = tmp_path / "tree.pkl"
    model.save(path)
    with TestClient(app) as client:
        assert client.post("/api/v1/predict", json={"features": {"x": 1}}).status_code == 503
        assert (
            client.post("/api/v1/load-model", params={"model_path": str(path)}).status_code == 200
        )
        result = client.post("/api/v1/predict", json={"features": {"x": 2}})
        assert result.status_code == 200
        assert result.json()["prediction"] >= 0
        batch = client.post("/api/v1/batch-predict", json={"features_list": [{"x": 2}, {"x": 3}]})
        assert batch.status_code == 200
        assert batch.json()["total_predictions"] == 2
        for values in [{}, {"x": "bad"}, {"x": None}]:
            assert client.post("/api/v1/predict", json={"features": values}).status_code == 422
        assert (
            client.post("/api/v1/batch-predict", json={"features_list": [{"x": 1}, {}]}).status_code
            == 422
        )
        assert client.post("/api/v1/batch-predict", json={"features_list": []}).status_code == 422
        assert (
            client.post(
                "/api/v1/load-model", params={"model_path": str(tmp_path / "missing.pkl")}
            ).status_code
            != 200
        )
        assert client.get("/api/v1/health").json()["model_loaded"]
        assert client.post("/api/v1/predict", json={"features": {"x": 2}}).status_code == 200


def test_panel_dates_do_not_cross_split_boundaries(panel):
    frame = panel.sample(frac=1, random_state=7)
    train, val, test = train_val_test_split(frame)
    assert train.date.max() < val.date.min() < test.date.min()
    assert val.date.max() < test.date.min()
    assert sum(map(len, [train, val, test])) == len(frame)
    sets = split_data(frame, Config())
    assert sets[0].date.max() < sets[2].date.min()
    assert sets[2].date.max() < sets[4].date.min()
    for X, y in zip(sets[::2], sets[1::2], strict=True):
        np.testing.assert_array_equal(frame.loc[X.index, "sales"], y)


def test_features_never_use_current_target_or_drop_no_event_rows(panel):
    panel["event_name_1"] = None
    engineer = FeatureEngineer()
    first = engineer.engineer_features(panel, lags=[1, 3], rolling_windows=[3])
    assert not first.empty
    changed = panel.copy()
    affected = changed.index[-1]
    changed.loc[affected, "sales"] = 999999
    second = engineer.engineer_features(changed, lags=[1, 3], rolling_windows=[3])
    history = [c for c in first if c.startswith("sales_")]
    np.testing.assert_array_equal(first.loc[affected, history], second.loc[affected, history])
    assert first.groupby("product_id").size().tolist() == [37, 37]


def test_dashboard_restores_concrete_model(panel, tmp_path):
    from forecastops.dashboard.streamlit_app import load_model as dashboard_load

    model = LinearRegressor().fit(panel[["x"]], panel.sales)
    path = tmp_path / "linear.pkl"
    model.save(path)
    restored = dashboard_load(str(path))
    np.testing.assert_allclose(restored.predict(panel[["x"]]), model.predict(panel[["x"]]))


def test_lstm_round_trip_and_series_boundaries(panel, tmp_path):
    from forecastops.models.lstm import LSTMForecaster

    model = LSTMForecaster(
        {
            "lstm": {
                "epochs": 1,
                "hidden_size": 4,
                "num_layers": 1,
                "seq_length": 3,
                "batch_size": 16,
            }
        }
    )
    X = panel.drop(columns="sales")
    model.fit(X, panel.sales)
    windows = model._windows(X, panel[["x"]].to_numpy())
    assert np.all(windows[40, :-1] == 0)
    path = tmp_path / "lstm.pkl"
    model.save(path)
    np.testing.assert_allclose(load_model(path).predict(X), model.predict(X))
    assert len(model.predict(X.iloc[:1])) == 1
    assert np.isfinite(model.predict(X)).all()


def test_settings_and_yaml(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:password@db:5433/test")
    assert DatabaseConfig().host == "db"
    assert DatabaseConfig().port == 5433
    monkeypatch.setenv("FEATURE_LAGS", "1,4")
    assert FeatureConfig().lags == [1, 4]
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  target_column: units\n  random_seed: 7\n")
    assert load_config(str(path)).model.target_column == "units"


def test_leakage_checks_reject_current_sales(panel):
    from forecastops.features.leakage import LeakagePreventer

    bad = panel.copy()
    bad["sales_lag_1"] = bad.sales
    bad["sales_rolling_mean_3"] = bad.sales
    checker = LeakagePreventer()
    assert checker.check_lag_leakage(bad, "sales", [1])
    assert checker.check_rolling_leakage(bad, "sales", [3])


def test_database_loader_defaults_and_connection_arguments(monkeypatch):
    from unittest.mock import Mock

    from forecastops.data.loader import DataLoader

    connect = Mock()
    monkeypatch.setattr("forecastops.data.loader.psycopg2.connect", connect)
    with DataLoader({"port": 15433}) as loader:
        assert loader.conn is connect.return_value
        assert connect.call_args.kwargs["dbname"] == "forecastops"
        assert connect.call_args.kwargs["port"] == 15433
    connect.return_value.close.assert_called_once()


def test_training_records_mlflow_metrics_and_model(panel, tmp_path):
    import mlflow

    from scripts.train import train_model

    config = Config()
    config.mlflow.tracking_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    config.mlflow.experiment_name = "regression"
    X = panel.drop(columns="sales")
    model, metrics = train_model("linear", X, panel.sales, X, panel.sales, config)
    experiment = mlflow.get_experiment_by_name("regression")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.iloc[0]["metrics.val_rmse"] == pytest.approx(metrics["rmse"])
    client = mlflow.MlflowClient()
    assert client.list_artifacts(runs.iloc[0].run_id, "model")[0].path == "model/model.pkl"


@pytest.mark.parametrize("method", ["rolling_origin_cv", "expanding_window_cv"])
def test_cross_validation_keeps_panel_dates_separate(panel, method):
    from forecastops.evaluation import metrics

    trained_ends = []

    class RecordingModel:
        def fit(self, X, y):
            self.end = X.date.max()
            trained_ends.append(self.end)

        def predict(self, X):
            assert self.end < X.date.min()
            return np.zeros(len(X))

    frame = panel.sample(frac=1, random_state=8)
    results = getattr(metrics, method)(
        frame.drop(columns="sales"),
        frame.sales,
        RecordingModel,
        initial_train_size=10,
        horizon=3,
        step_size=5,
    )
    assert results
    assert len(results) == len(trained_ends)


def test_seasonal_baseline_uses_each_series_calendar_position():
    frame = pd.DataFrame(
        [
            {"product_id": product, "store_id": "s", "date": date}
            for product in ["a", "b"]
            for date in pd.date_range("2024-01-01", periods=6)
        ]
    )
    model = SeasonalNaiveForecaster(season_length=3).fit(
        frame, pd.Series([1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16])
    )
    future = pd.DataFrame(
        [
            {"product_id": "a", "store_id": "s", "date": "2024-01-07"},
            {"product_id": "b", "store_id": "s", "date": "2024-01-07"},
            {"product_id": "a", "store_id": "s", "date": "2024-01-08"},
            {"product_id": "b", "store_id": "s", "date": "2024-01-08"},
        ]
    )
    np.testing.assert_array_equal(model.predict(future), [4, 14, 5, 15])
