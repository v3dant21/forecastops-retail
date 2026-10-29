"""Real-data integration and no-fallback/causal-planning regression checks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from forecastops.data.m5 import ARCHIVE_SHA256, download_m5, require_real_features
from forecastops.planning.forecast import KEYS, recursive_forecast, stock_plan
from forecastops.utils.config import Config
from scripts.train import load_features


@pytest.fixture(scope="module")
def actual_data():
    directory = Path("data/features")
    if not (directory / "history.parquet").exists():
        pytest.skip("Run make data for actual M5 integration checks")
    return tuple(
        pd.read_parquet(directory / name)
        for name in ["features.parquet", "history.parquet", "calendar.parquet", "prices.parquet"]
    )


def test_training_does_not_create_a_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="Actual M5"):
        load_features(Config(), str(tmp_path / "missing.parquet"))
    assert not list(tmp_path.iterdir())


def test_unverified_features_are_rejected():
    with pytest.raises(ValueError, match="verified M5"):
        require_real_features(pd.DataFrame({"sales": [1]}))


def test_wrong_archive_is_rejected_before_extraction(tmp_path):
    bad = tmp_path / "wrong.zip"
    bad.write_bytes(b"This is not the published M5 archive")
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_m5(tmp_path / "raw", bad)
    assert not list((tmp_path / "raw").iterdir())


def test_actual_source_and_causal_features(actual_data):
    features, history, calendar, prices = actual_data
    provenance = require_real_features(features)
    assert provenance["archive_sha256"] == ARCHIVE_SHA256
    assert features.product_id.str.match(r"^(FOODS|HOUSEHOLD|HOBBIES)_\d_\d{3}$").all()
    assert features.store_id.str.match(r"^(CA|TX|WI)_\d+$").all()
    assert (features.sales >= 0).all()
    assert {"snap_ca", "snap_tx", "snap_wi"}.issubset(calendar.columns)
    key = features.iloc[0][KEYS]
    rows = history[(history.product_id == key.product_id) & (history.store_id == key.store_id)]
    rows = rows.sort_values("date")
    expected = rows.assign(expected_lag=rows.sales.shift(28))
    checked = features.merge(expected[KEYS + ["date", "expected_lag"]], on=KEYS + ["date"])
    np.testing.assert_array_equal(checked.sales_lag_28, checked.expected_lag)


class LagProbe:
    """Test double that makes recursive dependencies observable."""

    def predict(self, frame):
        assert frame.sales.isna().all(), "Future sales must be unknown at inference"
        return frame.sales_lag_1.to_numpy() + 0.25


def one_series(actual_data):
    _, history, calendar, prices = actual_data
    last = history[history.date == history.date.max()].iloc[0]
    history = history[(history.product_id == last.product_id) & (history.store_id == last.store_id)]
    return history, calendar, prices


def test_rollout_uses_previous_predictions_not_future_sales(actual_data):
    history, calendar, prices = one_series(actual_data)
    last_sale = float(history.sort_values("date").sales.iloc[-1])
    result = recursive_forecast(LagProbe(), history, calendar, prices, horizon=3)
    np.testing.assert_allclose(
        result.prediction, [last_sale + 0.25, last_sale + 0.5, last_sale + 0.75]
    )
    assert result.date.min() > history.date.max()
    assert "sales" not in result


def test_unknown_future_prices_are_disclosed(actual_data):
    history, calendar, prices = one_series(actual_data)
    result = recursive_forecast(LagProbe(), history, calendar, prices.iloc[:0], horizon=2)
    assert result.price_carried_forward.all()
    np.testing.assert_allclose(result.sell_price, history.sort_values("date").sell_price.iloc[-1])


def test_stock_plan_never_orders_negative_units():
    assert stock_plan(10, on_hand=100)["suggested_order"] == 0
    assert stock_plan(10, on_hand=2, incoming=3, buffer_percent=20)["suggested_order"] == 7
    with pytest.raises(ValueError, match="cannot be negative"):
        stock_plan(10, on_hand=-1)


def test_published_plan_has_real_provenance_and_separate_holdout():
    path = Path("reports/retail_plan.json")
    if not path.exists():
        pytest.skip("Run make train for published-plan checks")
    report = json.loads(path.read_text())
    assert report["source"]["source_type"] == "real_m5"
    assert pd.Timestamp(report["validation_end"]) < pd.Timestamp(report["test_start"])
    assert pd.Timestamp(report["test_end"]) < pd.Timestamp(report["forecast_start"])
    forecast = pd.read_parquet("reports/forecast.parquet")
    assert forecast.date.nunique() == 28
    assert np.isfinite(forecast.prediction).all()
    assert forecast.prediction.ge(0).all()


def test_dashboard_filters_and_stock_calculator(actual_data, monkeypatch):
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: False
        if path == Path("reports/active_dataset.json")
        else original_exists(path),
    )
    if not Path("reports/retail_plan.json").exists():
        pytest.skip("Run make train for dashboard integration checks")
    app = AppTest.from_file(str(Path("src/forecastops/dashboard/streamlit_app.py").resolve())).run(
        timeout=45
    )
    assert not app.exception
    assert not app.error
    app.sidebar.selectbox[0].set_value("CA_1")
    app.sidebar.selectbox[1].set_value("FOODS")
    app.sidebar.selectbox[2].set_value(14)
    app.run(timeout=45)
    assert not app.exception
    for name in ["Product & stock planner", "Forecast reliability", "Data & assumptions"]:
        app.sidebar.radio[0].set_value(name).run(timeout=45)
        assert not app.exception
        assert not app.error
    app.sidebar.radio[0].set_value("Product & stock planner").run(timeout=45)
    app.button[0].click().run(timeout=45)
    assert "Enter your current stock" in app.error[0].value
    app.number_input[1].set_value(20)
    app.number_input[2].set_value(5)
    app.button[0].click().run(timeout=45)
    assert not app.error
    assert "Suggested order:" in app.success[0].value


def test_api_rejects_unverified_model_artifacts(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from forecastops.api import routes
    from forecastops.api.main import app
    from forecastops.models.baseline import LinearRegressor

    monkeypatch.setattr(routes, "predictor", None)
    model = LinearRegressor().fit(pd.DataFrame({"x": [1, 2, 3]}), pd.Series([2, 4, 6]))
    path = tmp_path / "unverified.pkl"
    model.save(path)
    with TestClient(app) as client:
        result = client.post("/api/v1/load-model", json={"model_path": str(path)})
        assert result.status_code == 422
        assert "verified M5" in result.json()["detail"]
        assert client.get("/api/v1/health").json()["model_loaded"] is False
