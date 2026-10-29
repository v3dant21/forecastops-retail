#!/usr/bin/env python3
"""Validate competing methods on actual M5 history, then build a retail plan."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecastops.data.m5 import require_real_features
from forecastops.evaluation.metrics import calculate_all_metrics
from forecastops.models.baseline import SeasonalNaiveForecaster
from forecastops.models.tree import XGBoostForecaster
from forecastops.planning.forecast import KEYS, recursive_forecast
from forecastops.tracking.tracker import MLflowTracker
from forecastops.utils.config import Config
from forecastops.utils.helpers import set_seed


def new_model(name):
    if name == "seasonal_naive":
        return SeasonalNaiveForecaster(7)
    return XGBoostForecaster(
        {
            "xgboost": {
                "objective": "count:poisson",
                "n_estimators": 250,
                "max_depth": 6,
                "learning_rate": 0.05,
                "nthread": 4,
            }
        }
    )


def fit(model, features):
    return model.fit(features.drop(columns=["sales", "id"], errors="ignore"), features.sales)


def score_rollout(model, history, actual, calendar, prices, horizon):
    forecast = recursive_forecast(model, history, calendar, prices, horizon)
    result = forecast.merge(
        actual[KEYS + ["date", "sales"]], on=KEYS + ["date"], how="left", validate="one_to_one"
    )
    if result.sales.isna().any():
        raise ValueError("Holdout forecast is missing actual outcomes")
    return result, {
        k: float(v)
        for k, v in calculate_all_metrics(
            result.sales.to_numpy(), result.prediction.to_numpy()
        ).items()
    }


def build_plan(features_dir="data/features", output_dir="reports", models_dir="models"):
    set_seed(42)
    source = Path(features_dir)
    features = pd.read_parquet(source / "features.parquet")
    provenance = require_real_features(features)
    history = pd.read_parquet(source / "history.parquet")
    calendar = pd.read_parquet(source / "calendar.parquet")
    prices = pd.read_parquet(source / "prices.parquet")
    latest = features.date.max()
    validation_origin = latest - pd.Timedelta(days=56)
    test_origin = latest - pd.Timedelta(days=28)
    if features.date.min() >= validation_origin - pd.Timedelta(days=56):
        raise ValueError("At least 112 feature days are required for retail validation")
    # Require a complete warmup at the earliest holdout origin. Exclude recently
    # launched products instead of inventing pre-launch history.
    eligible = features[features.date == validation_origin][KEYS]
    features = features.merge(eligible, on=KEYS, validate="many_to_one")
    history = history.merge(eligible, on=KEYS, validate="many_to_one")
    validation_actual = features[
        (features.date > validation_origin) & (features.date <= test_origin)
    ]
    validation_history = history[history.date <= validation_origin]
    candidates = {}
    for name in ["seasonal_naive", "xgboost"]:
        print(f"Fitting and checking {name} on a 28-day validation rollout...", flush=True)
        model = fit(new_model(name), features[features.date <= validation_origin])
        _, candidates[name] = score_rollout(
            model, validation_history, validation_actual, calendar, prices, 28
        )
    selected = min(candidates, key=lambda name: candidates[name]["wape"])
    print(
        f"Selected {selected} using validation only. Checking untouched final 28 days...",
        flush=True,
    )
    test_model = fit(new_model(selected), features[features.date <= test_origin])
    backtest, test_metrics = score_rollout(
        test_model,
        history[history.date <= test_origin],
        features[features.date > test_origin],
        calendar,
        prices,
        28,
    )
    print(
        "Refitting selected method on all observed sales and forecasting next 28 days...",
        flush=True,
    )
    final_model = fit(new_model(selected), features)
    future = recursive_forecast(final_model, history, calendar, prices, 28)
    fingerprint = provenance["files"]["sales_train_validation.csv"]
    report = {
        "source": provenance,
        "selected_model": selected,
        "candidate_validation": candidates,
        "test_metrics": test_metrics,
        "forecast_origin": str(latest.date()),
        "test_start": str((test_origin + pd.Timedelta(days=1)).date()),
        "test_end": str(latest.date()),
        "validation_end": str(test_origin.date()),
        "forecast_start": str(future.date.min().date()),
        "forecast_end": str(future.date.max().date()),
        "planning_series": int(len(eligible)),
        "planning_products": int(features.product_id.nunique()),
        "horizon": 28,
        "data_fingerprint": fingerprint,
        "evaluation": "28-day recursive holdout; no future sales used",
        "assumptions": [
            "Historical calendar and listed weekly prices are treated as known in advance.",
            "Where a future price is missing, the last observed price is carried forward.",
            "Sales are observed demand, not lost sales. Inventory and supplier lead "
            "times are not in M5.",
        ],
        "price_carried_forward_rows": int(future.price_carried_forward.sum()),
    }
    final_model.metadata = {
        "provenance": provenance,
        "metrics": candidates[selected],
        "test_metrics": test_metrics,
        "forecast_origin": report["forecast_origin"],
        "data_fingerprint": fingerprint,
        "evaluation": report["evaluation"],
        "version": "2.0.0",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models = Path(models_dir)
    models.mkdir(parents=True, exist_ok=True)
    # Publish the manifest last. Dashboard checks data/model origin consistency.
    backtest.to_parquet(output / "backtest.parquet", index=False)
    future.to_parquet(output / "forecast.parquet", index=False)
    final_model.save(str(models / "retail_model.pkl"))
    (output / "retail_plan.json").write_text(json.dumps(report, indent=2))
    config = Config()
    tracker = MLflowTracker(config)
    with tracker.start_run(run_name=f"actual_m5_{selected}"):
        tracker.log_parameters(
            {
                "method": selected,
                "source": "actual_m5",
                "horizon": 28,
                "series": len(eligible),
                "data_fingerprint": fingerprint,
            }
        )
        tracker.log_metrics(test_metrics, prefix="test_")
        tracker.log_artifact(str(models / "retail_model.pkl"), "model")
        tracker.log_artifact(str(output / "retail_plan.json"), "report")
    print(
        json.dumps(
            {"selected_model": selected, "test": test_metrics, "planning_series": len(eligible)},
            indent=2,
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()
    build_plan(args.features_dir, args.output_dir, args.models_dir)


if __name__ == "__main__":
    main()
