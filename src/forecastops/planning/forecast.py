"""Causal recursive demand forecasts and plain-language stock calculations."""

import math

import numpy as np
import pandas as pd

from forecastops.features.engineer import FeatureEngineer

KEYS = ["product_id", "store_id"]


def recursive_forecast(model, history, calendar, prices, horizon=28):
    """Roll forward without reading future sales; append our own predictions."""
    if not 1 <= horizon <= 28:
        raise ValueError("Forecast horizon must be between 1 and 28 days")
    history = history.sort_values(KEYS + ["date"]).copy()
    origin = history.date.max()
    latest = history[history.date == origin].copy()
    if latest.duplicated(KEYS).any():
        raise ValueError("Duplicate product/store observations at forecast origin")
    metadata = latest[KEYS + ["dept_id", "cat_id", "state_id"]]
    current = history.merge(metadata[KEYS], on=KEYS, validate="many_to_one")
    current = current.groupby(KEYS, sort=False).tail(56)
    outputs = []
    for step in range(1, horizon + 1):
        date = origin + pd.Timedelta(days=step)
        day = calendar[calendar.date == date]
        if len(day) != 1:
            raise ValueError(f"Calendar does not cover {date.date()}")
        future = metadata.merge(day, how="cross")
        future = future.merge(prices, on=KEYS + ["wm_yr_wk"], how="left", validate="many_to_one")
        last_prices = current.sort_values("date").groupby(KEYS).sell_price.last()
        fallback = future.set_index(KEYS).index.map(last_prices)
        carried = future.sell_price.isna()
        future["sell_price"] = future.sell_price.fillna(pd.Series(fallback, index=future.index))
        if future.sell_price.isna().any():
            raise ValueError("No known price for a product; forecast cannot be prepared")
        future["sales"] = np.nan
        candidate = pd.concat([current, future], ignore_index=True)
        engineered = FeatureEngineer().engineer_features(candidate)
        today = engineered[engineered.date == date]
        if len(today) != len(metadata):
            raise ValueError("Insufficient observed history for all selected product/store pairs")
        predicted = today[KEYS].copy()
        predicted["prediction"] = np.asarray(model.predict(today), dtype=float)
        if not np.isfinite(predicted.prediction).all():
            raise ValueError("Forecast produced non-finite values")
        future = future.merge(predicted, on=KEYS, validate="one_to_one")
        future["price_carried_forward"] = carried.to_numpy()
        outputs.append(future.drop(columns="sales"))
        future["sales"] = future.prediction
        current = pd.concat(
            [current, future.drop(columns=["prediction", "price_carried_forward"])],
            ignore_index=True,
        ).sort_values(KEYS + ["date"])
        current = current.groupby(KEYS, sort=False).tail(56)
    return pd.concat(outputs, ignore_index=True)


def stock_plan(demand, on_hand, incoming=0, buffer_percent=10):
    """Scenario only: entered inventory and buffer are retailer assumptions."""
    if min(demand, on_hand, incoming, buffer_percent) < 0:
        raise ValueError("Stock-planning inputs cannot be negative")
    target = math.ceil(demand * (1 + buffer_percent / 100))
    return {
        "target_stock": target,
        "suggested_order": max(0, target - on_hand - incoming),
        "available_stock": on_hand + incoming,
    }


def retail_summary(backtest):
    frame = backtest.copy()
    frame["absolute_error"] = (frame.sales - frame.prediction).abs()
    frame["missed_units"] = (frame.sales - frame.prediction).clip(lower=0)
    frame["excess_units"] = (frame.prediction - frame.sales).clip(lower=0)
    totals = frame.groupby(KEYS + ["cat_id"], as_index=False).agg(
        actual_units=("sales", "sum"),
        predicted_units=("prediction", "sum"),
        absolute_error=("absolute_error", "sum"),
        daily_error=("absolute_error", "mean"),
        missed_units=("missed_units", "sum"),
        excess_units=("excess_units", "sum"),
    )
    totals["error_percent"] = np.where(
        totals.actual_units > 0,
        100 * totals.absolute_error / totals.actual_units.replace(0, np.nan),
        np.nan,
    )
    return totals
