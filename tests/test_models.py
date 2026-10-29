"""Tests for model implementations."""

import pytest
import pandas as pd
import numpy as np

from forecastops.models.baseline import NaiveForecaster, SeasonalNaiveForecaster, LinearRegressor


class TestBaselineModels:
    """Test baseline models."""
    
    def test_naive_forecaster(self):
        """Test NaiveForecaster."""
        df = pd.DataFrame({
            "product_id": ["p1"] * 10,
            "store_id": ["s1"] * 10,
            "date": pd.date_range("2021-01-01", periods=10)
        })
        y_train = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        df_test = pd.DataFrame({
            "product_id": ["p1"],
            "store_id": ["s1"],
            "date": pd.date_range("2021-01-11", periods=1)
        })
        
        model = NaiveForecaster()
        model.fit(df, y_train)
        
        predictions = model.predict(df_test)
        
        assert len(predictions) == 1
        assert predictions[0] == 10
    
    def test_seasonal_naive_forecaster(self):
        """Test SeasonalNaiveForecaster."""
        df = pd.DataFrame({
            "product_id": ["p1"] * 21,
            "store_id": ["s1"] * 21,
            "date": pd.date_range("2021-01-01", periods=21)
        })
        y_train = pd.Series(list(range(21)))
        
        model = SeasonalNaiveForecaster(season_length=7)
        model.fit(df, y_train)
        
        df_test = pd.DataFrame({
            "product_id": ["p1"],
            "store_id": ["s1"],
            "date": pd.date_range("2021-01-22", periods=1)
        })
        
        predictions = model.predict(df_test)
        
        assert len(predictions) == 1
    
    def test_linear_regressor(self):
        """Test LinearRegressor."""
        df = pd.DataFrame({
            "feature_1": list(range(10)),
            "feature_2": [i * 2 for i in range(10)],
            "date": pd.date_range("2021-01-01", periods=10)
        })
        y_train = pd.Series([i * 3 for i in range(10)])
        
        model = LinearRegressor()
        model.fit(df, y_train)
        
        df_test = pd.DataFrame({
            "feature_1": [10],
            "feature_2": [20],
            "date": pd.date_range("2021-01-11", periods=1)
        })
        
        predictions = model.predict(df_test)
        
        assert len(predictions) == 1
        assert predictions[0] >= 0


from forecastops.models.tree import XGBoostForecaster, LightGBMForecaster, CatBoostForecaster


class TestTreeModels:
    """Test tree-based models."""
    
    def test_xgboost_forecaster(self):
        """Test XGBoostForecaster."""
        pytest.importorskip("xgboost")
        
        df = pd.DataFrame({
            "feature_1": list(range(20)),
            "feature_2": [i * 2 for i in range(20)],
            "date": pd.date_range("2021-01-01", periods=20)
        })
        y_train = pd.Series([i * 3 for i in range(20)])
        
        model = XGBoostForecaster({"xgboost": {"n_jobs": 1}})
        model.fit(df, y_train)
        
        df_test = pd.DataFrame({
            "feature_1": [20],
            "feature_2": [40],
            "date": pd.date_range("2021-01-21", periods=1)
        })
        
        predictions = model.predict(df_test)
        
        assert len(predictions) == 1
        assert predictions[0] >= 0
    
    def test_lightgbm_forecaster(self):
        """Test LightGBMForecaster."""
        pytest.importorskip("lightgbm")
        
        df = pd.DataFrame({
            "feature_1": list(range(20)),
            "feature_2": [i * 2 for i in range(20)],
            "date": pd.date_range("2021-01-01", periods=20)
        })
        y_train = pd.Series([i * 3 for i in range(20)])
        
        model = LightGBMForecaster({"lightgbm": {"n_jobs": 1}})
        model.fit(df, y_train)
        
        df_test = pd.DataFrame({
            "feature_1": [20],
            "feature_2": [40],
            "date": pd.date_range("2021-01-21", periods=1)
        })
        
        predictions = model.predict(df_test)
        
        assert len(predictions) == 1
        assert predictions[0] >= 0
    
    def test_catboost_forecaster(self):
        """Test CatBoostForecaster."""
        pytest.importorskip("catboost")
        
        df = pd.DataFrame({
            "feature_1": list(range(20)),
            "feature_2": [i * 2 for i in range(20)],
            "date": pd.date_range("2021-01-01", periods=20)
        })
        y_train = pd.Series([i * 3 for i in range(20)])
        
        model = CatBoostForecaster({"catboost": {"thread_count": 1}})
        model.fit(df, y_train)
        
        df_test = pd.DataFrame({
            "feature_1": [20],
            "feature_2": [40],
            "date": pd.date_range("2021-01-21", periods=1)
        })
        
        predictions = model.predict(df_test)
        
        assert len(predictions) == 1
        assert predictions[0] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
