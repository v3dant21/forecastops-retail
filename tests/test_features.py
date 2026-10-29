"""Tests for feature engineering module."""

import pytest
import pandas as pd
import numpy as np

from forecastops.features.engineer import FeatureEngineer
from forecastops.features.leakage import LeakagePreventer
from forecastops.utils.config import Config


class TestFeatureEngineer:
    """Test FeatureEngineer class."""
    
    def test_init(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer()
        assert len(engineer.feature_definitions) == 0
    
    def test_define_feature(self):
        """Test feature definition."""
        engineer = FeatureEngineer()
        engineer.define_feature("test_feature", "Test description")
        
        assert "test_feature" in engineer.feature_definitions
        assert engineer.feature_definitions["test_feature"] == "Test description"
    
    def test_get_feature_definitions(self):
        """Test feature definitions as DataFrame."""
        engineer = FeatureEngineer()
        engineer.define_feature("feature_1", "Description 1")
        engineer.define_feature("feature_2", "Description 2")
        
        df = engineer.get_feature_definitions()
        
        assert len(df) == 2
        assert "feature" in df.columns
        assert "description" in df.columns
    
    def test_engineer_features(self):
        """Test full feature engineering pipeline."""
        config = Config()
        engineer = FeatureEngineer(config)
        
        # Create test data
        df = pd.DataFrame({
            "product_id": ["p1"] * 30,
            "store_id": ["s1"] * 30,
            "date": pd.date_range("2021-01-01", periods=30),
            "sales": list(range(30)),
            "sell_price": [10.0 + i * 0.1 for i in range(30)],
            "snap_ca": [1] * 30,
            "event_name_1": [None] * 30
        })
        
        result = engineer.engineer_features(df)
        
        # Check date features
        assert "year" in result.columns
        assert "month" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
        
        # Check lag features
        assert "sales_lag_1" in result.columns
        assert "sales_lag_7" in result.columns
        
        # Check rolling features
        assert "sales_rolling_mean_7" in result.columns
        
        # Check price features
        assert "price_lag_1" in result.columns
        assert "price_change_1" in result.columns
        
        # Check event features
        assert "has_snap_benefit" in result.columns
        
        # Check no NaN values
        assert not result.empty
        assert result.select_dtypes(include=[np.number]).isna().sum().sum() == 0
    
    def test_engineer_features_with_custom_params(self):
        """Test feature engineering with custom parameters."""
        config = Config()
        engineer = FeatureEngineer(config)
        
        df = pd.DataFrame({
            "product_id": ["p1"] * 20,
            "store_id": ["s1"] * 20,
            "date": pd.date_range("2021-01-01", periods=20),
            "sales": list(range(20))
        })
        
        result = engineer.engineer_features(
            df,
            lags=[1, 3],
            rolling_windows=[3],
            rolling_functions=["mean"]
        )
        
        assert "sales_lag_1" in result.columns
        assert "sales_lag_3" in result.columns
        assert "sales_rolling_mean_3" in result.columns


class TestLeakagePreventer:
    """Test LeakagePreventer class."""
    
    def test_check_lag_leakage(self):
        """Test lag leakage detection."""
        preventer = LeakagePreventer()
        
        df = pd.DataFrame({
            "product_id": ["p1"] * 30,
            "store_id": ["s1"] * 30,
            "date": pd.date_range("2021-01-01", periods=30),
            "sales": list(range(30)),
            "sales_lag_1": [np.nan] + list(range(29))
        })
        
        result = preventer.check_lag_leakage(df, "sales", [1, 2, 7])
        assert result is False
    
    def test_check_rolling_leakage(self):
        """Test rolling leakage detection."""
        preventer = LeakagePreventer()
        
        df = pd.DataFrame({
            "product_id": ["p1"] * 30,
            "store_id": ["s1"] * 30,
            "date": pd.date_range("2021-01-01", periods=30),
            "sales": list(range(30)),
            "sales_rolling_mean_7": pd.Series(range(30)).shift(1).rolling(7, min_periods=1).mean()
        })
        
        result = preventer.check_rolling_leakage(df, "sales", [7])
        assert result is False
    
    def test_check_date_features_leakage(self):
        """Test date feature leakage detection."""
        preventer = LeakagePreventer()
        
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=30),
            "sales": list(range(30)),
            "is_weekend": [0, 1] * 15
        })
        
        result = preventer.check_date_features_leakage(df, "date")
        assert result is False
    
    def test_run_full_check(self):
        """Test full leakage check."""
        preventer = LeakagePreventer()
        
        df = pd.DataFrame({
            "product_id": ["p1"] * 30,
            "store_id": ["s1"] * 30,
            "date": pd.date_range("2021-01-01", periods=30),
            "sales": list(range(30)),
            "sell_price": [10.0] * 30,
            "snap_ca": [1] * 30,
            "day_of_week": list(range(7)) * 4 + list(range(2))
        })
        
        result = preventer.run_full_check(df, lags=[1, 2], windows=[7])
        
        assert result["leakage_detected"] is False
        assert result["checks_passed"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
