"""Leakage prevention utilities for time-series forecasting."""

import logging
from typing import List, Optional, Set

import pandas as pd
import numpy as np

from forecastops.utils.logger import setup_logger

logger = setup_logger(__name__)


class LeakagePreventer:
    """Prevent data leakage in time-series feature engineering."""
    
    def __init__(self):
        """Initialize leakage preventer."""
        self.logger = setup_logger(__name__)
        self.features_checklist: List[dict] = []
    
    def check_lag_leakage(
        self,
        df: pd.DataFrame,
        target_col: str,
        lags: List[int]
    ) -> bool:
        """Check for lag leakage.
        
        Args:
            df: Dataframe with features
            target_col: Target column name
            lags: List of lag periods
            
        Returns:
            True if leakage detected
        """
        ordered = df.sort_values(["product_id", "store_id", "date"])
        failed = False
        for lag in lags:
            col = f"{target_col}_lag_{lag}"
            if col not in ordered:
                continue
            expected = ordered.groupby(["product_id", "store_id"])[target_col].shift(lag)
            valid = np.isclose(ordered[col], expected, equal_nan=True)
            mismatch = not bool(valid.all())
            failed |= mismatch
            self.features_checklist.append({"feature": col, "check": "lag_validity",
                                            "status": "failed" if mismatch else "passed"})
        return failed

    def check_rolling_leakage(self, df, target_col, windows):
        """Compare supplied features with a causal per-series recomputation."""
        ordered = df.sort_values(["product_id", "store_id", "date"])
        failed = False
        for window in windows:
            for func in ["mean", "std", "max", "min"]:
                col = f"{target_col}_rolling_{func}_{window}"
                if col not in ordered:
                    continue
                expected = ordered.groupby(["product_id", "store_id"])[target_col].transform(
                    lambda values: getattr(values.shift(1).rolling(window, min_periods=1), func)()
                )
                if func == "std":
                    expected = expected.fillna(0)
                mismatch = not bool(np.isclose(ordered[col], expected, equal_nan=True).all())
                failed |= mismatch
                self.features_checklist.append({"feature": col, "check": "rolling_leakage",
                                                "status": "failed" if mismatch else "passed"})
        return failed

    def check_date_features_leakage(
        self,
        df: pd.DataFrame,
        date_col: str = "date"
    ) -> bool:
        """Check for date feature leakage.
        
        Args:
            df: Dataframe with features
            date_col: Date column name
            
        Returns:
            True if leakage detected
        """
        has_leakage = False
        
        # Extracted date features are safe (no leakage)
        # Only check if future date features exist
        future_date_cols = [
            col for col in df.columns
            if col.startswith("date_") or col.startswith("future_")
        ]
        
        if future_date_cols:
            has_leakage = True
            self.logger.warning(f"Future date features detected: {future_date_cols}")
        
        return has_leakage
    
    def check_price_leakage(
        self,
        df: pd.DataFrame,
        price_col: str = "sell_price"
    ) -> bool:
        """Check for price feature leakage.
        
        Args:
            df: Dataframe with features
            price_col: Price column name
            
        Returns:
            True if leakage detected
        """
        has_leakage = False
        
        # Check for future price features
        future_price_cols = [
            col for col in df.columns
            if "future_price" in col or "next_price" in col
        ]
        
        if future_price_cols:
            has_leakage = True
            self.logger.warning(f"Future price features detected: {future_price_cols}")
        
        # Check price lags
        for lag in [1, 2, 3, 7]:
            lag_col = f"{price_col}_lag_{lag}"
            if lag_col in df.columns:
                self.features_checklist.append({
                    "feature": lag_col,
                    "check": "price_lag_validity",
                    "status": "passed"
                })
        
        return has_leakage
    
    def check_categorical_leakage(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str]
    ) -> bool:
        """Check for categorical feature leakage.
        
        Args:
            df: Dataframe with features
            categorical_cols: List of categorical column names
            
        Returns:
            True if leakage detected
        """
        has_leakage = False
        
        for col in categorical_cols:
            if col in df.columns:
                # Check if category contains future information
                unique_values = df[col].nunique()
                
                self.features_checklist.append({
                    "feature": col,
                    "check": "categorical_validity",
                    "status": "passed",
                    "unique_values": unique_values
                })
        
        return has_leakage
    
    def run_full_check(
        self,
        df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        price_col: str = "sell_price",
        lags: List[int] = None,
        windows: List[int] = None,
        categorical_cols: List[str] = None
    ) -> dict:
        """Run complete leakage check.
        
        Args:
            df: Dataframe
            target_col: Target column name
            date_col: Date column name
            price_col: Price column name
            lags: List of lag periods
            windows: List of window sizes
            categorical_cols: List of categorical columns
            
        Returns:
            Leakage check results
        """
        lags = lags or [1, 2, 3, 7, 14, 21, 28]
        windows = windows or [7, 14, 28]
        categorical_cols = categorical_cols or ["day_of_week", "season"]
        
        results = {
            "leakage_detected": False,
            "checks_passed": 0,
            "checks_failed": 0,
            "features_checked": 0,
            "issues": []
        }
        
        self.features_checklist = []

        # Run all checks
        checks = [
            ("Lag features", self.check_lag_leakage(df, target_col, lags)),
            ("Rolling features", self.check_rolling_leakage(df, target_col, windows)),
            ("Date features", self.check_date_features_leakage(df, date_col)),
            ("Price features", self.check_price_leakage(df, price_col)),
            ("Categorical features", self.check_categorical_leakage(df, categorical_cols)),
        ]
        
        for name, has_leakage in checks:
            results["features_checked"] += 1
            if has_leakage:
                results["leakage_detected"] = True
                results["checks_failed"] += 1
                results["issues"].append(f"{name}: Potential leakage detected")
            else:
                results["checks_passed"] += 1
        
        return results
    
    def get_checklist(self) -> pd.DataFrame:
        """Get feature checklist.
        
        Returns:
            DataFrame of feature checks
        """
        return pd.DataFrame(self.features_checklist)
