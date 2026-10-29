"""Feature engineering module for ForecastOps."""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from forecastops.utils.config import Config
from forecastops.utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    """Feature engineer for time-series forecasting."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize feature engineer.
        
        Args:
            config: Optional configuration
        """
        self.config = config or Config()
        self.logger = setup_logger(__name__)
        self.feature_definitions: Dict[str, str] = {}
    
    def define_feature(self, name: str, description: str) -> None:
        """Define a feature and its description.
        
        Args:
            name: Feature name
            description: Feature description
        """
        self.feature_definitions[name] = description
    
    def get_feature_definitions(self) -> pd.DataFrame:
        """Get feature definitions as DataFrame.
        
        Returns:
            DataFrame with feature names and descriptions
        """
        return pd.DataFrame(
            list(self.feature_definitions.items()),
            columns=["feature", "description"]
        )
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        rolling_functions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Engineer all features for forecasting.
        
        Args:
            df: Input dataframe
            lags: List of lag periods
            rolling_windows: List of rolling window sizes
            rolling_functions: List of rolling functions
            
        Returns:
            Dataframe with engineered features
        """
        lags = lags or self.config.features.lags
        rolling_windows = rolling_windows or self.config.features.rolling_windows
        rolling_functions = rolling_functions or self.config.features.rolling_functions
        
        self.logger.info(f"Engineering features: lags={lags}, windows={rolling_windows}, functions={rolling_functions}")
        
        # Sort for correct feature calculation
        df = df.sort_values(["product_id", "store_id", "date"]).copy()
        
        # Stable numeric identifiers allow tree models to distinguish retail series.
        if "cat_id" in df:
            df["category_code"] = df["cat_id"].map({"FOODS": 1, "HOUSEHOLD": 2, "HOBBIES": 3}).fillna(0).astype(int)
        if "dept_id" in df:
            df["department_number"] = pd.to_numeric(df["dept_id"].str.extract(r"_(\d+)$")[0], errors="coerce").fillna(0)
        df["item_number"] = pd.to_numeric(df["product_id"].str.extract(r"_(\d+)$")[0], errors="coerce").fillna(0)
        df["store_number"] = pd.to_numeric(df["store_id"].str.extract(r"_(\d+)$")[0], errors="coerce").fillna(0)
        if "state_id" in df:
            df["state_code"] = df["state_id"].map({"CA": 1, "TX": 2, "WI": 3}).fillna(0).astype(int)

        # 1. Date features
        df = self._add_date_features(df)
        
        # 2. Lag features
        for lag in lags:
            df = self._add_lag_feature(df, "sales", lag)
        
        # 3. Rolling features
        for window in rolling_windows:
            for func in rolling_functions:
                df = self._add_rolling_feature(df, "sales", window, func)
        
        # 4. Price features
        if "sell_price" in df.columns:
            df = self._add_price_features(df)
        
        # 5. Event features
        if any(col in df.columns for col in ["snap_ca", "snap_tx", "snap_wi"]):
            df = self._add_event_features(df)
        
        # Remove rows with NaN (from lags/rolling)
        required = [c for c in df if c.startswith(("sales_lag_", "sales_rolling_", "price_"))]
        df = df.dropna(subset=required)
        
        df.attrs["feature_pipeline_version"] = 2
        self.logger.info(f"Final feature count: {len(df.columns)}")
        
        return df
    
    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add date-based features."""
        if "date" not in df.columns:
            raise ValueError("A date column is required")
        df["date"] = pd.to_datetime(df["date"])
        
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_month_start"] = (df["date"].dt.is_month_start).astype(int)
        df["is_month_end"] = (df["date"].dt.is_month_end).astype(int)
        
        # Season
        df["season"] = df["month"].apply(
            lambda m: "Q1" if m in [1, 2, 3] else 
                     "Q2" if m in [4, 5, 6] else 
                     "Q3" if m in [7, 8, 9] else "Q4"
        )
        
        # Define features
        for col in df.columns:
            if col.startswith("is_") or col in ["season", "day_of_week"]:
                self.define_feature(col, f"Calendar feature based on date")
        
        return df
    
    def _add_lag_feature(self, df: pd.DataFrame, column: str, lag: int) -> pd.DataFrame:
        """Add lag feature for a column."""
        lag_col = f"{column}_lag_{lag}"
        df[lag_col] = df.groupby(["product_id", "store_id"])[column].shift(lag)
        
        # Do not use current sales as historical information.
        # Leave unavailable history missing; never fill it with the target.
        
        self.define_feature(lag_col, f"{column} value from {lag} periods ago")
        
        return df
    
    def _add_rolling_feature(
        self,
        df: pd.DataFrame,
        column: str,
        window: int,
        func: str
    ) -> pd.DataFrame:
        """Add rolling feature for a column."""
        roll_col = f"{column}_rolling_{func}_{window}"
        
        if func == "mean":
            df[roll_col] = df.groupby(["product_id", "store_id"])[column].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        elif func == "std":
            df[roll_col] = df.groupby(["product_id", "store_id"])[column].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
            )
        elif func == "max":
            df[roll_col] = df.groupby(["product_id", "store_id"])[column].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).max()
            )
        elif func == "min":
            df[roll_col] = df.groupby(["product_id", "store_id"])[column].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).min()
            )
        
        self.define_feature(roll_col, f"{func} of {column} over {window}-period window")
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        if "sell_price" not in df.columns:
            return df
        
        df["price_lag_1"] = df.groupby(["product_id", "store_id"])["sell_price"].shift(1)
        df["price_change_1"] = df["sell_price"] - df["price_lag_1"]
        df["price_change_pct_1"] = df["price_change_1"] / df["price_lag_1"].replace(0, np.nan)
        df["price_change_pct_1"] = df["price_change_pct_1"].fillna(0)
        
        # Rolling price features
        df["price_rolling_mean_7"] = df.groupby(["product_id", "store_id"])["sell_price"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        )
        
        self.define_feature("price_lag_1", "Previous day's sell price")
        self.define_feature("price_change_1", "Absolute price change from previous day")
        self.define_feature("price_change_pct_1", "Percentage price change from previous day")
        self.define_feature("price_rolling_mean_7", "7-day rolling average of sell price")
        
        return df
    
    def _add_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add event-based features."""
        snap_cols = ["snap_ca", "snap_tx", "snap_wi"]
        existing_snap = [c for c in snap_cols if c in df.columns]
        
        if existing_snap:
            if "state_id" in df:
                df["has_snap_benefit"] = 0
                for state in ["CA", "TX", "WI"]:
                    column = f"snap_{state.lower()}"
                    if column in df:
                        mask = df.state_id.eq(state)
                        df.loc[mask, "has_snap_benefit"] = df.loc[mask, column].fillna(0).astype(int)
            else:
                df["has_snap_benefit"] = df[existing_snap].any(axis=1).astype(int)
            self.define_feature("has_snap_benefit", "Flag indicating SNAP benefit day")
        
        if "event_name_1" in df.columns:
            df["has_event"] = df["event_name_1"].notna().astype(int)
            self.define_feature("has_event", "Flag indicating presence of an event")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
        """Get list of feature columns.
        
        Args:
            df: Dataframe
            exclude_cols: Columns to exclude
            
        Returns:
            List of feature column names
        """
        exclude_cols = exclude_cols or []
        exclude_cols.extend(["date", "id", "sales", "product_id", "store_id"])
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        self.logger.info(f"Feature columns: {len(feature_cols)}")
        
        return feature_cols
