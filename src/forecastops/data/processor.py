"""Data processor for M5 Forecasting data."""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

from forecastops.utils.config import Config
from forecastops.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataProcessor:
    """Processor for M5 Forecasting data - joins and prepares data for feature engineering."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize processor.
        
        Args:
            config: Optional configuration
        """
        self.config = config or Config()
        self.logger = setup_logger(__name__)
        self.cache: Dict = {}
    
    def load_sales_data(self, db_conn) -> pd.DataFrame:
        """Load sales data from database.
        
        Args:
            db_conn: Database connection
            
        Returns:
            Sales dataframe
        """
        query = """
            SELECT 
                s.id,
                s.product_id,
                p.item_id,
                p.dept_id,
                p.cat_id,
                s.store_id,
                st.state_id,
                s.d,
                s.sales,
                c.date,
                c.wm_yr_wk,
                c.weekday,
                c.wday,
                c.month,
                c.year,
                c.snap_ca,
                c.snap_tx,
                c.snap_wi,
                c.event_name_1,
                c.event_type_1,
                c.event_name_2,
                c.event_type_2
            FROM sales s
            JOIN products p ON p.product_id = s.product_id
            JOIN stores st ON st.store_id = s.store_id
            JOIN calendar c ON s.d = c.d
        """
        
        df = pd.read_sql(query, db_conn)
        self.cache["sales_raw"] = df.copy()
        
        self.logger.info(f"Loaded {len(df)} sales records")
        return df
    
    def load_prices(self, db_conn) -> pd.DataFrame:
        """Load prices data from database.
        
        Args:
            db_conn: Database connection
            
        Returns:
            Prices dataframe
        """
        query = """
            SELECT 
                product_id,
                store_id,
                wm_yr_wk,
                sell_price,
                price_date
            FROM prices
        """
        
        df = pd.read_sql(query, db_conn)
        self.cache["prices"] = df.copy()
        
        self.logger.info(f"Loaded {len(df)} price records")
        return df
    
    def enrich_sales_with_prices(self, sales_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Enrich sales data with price information.
        
        Args:
            sales_df: Sales dataframe
            prices_df: Prices dataframe
            
        Returns:
            Enriched dataframe
        """
        # Merge on product_id, store_id, wm_yr_wk
        merged = sales_df.merge(
            prices_df,
            on=["product_id", "store_id", "wm_yr_wk"],
            how="left"
        )
        
        self.cache["sales_enriched"] = merged.copy()
        self.logger.info("Enriched sales with price data")
        
        return merged
    
    def filter_subset(self, df: pd.DataFrame, config: Optional[Config] = None) -> pd.DataFrame:
        """Filter data to subset for development.
        
        Args:
            df: Input dataframe
            config: Optional configuration override
            
        Returns:
            Filtered dataframe
        """
        config = config or self.config
        data_config = config.data
        
        result = df.copy()
        
        # Filter products
        if data_config.subset_products > 0 and data_config.subset_products < len(result["product_id"].unique()):
            products = result["product_id"].unique()[:data_config.subset_products]
            result = result[result["product_id"].isin(products)]
            self.logger.info(f"Filtered to {len(products)} products")
        
        # Filter stores
        if data_config.subset_stores > 0 and data_config.subset_stores < len(result["store_id"].unique()):
            stores = result["store_id"].unique()[:data_config.subset_stores]
            result = result[result["store_id"].isin(stores)]
            self.logger.info(f"Filtered to {len(stores)} stores")
        
        # Filter dates
        if data_config.subset_days > 0:
            result = result.sort_values("date").head(data_config.subset_days)
            self.logger.info(f"Filtered to {len(result)} records")
        
        self.cache["sales_subset"] = result.copy()
        return result
    
    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create date-based features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with added features
        """
        df = df.copy()
        
        # Ensure date column exists
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
        
        # Seasonal feature
        df["season"] = df["month"].apply(
            lambda m: "Q1" if m in [1, 2, 3] else 
                     "Q2" if m in [4, 5, 6] else 
                     "Q3" if m in [7, 8, 9] else "Q4"
        )
        
        self.logger.info("Created date features")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, lags: list = None) -> pd.DataFrame:
        """Create lagged sales features.
        
        Args:
            df: Input dataframe
            lags: List of lag periods (default: from config)
            
        Returns:
            Dataframe with lag features
        """
        df = df.copy()
        lags = lags or self.config.features.lags
        
        # Sort for correct lag calculation
        df = df.sort_values(["product_id", "store_id", "date"])
        
        for lag in lags:
            col_name = f"sales_lag_{lag}"
            df[col_name] = df.groupby(["product_id", "store_id"])["sales"].shift(lag)
        
        # Keep unavailable history missing.

        self.logger.info(f"Created lag features for periods: {lags}")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
        """Create rolling statistics features.
        
        Args:
            df: Input dataframe
            windows: List of window sizes (default: from config)
            
        Returns:
            Dataframe with rolling features
        """
        df = df.copy()
        windows = windows or self.config.features.rolling_windows
        functions = self.config.features.rolling_functions
        
        # Sort for correct rolling calculation
        df = df.sort_values(["product_id", "store_id", "date"])
        
        for window in windows:
            for func in functions:
                col_name = f"sales_rolling_{func}_{window}"
                if func == "mean":
                    df[col_name] = df.groupby(["product_id", "store_id"])["sales"].transform(
                        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                    )
                elif func == "std":
                    df[col_name] = df.groupby(["product_id", "store_id"])["sales"].transform(
                        lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
                    )
                elif func == "max":
                    df[col_name] = df.groupby(["product_id", "store_id"])["sales"].transform(
                        lambda x: x.shift(1).rolling(window, min_periods=1).max()
                    )
                elif func == "min":
                    df[col_name] = df.groupby(["product_id", "store_id"])["sales"].transform(
                        lambda x: x.shift(1).rolling(window, min_periods=1).min()
                    )
        
        self.logger.info(f"Created rolling features for windows: {windows}")
        return df
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with price features
        """
        df = df.copy()
        
        # Sort for correct price calculation
        df = df.sort_values(["product_id", "store_id", "date"])
        
        # Price lag features
        df["price_lag_1"] = df.groupby(["product_id", "store_id"])["sell_price"].shift(1)
        
        # Price change features
        df["price_change_1"] = df["sell_price"] - df["price_lag_1"]
        df["price_change_pct_1"] = df["price_change_1"] / df["price_lag_1"].replace(0, np.nan)
        
        # Rolling price features
        df["price_rolling_mean_7"] = df.groupby(["product_id", "store_id"])["sell_price"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        )
        
        self.logger.info("Created price features")
        return df
    
    def create_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create event-based features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with event features
        """
        df = df.copy()
        
        # Snap benefit flag
        df["has_snap_benefit"] = (
            df["snap_ca"].fillna(0) | df["snap_tx"].fillna(0) | df["snap_wi"].fillna(0)
        ).astype(int)
        
        # Has any event flag
        df["has_event"] = (
            df["event_name_1"].notna() | df["event_name_2"].notna()
        ).astype(int)
        
        self.logger.info("Created event features")
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with all features
        """
        self.logger.info("Creating features...")
        
        df = self.create_date_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_price_features(df)
        df = self.create_event_features(df)
        
        # Drop rows with NaN values (due to lags/rolling)
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        
        if dropped > 0:
            self.logger.warning(f"Dropped {dropped} rows with NaN values")
        
        self.cache["features_ready"] = df.copy()
        self.logger.info("Feature creation complete")
        
        return df
    
    def prepare_for_training(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare data for training with time-based split.
        
        Args:
            df: Full dataframe with features
            
        Returns:
            Tuple of (train, val, test) dataframes
        """
        from forecastops.utils.helpers import train_val_test_split
        train, val, test = train_val_test_split(
            df, self.config.training.test_split, self.config.training.validation_split
        )

        self.cache["train"] = train
        self.cache["val"] = val
        self.cache["test"] = test
        
        self.logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train, val, test
