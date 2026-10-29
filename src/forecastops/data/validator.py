"""Data validation module for ForecastOps."""

import logging
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

from forecastops.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataValidator:
    """Validator for M5 Forecasting data quality."""
    
    def __init__(self):
        """Initialize validator."""
        self.logger = setup_logger(__name__)
        self.validation_results: Dict = {}
    
    def validate_sales(self, df: pd.DataFrame) -> Dict:
        """Validate sales data.
        
        Args:
            df: Sales dataframe
            
        Returns:
            Validation results dictionary
        """
        results = {
            "total_rows": len(df),
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        # Check required columns
        required_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "sales"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            results["errors"].append(f"Missing columns: {missing_cols}")
            return results
        
        # Check for null values
        null_counts = df[["sales"]].isnull().sum().to_dict()
        for col, count in null_counts.items():
            if count > 0:
                results["warnings"].append(f"Column '{col}' has {count} null values")
        
        # Check sales values
        if "sales" in df.columns:
            results["metrics"]["min_sales"] = int(df["sales"].min())
            results["metrics"]["max_sales"] = int(df["sales"].max())
            results["metrics"]["mean_sales"] = float(df["sales"].mean())
            results["metrics"]["zero_sales_count"] = int((df["sales"] == 0).sum())
            
            if (df["sales"] < 0).any():
                results["errors"].append("Negative sales values found")
        
        # Check ID formats
        if "id" in df.columns:
            sample_ids = df["id"].head(5).tolist()
            self.logger.debug(f"Sample IDs: {sample_ids}")
        
        self.validation_results["sales"] = results
        return results
    
    def validate_calendar(self, df: pd.DataFrame) -> Dict:
        """Validate calendar data.
        
        Args:
            df: Calendar dataframe
            
        Returns:
            Validation results dictionary
        """
        results = {
            "total_rows": len(df),
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        required_cols = ["date", "d", "wm_yr_wk", "weekday", "wday", "month", "year"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            results["errors"].append(f"Missing columns: {missing_cols}")
            return results
        
        # Check date format
        try:
            df["date"] = pd.to_datetime(df["date"])
            results["metrics"]["date_range"] = {
                "start": str(df["date"].min()),
                "end": str(df["date"].max())
            }
        except Exception as e:
            results["errors"].append(f"Date parsing failed: {e}")
        
        # Check for duplicates
        date_duplicates = df["date"].duplicated().sum()
        if date_duplicates > 0:
            results["errors"].append(f"{date_duplicates} duplicate dates found")
        
        # Check date continuity
        sorted_dates = df["date"].sort_values()
        date_diffs = sorted_dates.diff().dropna()
        non_one_days = (date_diffs != pd.Timedelta(days=1)).sum()
        
        if non_one_days > 0:
            results["warnings"].append(f"{non_one_days} non-consecutive days found")
        
        self.validation_results["calendar"] = results
        return results
    
    def validate_prices(self, df: pd.DataFrame) -> Dict:
        """Validate prices data.
        
        Args:
            df: Prices dataframe
            
        Returns:
            Validation results dictionary
        """
        results = {
            "total_rows": len(df),
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        required_cols = ["product_id", "store_id", "wm_yr_wk", "sell_price"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            results["errors"].append(f"Missing columns: {missing_cols}")
            return results
        
        # Check for null prices
        null_prices = df["sell_price"].isnull().sum()
        if null_prices > 0:
            results["warnings"].append(f"{null_prices} null prices found")
        
        # Check price range
        results["metrics"]["min_price"] = float(df["sell_price"].min())
        results["metrics"]["max_price"] = float(df["sell_price"].max())
        results["metrics"]["mean_price"] = float(df["sell_price"].mean())
        
        # Check for zero or negative prices
        invalid_prices = (df["sell_price"] <= 0).sum()
        if invalid_prices > 0:
            results["errors"].append(f"{invalid_prices} zero or negative prices found")
        
        self.validation_results["prices"] = results
        return results
    
    def validate_all(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Validate all dataframes.
        
        Args:
            data: Dictionary of dataframe names to dataframes
            
        Returns:
            Combined validation results
        """
        results = {}
        
        for name, df in data.items():
            self.logger.info(f"Validating {name} data...")
            results[name] = getattr(self, f"validate_{name}")(df)
        
        # Check relationships
        self._validate_relationships(data, results)
        
        self.validation_results["combined"] = results
        return results
    
    def _validate_relationships(self, data: Dict[str, pd.DataFrame], results: Dict) -> None:
        """Validate relationships between tables.
        
        Args:
            data: Dictionary of dataframes
            results: Validation results dictionary
        """
        if "sales" in data and "products" in data:
            sales_products = set(data["sales"]["product_id"].unique())
            valid_products = set(data["products"]["product_id"].unique())
            
            invalid_products = sales_products - valid_products
            if invalid_products:
                results["sales"]["errors"].append(
                    f"{len(invalid_products)} sales products not in products table"
                )
        
        if "sales" in data and "stores" in data:
            sales_stores = set(data["sales"]["store_id"].unique())
            valid_stores = set(data["stores"]["store_id"].unique())
            
            invalid_stores = sales_stores - valid_stores
            if invalid_stores:
                results["sales"]["errors"].append(
                    f"{len(invalid_stores)} sales stores not in stores table"
                )
        
        if "prices" in data and "sales" in data:
            price_products = set(data["prices"]["product_id"].unique())
            sales_products = set(data["sales"]["product_id"].unique())
            
            missing_products = price_products - sales_products
            if missing_products:
                results["prices"]["warnings"].append(
                    f"{len(missing_products)} price products not in sales data"
                )

    def get_summary(self) -> str:
        """Get validation summary as string.
        
        Returns:
            Summary string
        """
        if not self.validation_results:
            return "No validation results available"
        
        lines = ["=" * 60, "DATA VALIDATION SUMMARY", "=" * 60]
        
        for name, results in self.validation_results.items():
            if name == "combined":
                continue
            
            lines.append(f"\n{name.upper()}:")
            lines.append(f"  Total rows: {results.get('total_rows', 'N/A')}")
            lines.append(f"  Errors: {len(results.get('errors', []))}")
            lines.append(f"  Warnings: {len(results.get('warnings', []))}")
            
            if results.get("errors"):
                for error in results["errors"][:5]:
                    lines.append(f"    - {error}")
            
            if results.get("warnings"):
                for warning in results["warnings"][:5]:
                    lines.append(f"    - {warning}")
        
        return "\n".join(lines)
