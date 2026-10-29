"""Load M5 wide sales and development long sales into the shared SQL schema."""

from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

from forecastops.utils.config import DatabaseConfig
from forecastops.utils.logger import setup_logger


class DataLoader:
    def __init__(self, config=None):
        self.config = config or {}
        values = self.config.get("database", self.config)
        self.db_config = (
            values
            if isinstance(values, DatabaseConfig)
            else DatabaseConfig(
                **{
                    k: v
                    for k, v in values.items()
                    if k in {"host", "port", "name", "user", "password"}
                }
            )
        )
        self.conn = None
        self.logger = setup_logger(__name__)

    def connect(self):
        if self.conn is None:
            self.conn = psycopg2.connect(
                host=self.db_config.host,
                port=self.db_config.port,
                dbname=self.db_config.name,
                user=self.db_config.user,
                password=self.db_config.password,
            )
            self.conn.autocommit = True

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    def _insert(self, table, columns, frame, conflict):
        self.connect()
        if frame.empty:
            return
        records = (
            frame[columns].astype(object).where(pd.notna(frame[columns]), None).values.tolist()
        )
        query = sql.SQL("INSERT INTO {} ({}) VALUES %s ON CONFLICT {} DO NOTHING").format(
            sql.Identifier(table),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(conflict),
        )
        with self.conn.cursor() as cur:
            execute_values(cur, query.as_string(self.conn), records)

    def load_calendar(self, csv_path):
        frame = pd.read_csv(csv_path)
        for col in ["snap_ca", "snap_tx", "snap_wi"]:
            frame[col] = frame[col].fillna(0).astype(bool)
        for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
            if col not in frame:
                frame[col] = None
        self._insert(
            "calendar",
            [
                "date",
                "d",
                "wm_yr_wk",
                "weekday",
                "wday",
                "month",
                "year",
                "snap_ca",
                "snap_tx",
                "snap_wi",
                "event_name_1",
                "event_type_1",
                "event_name_2",
                "event_type_2",
            ],
            frame,
            "(date)",
        )

    def load_stores(self, csv_path):
        frame = pd.read_csv(csv_path)[["store_id", "state_id"]].drop_duplicates()
        self._insert("stores", ["store_id", "state_id"], frame, "(store_id)")

    def load_products(self, csv_path):
        frame = pd.read_csv(csv_path)
        if "product_id" not in frame:
            frame["product_id"] = frame["item_id"]
        if "item_id" not in frame:
            frame["item_id"] = frame["product_id"]
        for column in ["dept_id", "cat_id"]:
            if column not in frame:
                frame[column] = None
        self._insert(
            "products",
            ["product_id", "item_id", "dept_id", "cat_id"],
            frame.drop_duplicates("product_id"),
            "(product_id)",
        )

    def _product_ids(self, frame):
        if "product_id" not in frame:
            with self.conn.cursor() as cur:
                cur.execute("SELECT item_id, product_id FROM products")
                mapping = dict(cur.fetchall())
            frame["product_id"] = frame["item_id"].map(mapping)
            if frame["product_id"].isna().any():
                raise ValueError("Load products before prices/sales; unknown item_id found")
        return frame

    def load_prices(self, csv_path):
        self.connect()
        frame = self._product_ids(pd.read_csv(csv_path))
        with self.conn.cursor() as cur:
            cur.execute("SELECT wm_yr_wk, MIN(date) FROM calendar GROUP BY wm_yr_wk")
            dates = dict(cur.fetchall())
        frame["price_date"] = frame["wm_yr_wk"].map(dates)
        self._insert(
            "prices",
            ["product_id", "store_id", "wm_yr_wk", "sell_price", "price_date"],
            frame,
            "(product_id, store_id, wm_yr_wk)",
        )

    def load_sales(self, csv_path):
        self.connect()
        frame = self._product_ids(pd.read_csv(csv_path))
        if not {"d", "sales"}.issubset(frame.columns):
            days = [c for c in frame if c.startswith("d_")]
            if not days:
                raise ValueError("Sales must contain d/sales or M5 d_* columns")
            frame = frame.melt(
                id_vars=["product_id", "store_id"],
                value_vars=days,
                var_name="d",
                value_name="sales",
            )
        for column, setting in [("product_id", "subset_products"), ("store_id", "subset_stores")]:
            count = self.config.get(setting, -1)
            if count > 0:
                frame = frame[frame[column].isin(frame[column].unique()[:count])]
        with self.conn.cursor() as cur:
            cur.execute("SELECT d, date FROM calendar")
            dates = dict(cur.fetchall())
        frame["date"] = frame["d"].map(dates)
        if frame["date"].isna().any():
            raise ValueError("Sales contains days missing from calendar")
        count = self.config.get("subset_days", -1)
        if count > 0:
            frame = frame[frame["date"].isin(sorted(frame["date"].unique())[:count])]
        self._insert(
            "sales",
            ["product_id", "store_id", "d", "date", "sales"],
            frame,
            "(product_id, store_id, d)",
        )

    def load_from_directory(self, data_dir):
        directory = Path(data_dir)
        sales = directory / "sales_train_validation.csv"
        self.load_calendar(directory / "calendar.csv")
        self.load_stores(directory / "stores.csv" if (directory / "stores.csv").exists() else sales)
        self.load_products(
            directory / "products.csv" if (directory / "products.csv").exists() else sales
        )
        self.load_prices(directory / "sell_prices.csv")
        self.load_sales(sales)
