#!/usr/bin/env python3
"""Prepare actual M5 files for training; PostgreSQL is optional archival storage."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecastops.data.m5 import prepare_m5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--features-output", default="data/features/features.parquet")
    parser.add_argument("--subset-products", type=int, default=100)
    parser.add_argument("--subset-stores", type=int, default=5)
    parser.add_argument("--subset-days", type=int, default=730)
    args = parser.parse_args()
    frame = prepare_m5(
        args.data_dir,
        args.features_output,
        args.subset_products,
        args.subset_stores,
        args.subset_days,
    )
    print(
        f"Prepared {len(frame):,} actual M5 observations across "
        f"{frame.product_id.nunique()} products and {frame.store_id.nunique()} stores"
    )


if __name__ == "__main__":
    main()
