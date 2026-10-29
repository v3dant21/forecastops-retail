#!/usr/bin/env python3
"""Clean a sale-invoice export and activate the food-sales dashboard."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecastops.data.invoices import import_invoices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook")
    parser.add_argument("--output-dir", default="data/retail")
    args = parser.parse_args()
    print(json.dumps(import_invoices(args.workbook, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
