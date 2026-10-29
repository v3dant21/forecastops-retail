#!/usr/bin/env python3
"""Inspect and clean a company register while preserving source-row lineage."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecastops.data.spreadsheet_review import inspect_workbook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--analysis-json")
    args = parser.parse_args()
    result = inspect_workbook(args.workbook, args.output_dir)
    if args.analysis_json:
        Path(args.analysis_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["report"], indent=2))


if __name__ == "__main__":
    main()
