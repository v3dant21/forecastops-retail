#!/usr/bin/env python3
"""Download the pinned, checksum-verified Walmart M5 archive."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecastops.data.m5 import download_m5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default="data/raw")
    parser.add_argument("--archive", help="Existing copy of the pinned M5 ZIP")
    args = parser.parse_args()
    manifest = download_m5(args.output, args.archive)
    print(f"Verified real M5 files from {manifest['source_page']}")


if __name__ == "__main__":
    main()
