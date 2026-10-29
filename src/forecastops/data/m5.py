"""Verified M5 ingestion. Runtime pipelines never create artificial sales."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests

from forecastops.features.engineer import FeatureEngineer

SOURCE_URL = "https://raw.githubusercontent.com/Nixtla/m5-forecasts/72b8e7fd3b565b3c538adcb1d1a05117d8562d7e/datasets/m5.zip"
SOURCE_PAGE = "https://github.com/Nixtla/m5-forecasts"
ARCHIVE_SHA256 = "cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774"
RAW_FILES = ("sales_train_validation.csv", "calendar.csv", "sell_prices.csv")
META = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_m5(output_dir="data/raw", archive_path=None):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    archive = Path(archive_path) if archive_path else directory / "m5.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        with requests.get(SOURCE_URL, stream=True, timeout=(15, 180)) as response:
            response.raise_for_status()
            with open(temporary, "wb") as target:
                for block in response.iter_content(1024 * 1024):
                    target.write(block)
        temporary.replace(archive)
    if checksum(archive) != ARCHIVE_SHA256:
        raise ValueError("M5 archive checksum mismatch; refusing to ingest unverified data")
    with zipfile.ZipFile(archive) as bundle:
        for name in RAW_FILES:
            # Extract only known filenames, never paths supplied by an archive.
            temporary = directory / (name + ".part")
            with bundle.open(name) as source, open(temporary, "wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(directory / name)
    manifest = {
        "source_type": "real_m5",
        "source_url": SOURCE_URL,
        "source_page": SOURCE_PAGE,
        "archive_sha256": ARCHIVE_SHA256,
        "files": {name: checksum(directory / name) for name in RAW_FILES},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_source(directory):
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.exists():
        raise ValueError(
            "Verified M5 data is missing. Run make data; artificial data is not supported."
        )
    manifest = json.loads(path.read_text())
    if manifest.get("source_type") != "real_m5" or manifest.get("archive_sha256") != ARCHIVE_SHA256:
        raise ValueError("Dataset provenance is not the verified M5 source")
    for name in RAW_FILES:
        if checksum(directory / name) != manifest["files"].get(name):
            raise ValueError(f"Source file changed: {name}; download verified M5 data again")
    return manifest


def require_real_features(frame):
    provenance = frame.attrs.get("provenance", {})
    if (
        provenance.get("source_type") != "real_m5"
        or provenance.get("archive_sha256") != ARCHIVE_SHA256
    ):
        raise ValueError("Features lack verified M5 provenance. Run make data before training.")
    if frame.empty:
        raise ValueError("M5 feature dataset is empty")
    return provenance


def prepare_m5(
    directory="data/raw", output="data/features/features.parquet", products=100, stores=5, days=730
):
    """Select actual series before melting wide sales to keep RAM bounded."""
    directory = Path(directory)
    manifest = verify_source(directory)
    sales_path = directory / "sales_train_validation.csv"
    metadata = pd.read_csv(sales_path, usecols=META)
    available_products = int(metadata.item_id.nunique())
    metadata = metadata[metadata.cat_id.eq("FOODS")].copy()
    if metadata.empty:
        raise ValueError("The source contains no food items")
    items = metadata[["item_id", "dept_id"]].drop_duplicates().sort_values("item_id")
    # Interleave departments so a small subset still spans food departments.
    items["rank"] = items.groupby("dept_id").cumcount()
    items = items.sort_values(["rank", "dept_id"])
    item_ids = items.item_id.tolist()[:products] if products > 0 else items.item_id.tolist()
    store_frame = metadata[["store_id", "state_id"]].drop_duplicates().sort_values("store_id")
    store_frame["rank"] = store_frame.groupby("state_id").cumcount()
    store_ids = store_frame.sort_values(["rank", "state_id"]).store_id.tolist()
    if stores > 0:
        store_ids = store_ids[:stores]
    header = pd.read_csv(sales_path, nrows=0).columns
    day_columns = sorted((c for c in header if c.startswith("d_")), key=lambda c: int(c[2:]))
    if days > 0:
        day_columns = day_columns[-(days + 28) :]
    pieces = []
    for chunk in pd.read_csv(sales_path, usecols=META + day_columns, chunksize=1000):
        selected = chunk[chunk.item_id.isin(item_ids) & chunk.store_id.isin(store_ids)]
        if not selected.empty:
            pieces.append(selected)
    wide = pd.concat(pieces, ignore_index=True)
    if not wide.item_id.str.match(r"^FOODS_\d_\d{3}$").all():
        raise ValueError("Unexpected M5 product identifiers")
    history = wide.melt(id_vars=META, value_vars=day_columns, var_name="d", value_name="sales")
    history = history.rename(columns={"item_id": "product_id"})
    calendar = pd.read_csv(directory / "calendar.csv", parse_dates=["date"])
    calendar = calendar.rename(columns={c: c.lower() for c in calendar if c.startswith("snap_")})
    if "d" not in calendar:
        calendar["d"] = "d_" + ((calendar.date - pd.Timestamp("2011-01-29")).dt.days + 1).astype(
            str
        )
    history = history.merge(calendar, on="d", how="left", validate="many_to_one")
    prices = []
    for chunk in pd.read_csv(directory / "sell_prices.csv", chunksize=250000):
        chosen = chunk[chunk.item_id.isin(item_ids) & chunk.store_id.isin(store_ids)]
        if not chosen.empty:
            prices.append(chosen)
    prices = pd.concat(prices, ignore_index=True).rename(columns={"item_id": "product_id"})
    history = history.merge(
        prices, on=["product_id", "store_id", "wm_yr_wk"], how="left", validate="many_to_one"
    )
    history = history.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)
    features = FeatureEngineer().engineer_features(history)
    if days > 0:
        cutoff = history.date.max() - pd.Timedelta(days=days - 1)
        features = features[features.date >= cutoff].copy()
    # Sparse/unlaunched items may have insufficient price history; only retain
    # series that are still observed at the planning origin.
    latest = features.date.max()
    active = features.loc[features.date == latest, ["product_id", "store_id"]]
    features = features.merge(active, on=["product_id", "store_id"], validate="many_to_one")
    provenance = {
        **manifest,
        "products": int(features.product_id.nunique()),
        "stores": sorted(features.store_id.unique().tolist()),
        "series": int(len(active)),
        "available_products": available_products,
        "category_scope": ["FOODS"],
        "available_stores": int(metadata.store_id.nunique()),
        "start_date": str(features.date.min().date()),
        "end_date": str(latest.date()),
        "rows": int(len(features)),
        "scope": "Actual M5 food observations; selected products/stores",
    }
    features.attrs = {"provenance": provenance, "feature_pipeline_version": 3}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(target, index=False)
    # Keep the exact raw history and known calendar/prices needed for causal rollouts.
    history.attrs = features.attrs
    history.to_parquet(target.parent / "history.parquet", index=False)
    calendar.to_parquet(target.parent / "calendar.parquet", index=False)
    prices.to_parquet(target.parent / "prices.parquet", index=False)
    (target.parent / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return features
