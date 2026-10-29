"""Clean nested invoice exports without inventing sales or mixing quantity units."""

import hashlib
import json
import re
import shutil
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from forecastops.data.resolutions import coverage_status, resolution_history
from forecastops.data.retail_database import (
    connect_database,
    invoice_key,
    load_database,
    store_import,
)

FOOD_CHAPTERS = {
    "02",
    "03",
    "04",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
}
UNIT_MAP = {"KGS": "KG", "KG": "KG", "PCS": "PCS", "PC": "PCS", "BGS": "BAG", "BAG": "BAG"}


def text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def product_name(value):
    name = text(value).upper()
    return re.sub(r"(\d)\s*(GM|KG)\b", r"\1 \2", name)


def number(value):
    if pd.isna(value) or text(value) == "":
        return np.nan
    normalized = text(value).replace(",", "").replace("₹", "").replace(" ", "")
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = "-" + normalized[1:-1]
    try:
        return float(normalized)
    except ValueError:
        return np.nan


def hsn_code(value):
    if pd.isna(value):
        return ""
    value = text(value).removesuffix(".0")
    return value.zfill(8) if value.isdigit() and len(value) <= 8 else value


def parse_item_row(row, current, source_row):
    name = product_name(row.iloc[1])
    code = hsn_code(row.iloc[3])
    unit_original = text(row.iloc[4]).upper()
    unit = UNIT_MAP.get(unit_original, unit_original or "UNSPECIFIED")
    chapter = code[:2] if len(code) == 8 and code.isdigit() else ""
    scope = "food" if chapter in FOOD_CHAPTERS else "non_food" if chapter else "unclassified"
    issues = []
    if not name:
        issues.append("missing_product_name")
    if unit == "UNSPECIFIED":
        issues.append("missing_quantity_unit")
    if re.search(r"(?:GM|KG)[A-Z0-9]+$", name) or re.search(r"[A-Z]\d+$", name):
        issues.append("suspicious_product_suffix")
    if not chapter:
        issues.append("missing_or_invalid_hsn")
    qty, rate, taxable = number(row.iloc[5]), number(row.iloc[6]), number(row.iloc[8])
    taxes = [number(row.iloc[c]) for c in [9, 11, 13, 17]]
    tax = sum(taxes) if all(np.isfinite(taxes)) else np.nan
    item_total = number(row.iloc[18])
    if not all(np.isfinite([qty, rate, taxable, tax, item_total])):
        issues.append("invalid_numeric_value")
    if qty <= 0:
        issues.append("non_positive_quantity")
    if pd.isna(current["date"]):
        issues.append("invalid_invoice_date")
    if np.isfinite(taxable) and np.isfinite(tax) and abs(taxable + tax - item_total) > 0.02:
        issues.append("item_total_mismatch")
    if np.isfinite(qty) and np.isfinite(rate) and abs(qty * rate - taxable) > 0.02:
        issues.append("quantity_rate_taxable_difference")
    return {
        "invoice_id": current["invoice_id"],
        "invoice_key": current["invoice_key"],
        "date": current["date"],
        "customer": current["customer"],
        "source_row": source_row,
        "original_product_name": text(row.iloc[1]),
        "product_name": name,
        "hsn_code": code,
        "scope": scope,
        "unit_original": unit_original,
        "unit": unit,
        "quantity": qty,
        "unit_rate": rate,
        "taxable_value": taxable,
        "tax_amount": tax,
        "item_total": item_total,
        "issues": issues,
    }


def annotate_items(items):
    conflicting = set(items.groupby("product_name").unit.nunique().loc[lambda x: x > 1].index)
    identity = "invoice_key" if "invoice_key" in items else "invoice_id"
    duplicate_key = [identity, "product_name", "unit", "quantity", "unit_rate", "taxable_value"]
    duplicates = items.duplicated(duplicate_key, keep=False)
    for index in items.index:
        if items.at[index, "product_name"] in conflicting:
            items.at[index, "issues"].append("mixed_units_for_product")
        if duplicates.loc[index]:
            items.at[index, "issues"].append("repeated_item_line_needs_review")
    items["product_id"] = [
        "FOOD_" + hashlib.sha256(f"{n}|{u}|{h}".encode()).hexdigest()[:12]
        for n, u, h in zip(items.product_name, items.unit, items.hsn_code, strict=True)
    ]
    items["cat_id"] = items.scope.map(
        {"food": "FOODS", "non_food": "EXCLUDED", "unclassified": "REVIEW"}
    )
    items["needs_review"] = items.issues.map(bool)
    items["issues"] = items.issues.map(lambda values: "; ".join(values))
    return items


def validate_export_layout(frame):
    if frame.empty or frame.shape[1] < 39:
        raise ValueError("Expected the detailed 41-column sale-invoice export")
    if text(frame.iloc[0, 0]).casefold() != "invoice id":
        raise ValueError("Invoice Id header is missing")


def parse_invoice_rows(frame):
    validate_export_layout(frame)
    invoices, lines, structural = [], [], []
    current = None
    for index, row in frame.iloc[1:].iterrows():
        source_row = int(index) + 1
        first = text(row.iloc[0])
        if re.fullmatch(r"\d+(?:\.0)?", first):
            invoice_id = first.removesuffix(".0")
            date = pd.to_datetime(text(row.iloc[1]), format="%d-%b-%y", errors="coerce")
            if pd.isna(date):
                date = pd.to_datetime(row.iloc[1], errors="coerce", dayfirst=True)
            current = {
                "invoice_id": invoice_id,
                "invoice_key": invoice_key(invoice_id, date),
                "date": date,
                "customer": text(row.iloc[2]),
                "source_invoice_row": source_row,
                "invoice_quantity": number(row.iloc[29]),
                "invoice_taxable": number(row.iloc[31]),
                "invoice_tax": number(row.iloc[32]),
                "invoice_total": number(row.iloc[30]),
                "invoice_grand_total": number(row.iloc[38]),
            }
            invoices.append(current.copy())
        elif first == "--------" and text(row.iloc[1]).casefold() == "product name":
            structural.append(source_row)
        elif first == "--------":
            if current is None:
                raise ValueError(f"Item at row {source_row} has no invoice header")
            lines.append(parse_item_row(row, current, source_row))
        elif first or row.notna().any():
            raise ValueError(f"Unrecognized populated row {source_row}; import stopped for review")
    if not lines:
        raise ValueError("No invoice item rows found")
    items = pd.DataFrame(lines)
    headers = pd.DataFrame(invoices)
    if headers.invoice_key.duplicated().any():
        raise ValueError("Repeated invoice IDs on the same date require review")
    items = annotate_items(items)
    audit = reconcile_invoices(items, headers, key="invoice_key")
    return items, audit, len(structural)


def reconcile_invoices(items, headers, key="invoice_id"):
    totals = items.groupby(key).agg(
        line_quantity=("quantity", "sum"),
        line_taxable=("taxable_value", "sum"),
        line_tax=("tax_amount", "sum"),
        line_total=("item_total", "sum"),
    )
    audit = headers.merge(totals, on=key, validate="one_to_one")
    audit["quantity_difference"] = audit.line_quantity - audit.invoice_quantity
    audit["taxable_difference"] = audit.line_taxable - audit.invoice_taxable
    audit["tax_difference"] = audit.line_tax - audit.invoice_tax
    audit["total_difference"] = audit.line_total - audit.invoice_total
    audit["grand_total_rounding"] = audit.invoice_grand_total - audit.invoice_total
    audit["reconciles"] = (
        audit[
            ["quantity_difference", "taxable_difference", "tax_difference", "total_difference"]
        ].abs()
        <= 0.02
    ).all(axis=1)
    return audit


def import_invoices(path, destination="data/retail", reports="reports", archive_source=True):
    path = Path(path)
    if not zipfile.is_zipfile(path):
        size = path.stat().st_size
        if size < 16:
            raise ValueError(
                f"{path.name} is empty or incomplete ({size} bytes). "
                "Choose the actual Excel workbook, then upload it again."
            )
        with path.open("rb") as source:
            signature = source.read(8)
        if signature == bytes.fromhex("D0CF11E0A1B11AE1"):
            raise ValueError(
                "This is an older .xls workbook. Open it in Excel or LibreOffice and "
                "save it as .xlsx before uploading."
            )
        raise ValueError(
            f"{path.name} is not a valid .xlsx workbook. Check that it is a real Excel "
            "file and not a renamed or partially downloaded file."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    if len(sheets) != 1:
        raise ValueError("Select one worksheet before importing a multi-sheet workbook")
    sheet, frame = next(iter(sheets.items()))
    items, audit, repeated_headers = parse_invoice_rows(frame)
    import_start, import_end = audit.date.min(), audit.date.max()
    out, report_dir = Path(destination), Path(reports)
    out.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if archive_source:
        archive = out / "source" / digest
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive / "source.xlsx")
    database = out / "retail.sqlite3"
    # Upgrade the previous file-backed dataset without losing its history.
    previous_report = out / "cleaning_report.json"
    if not database.exists() and previous_report.exists():
        previous = json.loads(previous_report.read_text())
        store_import(
            database,
            pd.read_parquet(out / "all_items.parquet"),
            pd.read_parquet(out / "invoice_reconciliation.parquet"),
            {**previous, "repeated_headers": previous["repeated_item_headers_removed"]},
        )
    stats = store_import(
        database,
        items,
        audit,
        {
            "source_sha256": digest,
            "source_filename": path.name,
            "source_sheet": sheet,
            "repeated_headers": repeated_headers,
        },
    )
    return refresh_invoice_outputs(
        destination,
        reports,
        {
            "source_filename": path.name,
            "source_sha256": digest,
            "source_sheet": sheet,
            "last_import": stats,
            "latest_source_start_date": str(import_start.date()),
            "latest_source_end_date": str(import_end.date()),
        },
    )


def refresh_invoice_outputs(destination="data/retail", reports="reports", metadata=None):
    out, report_dir = Path(destination), Path(reports)
    if out.is_absolute() and out.is_relative_to(Path.cwd()):
        out = out.relative_to(Path.cwd())
    out.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if metadata is None:
        metadata = json.loads((out / "cleaning_report.json").read_text())
    database = out / "retail.sqlite3"
    items, audit, sources = load_database(database)
    corrections = resolution_history(database)
    with connect_database(database, readonly=True) as connection:
        raw_lines = connection.execute("SELECT COUNT(*) FROM retail_invoice_lines").fetchone()[0]
        raw_food = connection.execute("SELECT COUNT(*) FROM retail_food_sales").fetchone()[0]
    coverage = coverage_status(
        items[items.scope.eq("food")],
        items.date.min(),
        items.date.max(),
        resolution_history(database, latest=True),
    )
    repeated_headers = sum(source["repeated_headers"] for source in sources)
    food = items[items.scope.eq("food")].copy()
    if food.empty:
        raise ValueError("No food lines passed the food-chapter filter")
    # Unknown units remain separate. Days without invoices are NOT filled with zeros.
    daily = food.groupby(["date", "product_id", "product_name", "unit"], as_index=False).agg(
        quantity=("quantity", "sum"),
        taxable_value=("taxable_value", "sum"),
        tax_amount=("tax_amount", "sum"),
        item_total=("item_total", "sum"),
        item_lines=("source_row", "count"),
        needs_review=("needs_review", "max"),
    )
    start, end = food.date.min(), food.date.max()
    span = (end - start).days + 1
    observed_days = food.date.nunique()
    issue_counts = (
        items.loc[items.needs_review & ~items.scope.eq("non_food"), "issues"]
        .str.split("; ")
        .explode()
        .value_counts()
        .to_dict()
    )
    report = {
        "source_type": "retail_invoice",
        "corrections_revision": corrections[-1]["resolution_id"] if corrections else 0,
        "correction_count": len(corrections),
        "coverage": coverage,
        "coverage_rules": [
            r for r in resolution_history(database, latest=True) if r["kind"] == "coverage"
        ],
        "source_filename": metadata["source_filename"],
        "source_sha256": metadata["source_sha256"],
        "source_sheet": metadata["source_sheet"],
        "storage": "sqlite",
        "database_path": str(database),
        "source_count": len(sources),
        "sources": sources,
        "last_import": metadata["last_import"],
        "latest_source_start_date": metadata["latest_source_start_date"],
        "latest_source_end_date": metadata["latest_source_end_date"],
        "currency": "INR",
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "calendar_days": int(span),
        "invoice_dates": int(observed_days),
        "unobserved_dates": int(span - observed_days),
        "invoices": int(len(audit)),
        "source_item_lines": raw_lines,
        "source_food_lines": raw_food,
        "removed_duplicate_lines": items.attrs.get("removed_duplicate_lines", 0),
        "excluded_placeholder_lines": items.attrs.get("excluded_placeholder_lines", 0),
        "excluded_unidentified_lines": items.attrs.get("excluded_unidentified_lines", 0),
        "effective_item_lines": int(len(items)),
        "food_lines": int(len(food)),
        "excluded_non_food_lines": int(items.scope.eq("non_food").sum()),
        "unclassified_lines": int(items.scope.eq("unclassified").sum()),
        "food_product_names": int(food.product_name.nunique()),
        "food_product_unit_combinations": int(food.product_id.nunique()),
        "customers": int(food.customer.nunique()),
        "food_taxable_value": float(food.taxable_value.sum()),
        "food_tax_amount": float(food.tax_amount.sum()),
        "food_item_total": float(food.item_total.sum()),
        "food_lines_needing_review": int(food.needs_review.sum()),
        "issue_counts": issue_counts,
        "repeated_item_headers_removed": repeated_headers,
        "product_labels_normalized": int(items.original_product_name.ne(items.product_name).sum()),
        "invoice_reconciliation_failures": int((~audit.reconciles).sum()),
        "forecast_ready": False,
        "forecast_readiness": (
            "Data checks are clear; validate a model on continuous daily history "
            "before forecasting."
            if coverage["unresolved_days"] == 0 and not food.needs_review.any()
            else "Confirm non-invoice days, resolve unit/name issues and "
            "validate a model on continuous daily history before forecasting."
        ),
        "cleaning_rules": [
            "Inherit date and customer only within each invoice block.",
            "Remove repeated structural headers; preserve original line numbers and names.",
            "Normalize whitespace, case, numeric formatting, HSN text and unit abbreviations.",
            "Include only food HSN chapters; exclude non-food and quarantine "
            "missing classifications.",
            "Keep quantities in their recorded units; never convert a 500 GM label "
            "into inferred pack counts.",
            "Flag suspicious suffixes and mixed units without automatically "
            "merging different products.",
            (
                f"Exclude {items.attrs.get('removed_duplicate_lines', 0)} repeated item entries "
                "according to the reviewed keep-one decision; preserve original SQL rows."
                if items.attrs.get("removed_duplicate_lines", 0)
                else "Flag repeated item lines without deleting potentially legitimate sales."
            ),
            "Reconcile every complete invoice before food filtering.",
            (
                f"Treat {coverage['confirmed_no_trading_days']} confirmed dates without invoices "
                "as no trading; show zero sales for those dates."
                if coverage["unresolved_days"] == 0
                else "Do not infer zero sales on dates absent from the export; "
                "confirm coverage first."
            ),
        ],
        "quantity_note": (
            "KG/KGS means kilograms and PCS means pieces, as confirmed for this export. "
            "Quantities remain in their recorded unit; no unit conversion is applied."
            if any(r["kind"] == "units" for r in resolution_history(database, latest=True))
            else "KG, PCS, BAG and UNSPECIFIED are separate measures; confirm their "
            "meanings before ordering."
        ),
        "scope_note": "Customer names identify buyers on invoices, not the "
        "seller's store locations.",
    }
    for name, table in [
        ("food_items", food),
        ("all_items", items),
        ("daily_food_sales", daily),
        ("invoice_reconciliation", audit),
        ("review_items", items[items.needs_review | ~items.scope.eq("food")]),
    ]:
        table.to_parquet(out / f"{name}.parquet", index=False)
        table.to_csv(out / f"{name}.csv", index=False)
    temporary_report = out / "cleaning_report.json.tmp"
    temporary_report.write_text(json.dumps(report, indent=2))
    temporary_report.replace(out / "cleaning_report.json")
    # Activate only after all reviewed tables are safely written.
    temporary_active = report_dir / "active_dataset.json.tmp"
    temporary_active.write_text(
        json.dumps(
            {
                "source_type": "retail_invoice",
                "directory": str(out),
                "source_sha256": metadata["source_sha256"],
            },
            indent=2,
        )
    )
    temporary_active.replace(report_dir / "active_dataset.json")
    return report
