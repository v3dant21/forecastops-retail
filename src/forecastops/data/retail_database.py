"""Transactional SQL ledger for real invoices, with overlap detection and provenance."""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import pandas as pd

DERIVED_ISSUES = {"mixed_units_for_product", "repeated_item_line_needs_review"}


@contextmanager
def connect_database(path, readonly=False):
    path = Path(path).resolve()
    if readonly:
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(Path(__file__).with_name("retail_schema.sql").read_text())
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def paise(value):
    if pd.isna(value):
        return None
    return int((Decimal(str(value)) * 100).quantize(Decimal("1")))


def invoice_key(invoice_id, date):
    return f"{pd.Timestamp(date).date()}|{invoice_id}"


def base_issues(value):
    return "; ".join(issue for issue in value.split("; ") if issue and issue not in DERIVED_ISSUES)


def content_digest(header, lines):
    """Exclude export row positions and derived review flags; retain actual invoice content."""
    columns = [
        "original_product_name",
        "product_id",
        "unit_original",
        "quantity",
        "unit_rate",
        "taxable_value",
        "tax_amount",
        "item_total",
    ]
    payload = {
        "header": {
            key: str(header[key])
            for key in [
                "customer",
                "invoice_quantity",
                "invoice_taxable",
                "invoice_tax",
                "invoice_total",
                "invoice_grand_total",
            ]
        },
        "lines": sorted(tuple(str(row[c]) for c in columns) for row in lines.to_dict("records")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def store_import(path, items, audit, metadata):
    """Commit a whole import or nothing; identical overlaps share the existing invoice."""
    if items.date.isna().any() or audit.date.isna().any():
        raise ValueError("Invalid invoice dates must be corrected before SQL import")
    digest = metadata["source_sha256"]
    stats = {"inserted_invoices": 0, "overlapping_invoices": 0}
    with connect_database(path) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO retail_imports "
            "(source_sha256, source_filename, source_sheet, repeated_headers) VALUES (?, ?, ?, ?)",
            (
                digest,
                metadata["source_filename"],
                metadata["source_sheet"],
                metadata["repeated_headers"],
            ),
        )
        grouped = {
            (str(date.date()), str(identity)): frame
            for (date, identity), frame in items.groupby(["date", "invoice_id"], sort=False)
        }
        for header in audit.to_dict("records"):
            lines = grouped[(str(header["date"].date()), header["invoice_id"])]
            key = invoice_key(header["invoice_id"], header["date"])
            fingerprint = content_digest(header, lines)
            existing = connection.execute(
                "SELECT content_sha256 FROM retail_invoices WHERE invoice_key = ?", (key,)
            ).fetchone()
            if existing and existing[0] != fingerprint:
                raise ValueError(
                    f"Invoice {key} conflicts with stored data. No changes committed; "
                    "review the corrected invoice before importing."
                )
            if existing:
                stats["overlapping_invoices"] += 1
            else:
                _insert_invoice(connection, header, lines, key, digest, fingerprint)
                stats["inserted_invoices"] += 1
            connection.execute(
                "INSERT OR IGNORE INTO retail_invoice_sources VALUES (?, ?, ?)",
                (digest, key, int(header["source_invoice_row"])),
            )
    return stats


def _insert_invoice(connection, header, lines, key, digest, fingerprint):
    customer = header["customer"]
    customer_id = hashlib.sha256(customer.encode()).hexdigest()[:24]
    connection.execute(
        "INSERT OR IGNORE INTO retail_customers VALUES (?, ?)", (customer_id, customer)
    )
    amounts = [
        paise(header[f"invoice_{field}"]) for field in ("taxable", "tax", "total", "grand_total")
    ]
    connection.execute(
        "INSERT INTO retail_invoices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            key,
            header["invoice_id"],
            str(pd.Timestamp(header["date"]).date()),
            customer_id,
            digest,
            int(header["source_invoice_row"]),
            header["invoice_quantity"],
            *amounts,
            fingerprint,
        ),
    )
    for ordinal, row in enumerate(lines.to_dict("records"), 1):
        connection.execute(
            "INSERT OR IGNORE INTO retail_products VALUES (?, ?, ?, ?, ?)",
            (row["product_id"], row["product_name"], row["unit"], row["hsn_code"], row["scope"]),
        )
        connection.execute(
            "INSERT INTO retail_invoice_lines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                ordinal,
                row["product_id"],
                int(row["source_row"]),
                row["original_product_name"],
                row["unit_original"],
                row["quantity"],
                row["unit_rate"],
                paise(row["taxable_value"]),
                paise(row["tax_amount"]),
                paise(row["item_total"]),
                base_issues(row["issues"]),
            ),
        )


def load_database(path):
    """Return dashboard-compatible data from SQL and recalculate cross-import quality flags."""
    # Defer to avoid a cycle with the importer, which owns the shared cleaning rules.
    from forecastops.data.invoices import annotate_items, reconcile_invoices  # noqa: PLC0415
    from forecastops.data.resolutions import (  # noqa: PLC0415
        apply_resolutions,
        clear_verified_flags,
        resolution_history,
    )

    with connect_database(path, readonly=True) as connection:
        items = pd.read_sql_query(
            "SELECT i.invoice_key, i.invoice_id, i.date, c.customer_name AS customer, "
            "i.source_sha256, r.source_filename, l.source_row, l.original_product_name, "
            "p.product_name, p.product_id, l.line_number, "
            "p.hsn_code, p.scope, l.unit_original, p.unit, l.quantity, l.unit_rate, "
            "l.taxable_paise / 100.0 AS taxable_value, l.tax_paise / 100.0 AS tax_amount, "
            "l.total_paise / 100.0 AS item_total, l.base_issues AS issues "
            "FROM retail_invoice_lines l JOIN retail_invoices i USING(invoice_key) "
            "JOIN retail_products p USING(product_id) JOIN retail_customers c USING(customer_id) "
            "JOIN retail_imports r ON r.source_sha256 = i.source_sha256 "
            "ORDER BY i.date, i.invoice_id, l.line_number",
            connection,
        )
        headers = pd.read_sql_query(
            "SELECT i.invoice_key, i.invoice_id, i.date, c.customer_name AS customer, "
            "i.source_sha256, r.source_filename, i.source_invoice_row, i.invoice_quantity, "
            "i.invoice_taxable_paise / 100.0 AS invoice_taxable, "
            "i.invoice_tax_paise / 100.0 AS invoice_tax, "
            "i.invoice_total_paise / 100.0 AS invoice_total, "
            "i.invoice_grand_total_paise / 100.0 AS invoice_grand_total "
            "FROM retail_invoices i JOIN retail_customers c USING(customer_id) "
            "JOIN retail_imports r ON r.source_sha256 = i.source_sha256 "
            "ORDER BY i.date, i.invoice_id",
            connection,
        )
        sources = pd.read_sql_query(
            "SELECT r.*, MIN(i.date) AS start_date, MAX(i.date) AS end_date, "
            "COUNT(s.invoice_key) AS invoices FROM retail_imports r "
            "JOIN retail_invoice_sources s USING(source_sha256) "
            "JOIN retail_invoices i USING(invoice_key) "
            "GROUP BY r.source_sha256 ORDER BY r.imported_at, r.source_sha256",
            connection,
        ).to_dict("records")
    items["date"] = pd.to_datetime(items.date)
    headers["date"] = pd.to_datetime(headers.date)
    items, headers = apply_resolutions(items, headers, resolution_history(path, latest=True))
    items = annotate_items(items)
    items = clear_verified_flags(items)
    audit = reconcile_invoices(items, headers, key="invoice_key")
    return items, audit, sources
