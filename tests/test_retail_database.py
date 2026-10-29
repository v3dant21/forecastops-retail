"""SQL integrity, cumulative imports, repeat safety and invoice conflict handling."""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from forecastops.data.retail_database import connect_database, load_database, store_import


@pytest.fixture
def actual_invoice():
    root = Path("data/retail")
    if not (root / "all_items.parquet").exists():
        pytest.skip("Import actual invoices first")
    items = pd.read_parquet(root / "all_items.parquet")
    audit = pd.read_parquet(root / "invoice_reconciliation.parquet")
    header = audit.iloc[[0]].copy()
    lines = items[
        items.invoice_id.eq(header.iloc[0].invoice_id) & items.date.eq(header.iloc[0].date)
    ].copy()
    metadata = {
        "source_sha256": "a" * 64,
        "source_filename": "test-source.xlsx",
        "source_sheet": "Worksheet",
        "repeated_headers": 1,
    }
    return lines, header, metadata


def test_repeat_and_overlapping_exports_are_not_double_counted(tmp_path, actual_invoice):
    lines, header, metadata = actual_invoice
    database = tmp_path / "retail.sqlite3"
    first = store_import(database, lines, header, metadata)
    repeat = store_import(database, lines, header, metadata)
    assert first == {"inserted_invoices": 1, "overlapping_invoices": 0}
    assert repeat == {"inserted_invoices": 0, "overlapping_invoices": 1}
    # A differently positioned block in another export is still the same invoice.
    lines["source_row"] += 200
    header["source_invoice_row"] += 200
    store_import(database, lines, header, {**metadata, "source_sha256": "b" * 64})
    restored, audit, sources = load_database(database)
    assert len(restored) == len(lines)
    assert len(audit) == 1
    assert len(sources) == 2
    assert restored.taxable_value.sum() == pytest.approx(lines.taxable_value.sum())
    with connect_database(database, readonly=True) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT COUNT(*) FROM retail_invoice_sources").fetchone()[0] == 2


def test_conflicting_invoice_rolls_back_entire_import(tmp_path, actual_invoice):
    lines, header, metadata = actual_invoice
    database = tmp_path / "retail.sqlite3"
    store_import(database, lines, header, metadata)
    changed = lines.copy()
    changed.loc[changed.index[0], "quantity"] += 1
    with pytest.raises(ValueError, match="conflicts with stored data"):
        store_import(database, changed, header, {**metadata, "source_sha256": "c" * 64})
    restored, _, sources = load_database(database)
    assert len(sources) == 1
    assert restored.quantity.sum() == pytest.approx(lines.quantity.sum())


def test_invoice_numbers_can_repeat_on_another_date(tmp_path, actual_invoice):
    lines, header, metadata = actual_invoice
    database = tmp_path / "retail.sqlite3"
    store_import(database, lines, header, metadata)
    next_year = pd.DateOffset(years=1)
    lines["date"] += next_year
    header["date"] += next_year
    store_import(database, lines, header, {**metadata, "source_sha256": "d" * 64})
    restored, audit, _ = load_database(database)
    assert len(audit) == 2
    assert audit.invoice_key.is_unique
    assert len(restored) == 2 * len(lines)


def test_sql_constraints_reject_orphan_lines(tmp_path, actual_invoice):
    lines, header, metadata = actual_invoice
    database = tmp_path / "retail.sqlite3"
    store_import(database, lines, header, metadata)
    with pytest.raises(sqlite3.IntegrityError), connect_database(database) as connection:
        connection.execute("UPDATE retail_invoice_lines SET invoice_key = ?", ("missing",))


def test_active_dashboard_data_matches_sql_ledger():
    root = Path("data/retail")
    if not (root / "retail.sqlite3").exists():
        pytest.skip("Import SQL invoices first")
    report = json.loads((root / "cleaning_report.json").read_text())
    items, audit, sources = load_database(root / "retail.sqlite3")
    assert len(items) == report["effective_item_lines"]
    assert len(audit) == report["invoices"]
    assert len(sources) == report["source_count"]
    assert (~audit.reconciles).sum() == report["invoice_reconciliation_failures"]
    assert items[items.scope.eq("food")].taxable_value.sum() == pytest.approx(
        report["food_taxable_value"]
    )
    with connect_database(root / "retail.sqlite3", readonly=True) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert (
            connection.execute("SELECT COUNT(*) FROM retail_food_sales").fetchone()[0]
            == report["source_food_lines"]
        )
        assert (
            connection.execute(
                "SELECT typeof(taxable_paise) FROM retail_invoice_lines LIMIT 1"
            ).fetchone()[0]
            == "integer"
        )
