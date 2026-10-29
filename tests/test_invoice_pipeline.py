"""Actual invoice cleaning, food filtering and active-data dashboard checks."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from forecastops.data.invoices import number, parse_invoice_rows, product_name


def test_label_and_number_cleaning():
    assert number(" - 0.0000") == 0
    assert number("1,234.50") == 1234.5
    assert product_name(" BESAN  500gm ") == "BESAN 500 GM"
    assert product_name("DALIA 500 GM5") == "DALIA 500 GM5"


def test_orphan_item_rejected():
    frame = pd.DataFrame([["Invoice Id"] + [None] * 40, ["--------", "BESAN 500GM"] + [None] * 39])
    with pytest.raises(ValueError, match="no invoice header"):
        parse_invoice_rows(frame)


def test_actual_invoice_reconciliation_and_food_scope():
    root = Path("data/retail")
    if not (root / "cleaning_report.json").exists():
        pytest.skip("Import actual invoices first")
    report = json.loads((root / "cleaning_report.json").read_text())
    items = pd.read_parquet(root / "all_items.parquet")
    food = pd.read_parquet(root / "food_items.parquet")
    audit = pd.read_parquet(root / "invoice_reconciliation.parquet")
    assert len(items) == report["effective_item_lines"]
    assert len(food) == report["food_lines"]
    assert food.scope.eq("food").all()
    assert not food.hsn_code.str.startswith("33").any()
    assert not items.duplicated(["source_sha256", "source_row"]).any()
    assert len(audit) == report["invoices"]
    assert (~audit.reconciles).sum() == report["invoice_reconciliation_failures"]
    assert food.taxable_value.sum() == pytest.approx(report["food_taxable_value"])
    assert food.item_total.sum() == pytest.approx(report["food_item_total"])
    assert food[food.unit.eq("UNSPECIFIED")].needs_review.all()
    assert "DALIA 500 GM5" in food.original_product_name.values
    assert not report["forecast_ready"]
    source = root / "source" / report["source_sha256"] / "source.xlsx"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report["source_sha256"]


def test_own_data_dashboard_all_views():
    if not Path("reports/active_dataset.json").exists():
        pytest.skip("Import actual invoices first")
    app = AppTest.from_file(str(Path("src/forecastops/dashboard/streamlit_app.py").resolve())).run(
        timeout=45
    )
    assert not app.exception
    assert not app.error
    assert "Your food sales" in app.title[0].value
    report = json.loads(Path("data/retail/cleaning_report.json").read_text())
    assert f"{report['food_taxable_value']:,.2f}" in app.metric[0].value
    for page in ["Food items", "Resolve warnings", "Data cleaning", "Forecast readiness"]:
        app.sidebar.radio[0].set_value(page).run(timeout=45)
        assert not app.exception
        if page != "Data cleaning":
            assert not app.error
    assert "Validated demand forecasts" in app.info[0].value
