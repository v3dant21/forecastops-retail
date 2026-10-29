"""Purchase quantity correctness and invoice UI download gating."""

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from forecastops.planning.purchase import draft_lines, purchase_quantity, sales_order_inputs


def test_pack_rounding_stock_and_minimum_orders():
    # 10/day * (7 + 7 + 3) - 20 in stock - 15 incoming = 135; packs of 12 => 144.
    assert purchase_quantity(10, 20, 15, 7, 7, 3, 12, 0) == (170, 144)
    assert purchase_quantity(10, 200, 0, 7, 7, 3, 12, 100) == (170, 0)
    assert purchase_quantity(1, 0, 0, 1, 1, 0, 12, 25) == (2, 36)
    assert purchase_quantity(0.1, 0, 0, 1, 1, 1, 0.1, 0) == (0.3, 0.3)
    for bad in [-1, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="finite and non-negative"):
            purchase_quantity(10, bad, 0, 7, 7, 3, 12, 0)
    with pytest.raises(ValueError, match="greater than zero"):
        purchase_quantity(10, 0, 0, 7, 7, 3, 0, 0)


def test_period_denominator_and_blocking_across_entire_source():
    food = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-10-01", "2026-10-03", "2026-10-01", "2026-10-03"]),
            "product_id": ["a", "a", "b", "c"],
            "product_name": ["A", "A", "B", "B"],
            "unit": ["KG", "KG", "KG", "PCS"],
            "quantity": [10, 30, 10, 20],
            "needs_review": [False, True, False, False],
        }
    )
    rows, days = sales_order_inputs(food, "2026-10-01", "2026-10-02")
    assert days == 2
    assert rows.set_index("product_id").loc["a", "daily_sales"] == 5
    assert rows.blocked.all()  # A has a flagged row outside the period; B has mixed units.


def test_draft_excludes_blocked_unselected_and_covered_stock():
    inputs = pd.DataFrame(
        [
            {
                "product_id": str(i),
                "product_name": str(i),
                "unit": "KG",
                "sold_quantity": 290,
                "daily_sales": 10,
                "include": i != 1,
                "blocked": i == 2,
                "on_hand": 1000 if i == 3 else 0,
                "incoming": 0,
                "pack_size": 12.0,
                "minimum_order": 0,
                "unit_cost": None if i == 4 else 20,
            }
            for i in range(5)
        ]
    )
    lines = draft_lines(inputs, 7, 7, 3)
    assert list(lines.product_id) == ["0", "4"]
    assert list(lines.order_quantity) == [180, 180]
    assert lines.iloc[0].line_value_before_tax == 3600
    assert pd.isna(lines.iloc[1].line_value_before_tax)
    inputs.loc[0, ["unit", "pack_size"]] = ["PCS", 0.5]
    with pytest.raises(ValueError, match="whole units"):
        draft_lines(inputs, 7, 7, 3)


def test_live_invoice_purchase_order_download_gating():
    report_path = Path("data/retail/cleaning_report.json")
    if not report_path.exists():
        pytest.skip("Import invoices first")
    report = json.loads(report_path.read_text())
    app = AppTest.from_file(
        str(Path("src/forecastops/dashboard/streamlit_app.py").resolve())
    ).run(timeout=45)
    app.sidebar.radio[0].set_value("Purchase order").run(timeout=45)
    assert not app.exception
    assert "Build a food purchase order" in app.title[0].value
    assert len(app.get("download_button")) == 0
    period = tuple(pd.Timestamp(report[k]).date() for k in ("start_date", "end_date"))
    key = f"purchase-inputs-{report['source_sha256']}-{period}"
    app.session_state[key] = {
        "edited_rows": {0: {"include": True, "on_hand": 5.0, "unit_cost": 20.0}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run(timeout=45)
    assert not app.exception
    assert any("preview" in s.value for s in app.subheader)
    assert len(app.get("download_button")) == 0
    for widget, value in zip(
        app.text_input, ["Test buyer", "Test supplier", "PO-TEST"], strict=True
    ):
        widget.set_value(value)
    app.text_area[0].set_value("Test delivery address")
    for checkbox in app.checkbox:
        if not checkbox.disabled:
            checkbox.check()
    app.run(timeout=45)
    assert not app.exception
    assert len(app.get("download_button")) == 1
    assert any("source workbook(s)" in c.value for c in app.caption)
