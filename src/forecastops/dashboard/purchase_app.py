"""Retailer-controlled purchase order drafts from the active food invoices."""

import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from forecastops.data.resolutions import coverage_status
from forecastops.planning.purchase import draft_lines, sales_order_inputs


def invoice_purchase_order(food, report):
    st.title("Build a food purchase order")
    st.write("Choose items, enter stock and supplier terms, then download a draft for review.")
    st.info(
        "This is a sales-based restocking estimate, not a validated "
        "demand forecast. Review seasonality, stockouts and shelf life before ordering."
    )
    period = st.date_input(
        "Sales period used for this order",
        value=(pd.Timestamp(report["start_date"]).date(), pd.Timestamp(report["end_date"]).date()),
        min_value=pd.Timestamp(report["start_date"]).date(),
        max_value=pd.Timestamp(report["end_date"]).date(),
    )
    if len(period) != 2:
        st.info("Select both dates to prepare the order.")
        return
    inputs, days = sales_order_inputs(food, *period)
    if inputs.empty:
        st.info("No food sales in this period.")
        return
    coverage = coverage_status(food, *period, report.get("coverage_rules", []))
    known_complete = coverage["unresolved_days"] == 0
    complete = st.checkbox(
        "I confirm this export includes all sales in the selected period; "
        "dates without invoices had no sales.",
        value=known_complete,
        disabled=known_complete,
    )
    st.caption(
        f"Quantity sold is the actual total for {period[0]} to {period[1]}, "
        "in the unit shown (KG, PCS, etc.). It is not a pack count inferred from the item name."
    )
    missing = (
        days
        - food[food.date.between(pd.Timestamp(period[0]), pd.Timestamp(period[1]))].date.nunique()
    )
    if known_complete:
        st.caption(
            f"Saved review: {missing} dates without food invoices are confirmed no-trading days."
        )
    else:
        st.warning(
            f"{coverage['unresolved_days']} dates need confirmation. "
            "Open Resolve warnings → Sales dates."
        )
    left, right = st.columns(2)
    with left:
        buyer = st.text_input("Your business / buyer name")
        supplier = st.text_input("Supplier for this draft")
        ship_to = st.text_area("Delivery address")
    with right:
        order_date = st.date_input("Order date", value=date.today())
        reference = st.text_input("Order reference", value=f"PO-{order_date:%Y%m%d}-DRAFT")
        notes = st.text_area("Supplier notes / payment terms")
    cols = st.columns(3)
    lead = cols[0].number_input("Supplier lead time (days)", min_value=0, value=7, step=1)
    review = cols[1].number_input("Days until next stock review", min_value=1, value=7, step=1)
    safety = cols[2].number_input("Extra safety stock (days)", min_value=0, value=3, step=1)
    st.caption(
        "The suggested target prorates the selected-period total to cover supplier lead time, "
        "the next stock review and safety days. It then subtracts stock on hand and incoming "
        "stock, and rounds up to the supplier pack size and minimum order."
    )
    blocked = inputs[inputs.blocked]
    if not blocked.empty:
        st.warning(
            f"{len(blocked)} product/unit rows blocked until source labels or units are resolved."
        )
        with st.expander("See blocked items"):
            st.dataframe(blocked[["product_name", "unit"]], hide_index=True)
    editable = inputs[~inputs.blocked].copy()
    if editable.empty:
        st.info("Resolve source data issues before preparing an order.")
        return
    editable["include"] = False
    editable["on_hand"] = 0.0
    editable["incoming"] = 0.0
    editable["pack_size"] = 1.0
    editable["minimum_order"] = 0.0
    editable["unit_cost"] = pd.Series(float("nan"), index=editable.index, dtype=float)
    st.subheader("Select items and enter stock in the displayed unit")
    st.caption(
        "Stock defaults are placeholders. Pack size is in the recorded sales unit "
        "(for example, 5 means 5 KG for a KG row). Purchase cost is per recorded unit, "
        "before tax; sales prices are never used as supplier costs."
    )
    edited = st.data_editor(
        editable,
        key=f"purchase-inputs-{report['source_sha256']}-{period}",
        hide_index=True,
        width="stretch",
        disabled=["product_id", "product_name", "unit", "sold_quantity", "daily_sales", "blocked"],
        column_config={
            "product_id": None,
            "blocked": None,
            "product_name": "Food item",
            "unit": "Unit",
            "include": "Include",
            "sold_quantity": f"Quantity sold · {days} days",
            "daily_sales": None,
            "on_hand": st.column_config.NumberColumn("Stock on hand", min_value=0, required=True),
            "incoming": st.column_config.NumberColumn(
                "Incoming within lead time", min_value=0, required=True
            ),
            "pack_size": st.column_config.NumberColumn(
                "Supplier pack size", min_value=0.001, required=True
            ),
            "minimum_order": st.column_config.NumberColumn(
                "Minimum order", min_value=0, required=True
            ),
            "unit_cost": st.column_config.NumberColumn("Purchase cost / unit (₹)", min_value=0),
        },
    )
    confirmed = st.checkbox(
        "I checked stock, incoming quantities, supplier packs and units for every selected item."
    )
    try:
        lines = draft_lines(edited, lead, review, safety)
    except (ValueError, TypeError) as error:
        st.error(f"Check the selected inputs: {error}")
        return
    if lines.empty:
        st.info("Select items to draft an order. Items already covered by stock need no purchase.")
        return
    st.subheader("Purchase order preview · DRAFT")
    st.caption(
        f"Sales basis: {period[0]} to {period[1]}. Expected arrival: "
        f"{order_date + timedelta(days=int(lead))}. Confirm availability with your supplier."
    )
    st.dataframe(
        lines.drop(columns=["product_id", "daily_sales"]),
        hide_index=True,
        width="stretch",
        column_config={
            "sold_quantity": f"Quantity sold · {days} days",
            "target_stock": "Target stock for cover period",
        },
    )
    st.metric("Items to purchase", len(lines))
    if lines.purchase_unit_cost.isna().any():
        st.warning("Some purchase costs are missing. The draft is unpriced; no total is shown.")
    else:
        st.metric(
            "Estimated purchase value before tax / freight",
            f"₹{lines.line_value_before_tax.sum():,.2f}",
        )
    if not all(
        (complete, confirmed, buyer.strip(), supplier.strip(), reference.strip(), ship_to.strip())
    ):
        st.info(
            "Complete buyer, supplier, reference and delivery address, then confirm "
            "sales completeness and stock to download."
        )
        return
    download_purchase_order(
        lines.drop(columns="daily_sales"),
        {
            "status": "DRAFT - retailer review required",
            "order_reference": reference,
            "buyer": buyer,
            "supplier": supplier,
            "delivery_address": ship_to,
            "order_date": str(order_date),
            "expected_arrival": str(order_date + timedelta(days=int(lead))),
            "sales_start": str(period[0]),
            "sales_end": str(period[1]),
            "sales_calendar_days": days,
            "lead_days": lead,
            "review_days": review,
            "safety_days": safety,
            "supplier_notes": notes,
            "source_filename": report["source_filename"],
            "source_sha256": report["source_sha256"],
            "all_sources": json.dumps(report.get("sources", [])),
        },
    )


def download_purchase_order(lines, metadata):
    export = lines.copy()
    for column, value in metadata.items():
        export[column] = value
    # Keep user/source text inert when the CSV is opened in spreadsheet software.
    for column in export:
        export[column] = export[column].map(
            lambda value: "'" + value
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
            else value
        )
    st.download_button(
        "Download purchase order draft (CSV)",
        export.to_csv(index=False).encode("utf-8-sig"),
        file_name="food-purchase-order-draft.csv",
        mime="text/csv",
    )
    st.caption(
        "Downloading creates a draft only. No order is sent to a supplier or added to inventory."
    )
