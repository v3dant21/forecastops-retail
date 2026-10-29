"""Business-input forms that resolve data warnings and retain a SQL audit trail."""

import pandas as pd
import streamlit as st

from forecastops.data.invoices import refresh_invoice_outputs
from forecastops.data.resolutions import (
    INVOICE_FIELDS,
    LINE_FIELDS,
    resolution_history,
    save_resolutions,
)
from forecastops.data.retail_database import load_database


def persist_review(root, records, reviewer, reason):
    try:
        save_resolutions(root / "retail.sqlite3", records, reviewer, reason)
        refresh_invoice_outputs(root)
    except (ValueError, KeyError, OSError) as error:
        st.error(str(error))
        return
    st.cache_data.clear()
    st.session_state["resolution_saved"] = (
        "Saved to SQL. Warnings and order eligibility recalculated."
    )
    st.rerun()


def resolution_workspace(root, report):
    st.title("Resolve warnings with your business records")
    st.write(
        "Supply the missing facts below. Saved reviews update sales, order quantities and "
        "reconciliation checks; the original invoices remain unchanged."
    )
    if message := st.session_state.pop("resolution_saved", None):
        st.success(message)
    columns = st.columns(3)
    columns[0].metric("Food lines awaiting input", report["food_lines_needing_review"])
    columns[1].metric("Invoices to reconcile", report["invoice_reconciliation_failures"])
    columns[2].metric("Dates to confirm", report.get("coverage", {}).get("unresolved_days", 0))
    topic = st.selectbox(
        "What do you want to resolve?",
        [
            "Product names, units and HSN",
            "Sales dates",
            "Recorded unit meanings",
            "Repeated entries",
            "Invoice or item amounts",
            "Review history",
        ],
    )
    database = root / "retail.sqlite3"
    if topic == "Review history":
        history = resolution_history(database)
        if history:
            st.dataframe(
                pd.DataFrame(history).drop(columns="values"), hide_index=True, width="stretch"
            )
        else:
            st.info("No reviews have been saved.")
        return
    reviewer = st.text_input("Reviewed by", help="Your name or role, retained with the correction.")
    reason = st.text_input(
        "Reason / evidence", help="For example: verified against supplier invoice."
    )
    items, audit, _ = load_database(database)
    actions = {
        "Product names, units and HSN": product_review,
        "Sales dates": date_review,
        "Recorded unit meanings": unit_review,
        "Repeated entries": duplicate_review,
        "Invoice or item amounts": amount_review,
    }
    actions[topic](root, items, audit, report, reviewer, reason)


def product_review(root, items, audit, report, reviewer, reason):
    st.subheader("Supply the correct name, unit or classification")
    st.caption(
        "One saved mapping applies to every occurrence of the selected source product. "
        "Changing a unit never converts its quantity. Correct quantities separately if needed."
    )
    candidates = items[
        (items.needs_review | items.scope.eq("unclassified")) & ~items.scope.eq("non_food")
    ].copy()
    if candidates.empty:
        st.success("No product fields need review.")
        return
    groups = candidates.groupby("source_product_id", sort=False).first()
    selected = st.selectbox(
        "Source product to correct",
        list(groups.index),
        format_func=lambda key: f"{groups.loc[key, 'product_name'] or '(blank name)'} · "
        f"{groups.loc[key, 'unit']} · HSN {groups.loc[key, 'hsn_code'] or '(missing)'} · {key}",
    )
    rows = items[items.source_product_id.eq(selected)]
    row = rows.iloc[0]
    st.dataframe(
        rows[
            [
                "source_filename",
                "source_row",
                "date",
                "invoice_id",
                "original_product_name",
                "unit",
                "quantity",
                "taxable_value",
                "issues",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    with st.form(f"product-review-{selected}"):
        name = st.text_input("Correct product name", value=row.product_name)
        unit = st.selectbox(
            "Correct recorded unit",
            ["Leave unresolved", "KG", "PCS", "BAG", "SET"],
            index=["Leave unresolved", "KG", "PCS", "BAG", "SET"].index(row.unit)
            if row.unit in {"KG", "PCS", "BAG", "SET"}
            else 0,
        )
        hsn = st.text_input("Correct HSN (8 digits, if known)", value=row.hsn_code)
        checked = st.checkbox("I checked this mapping for all displayed source rows.")
        submitted = st.form_submit_button("Save product correction")
    if submitted:
        if not checked:
            st.error("Check the source rows before saving this mapping.")
            return
        values = {"product_name": name.strip()}
        if unit != "Leave unresolved":
            values["unit"] = unit
        if hsn.strip():
            values["hsn_code"] = hsn.strip()
        persist_review(root, [("product", selected, values)], reviewer, reason)


def date_review(root, items, audit, report, reviewer, reason):
    st.subheader("Were absent invoice dates closed days or missing records?")
    with st.form("coverage-review"):
        bounds = st.date_input(
            "Period to confirm",
            value=(
                pd.Timestamp(report["start_date"]).date(),
                pd.Timestamp(report["end_date"]).date(),
            ),
        )
        status = st.radio(
            "Dates without invoices in this period mean",
            [
                "No trading / no sales",
                "Missing records — keep unresolved",
            ],
        )
        st.caption(
            "Existing sales are never erased by this setting. "
            "Only dates without invoices are classified."
        )
        submitted = st.form_submit_button("Save date coverage")
    if submitted:
        if len(bounds) != 2:
            st.error("Select start and end dates.")
            return
        values = {
            "start": str(bounds[0]),
            "end": str(bounds[1]),
            "status": "no_trading" if status == "No trading / no sales" else "missing",
        }
        persist_review(root, [("coverage", f"{bounds[0]}/{bounds[1]}", values)], reviewer, reason)


def unit_review(root, items, audit, report, reviewer, reason):
    st.subheader("Confirm KG and PCS are separate valid measures")
    st.write(
        "This clears mixed-unit warnings for confirmed KG/PCS rows "
        "while keeping quantities separate."
    )
    with st.form("unit-meanings"):
        checked = st.checkbox("KG/KGS means kilograms; PCS means pieces. Neither means the other.")
        submitted = st.form_submit_button("Save unit meanings")
    if submitted:
        if not checked:
            st.error("Confirm the unit meanings before saving.")
            return
        persist_review(
            root,
            [("units", "recorded_units", {"KG": "kilograms", "PCS": "pieces"})],
            reviewer,
            reason,
        )


def duplicate_review(root, items, audit, report, reviewer, reason):
    st.subheader("Review repeated entries")
    repeated = items[items.issues.str.contains("repeated_item_line_needs_review")]
    if repeated.empty:
        st.info("No unresolved repeated entries. Previous decisions are in Review history.")
        return
    key = st.selectbox("Invoice containing repeated entries", list(repeated.invoice_key.unique()))
    st.dataframe(repeated[repeated.invoice_key.eq(key)], hide_index=True, width="stretch")
    with st.form(f"duplicate-{key}"):
        decision = st.radio(
            "Correct treatment", ["Both entries are valid", "Keep one of each identical entry"]
        )
        st.caption(
            "Keeping one recalculates the working invoice totals without extra rounding. "
            "Original entries stay in SQL."
        )
        submitted = st.form_submit_button("Save repeated-entry decision")
    if submitted:
        values = (
            {"both_valid": True} if decision == "Both entries are valid" else {"keep_one": True}
        )
        persist_review(root, [("duplicate", key, values)], reviewer, reason)


def amount_review(root, items, audit, report, reviewer, reason):
    st.subheader("Enter amounts verified against the invoice")
    key = st.selectbox("Invoice date and number", list(audit.invoice_key))
    lines = items[items.invoice_key.eq(key)]
    header = audit[audit.invoice_key.eq(key)].iloc[0]
    st.dataframe(
        lines[
            [
                "line_number",
                "source_row",
                "product_name",
                "quantity",
                "unit_rate",
                "taxable_value",
                "tax_amount",
                "item_total",
                "issues",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    kind = st.radio("Correct a value in", ["Item line", "Invoice header"])
    if kind == "Item line":
        ordinal = st.selectbox("Item line number", list(lines.line_number))
        fields, target, record_kind = sorted(LINE_FIELDS), f"{key}#{ordinal}", "line"
    else:
        fields, target, record_kind = sorted(INVOICE_FIELDS), key, "invoice"
        st.dataframe(
            pd.DataFrame([header])[sorted(INVOICE_FIELDS)], hide_index=True, width="stretch"
        )
    with st.form(f"amount-{target}-{record_kind}"):
        field = st.selectbox("Field to correct", fields)
        # A separate confirmation avoids accidentally saving a default as a correction.
        value = st.number_input(
            "Verified amount / quantity", min_value=0.0, value=None, format="%.4f"
        )
        st.caption(
            "Current values are shown above. Correct related item/header totals as needed; "
            "checks remain until they agree."
        )
        submitted = st.form_submit_button("Save verified amount")
    if submitted:
        if value is None:
            st.error("Enter the verified value.")
            return
        persist_review(root, [(record_kind, target, {field: value})], reviewer, reason)
