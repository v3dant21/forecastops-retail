"""Sales and cleaning views for the retailer's own food invoice export."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from forecastops.dashboard.purchase_app import invoice_purchase_order
from forecastops.dashboard.resolution_app import resolution_workspace
from forecastops.data.retail_database import load_database


def load_invoice_data(directory, stamp):
    root = Path(directory)
    report = json.loads((root / "cleaning_report.json").read_text())
    if report.get("storage") == "sqlite":
        items, audit, sources = load_database(root / "retail.sqlite3")
        if {s["source_sha256"] for s in sources} != {s["source_sha256"] for s in report["sources"]}:
            raise ValueError(
                "SQL history changed without a refreshed report. Re-run invoice import."
            )
        food = items[items.scope.eq("food")].copy()
        review = items[items.needs_review | ~items.scope.eq("food")].copy()
    else:
        food = pd.read_parquet(root / "food_items.parquet")
        review = pd.read_parquet(root / "review_items.parquet")
        audit = pd.read_parquet(root / "invoice_reconciliation.parquet")
    return report, food, review, audit


def discard_session_sales():
    """Delete the temporary invoice workspace and return to the upload screen."""
    workspace = st.session_state.pop("sales_workspace", None)
    if workspace is not None:
        workspace.cleanup()
    for key in ("active_invoice_dataset", "invoice_import_notice", "sales_intake_complete"):
        st.session_state.pop(key, None)
    st.session_state.pop("invoice_workspace_page", None)
    st.cache_data.clear()
    st.rerun()


def session_data_controls(report):
    if report.get("storage") == "sqlite":
        st.sidebar.caption("Working data: temporary for this session")
    if st.sidebar.button("Discard session data and start over"):
        discard_session_sales()


def show_invoice_workspace(active):
    root = Path(active["directory"])
    report, food, review, audit = load_invoice_data(
        str(root), (root / "cleaning_report.json").stat().st_mtime_ns
    )
    if report["source_sha256"] != active["source_sha256"]:
        st.error("The active invoice source does not match the cleaning report. Re-import it.")
        return
    st.sidebar.title("Food sales")
    page = st.sidebar.radio(
        "Workspace",
        [
            "Sales overview",
            "Food items",
            "Purchase order",
            "Resolve warnings",
            "Data cleaning",
            "Forecast readiness",
        ],
        key="invoice_workspace_page",
    )
    st.sidebar.caption(
        f"Source: {report.get('source_count', 1)} invoice workbook(s)\n\nFood items only"
    )
    session_data_controls(report)
    st.markdown('<p class="eyebrow">YOUR BUSINESS / FOOD SALES</p>', unsafe_allow_html=True)
    st.caption(
        f"{report.get('source_count', 1)} source workbook(s) · {report['start_date']} to "
        f"{report['end_date']} · Amounts in INR"
    )
    if page in {"Sales overview", "Food items"}:
        customers = ["All customers"] + sorted(food.customer.unique())
        customer = st.sidebar.selectbox("Customer", customers)
        if customer != "All customers":
            food = food[food.customer == customer]
        bounds = st.sidebar.date_input(
            "Invoice dates",
            value=(
                pd.Timestamp(report["start_date"]).date(),
                pd.Timestamp(report["end_date"]).date(),
            ),
            min_value=pd.Timestamp(report["start_date"]).date(),
            max_value=pd.Timestamp(report["end_date"]).date(),
        )
        if len(bounds) != 2:
            st.info("Select a start and end date.")
            return
        food = food[food.date.between(pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))]
        if food.empty:
            st.info("No food invoices match this selection.")
            return
    if page == "Data cleaning":
        invoice_cleaning(review, audit, report)
    elif page == "Forecast readiness":
        invoice_readiness(report)
    elif page == "Resolve warnings":
        resolution_workspace(root, report)
    else:
        {
            "Sales overview": invoice_overview,
            "Food items": invoice_food_items,
            "Purchase order": invoice_purchase_order,
        }[page](food, report)
    st.sidebar.caption(f"Source verified by SHA-256\n\n{report['source_sha256'][:16]}…")


def invoice_overview(food, report):
    st.title("Your food sales, clearly accounted for.")
    st.write(
        "Actual invoiced food sales. Non-food items are excluded; uncertain "
        "records remain visible for review."
    )
    amounts = st.columns(2)
    amounts[0].metric("Food sales before tax", f"₹{food.taxable_value.sum():,.2f}")
    amounts[1].metric("Food value including item tax", f"₹{food.item_total.sum():,.2f}")
    columns = st.columns(2)
    identity = "invoice_key" if "invoice_key" in food else "invoice_id"
    columns[0].metric("Invoices containing food", f"{food[identity].nunique():,}")
    columns[1].metric("Food product names", f"{food.product_name.nunique():,}")
    left, right = st.columns([2, 1], gap="large")
    with left:
        st.subheader("Sales on recorded invoice dates")
        daily = food.groupby("date", as_index=False).taxable_value.sum()
        coverage = report.get("coverage", {})
        dates_confirmed = coverage.get("unresolved_days", 1) == 0
        if dates_confirmed:
            all_days = pd.date_range(daily.date.min(), daily.date.max(), freq="D")
            daily = (
                daily.set_index("date").reindex(all_days, fill_value=0)
                .rename_axis("date").reset_index()
            )
        chart = px.bar(
            daily,
            x="date",
            y="taxable_value",
            color_discrete_sequence=["#20786b"],
            labels={"taxable_value": "Food sales before tax (₹)", "date": "Invoice date"},
        )
        chart.update_layout(template="plotly_white", height=330, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(chart, width="stretch")
        if dates_confirmed:
            st.caption(
                "Dates without invoices were confirmed as no trading; "
                "these days appear as zero food sales."
            )
        else:
            st.caption(
                "Dates without invoices need confirmation; the chart shows "
                "recorded invoice dates only."
            )
    with right:
        st.subheader("What this tells you")
        ranked = food.groupby("product_name").taxable_value.sum().sort_values(ascending=False)
        share = 100 * ranked.iloc[0] / ranked.sum()
        st.write(
            f"**{ranked.index[0]}** has the highest invoiced food value in this selection: "
            f"**₹{ranked.iloc[0]:,.2f}**, or **{share:.1f}%** of food sales before tax."
        )
        st.write(f"The selected records cover **{food.customer.nunique()} buying customers**.")
        st.caption("This shows sales value, not profit or customer demand lost to stockouts.")
    if report["food_lines_needing_review"]:
        st.warning(
            f"{report['food_lines_needing_review']} food lines still need your input. "
            "Open Resolve warnings to supply the missing facts."
        )
    else:
        st.success("Food quantities and labels have passed the current review checks.")
    if report.get("removed_duplicate_lines", 0):
        st.caption(
            f"{report['removed_duplicate_lines']} repeated item entries excluded following review. "
            "Working invoice totals were recalculated; original records remain in SQL."
        )
    if report.get("excluded_placeholder_lines", 0) or report.get("excluded_unidentified_lines", 0):
        st.caption(
            f"Excluded from sales analysis: {report.get('excluded_placeholder_lines', 0)} blank, "
            f"zero-value placeholders and {report.get('excluded_unidentified_lines', 0)} "
            "unidentified zero-value product code. Original source rows remain in SQL."
        )
    if report["invoice_reconciliation_failures"]:
        st.warning(
            f"{report['invoice_reconciliation_failures']} invoice(s) have a totals discrepancy. "
            "See Data cleaning; supplied values have not been corrected by guessing."
        )
    st.subheader("Best-selling food items by sales value")
    show_product_table(food)
    st.download_button(
        "Download cleaned food invoice lines",
        food.to_csv(index=False),
        file_name="cleaned-food-invoice-lines.csv",
        mime="text/csv",
    )


def invoice_food_items(food, report):
    st.title("Review each food item in its recorded unit.")
    st.caption(report["quantity_note"])
    item = st.selectbox("Food item", sorted(food.product_name.unique()))
    selected = food[food.product_name == item]
    st.metric("Invoiced value before tax", f"₹{selected.taxable_value.sum():,.2f}")
    quantities = (
        selected.groupby("unit", as_index=False)
        .quantity.sum()
        .rename(columns={"unit": "Recorded unit", "quantity": "Recorded quantity"})
    )
    st.dataframe(quantities, hide_index=True, width="stretch")
    if selected.needs_review.any():
        st.warning(
            "Some lines for this item need review. Confirm the recorded units and "
            "product labels before using quantities for stock planning."
        )
    st.dataframe(
        selected[
            (["source_filename"] if "source_filename" in selected else [])
            + [
                "date",
                "invoice_id",
                "customer",
                "original_product_name",
                "unit",
                "quantity",
                "unit_rate",
                "taxable_value",
                "issues",
                "source_row",
            ]
        ],
        hide_index=True,
        width="stretch",
    )


def invoice_cleaning(review, audit, report):
    st.title("Every cleaned line has a source row.")
    st.write(
        "The original workbook is preserved. Repeated invoice/item headers "
        "were parsed into invoice records and item lines."
    )
    columns = st.columns(4)
    columns[0].metric("Source item lines", report["source_item_lines"])
    columns[1].metric("Food lines retained", report["food_lines"])
    columns[2].metric("Non-food lines excluded", report["excluded_non_food_lines"])
    columns[3].metric("Unclassified lines for review", report["unclassified_lines"])
    st.subheader("Cleaning actions")
    for rule in report["cleaning_rules"]:
        st.write("• " + rule)
    st.subheader("Remaining review flags")
    if report["invoice_reconciliation_failures"]:
        st.error(
            f"{report['invoice_reconciliation_failures']} invoice(s) do not reconcile. "
            "Source values are preserved; review the reconciliation table before using totals."
        )
    if not report["food_lines_needing_review"] and not report["invoice_reconciliation_failures"]:
        st.success("No food-item review flags or invoice reconciliation differences remain.")
    issue_rows = [
        {"Issue": key.replace("_", " "), "Lines": value}
        for key, value in report["issue_counts"].items()
    ]
    if issue_rows:
        st.caption("Issue counts can overlap on the same line.")
        st.dataframe(pd.DataFrame(issue_rows), hide_index=True, width="stretch")
    st.subheader("Applied review decisions")
    st.write(
        f"{report.get('correction_count', 0)} reviewed decisions are recorded "
        "in the SQL audit history. "
        f"{report.get('removed_duplicate_lines', 0)} repeated entries, "
        f"{report.get('excluded_placeholder_lines', 0)} blank placeholders and "
        f"{report.get('excluded_unidentified_lines', 0)} unidentified zero-value item "
        "are excluded from the working sales analysis. Raw rows remain available in SQL."
    )
    view = st.selectbox(
        "Review group",
        [
            "All review lines",
            "Excluded non-food",
            "Missing classification",
            "Food lines needing review",
        ],
    )
    rows = review
    if view == "Excluded non-food":
        rows = review[review.scope.eq("non_food")]
    elif view == "Missing classification":
        rows = review[review.scope.eq("unclassified")]
    elif view == "Food lines needing review":
        rows = review[review.scope.eq("food")]
    st.dataframe(
        rows[
            (["source_filename"] if "source_filename" in rows else [])
            + [
                "source_row",
                "invoice_id",
                "original_product_name",
                "product_name",
                "hsn_code",
                "unit_original",
                "unit",
                "scope",
                "issues",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    st.subheader("Invoice reconciliation")
    st.write(
        f"**{len(audit) - report['invoice_reconciliation_failures']} of {len(audit)} invoices** "
        "reconcile to the source quantities, taxable values, taxes and totals within ₹0.02. "
        "This check includes all source items before food filtering."
    )
    st.caption(
        "Recorded invoice quantity totals are checked arithmetically only; "
        "they are not a valid combined KG/PCS measure."
    )
    st.dataframe(
        audit[
            (["source_filename"] if "source_filename" in audit else [])
            + [
                "invoice_id",
                "date",
                "quantity_difference",
                "taxable_difference",
                "tax_difference",
                "total_difference",
                "grand_total_rounding",
                "reconciles",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download rows needing review",
        review.to_csv(index=False),
        file_name="invoice-lines-for-review.csv",
        mime="text/csv",
    )


def invoice_readiness(report):
    st.title("Check your history before forecasting.")
    st.info(
        "Validated demand forecasts are unavailable until history and data issues are checked. "
        "The Purchase order page offers a sales-based draft with your stock and supplier inputs."
    )
    st.write(
        f"The export covers **{report['calendar_days']} calendar days**, with invoices on "
        f"**{report['invoice_dates']} dates** and "
        f"**{report['unobserved_dates']} dates without invoice records."
    )
    if report.get("sources"):
        st.subheader("Imported source coverage")
        st.dataframe(
            pd.DataFrame(report["sources"])[
                ["source_filename", "start_date", "end_date", "invoices"]
            ],
            hide_index=True,
            width="stretch",
        )
        coverage = report.get("coverage", {})
        if coverage.get("unresolved_days", report["unobserved_dates"]):
            st.warning("Some dates need your confirmation. Open Resolve warnings → Sales dates.")
        else:
            st.success(
                f"{coverage.get('confirmed_no_trading_days', 0)} dates without invoices "
                "are confirmed no-trading days."
            )
    st.write(report.get("forecast_readiness", ""))
    st.write(
        "The existing forecasting design needs roughly five months of daily "
        "history for warmup and separate validation/test windows. "
        "A longer export supports sales analysis, but does not by itself establish "
        "that model’s forecast accuracy."
    )
    st.caption(
        "No Walmart forecasts or generated sales are being shown as forecasts for your business."
    )


def show_product_table(food):
    identity = "invoice_key" if "invoice_key" in food else "invoice_id"
    table = food.groupby(["product_name", "unit"], as_index=False).agg(
        quantity=("quantity", "sum"),
        sales_before_tax=("taxable_value", "sum"),
        invoices=(identity, "nunique"),
        review_lines=("needs_review", "sum"),
    )
    table = table.sort_values("sales_before_tax", ascending=False).rename(
        columns={
            "product_name": "Food item",
            "unit": "Recorded unit",
            "quantity": "Recorded quantity",
            "sales_before_tax": "Sales before tax (₹)",
            "invoices": "Invoices",
            "review_lines": "Lines needing review",
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={"Sales before tax (₹)": st.column_config.NumberColumn(format="%.2f")},
    )
