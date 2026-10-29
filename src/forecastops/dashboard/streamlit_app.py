"""Retailer sales intake and planning dashboard, with an optional M5 benchmark."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecastops.dashboard.invoice_app import show_invoice_workspace
from forecastops.data.invoices import import_invoices
from forecastops.data.m5 import require_real_features
from forecastops.models.io import load_model as restore_model
from forecastops.planning.forecast import KEYS, retail_summary, stock_plan

CATEGORY_NAMES = {"FOODS": "Food"}
ACCENT = "#20786b"


@st.cache_resource
def load_model(path, modified_ns=0):
    """Keep the same restoration path as API and evaluation."""
    return restore_model(path)


@st.cache_data
def load_plan(stamps):
    report = json.loads(Path("reports/retail_plan.json").read_text())
    history = pd.read_parquet("data/features/history.parquet")
    provenance = require_real_features(history)
    if report["data_fingerprint"] != provenance["files"]["sales_train_validation.csv"] or report[
        "forecast_origin"
    ] != str(history.date.max().date()):
        raise ValueError(
            "The retail plan does not match the prepared sales history. Rebuild the plan."
        )
    forecast = pd.read_parquet("reports/forecast.parquet")
    backtest = pd.read_parquet("reports/backtest.parquet")
    eligible = forecast[KEYS].drop_duplicates()
    history = history.merge(eligible, on=KEYS, validate="many_to_one")
    return report, history, forecast, backtest


def style():
    st.markdown(
        """<style>
    .stApp {background:#f7f8f6;color:#202b28;font-family:'Segoe UI',Arial,sans-serif;}
    [data-testid="stSidebar"] {background:#edf1ed;border-right:1px solid #d9e0da;}
    .block-container {max-width:1400px;padding-top:2.4rem;padding-bottom:3rem;}
    h1 {font-size:2.3rem!important;letter-spacing:-.065rem;font-weight:650!important;}
    h2 {font-size:1.35rem!important;letter-spacing:-.025rem;}
    h3 {font-size:1.05rem!important;}
    [data-testid="stMetricValue"] {font-variant-numeric:tabular-nums;font-size:2rem;}
    [data-testid="stMetricLabel"] {color:#516159;}
    [data-testid="stMetric"] {padding:1rem 0;border-top:1px solid #cdd7ce;}
    .eyebrow {color:#20786b;font-size:.75rem;letter-spacing:.13rem;font-weight:650;}
    .reading {border-left:3px solid #20786b;padding:.2rem 0 .2rem 1rem;margin:1rem 0;}
    .stButton button,.stDownloadButton button {border-radius:6px;transition:transform .15s;}
    .stButton button:active,.stDownloadButton button:active {transform:translateY(1px);}
    @media(max-width:768px){.block-container{padding:1.2rem;}h1{font-size:1.8rem!important;}}
    </style>""",
        unsafe_allow_html=True,
    )


def import_sales_workbook(uploaded):
    """Feed an uploaded workbook into the existing invoice cleaning pipeline."""
    filename = Path(uploaded.name.replace("\\", "/")).name
    workspace = st.session_state.get("sales_workspace")
    if workspace is None:
        workspace = tempfile.TemporaryDirectory(prefix="forecastops-sales-session-")
        st.session_state["sales_workspace"] = workspace
    work_root = Path(workspace.name)
    retail_dir = work_root / "retail"
    report_dir = work_root / "reports"
    try:
        with tempfile.TemporaryDirectory(prefix="forecastops-upload-") as temporary:
            source = Path(temporary) / filename
            source.write_bytes(uploaded.getvalue())
            report = import_invoices(
                source,
                destination=retail_dir,
                reports=report_dir,
                archive_source=False,
            )
    except Exception as error:
        if "active_invoice_dataset" not in st.session_state:
            workspace = st.session_state.pop("sales_workspace", None)
            if workspace is not None:
                workspace.cleanup()
        st.error(f"The workbook could not be imported: {error}")
        return
    # No source workbook archive is kept; only temporary cleaned working data remain.
    active = json.loads((report_dir / "active_dataset.json").read_text())
    st.session_state["active_invoice_dataset"] = active
    stats = report.get("last_import", {})
    st.session_state["invoice_import_notice"] = (
        f"{filename} processed. {stats.get('inserted_invoices', 0):,} new invoices; "
        f"{stats.get('overlapping_invoices', 0):,} duplicates matched in this session. "
        "Review the cleaning and reconciliation results below."
    )
    st.session_state["sales_intake_complete"] = True
    st.session_state["invoice_workspace_page"] = "Data cleaning"
    st.cache_data.clear()
    st.rerun()


def sales_workbook_upload():
    """Keep workbook intake available for adding later sales exports."""
    with st.sidebar.expander("Upload another sales spreadsheet"):
        st.caption(
            "The workbook is processed in this session and is not added "
            "to a permanent database."
        )
        with st.form("additional-sales-upload", clear_on_submit=True):
            uploaded = st.file_uploader(
                "Sales workbook (.xlsx)",
                type=["xlsx"],
                key="sales_workbook_upload",
            )
            submitted = st.form_submit_button("Import and clean sales data")
        if submitted and uploaded is None:
            st.error("Choose an .xlsx workbook first.")
        elif submitted:
            import_sales_workbook(uploaded)


def first_sales_upload():
    """Start each fresh dashboard session with sales intake, never sample data."""
    st.markdown('<p class="eyebrow">FORECASTOPS / SALES INTAKE</p>', unsafe_allow_html=True)
    st.title("Start by uploading your sales spreadsheet")
    st.write(
        "Upload a sales invoice workbook to begin. ForecastOps will clean the records, "
        "check invoice totals, and then open the cleaning review before sales and "
        "purchase-order planning."
    )
    st.info(
        "This intake currently supports one .xlsx invoice workbook in the existing "
        "invoice-export layout. It does not automatically map arbitrary spreadsheet columns."
    )
    with st.form("first-sales-upload", clear_on_submit=True):
        uploaded = st.file_uploader(
            "Choose sales invoice workbook (.xlsx)",
            type=["xlsx"],
            key="first_sales_workbook_upload",
        )
        submitted = st.form_submit_button("Upload, clean and continue")
    if submitted and uploaded is None:
        st.error("Choose an .xlsx workbook first.")
    elif submitted:
        import_sales_workbook(uploaded)
    st.caption(
        "No preloaded sales dashboard is shown before upload. Raw uploads and "
        "processed sales are kept only in this session and discarded when you start over."
    )


def require_sales_intake():
    if st.session_state.get("sales_intake_complete", False):
        return True
    first_sales_upload()
    return False


def subset(frame, store, category):
    result = frame
    if store != "All selected stores":
        result = result[result.store_id == store]
    if category != "All categories":
        result = result[result.cat_id == category]
    return result.copy()


def demand_change(expected, recent):
    if recent <= 0:
        return None
    return 100 * (expected - recent) / recent


def change_words(change):
    if change is None:
        return "No recent sales for comparison"
    return f"{abs(change):.1f}% {'more' if change >= 0 else 'fewer'} units than the previous period"


def trend_chart(history, forecast):
    cutoff = history.date.max() - pd.Timedelta(days=55)
    actual = history[history.date >= cutoff].groupby("date").sales.sum()
    expected = forecast.groupby("date").prediction.sum()
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual.values,
            name="Units sold",
            mode="lines",
            line={"color": "#87938c", "width": 2},
        )
    )
    chart.add_trace(
        go.Scatter(
            x=expected.index,
            y=expected.values,
            name="Expected units",
            mode="lines",
            line={"color": ACCENT, "width": 3, "dash": "dash"},
        )
    )
    chart.add_vline(x=history.date.max().timestamp() * 1000, line_dash="dot", line_color="#acb7af")
    chart.update_layout(
        height=340,
        margin={"l": 0, "r": 10, "t": 15, "b": 0},
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Units per day",
        xaxis_title=None,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.1},
        font={"family": "Arial", "color": "#35483f"},
    )
    st.plotly_chart(chart, width="stretch")


def overview(history, future, backtest, horizon):
    st.title("Know what your shelves will need.")
    st.write(f"A {horizon}-day demand outlook for the products and stores you selected.")
    origin = history.date.max()
    recent = history[history.date > origin - pd.Timedelta(days=horizon)]
    expected, previous = float(future.prediction.sum()), float(recent.sales.sum())
    change = demand_change(expected, previous)
    metrics = st.columns(4)
    metrics[0].metric(f"Expected units · next {horizon} days", f"{expected:,.0f}")
    metrics[1].metric(f"Units sold · previous {horizon} days", f"{previous:,.0f}")
    metrics[2].metric("Average expected each day", f"{expected / horizon:,.0f}")
    metrics[3].metric(
        "Product / store combinations", f"{future[KEYS].drop_duplicates().shape[0]:,}"
    )
    left, right = st.columns([2.2, 1], gap="large")
    with left:
        st.subheader("Sales history and the planning outlook")
        trend_chart(history, future)
    with right:
        st.subheader("What this means for your store")
        st.markdown(
            f"Expect **{expected:,.0f} units** across this selection over the next "
            f"**{horizon} days**."
        )
        st.write(change_words(change) + ".")
        if change is not None and change > 15:
            st.write(
                "Demand is expected to rise. Review shelf availability and supplier "
                "delivery dates before placing orders."
            )
        elif change is not None and change < -15:
            st.write(
                "Demand is expected to ease. Check open orders before adding more stock, "
                "especially for perishable products."
            )
        else:
            st.write(
                "Demand is broadly steady. Use your normal replenishment cycle and check "
                "individual products below."
            )
        st.caption(
            "These are demand signals. Stock shortages cannot be identified without "
            "your inventory counts."
        )
    test_total = float(backtest.sales.sum())
    if test_total > 0:
        error_per_100 = 100 * (backtest.sales - backtest.prediction).abs().sum() / test_total
        if error_per_100 > 40:
            st.warning(
                f"Manual review recommended: the historical check missed {error_per_100:.0f} units "
                "in absolute error for every 100 units sold. Treat this outlook as a starting "
                "point and check product-level reliability before ordering."
            )
    st.divider()
    st.subheader("Products to review first")
    st.caption(
        "Ordered by expected sales volume. Changes compare equal-length periods; "
        "they do not prove stock shortages."
    )
    expected_items = future.groupby(["product_id", "cat_id"], as_index=False).prediction.sum()
    previous_items = recent.groupby("product_id", as_index=False).sales.sum()
    table = expected_items.merge(previous_items, on="product_id", how="left").fillna({"sales": 0})
    table["change"] = [
        demand_change(a, b) for a, b in zip(table.prediction, table.sales, strict=True)
    ]

    def note(value):
        if pd.isna(value):
            return "No recent sales — review this item"
        if value > 15:
            return "Rising demand — check availability"
        if value < -15:
            return "Easing demand — review open orders"
        return "Steady demand — follow normal replenishment"

    table["Planning note"] = table.change.map(note)
    table["Category"] = table.cat_id.map(CATEGORY_NAMES)
    table["Change"] = table.change.map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "No recent sales")
    table = (
        table.sort_values("prediction", ascending=False)
        .head(15)
        .rename(
            columns={"product_id": "SKU", "prediction": "Expected units", "sales": "Recent units"}
        )
    )
    st.dataframe(
        table[["SKU", "Category", "Expected units", "Recent units", "Change", "Planning note"]],
        hide_index=True,
        width="stretch",
        column_config={
            "Expected units": st.column_config.NumberColumn(format="%.0f"),
            "Recent units": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.download_button(
        "Download this demand outlook",
        future[KEYS + ["date", "prediction"]]
        .rename(columns={"prediction": "expected_units"})
        .to_csv(index=False),
        file_name=f"demand-outlook-{origin.date()}.csv",
        mime="text/csv",
    )


def product_planner(history, all_future, backtest, horizon):
    st.title("Turn expected demand into a stock plan.")
    st.write("Choose a product, check its outlook, then enter your stock position.")
    products = sorted(all_future.product_id.unique())
    product = st.selectbox("Product SKU", products)
    stores = sorted(all_future.loc[all_future.product_id == product, "store_id"].unique())
    store = st.selectbox("Store for this product", stores)

    def mask(frame):
        return (frame.product_id == product) & (frame.store_id == store)

    observed = history[mask(history)]
    outlook = all_future[mask(all_future)].sort_values("date")
    displayed = outlook.head(horizon)
    category = CATEGORY_NAMES[outlook.cat_id.iloc[0]]
    st.subheader(f"{category} · {product} · {store}")
    st.caption("M5 uses anonymous item codes. Product names and pack sizes are not supplied.")
    expected = displayed.prediction.sum()
    recent = observed[observed.date > observed.date.max() - pd.Timedelta(days=horizon)].sales.sum()
    first, second = st.columns(2)
    first.metric(f"Expected sales over {horizon} days", f"{expected:,.1f} units")
    second.metric(f"Sold in the previous {horizon} days", f"{recent:,.0f} units")
    st.write(change_words(demand_change(expected, recent)) + ".")
    trend_chart(observed, displayed)
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.subheader("Replenishment calculator")
        st.caption(
            "Your inputs are assumptions; this dataset has no live inventory or "
            "supplier lead times."
        )
        with st.form("stock_plan"):
            days = st.number_input(
                "Days to cover (delivery time + review period)",
                min_value=1,
                max_value=28,
                value=horizon,
            )
            on_hand = st.number_input("Units currently on hand", min_value=0, value=None, step=1)
            incoming = st.number_input(
                "Units already on order, arriving within this period", min_value=0, value=0, step=1
            )
            buffer = st.slider(
                "Extra stock allowance (%)",
                min_value=0,
                max_value=50,
                value=10,
                help="A planning allowance you choose, not a statistical confidence interval.",
            )
            submitted = st.form_submit_button("Calculate suggested order")
        if submitted:
            if on_hand is None:
                st.error("Enter your current stock count to calculate an order.")
            else:
                demand = float(outlook.head(days).prediction.sum())
                plan = stock_plan(demand, on_hand, incoming, buffer)
                st.success(f"Suggested order: {plan['suggested_order']:,} units")
                st.write(
                    f"Expected demand: {demand:,.1f} units. With your {buffer}% allowance, "
                    f"target stock is {plan['target_stock']:,} units. "
                    f"You entered {plan['available_stock']:,} units on hand or arriving."
                )
                st.caption(
                    "Check expiry dates, case sizes, delivery timing and storage space before "
                    "ordering."
                )
    with right:
        st.subheader("How much trust to place in this outlook")
        tested = backtest[mask(backtest)]
        error = (tested.sales - tested.prediction).abs().mean()
        st.write(
            f"In the final historical 28-day check, the forecast missed this product’s daily sales "
            f"by **{error:.1f} units per day on average**."
        )
        missed = float((tested.sales - tested.prediction).sum())
        st.write(
            f"Across that check it {'underestimated' if missed >= 0 else 'overestimated'} "
            f"sales by **{abs(missed):,.1f} units in total**."
        )
        st.caption(
            "An average error is not a guarantee or a prediction interval. Review "
            "erratic or newly launched products manually."
        )


def reliability(backtest):
    st.title("See where the forecast needs a closer look.")
    st.write("These results come from the final 28 days held out of model fitting for this check.")
    error = (backtest.sales - backtest.prediction).abs()
    total = float(backtest.sales.sum())
    missed = float((backtest.sales - backtest.prediction).sum())
    columns = st.columns(3)
    columns[0].metric("Average miss per product / store / day", f"{error.mean():.2f} units")
    columns[1].metric(
        "Absolute error per 100 units sold",
        f"{100 * error.sum() / total:.1f}" if total else "No sales",
    )
    columns[2].metric(
        "Net bias across the check",
        f"{abs(missed):,.0f} units",
        help="Underestimation means the forecast was lower than observed sales in total.",
    )
    st.write(
        f"The method {'underestimated' if missed >= 0 else 'overestimated'} total "
        f"sales in this selection. "
        "Use the product-level check below to decide where manual review matters most."
    )
    st.caption(
        "Error per 100 units is not an accuracy score. It can exceed 100 for "
        "difficult, low-volume items."
    )
    daily = backtest.groupby("date", as_index=False)[["sales", "prediction"]].sum()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily.date, y=daily.sales, name="Units actually sold", line={"color": "#87938c"}
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily.date,
            y=daily.prediction,
            name="Forecast made without those sales",
            line={"color": ACCENT, "dash": "dash"},
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=330,
        yaxis_title="Units per day",
        paper_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h"},
    )
    st.plotly_chart(fig, width="stretch")
    table = retail_summary(backtest).sort_values("absolute_error", ascending=False)
    table = table.rename(
        columns={
            "product_id": "SKU",
            "store_id": "Store",
            "actual_units": "Units sold",
            "predicted_units": "Units forecast",
            "daily_error": "Average daily miss",
            "missed_units": "Underestimated units",
            "excess_units": "Overestimated units",
        }
    )
    st.subheader("Largest misses to review")
    st.dataframe(
        table[
            [
                "SKU",
                "Store",
                "Units sold",
                "Units forecast",
                "Average daily miss",
                "Underestimated units",
                "Overestimated units",
            ]
        ].head(30),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Underestimated units are forecast errors, not confirmed lost sales. Stock "
        "availability was not recorded."
    )


def data_notes(report):
    source = report["source"]
    st.title("Real observations. Clear boundaries.")
    st.write(
        "This workspace uses historical Walmart M5 sales, calendar events and weekly prices. "
        "There are no generated sales in the planning pipeline."
    )
    st.markdown(
        f"[Walmart M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) · "
        f"[Pinned public data mirror]({source['source_page']})"
    )
    st.write(
        f"Prepared scope: **{source['products']} actual products**, "
        f"**{len(source['stores'])} stores**, "
        f"from **{source['start_date']}** through **{source['end_date']}**. "
        f"The source contains {source['available_products']:,} products across "
        f"{source['available_stores']} stores."
    )
    st.write(
        f"The retail plan covers **{report['planning_products']} products** and "
        f"**{report['planning_series']} product/store combinations** with enough "
        f"observed history for validation."
    )
    st.subheader("How the plan was checked")
    st.write(
        f"Two methods were compared on an earlier 28-day validation period ending "
        f"{report['validation_end']}. "
        f"The selected method was then checked separately from "
        f"{report['test_start']} to {report['test_end']}. "
        "During each 28-day rollout, it used its own earlier predictions rather "
        "than looking at future sales. "
        "The planning model was then refitted using all sales available at the "
        "displayed historical date."
    )
    st.subheader("What to keep in mind")
    for assumption in report["assumptions"]:
        st.write("• " + assumption)
    st.write(
        "• Product codes are anonymous; product names, case sizes and expiry dates are unavailable."
    )
    st.write("• This archive is historical. It is not a current forecast for your own shop.")
    st.caption(f"Source archive SHA-256: {source['archive_sha256']}")


def render_dashboard_workspace():
    st.sidebar.markdown('<p class="eyebrow">FORECASTOPS</p>', unsafe_allow_html=True)
    sales_workbook_upload()
    active = st.session_state.get("active_invoice_dataset")
    if active and active.get("source_type") == "retail_invoice":
        if notice := st.session_state.pop("invoice_import_notice", None):
            st.success(notice)
        show_invoice_workspace(active)
        return
    st.sidebar.title("Food planning")
    page = st.sidebar.radio(
        "Workspace",
        [
            "Demand overview",
            "Product & stock planner",
            "Forecast reliability",
            "Data & assumptions",
        ],
    )
    required = [
        Path(p)
        for p in [
            "reports/retail_plan.json",
            "reports/forecast.parquet",
            "reports/backtest.parquet",
            "data/features/history.parquet",
        ]
    ]
    if not all(path.exists() for path in required):
        st.title("Your real-data plan is being prepared.")
        st.write(
            "The dashboard needs verified M5 observations and a completed historical "
            "validation run."
        )
        st.info(
            "Run make data, then make train. The application will not substitute artificial sales."
        )
        return
    try:
        report, history, forecast, backtest = load_plan(
            tuple(path.stat().st_mtime_ns for path in required)
        )
    except (ValueError, OSError, KeyError) as error:
        st.error(str(error))
        return
    st.sidebar.divider()
    store = st.sidebar.selectbox(
        "Store", ["All selected stores"] + sorted(forecast.store_id.unique())
    )
    categories = ["FOODS"]
    category = st.sidebar.selectbox(
        "Category",
        ["All categories"] + categories,
        format_func=lambda value: CATEGORY_NAMES.get(value, value),
    )
    horizon = st.sidebar.selectbox(
        "Planning period", [7, 14, 28], format_func=lambda days: f"Next {days} days"
    )
    st.sidebar.caption(f"Planning date: {report['forecast_origin']}\n\nHistorical Walmart M5 data")
    st.markdown(
        '<p class="eyebrow">DEMAND PLANNING / HISTORICAL M5 WORKSPACE</p>', unsafe_allow_html=True
    )
    st.caption(
        f"As of {report['forecast_origin']} · Actual sales archive · Forecast dates "
        f"are historical, not today’s trading outlook"
    )
    history = history[history.cat_id.eq("FOODS")]
    forecast = forecast[forecast.cat_id.eq("FOODS")]
    backtest = backtest[backtest.cat_id.eq("FOODS")]
    history = subset(history, store, category)
    forecast = subset(forecast, store, category)
    backtest = subset(backtest, store, category)
    if forecast.empty:
        st.info("No products match this store and category. Choose a different selection.")
        return
    visible = forecast[
        forecast.date <= pd.Timestamp(report["forecast_origin"]) + pd.Timedelta(days=horizon)
    ]
    if page == "Demand overview":
        overview(history, visible, backtest, horizon)
    elif page == "Product & stock planner":
        product_planner(history, forecast, backtest, horizon)
    elif page == "Forecast reliability":
        reliability(backtest)
    else:
        data_notes(report)


def main():
    st.set_page_config(page_title="ForecastOps · Retail planning", page_icon=None, layout="wide")
    style()
    if require_sales_intake():
        render_dashboard_workspace()


if __name__ == "__main__":
    main()
