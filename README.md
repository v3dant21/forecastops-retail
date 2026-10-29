# ForecastOps

Retail demand planning using actual retail  sales.

The dashboard answers four questions:

- How many units are likely to sell over the next 7, 14 or 28 days?
- Which products have rising or easing demand?
- Given stock counts I enter, how many units should I consider ordering?
- Where did the forecast miss actual sales, and by how much?

## Run

Requires Python 3.10+ and the dependencies in `pyproject.toml`.

```bash
pip install -e .
make data
make train
make dashboard
```

Dashboard: http://localhost:8501. Optional API: `make api`, documentation at
http://localhost:8000/docs. Load `models/retail_model.pkl` with
`POST /api/v1/load-model`, passing `{"model_path":"models/retail_model.pkl"}`.
Only models with verified M5 provenance can be served.

`make data` downloads a pinned [Nixtla M5 mirror](https://github.com/Nixtla/m5-forecasts),
checks the archive SHA-256 and records per-file checksums. The original dataset is
from the [M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy).
The mirror retains actual observations, with redundant ID/day labels reconstructed
from item/store keys and dates where needed. The competition data has its own use
terms; this repository does not grant a new license to the data.

The default preparation selects 100 actual products across departments, five stores
across CA/TX/WI, and the latest 730 observed days. This is a real-data subset, not
full-catalogue coverage. To change scope:

```bash
python scripts/prepare_data.py --subset-products 200 --subset-stores 10 --subset-days 1000
make train
```

Use `-1` for all products, stores or days; the full panel requires considerably more
memory. Raw sales and prices are read in chunks and selection happens before melting.

## Forecast validation

`make train` runs `scripts/build_retail_plan.py`:

1. Validate dataset provenance and select series with sufficient observed history.
2. Compare seasonal-naive and XGBoost on an earlier 28-day validation period.
3. Select a method using validation WAPE, then check it on the final untouched 28 days.
4. Refit that method on all observed history and forecast the following 28 days.
5. Save the retail plan, held-out predictions, final model and MLflow record.

Every rollout is recursive: earlier model predictions become later lag inputs.
Future actual sales never become forecast inputs. Calendar events and recorded
weekly prices are treated as known in advance; missing future prices are carried
forward and counted in the report. The dashboard displays the historical planning
date. These are historical forecasts, not current predictions for your own store.

The stock calculator uses entered on-hand/incoming stock and a user-chosen buffer.
The buffer is not a confidence interval. M5 does not contain live inventory,
lead times, case sizes or expiry dates, so demand signals cannot establish stockouts
or automatic purchase quantities. Item codes are anonymous; no product names are invented.

## Repository map

- `src/forecastops/data/m5.py`: verified download, source checks and real-data preparation.
- `src/forecastops/features/`: causal sales history, calendar, price and identity features.
- `src/forecastops/models/`: baseline, tree and LSTM wrappers with complete persistence.
- `src/forecastops/planning/`: recursive forecasts and retailer stock calculations.
- `src/forecastops/dashboard/`: demand overview, product planner, reliability and source notes.
- `src/forecastops/api/`, `serving/`: validated model loading and prediction endpoints.
- `src/forecastops/evaluation/`, `tracking/`, `utils/`: metrics, MLflow and configuration.
- `scripts/`: download, prepare, retail-plan build and optional model experiments.
- `sql/`: optional PostgreSQL archival schema; `data/loader.py` supports ingestion.
- `tests/`: unit and regression checks. Small constructed fixtures are isolated tests,
  never application data or retailer forecasts.

Generated runtime directories are `data/raw`, `data/features`, `models`, `reports`
and `mlruns`. Do not commit their data or artifacts. Unused scaffolding and old
artificial artifacts have been removed from the active project.

## Optional experiments and services

`python scripts/train.py --model lightgbm` and `scripts/tune.py` run experiments on
verified observations. Their simple chronological one-step metrics differ from the
retail plan's recursive 28-day validation. Use `make train` to publish the dashboard plan.
The standalone evaluation/benchmark commands inspect saved model artifacts; rerunning
evaluation on a model fitted on all history is not a fresh held-out accuracy check.

`docker compose up --build api dashboard` runs the UI and API using the local prepared
artifacts, mounted read-only. PostgreSQL and an MLflow server are optional profiles:
`docker compose --profile database --profile tracking up -d postgres mlflow`.
The full image includes large ML dependencies. Local training defaults to SQLite MLflow.

## Check

```bash
python -m pytest tests/ -q
python -m compileall -q src scripts
```

New source/provenance, recursive-rollout, stock-planning and dashboard checks live
in `tests/test_retail_pipeline.py`. A missing or unverified dataset must fail clearly;
it must never cause invented sales to appear.

## Restart the saved version

No download or retraining is needed to reopen an existing retail plan:

```bash
cd /home/vedant/forecastops
python3 -m streamlit run src/forecastops/dashboard/streamlit_app.py
```

Open http://localhost:8501. Stop the process with Ctrl+C. To rebuild the plan using
already prepared observations, run `python3 scripts/build_retail_plan.py` before
starting the dashboard. `make data` is only necessary to download/prepare M5 again
or change the selected scope. Avoid `make clean` when preserving the saved plan:
it removes prepared features, models, reports and experiment records.

For a fresh Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Activate that environment again in each new terminal if you chose this setup.

Optional API, in a separate terminal:

```bash
cd /home/vedant/forecastops
python3 -m uvicorn forecastops.api.main:app --host 127.0.0.1 --port 8000
```

Load the saved retail model after each API restart:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/load-model \
  -H 'Content-Type: application/json' \
  -d '{"model_path":"models/retail_model.pkl"}'
```

## Adapting this version to a retailer's own sales

Custom retail imports are not implemented in this M5-specific version. Do not
replace the pinned M5 files or forge their manifest. Add a separate, validated
source adapter with its own provenance and configurable product/category labels.

A suitable canonical daily-sales format is:

```csv
date,product_id,store_id,sales,sell_price
2026-10-01,SKU_001,STORE_01,12,49.00
2026-10-02,SKU_001,STORE_01,9,49.00
```

This is a format illustration, not an application dataset. `sales` means units
sold, with one row per product/store/date; `sell_price` means price per unit.
Optional attributes include category, department, promotion, holiday/store closure,
on-hand inventory, supplier lead time and incoming purchase orders.

Implementation work:

1. Add `src/forecastops/data/retail.py` to validate source CSV/POS/database exports,
   daily aggregation, duplicate rows, dates, units, currencies and missing periods.
   A missing record becomes zero only when it is confirmed to mean no sales.
2. Generalize source validation in `data/m5.py`, training scripts and API loading
   to allow a separate verified retail source. Record file hashes, date scope,
   product/store counts and import time; preserve the M5 checks for M5 sources.
3. Replace M5-specific product/store encodings, categories and state SNAP features
   in `features/engineer.py` with mappings learned from the retailer's training data.
   Supply any category/department/state fields still required by the current rollout,
   or generalize that metadata contract to make them optional.
4. Supply the next 28 days of calendar information and planned prices/promotions.
   Disclose any carried-forward price assumption.
5. Reuse the causal validation/refit flow in `scripts/build_retail_plan.py` and
   `planning/forecast.py`. The current design needs more than 112 usable feature
   days plus historical warmup; roughly five months of continuous daily history
   is a sensible minimum for satisfying that code path, not an accuracy guarantee.
6. Make source labels, categories and dates configurable in the dashboard. Keep
   inventory assumptions distinct from observed sales. Stockouts can suppress
   recorded sales, so sales alone cannot establish unconstrained customer demand.

After the adapter is implemented, daily operations become: import actual new
sales → validate → prepare features → backtest/retrain → publish the updated plan.
An automated refresh is not configured in this version.

## Import the detailed sale-invoice workbook

```bash
python3 scripts/import_invoices.py "/path/to/Sale-Invoice-2026-10-29.xlsx"
python3 -m streamlit run src/forecastops/dashboard/streamlit_app.py
```

The import parses invoice headers and item blocks, preserves the original file by
SHA-256, normalizes labels/numbers/units, filters food HSN chapters, and writes
`data/retail/food_items`, `daily_food_sales`, `review_items` and
`invoice_reconciliation` in CSV and Parquet. `reports/active_dataset.json` switches
the dashboard to the retailer's own data. Food analysis never combines KG, PCS,
bags or unspecified quantities. Ambiguous product suffixes and units require
review; an original row number is preserved for every item. Re-importing the same
workbook replaces the same outputs without appending duplicate sales.

This October export supplies sales analysis and a cleaning audit, not a validated
forecast. Dates absent from an export remain unknown unless the business confirms
that they were no-trading days; confirmed closed dates can then be displayed as zero.
The dashboard and prediction API do not substitute M5 forecasts for this business.
The original M5 files remain a separate historical benchmark.

For the supplied file, rerun the exact import with:

```bash
cd /home/vedant/forecastops
python3 scripts/import_invoices.py "/home/vedant/Downloads/Sale-Invoice-2026-10-29.xlsx"
```

The active business dataset has 929 food lines from October 1–29, 2026. Twelve
non-food lines are excluded and one unclassified line is retained for review.
All 161 original invoice blocks reconcile before filtering. Seventy-two food
lines require unit/label review. The original source and its checksum are retained.

## Build a purchase order from food sales

Open **Purchase order** in the invoice dashboard sidebar. Choose a sales period,
enter your business, supplier and delivery details, then select the items supplied
by that supplier. Enter current stock, incoming stock due within the lead time,
supplier pack size, minimum order and optional purchase cost **in each displayed unit**.
Supplier information and stock are manual inputs; sales invoices do not contain them.

The draft uses recorded quantity divided by calendar days in the selected period.
Target stock covers supplier lead time + time until the next stock review + extra
safety days. Stock on hand and incoming stock are subtracted, and a positive gap
is rounded up to the supplier's pack size and minimum order. No order is suggested
when existing stock already covers the target. KG, PCS and BAG remain separate.
Any product with unresolved labels or units in the imported source is blocked.

Confirm that the sales export covers the whole period (including genuine days
with no sales) and that you checked the inventory and supplier inputs to enable
the **Download purchase order draft (CSV)** button. The export contains buyer,
supplier, item quantities, optional costs, planning assumptions and source checksum.
Missing costs leave the draft unpriced; sales prices are not treated as costs.
Taxes and freight are excluded. This is a reviewable restocking estimate from
observed sales, not a validated model forecast or an order sent to a supplier.
Form entries last for the current browser session; download drafts to retain them.

## Session-only sales intake

The retailer dashboard does not keep uploaded sales in a permanent database.
Each fresh Streamlit session opens at the upload screen. A supported single-sheet
`.xlsx` invoice workbook is parsed through a temporary SQLite working database so
the existing cleaning, reconciliation, and warning-review pipeline can run. The
original upload is not archived. Cleaned sales remain available only while that
dashboard session is active so the retailer can review results and prepare an
order draft.

```bash
cd /home/vedant/forecastops
python3 -m streamlit run src/forecastops/dashboard/streamlit_app.py
```

After import, the app opens **Data cleaning**. Additional workbooks can be added
from the sidebar and join the same temporary workspace. Use **Discard session data
and start over** to delete that workspace immediately and return to upload; the
temporary workspace is also cleaned up when the session ends. No uploaded workbook,
cleaned sales table, review decision, or purchase draft is written to the project's
permanent `data/` or `reports/` directories. Download a purchase-order CSV if you
need to retain that output outside the session.

The SQL schema in `src/forecastops/data/retail_schema.sql` is used only for the
temporary working database. It is not a persistent retailer ledger. The separate
M5 files and model artifacts remain the historical benchmark path; the session-only
invoice workflow does not use M5 sales as a fallback.
