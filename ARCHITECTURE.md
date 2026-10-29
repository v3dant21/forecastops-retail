# ForecastOps data flow

Verified pinned M5 archive → chunked selection of actual sales/weekly prices →
calendar join → past-only features → earlier 28-day method validation → separate
28-day recursive test → refit on observed history → 28-day retail demand outlook.

The main path is file-based and reproducible. Parquet preserves source provenance;
JSON manifests record checksums and scope. PostgreSQL is optional archival storage,
not a dependency of local planning. No code path generates substitute sales.

The retailer dashboard reads a published plan, historical observations and held-out
predictions. It rejects a plan whose source fingerprint or planning date does not
match its history. It never calculates accuracy by re-predicting training data.
The stock calculator uses retailer-entered assumptions; M5 supplies no live inventory.

Forecasts use the same feature engineer as training. At every forecast step, current
sales are unknown, lag/rolling values come only from past observations or previous
predictions, and product/store series remain separate. Calendar and recorded prices
are exogenous assumptions explicitly disclosed in the UI.

Artifacts:

- `data/raw/manifest.json`: source URL, pinned archive and file hashes.
- `data/features/features.parquet`: actual observations with causal predictors.
- `data/features/history.parquet`: actual source history for recursive rollouts.
- `data/features/calendar.parquet`, `prices.parquet`: known covariates.
- `reports/retail_plan.json`: selection, test metrics, origin, scope and assumptions.
- `reports/backtest.parquet`: predictions made before revealing final 28-day outcomes.
- `reports/forecast.parquet`: forecast days after the final observed date.
- `models/retail_model.pkl`: final fitted forecasting wrapper.
- `mlruns/`: local SQLite experiment records and model/report artifacts.

`make train` builds the retail plan. Optional `scripts/train.py`, `tune.py`,
`evaluate.py` and `benchmark.py` support model experiments, using verified features.
The source model package is distinct from the generated root `models/` directory.
