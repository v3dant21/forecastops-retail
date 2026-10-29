# Actual Walmart M5 data

Run `make data` to download and verify the pinned public Nixtla archive, then
prepare actual observations. `raw/manifest.json` records the source URL, pinned
archive SHA-256 and per-file checksums. `features/provenance.json` records the exact
product/store/date scope. Training and the dashboard reject unverified inputs.

The default is 100 actual products, five stores and 730 days, rather than the full
3,049-product, ten-store source. No generated sales are used. Historical forecasts
are dated relative to the archive, not to today's date.
