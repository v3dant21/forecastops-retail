"""Reviewed corrections are overlays: source invoices and import fingerprints remain immutable."""

import json
import math
import re

import pandas as pd

from forecastops.data.retail_database import connect_database

LINE_FIELDS = {"quantity", "unit_rate", "taxable_value", "tax_amount", "item_total"}
INVOICE_FIELDS = {
    "invoice_quantity",
    "invoice_taxable",
    "invoice_tax",
    "invoice_total",
    "invoice_grand_total",
}


def resolution_history(path, latest=False):
    with connect_database(path, readonly=True) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'retail_resolutions'"
        ).fetchone()
        if not exists:
            return []
        query = "SELECT * FROM retail_resolutions"
        if latest:
            query += (
                " WHERE resolution_id IN (SELECT MAX(resolution_id) FROM retail_resolutions "
                "GROUP BY kind, target)"
            )
        rows = pd.read_sql_query(query + " ORDER BY resolution_id", connection).to_dict("records")
    for row in rows:
        row["values"] = json.loads(row["values_json"])
    return rows


def save_resolutions(path, records, reviewer, reason):
    if not reviewer.strip() or not reason.strip() or not records:
        raise ValueError("Reviewer, reason and at least one correction are required")
    with connect_database(path) as connection:
        prepared = []
        for kind, target, supplied in records:
            values = dict(supplied)
            validate_resolution(connection, kind, target, values)
            if kind in {"product", "line", "invoice"}:
                previous = connection.execute(
                    "SELECT values_json FROM retail_resolutions WHERE kind = ? AND target = ? "
                    "ORDER BY resolution_id DESC LIMIT 1",
                    (kind, target),
                ).fetchone()
                values = {**(json.loads(previous[0]) if previous else {}), **values}
            prepared.append(
                (
                    kind,
                    target,
                    json.dumps(values, allow_nan=False),
                    reviewer.strip(),
                    reason.strip(),
                )
            )
        connection.executemany(
            "INSERT INTO retail_resolutions (kind, target, values_json, reviewer, reason) "
            "VALUES (?, ?, ?, ?, ?)",
            prepared,
        )


def validate_resolution(connection, kind, target, values):
    if not isinstance(values, dict) or not values:
        raise ValueError("Correction values must be supplied")
    validators = {
        "coverage": validate_coverage,
        "units": validate_units,
        "product": validate_product,
        "line": validate_amount,
        "invoice": validate_amount,
        "duplicate": validate_amount,
    }
    if kind not in validators:
        raise ValueError("Unknown correction kind")
    validators[kind](connection, kind, target, values)


def validate_coverage(connection, kind, target, values):
    start, end = pd.Timestamp(values["start"]), pd.Timestamp(values["end"])
    if pd.isna(start) or pd.isna(end) or start > end or (end - start).days > 18250:
        raise ValueError("Coverage needs valid dates spanning at most 50 years")
    if values.get("status") not in {"no_trading", "missing"}:
        raise ValueError("Choose confirmed no trading or missing records")


def validate_units(connection, kind, target, values):
    if target != "recorded_units" or values != {"KG": "kilograms", "PCS": "pieces"}:
        raise ValueError("Confirm that recorded KG means kilograms and PCS means pieces")


def validate_product(connection, kind, target, values):
    if not connection.execute(
        "SELECT 1 FROM retail_products WHERE product_id = ?", (target,)
    ).fetchone():
        raise ValueError("Unknown source product")
    if set(values) - {
        "product_name", "unit", "hsn_code", "is_placeholder", "exclude_from_analysis"
    }:
        raise ValueError("Unsupported product correction")
    if any(
        not isinstance(v, str) or not v.strip()
        for k, v in values.items()
        if k not in {"is_placeholder", "exclude_from_analysis"}
    ):
        raise ValueError("Product values cannot be blank")
    if "is_placeholder" in values and values["is_placeholder"] is not True:
        raise ValueError("Placeholder exclusion must be explicitly confirmed")
    if "exclude_from_analysis" in values:
        rows = connection.execute(
            "SELECT l.original_product_name, l.quantity, l.unit_rate, l.taxable_paise, "
            "l.tax_paise, l.total_paise FROM retail_invoice_lines l WHERE l.product_id = ?",
            (target,),
        ).fetchall()
        if values["exclude_from_analysis"] is not True or not rows or any(
            row[0] != "267621" or row[1] != 1 or row[2] != 0
            or any(row[i] != 0 for i in (3, 4, 5)) for row in rows
        ):
            raise ValueError(
                "Only the verified zero-value 267621 row can be excluded by this review"
            )
    if "unit" in values and values["unit"] not in {"KG", "PCS", "BAG", "SET"}:
        raise ValueError("Choose KG, PCS, BAG or SET")
    if "hsn_code" in values and not re.fullmatch(r"\d{8}", values["hsn_code"]):
        raise ValueError("HSN must have eight digits")


def validate_amount(connection, kind, target, values):
    key = target.rsplit("#", 1)[0] if kind == "line" else target
    if not connection.execute(
        "SELECT 1 FROM retail_invoices WHERE invoice_key = ?", (key,)
    ).fetchone():
        raise ValueError("Unknown invoice")
    if kind == "line":
        ordinal = target.rsplit("#", 1)[1]
        if not connection.execute(
            "SELECT 1 FROM retail_invoice_lines WHERE invoice_key = ? AND line_number = ?",
            (key, ordinal),
        ).fetchone():
            raise ValueError("Unknown invoice item line")
    if kind == "duplicate":
        if values not in ({"both_valid": True}, {"keep_one": True}):
            raise ValueError("Choose whether repeated entries are valid or keep only one")
        return
    allowed = LINE_FIELDS if kind == "line" else INVOICE_FIELDS
    if set(values) - allowed or any(
        isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0
        for v in values.values()
    ):
        raise ValueError("Use finite non-negative amounts and quantities")


def apply_resolutions(items, headers, records):
    """Recalculate warnings from effective facts rather than blanket-hiding them."""
    from forecastops.data.invoices import FOOD_CHAPTERS  # noqa: PLC0415

    items = items.copy()
    headers = headers.copy()
    items["source_product_id"] = items.product_id
    items["recorded_product_name"] = items.product_name
    items["recorded_unit"] = items.unit
    items["unit_verified"] = False
    items["duplicate_verified"] = False
    items["is_placeholder"] = False
    items["explicitly_excluded"] = False
    for field in LINE_FIELDS:
        items[f"recorded_{field}"] = items[field]
    for field in INVOICE_FIELDS:
        headers[f"recorded_{field}"] = headers[field]
    for record in records:
        apply_record(items, headers, record)
    raw_count = len(items)
    items, headers = apply_duplicate_decisions(items, headers, records)
    removed_duplicates = raw_count - len(items)
    placeholder = (
        items.is_placeholder
        & items.recorded_product_name.eq("")
        & items.hsn_code.eq("")
        & items.taxable_value.eq(0)
        & items.tax_amount.eq(0)
        & items.item_total.eq(0)
    )
    explicitly_excluded = pd.Series(False, index=items.index)
    for record in records:
        if record["kind"] == "product" and record["values"].get("exclude_from_analysis"):
            explicitly_excluded |= (
                items.source_product_id.eq(record["target"])
                & items.recorded_product_name.eq("267621")
                & items.taxable_value.eq(0)
                & items.tax_amount.eq(0)
                & items.item_total.eq(0)
            )
    excluded_quantities = (
        items.loc[placeholder | explicitly_excluded].groupby("invoice_key").quantity.sum()
    )
    items["explicitly_excluded"] = explicitly_excluded
    items = items[~(placeholder | explicitly_excluded)].copy()
    # Zero-value placeholders still contribute an arithmetic quantity to raw invoice totals.
    # Adjust only that expected quantity; money amounts remain unchanged.
    headers["invoice_quantity"] -= headers.invoice_key.map(excluded_quantities).fillna(0)
    items["scope"] = items.hsn_code.map(
        lambda code: "food"
        if code.isdigit() and len(code) == 8 and code[:2] in FOOD_CHAPTERS
        else "non_food"
        if re.fullmatch(r"\d{8}", code)
        else "unclassified"
    )
    items["issues"] = [effective_issues(row) for row in items.to_dict("records")]
    items.attrs["removed_duplicate_lines"] = removed_duplicates
    items.attrs["excluded_placeholder_lines"] = int(placeholder.sum())
    items.attrs["excluded_unidentified_lines"] = int(explicitly_excluded.sum())
    return items, headers


def apply_record(items, headers, record):
    from forecastops.data.invoices import product_name  # noqa: PLC0415

    kind, target, values = record["kind"], record["target"], record["values"]
    if kind == "units":
        items.loc[items.unit.isin(["KG", "PCS"]), "unit_verified"] = True
    elif kind == "product":
        mask = items.source_product_id.eq(target)
        for field, value in values.items():
            items.loc[mask, field] = product_name(value) if field == "product_name" else value
        if "unit" in values:
            items.loc[mask, "unit_verified"] = True
    elif kind == "line":
        key, ordinal = target.rsplit("#", 1)
        mask = items.invoice_key.eq(key) & items.line_number.eq(int(ordinal))
        for field, value in values.items():
            items.loc[mask, field] = value
    elif kind == "invoice":
        for field, value in values.items():
            headers.loc[headers.invoice_key.eq(target), field] = value
    elif kind == "duplicate":
        items.loc[items.invoice_key.eq(target), "duplicate_verified"] = True


def apply_duplicate_decisions(items, headers, records):
    # Only remove repetitions of original source facts, not newly merged labels.
    duplicate_columns = [
        "invoice_key",
        "recorded_product_name",
        "recorded_unit",
        "recorded_quantity",
        "recorded_unit_rate",
        "recorded_taxable_value",
    ]
    repeated = items.duplicated(duplicate_columns, keep="first")
    dedup_invoices = {
        r["target"] for r in records if r["kind"] == "duplicate" and r["values"].get("keep_one")
    }
    items = items[~(items.invoice_key.isin(dedup_invoices) & repeated)].copy()
    for key in dedup_invoices:
        retained = items[items.invoice_key.eq(key)]
        for target, source in {
            "invoice_quantity": "quantity",
            "invoice_taxable": "taxable_value",
            "invoice_tax": "tax_amount",
            "invoice_total": "item_total",
            "invoice_grand_total": "item_total",
        }.items():
            headers.loc[headers.invoice_key.eq(key), target] = round(retained[source].sum(), 2)
    return items, headers


def effective_issues(row):
    issues = []
    if not row["product_name"]:
        issues.append("missing_product_name")
    elif row["product_name"].isdigit():
        issues.append("unidentified_product_label")
    if row["unit"] == "UNSPECIFIED":
        issues.append("missing_quantity_unit")
    # KGS is a valid spelling of kilograms, not a damaged product suffix.
    name = row["product_name"]
    if (re.search(r"(?:GM|KG)[A-Z0-9]+$", name) and not name.endswith(" KGS")) or re.search(
        r"[A-Z]\d+$", name
    ):
        issues.append("suspicious_product_suffix")
    if not re.fullmatch(r"\d{8}", row["hsn_code"]):
        issues.append("missing_or_invalid_hsn")
    if not all(math.isfinite(row[field]) for field in LINE_FIELDS):
        issues.append("invalid_numeric_value")
    if row["quantity"] <= 0:
        issues.append("non_positive_quantity")
    if abs(row["taxable_value"] + row["tax_amount"] - row["item_total"]) > 0.02:
        issues.append("item_total_mismatch")
    if abs(row["quantity"] * row["unit_rate"] - row["taxable_value"]) > 0.02:
        issues.append("quantity_rate_taxable_difference")
    return issues


def clear_verified_flags(items):
    def remaining(row):
        values = row["issues"].split("; ")
        if row["unit_verified"]:
            values = [v for v in values if v != "mixed_units_for_product"]
        if row["duplicate_verified"]:
            values = [v for v in values if v != "repeated_item_line_needs_review"]
        return "; ".join(v for v in values if v)

    items["issues"] = [remaining(row) for row in items.to_dict("records")]
    items["needs_review"] = items.issues.ne("")
    return items


def coverage_status(food, start, end, records):
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    absent = set(days) - set(food.date)
    statuses = {}
    for record in records:
        if record["kind"] == "coverage":
            values = record["values"]
            for day in pd.date_range(values["start"], values["end"], freq="D"):
                statuses[day] = values["status"]
    confirmed = {d for d in absent if statuses.get(d) == "no_trading"}
    return {
        "absent_days": len(absent),
        "confirmed_no_trading_days": len(confirmed),
        "unresolved_days": len(absent - confirmed),
    }
