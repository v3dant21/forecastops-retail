"""Reviewable purchase drafts from observed invoice quantities, without model forecasts."""

import math
from decimal import ROUND_CEILING, Decimal

import pandas as pd


def sales_order_inputs(food, start, end):
    """Keep product/unit pairs intact and block every item with unresolved source issues."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end:
        raise ValueError("Start date must be on or before end date")
    selected = food[food.date.between(start, end)]
    days = (end - start).days + 1
    # Review the entire import, even if the chosen period hides a problematic line.
    blocked_ids = set(food.loc[food.needs_review | food.unit.eq("UNSPECIFIED"), "product_id"])
    mixed = food.groupby("product_name").unit.nunique()
    blocked_names = set(mixed[mixed > 1].index)
    rows = selected.groupby(["product_id", "product_name", "unit"], as_index=False).agg(
        sold_quantity=("quantity", "sum")
    )
    verified = (
        food.groupby("product_id").unit_verified.all()
        if "unit_verified" in food
        else pd.Series(dtype=bool)
    )
    rows["blocked"] = rows.product_id.isin(blocked_ids) | (
        rows.product_name.isin(blocked_names)
        & ~rows.product_id.map(verified).astype("boolean").fillna(False)
    )
    rows["daily_sales"] = rows.sold_quantity / days
    return rows, days


def purchase_quantity(
    daily_sales, on_hand, incoming, lead_days, review_days, safety_days, pack_size, minimum_order
):
    """Order up to lead + review + safety coverage; round positive gaps to supplier packs."""
    values = (
        daily_sales,
        on_hand,
        incoming,
        lead_days,
        review_days,
        safety_days,
        pack_size,
        minimum_order,
    )
    if any(not math.isfinite(float(value)) or value < 0 for value in values):
        raise ValueError("Planning values must be finite and non-negative")
    if pack_size <= 0:
        raise ValueError("Pack size must be greater than zero")
    target = Decimal(str(daily_sales)) * sum(
        (Decimal(str(days)) for days in (lead_days, review_days, safety_days)), Decimal(0)
    )
    gap = max(Decimal(0), target - Decimal(str(on_hand)) - Decimal(str(incoming)))
    if gap == 0:
        return float(target), 0.0
    pack = Decimal(str(pack_size))
    quantity = (max(gap, Decimal(str(minimum_order))) / pack).to_integral_value(
        rounding=ROUND_CEILING
    ) * pack
    return float(target), float(quantity)


def draft_lines(inputs, lead_days, review_days, safety_days):
    """Only explicitly selected, source-cleared lines can enter a draft."""
    result = []
    for row in inputs.to_dict("records"):
        if not row["include"] or row["blocked"]:
            continue
        if row["unit"] in {"PCS", "BAG"} and not float(row["pack_size"]).is_integer():
            raise ValueError("PCS and BAG supplier packs must contain whole units")
        target, quantity = purchase_quantity(
            row["daily_sales"],
            row["on_hand"],
            row["incoming"],
            lead_days,
            review_days,
            safety_days,
            row["pack_size"],
            row["minimum_order"],
        )
        if quantity <= 0:
            continue
        cost = row.get("unit_cost")
        if pd.notna(cost) and (not math.isfinite(float(cost)) or cost < 0):
            raise ValueError("Purchase cost must be finite and non-negative")
        result.append(
            {
                "product_id": row["product_id"],
                "food_item": row["product_name"],
                "unit": row["unit"],
                "sold_quantity": row["sold_quantity"],
                "daily_sales": row["daily_sales"],
                "on_hand": row["on_hand"],
                "incoming": row["incoming"],
                "target_stock": target,
                "pack_size": row["pack_size"],
                "minimum_order": row["minimum_order"],
                "order_quantity": quantity,
                "purchase_unit_cost": cost,
                "line_value_before_tax": quantity * cost if pd.notna(cost) else None,
            }
        )
    return pd.DataFrame(result)
