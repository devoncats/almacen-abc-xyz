"""Inventory performance metrics, all expressed in units."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from almacen.simulation import SimResult

METRIC_COLUMNS = [
    "fill_rate",
    "stockout_weeks",
    "stockout_units",
    "avg_inventory",
    "days_of_cover",
    "orders_placed",
]


@dataclass(frozen=True)
class Metrics:
    fill_rate: float  # sum(sold) / sum(demand)
    stockout_weeks: int  # weeks with lost > 0
    stockout_units: float  # sum(lost)
    avg_inventory: float  # mean(on_hand_end)
    days_of_cover: (
        float  # avg_inventory / (sum(demand) / (7 * n_weeks)); inf if zero demand
    )
    orders_placed: int  # secondary: replenishment workload


def _days_of_cover(avg_inventory: float, total_demand: float, n_weeks: int) -> float:
    if total_demand <= 0:
        return float("inf") if avg_inventory > 0 else 0.0
    return avg_inventory / (total_demand / (7 * n_weeks))


def item_metrics(result: SimResult) -> Metrics:
    tl = result.timeline
    demand = tl["demand"].sum()
    avg_inv = float(tl["on_hand_end"].mean())
    return Metrics(
        fill_rate=float(tl["sold"].sum() / demand) if demand > 0 else 1.0,
        stockout_weeks=int((tl["lost"] > 1e-9).sum()),
        stockout_units=float(tl["lost"].sum()),
        avg_inventory=avg_inv,
        days_of_cover=_days_of_cover(avg_inv, float(demand), len(tl)),
        orders_placed=int((tl["order_placed"] > 0).sum()),
    )


def item_table(results: dict[int, SimResult], seg: pd.DataFrame) -> pd.DataFrame:
    """One row per item with its metrics plus the raw sums used for aggregation."""
    rows = []
    for item, r in results.items():
        tl = r.timeline
        rows.append(
            {
                "item_nbr": item,
                **asdict(item_metrics(r)),
                "demand": tl["demand"].sum(),
                "sold": tl["sold"].sum(),
                "n_weeks": len(tl),
            }
        )
    return pd.DataFrame(rows).merge(
        seg[["item_nbr", "abc", "xyz", "segment"]], on="item_nbr", how="left"
    )


def aggregate(
    results: dict[int, SimResult], seg: pd.DataFrame, by: str | None = "segment"
) -> pd.DataFrame:
    """Aggregate per segment (or any seg column) or overall when by=None.

    Fill rate is demand-weighted; stockouts are summed; average inventory is summed across items
    (warehouse-level stock); days of cover uses aggregate inventory over aggregate daily demand.
    """
    items = item_table(results, seg)
    grouped = items.groupby(by) if by else [("overall", items)]
    rows = []
    for key, g in grouped:
        demand = g["demand"].sum()
        n_weeks = int(g["n_weeks"].max())
        inv = g["avg_inventory"].sum()
        row: dict[str, Any] = {by: key} if by else {"scope": key}
        row.update(
            {
                "n_items": len(g),
                "fill_rate": float(g["sold"].sum() / demand) if demand > 0 else 1.0,
                "stockout_weeks": int(g["stockout_weeks"].sum()),
                "stockout_units": float(g["stockout_units"].sum()),
                "avg_inventory": float(inv),
                "days_of_cover": _days_of_cover(inv, float(demand), n_weeks),
                "orders_placed": int(g["orders_placed"].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compare(expert: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Columns: <metric>_expert, <metric>_baseline, <metric>_delta (expert - baseline)."""
    key = [c for c in expert.columns if c not in METRIC_COLUMNS]
    merged = expert.merge(baseline, on=key, suffixes=("_expert", "_baseline"))
    ordered = list(key)
    for m in METRIC_COLUMNS:
        merged[f"{m}_delta"] = merged[f"{m}_expert"] - merged[f"{m}_baseline"]
        ordered += [f"{m}_expert", f"{m}_baseline", f"{m}_delta"]
    return merged[ordered].replace([np.inf, -np.inf], np.nan)
