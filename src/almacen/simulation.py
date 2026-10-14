"""Weekly lost-sales inventory simulation over the hold-out weeks."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from almacen.config import SimConfig
from almacen.policy import PolicyParams

TIMELINE_COLUMNS = [
    "week",
    "demand",
    "on_hand_start",
    "received",
    "sold",
    "lost",
    "on_hand_end",
    "order_placed",
    "on_order",
]


@dataclass
class SimResult:
    item_nbr: int
    timeline: pd.DataFrame  # TIMELINE_COLUMNS


def _initial_stock(params: PolicyParams, mode: str) -> float:
    if mode == "zero":
        return 0.0
    if params.review_type == "continuous_sQ":
        return float(params.reorder_point + params.order_qty)
    return float(params.order_up_to)


def _order_qty(params: PolicyParams, week_idx: int, position: float) -> float:
    if params.review_type == "continuous_sQ":
        # weekly check approximating continuous review: order n·Q so position rises above s
        if position > params.reorder_point:
            return 0.0
        n = math.floor((params.reorder_point - position) / params.order_qty) + 1
        return n * params.order_qty
    # periodic (R,S) and on-demand: review at the end of every R-th week
    if (week_idx + 1) % params.review_weeks != 0:
        return 0.0
    return max(0.0, params.order_up_to - position)


def simulate_item(
    demand: np.ndarray,
    weeks: pd.DatetimeIndex,
    params: PolicyParams,
    initial_stock: str,
) -> SimResult:
    """Lost-sales model (no backorders).

    Orders are placed at the end of a week and arrive at the start of the week L+1 periods later,
    so the protection interval is L+R weeks, matching the parameter formulas.
    """
    on_hand = _initial_stock(params, initial_stock)
    pipeline: dict[int, float] = defaultdict(float)
    rows = []
    for t, d in enumerate(demand):
        received = pipeline.pop(t, 0.0)
        start = on_hand + received
        sold = min(float(d), start)
        lost = float(d) - sold
        on_hand = start - sold
        on_order = sum(pipeline.values())
        qty = _order_qty(params, t, on_hand + on_order)
        if qty > 0:
            pipeline[t + params.lead_time_weeks + 1] += qty
            on_order += qty
        rows.append(
            (weeks[t], float(d), start, received, sold, lost, on_hand, qty, on_order)
        )
    return SimResult(
        item_nbr=params.item_nbr, timeline=pd.DataFrame(rows, columns=TIMELINE_COLUMNS)
    )


def simulate(
    test: pd.DataFrame, params: dict[int, PolicyParams], cfg: SimConfig
) -> dict[int, SimResult]:
    """Simulate every item that has parameters over the full test window."""
    weeks = pd.DatetimeIndex(np.sort(test["week"].unique()))
    panel = (
        test.pivot_table(
            index="item_nbr", columns="week", values="units", aggfunc="sum"
        )
        .reindex(columns=weeks)
        .fillna(0.0)
    )
    results: dict[int, SimResult] = {}
    for item, p in params.items():
        demand = (
            panel.loc[item].to_numpy() if item in panel.index else np.zeros(len(weeks))
        )
        results[item] = simulate_item(demand, weeks, p, cfg.initial_stock)
    return results
