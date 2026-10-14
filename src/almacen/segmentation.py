"""ABC (volume share) and XYZ (coefficient of variation) classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

from almacen.config import ABCConfig, XYZConfig
from almacen.schemas import SEGMENT_COLUMNS

ABC_ORDER = ["A", "B", "C"]
XYZ_ORDER = ["X", "Y", "Z"]
SEGMENT_ORDER = [a + x for a in ABC_ORDER for x in XYZ_ORDER]


def classify_abc(weekly_train: pd.DataFrame, cfg: ABCConfig) -> pd.DataFrame:
    """Columns: item_nbr, total_units, share, cum_share, abc.

    An item belongs to the class whose cut its *preceding* cumulative share has not yet reached,
    so the item that crosses a cut stays in the higher class.
    """
    totals = (
        weekly_train.groupby("item_nbr")["units"]
        .sum()
        .rename("total_units")
        .sort_values(ascending=False, kind="mergesort")
        .reset_index()
    )
    grand_total = totals["total_units"].sum()
    if grand_total <= 0:
        raise ValueError("total demand in the training window is zero")
    totals["share"] = totals["total_units"] / grand_total
    totals["cum_share"] = totals["share"].cumsum()
    preceding = totals["cum_share"] - totals["share"]
    totals["abc"] = np.select(
        [preceding < cfg.a_cut, preceding < cfg.b_cut], ["A", "B"], default="C"
    )
    totals.loc[totals["total_units"] <= 0, "abc"] = "C"
    return totals


def classify_xyz(weekly_train: pd.DataFrame, cfg: XYZConfig) -> pd.DataFrame:
    """Columns: item_nbr, mean_w, std_w, cv, xyz. Zero mean -> Z."""
    stats = (
        weekly_train.groupby("item_nbr")["units"]
        .agg(mean_w="mean", std_w="std")
        .reset_index()
    )
    stats["std_w"] = stats["std_w"].fillna(0.0)  # single-week history
    with np.errstate(divide="ignore", invalid="ignore"):
        stats["cv"] = np.where(
            stats["mean_w"] > 0, stats["std_w"] / stats["mean_w"], np.inf
        )
    stats["xyz"] = np.select(
        [stats["cv"] <= cfg.x_max_cv, stats["cv"] <= cfg.y_max_cv],
        ["X", "Y"],
        default="Z",
    )
    return stats


def segment(
    weekly_train: pd.DataFrame, abc_cfg: ABCConfig, xyz_cfg: XYZConfig
) -> pd.DataFrame:
    """Return SEGMENT_COLUMNS sorted by descending volume."""
    seg = classify_abc(weekly_train, abc_cfg).merge(
        classify_xyz(weekly_train, xyz_cfg), on="item_nbr", how="inner"
    )
    seg["segment"] = seg["abc"] + seg["xyz"]
    return seg[SEGMENT_COLUMNS]


def segment_matrix(seg: pd.DataFrame) -> pd.DataFrame:
    """3x3 matrix in long-to-wide form: one row per ABC class, item count and volume share per XYZ."""
    grouped = seg.groupby(["abc", "xyz"]).agg(
        n_items=("item_nbr", "size"), volume_share=("share", "sum")
    )
    full = pd.MultiIndex.from_product([ABC_ORDER, XYZ_ORDER], names=["abc", "xyz"])
    grouped = grouped.reindex(full, fill_value=0)
    items = grouped["n_items"].unstack("xyz").add_prefix("items_")
    share = (
        (grouped["volume_share"] * 100)
        .round(2)
        .unstack("xyz")
        .add_prefix("volume_pct_")
    )
    matrix = items.join(share)
    matrix["items_total"] = items.sum(axis=1)
    matrix["volume_pct_total"] = share.sum(axis=1).round(2)
    return matrix.reset_index()
