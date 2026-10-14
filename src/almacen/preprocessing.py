"""Turn raw daily store sales into a weekly DC-level demand panel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from almacen.config import OutlierConfig
from almacen.schemas import ITEM_COLUMNS, WEEKLY_COLUMNS, SchemaError

MIN_TRAIN_WEEKS = 8


@dataclass
class Prepared:
    weekly: (
        pd.DataFrame
    )  # WEEKLY_COLUMNS, item x week panel zero-filled from each item's first week
    items: pd.DataFrame  # ITEM_COLUMNS
    report: dict[str, float]  # data-quality counters and percentages


def _week_start(dates: pd.Series) -> pd.Series:
    return (dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")).dt.normalize()


def age_in_weeks(first_week: pd.Series, reference_week: pd.Timestamp) -> pd.Series:
    """Whole weeks of history up to and including reference_week."""
    return ((reference_week - first_week).dt.days // 7 + 1).clip(lower=0).astype(int)


def prepare(raw: pd.DataFrame, city: str | None, new_item_weeks: int) -> Prepared:
    report: dict[str, float] = {"rows_read": len(raw)}

    df = raw
    if city is not None:
        df = df[df["city"].str.casefold() == city.casefold()]
        if df.empty:
            raise SchemaError(
                f"no rows for city '{city}'; available: {sorted(raw['city'].unique())}"
            )
    report["rows_in_scope"] = len(df)
    report["stores_in_scope"] = int(df["store_nbr"].nunique())

    returns = df["unit_sales"] < 0
    report["returns_dropped"] = int(returns.sum())
    df = df[~returns & (df["unit_sales"] > 0)].copy()

    # keep only complete Monday-Sunday weeks
    df["week"] = _week_start(df["date"])
    first_day, last_day = df["date"].min().normalize(), df["date"].max().normalize()
    complete = (df["week"] >= first_day) & (
        df["week"] + pd.Timedelta(days=6) <= last_day
    )
    report["rows_in_partial_weeks"] = int((~complete).sum())
    df = df[complete]
    if df.empty:
        raise SchemaError("no complete weeks in the data")

    weekly = df.groupby(["item_nbr", "week"], as_index=False).agg(
        units=("unit_sales", "sum"), promo_share=("onpromotion", "mean")
    )

    # zero-fill weeks without sales, starting at each item's first selling week
    all_weeks = pd.date_range(weekly["week"].min(), weekly["week"].max(), freq="7D")
    first_week = weekly.groupby("item_nbr")["week"].min()
    panel = pd.MultiIndex.from_product(
        [first_week.index, all_weeks], names=["item_nbr", "week"]
    )
    weekly = weekly.set_index(["item_nbr", "week"]).reindex(panel).reset_index()
    weekly = weekly[weekly["week"] >= weekly["item_nbr"].map(first_week)]
    report["weeks_zero_filled"] = int(weekly["units"].isna().sum())
    weekly["units"] = weekly["units"].fillna(0.0)
    weekly["promo_share"] = weekly["promo_share"].fillna(0.0).astype(float)
    weekly = (
        weekly[WEEKLY_COLUMNS].sort_values(["item_nbr", "week"]).reset_index(drop=True)
    )

    attrs = df.groupby("item_nbr").agg(
        family=("family", "first"),
        **{"class": ("class", "first")},
        perishable=("perishable", "max"),
        is_weighted=("unit_sales", lambda s: bool((~np.isclose(s, np.round(s))).any())),
    )
    items = attrs.join(first_week.rename("first_week")).reset_index()
    items["age_weeks"] = age_in_weeks(items["first_week"], all_weeks[-1])
    items = items[ITEM_COLUMNS]

    report["items"] = len(items)
    report["weeks"] = len(all_weeks)
    report["weighted_items"] = int(items["is_weighted"].sum())
    report["new_items"] = int((items["age_weeks"] < new_item_weeks).sum())
    return Prepared(weekly=weekly, items=items, report=report)


def split_train_test(
    weekly: pd.DataFrame, test_weeks: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last `test_weeks` calendar weeks for simulation."""
    weeks = np.sort(weekly["week"].unique())
    if len(weeks) < test_weeks + MIN_TRAIN_WEEKS:
        raise SchemaError(
            f"need at least {test_weeks + MIN_TRAIN_WEEKS} weeks of data, found {len(weeks)}"
        )
    cutoff = weeks[-test_weeks]
    train = weekly[weekly["week"] < cutoff].reset_index(drop=True)
    test = weekly[weekly["week"] >= cutoff].reset_index(drop=True)
    return train, test


def cap_outliers(
    weekly_train: pd.DataFrame, cfg: OutlierConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap each item's weekly training demand from above.

    Returns the capped panel and a log with one row per capped week
    (item_nbr, week, units_original, units_capped, cap).
    """
    log_columns = ["item_nbr", "week", "units_original", "units_capped", "cap"]
    if cfg.method == "none":
        return weekly_train, pd.DataFrame(columns=log_columns)

    units = weekly_train.groupby("item_nbr")["units"]
    if cfg.method == "percentile":
        cap = units.transform(lambda s: s.quantile(cfg.percentile))
    else:  # mad
        median = units.transform("median")
        mad = (
            (weekly_train["units"] - median)
            .abs()
            .groupby(weekly_train["item_nbr"])
            .transform("median")
        )
        # MAD = 0 (mostly-constant or intermittent items) would cap everything to the median: skip them
        cap = (median + cfg.mad_k * 1.4826 * mad).where(mad > 0, np.inf)

    over = weekly_train["units"] > cap
    log = weekly_train.loc[over, ["item_nbr", "week", "units"]].rename(
        columns={"units": "units_original"}
    )
    log["units_capped"] = cap[over].to_numpy()
    log["cap"] = log["units_capped"]
    capped = weekly_train.copy()
    capped["units"] = np.minimum(capped["units"], cap)
    return capped, log[log_columns].reset_index(drop=True)
