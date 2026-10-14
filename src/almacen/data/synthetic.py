"""Synthetic daily sales generator that mimics the Favorita schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FAMILIES = {
    "GROCERY I": False,
    "BEVERAGES": False,
    "CLEANING": False,
    "PERSONAL CARE": False,
    "DAIRY": True,
    "PRODUCE": True,
    "BREAD/BAKERY": True,
    "MEATS": True,
    "POULTRY": True,
}
STORE_TYPES = ("A", "B", "C", "D", "E")
WEEKDAY_FACTOR = np.array([0.90, 0.90, 0.95, 1.00, 1.10, 1.25, 0.90])  # Mon..Sun

# profile -> (probability, gamma shape for item-week noise)
PROFILES = {
    "stable": (0.45, 60.0),
    "seasonal": (0.20, 25.0),
    "erratic": (0.20, 1.2),
    "intermittent": (0.15, 3.0),
}


@dataclass
class SyntheticSpec:
    n_items: int = 300
    n_stores: int = 5
    n_weeks: int = 104
    start: str = "2015-01-05"  # a Monday
    cities: tuple[str, ...] = ("Quito", "Guayaquil")
    pct_perishable: float = 0.25
    pct_new: float = 0.05
    pct_weighted: float = 0.05
    promo_rate: float = 0.04
    return_rate: float = 0.002
    seed: int = 42


def generate(spec: SyntheticSpec) -> pd.DataFrame:
    """Generate DAILY sales with RAW_COLUMNS.

    Pareto-shaped volume; stable / seasonal / erratic / intermittent demand profiles; promo uplift;
    injects returns (<0) and omits zero-sale days to exercise preprocessing.
    """
    rng = np.random.default_rng(spec.seed)
    n_i, n_s, n_d = spec.n_items, spec.n_stores, spec.n_weeks * 7
    n_w = spec.n_weeks
    dates = pd.date_range(spec.start, periods=n_d, freq="D")
    day_idx = np.arange(n_d)
    week_of_day = day_idx // 7

    # --- items --------------------------------------------------------------------------------
    item_nbr = rng.choice(np.arange(100_000, 1_000_000), size=n_i, replace=False)
    perishable_families = [f for f, p in FAMILIES.items() if p]
    stable_families = [f for f, p in FAMILIES.items() if not p]
    perishable = rng.random(n_i) < spec.pct_perishable
    family = np.where(
        perishable,
        rng.choice(perishable_families, size=n_i),
        rng.choice(stable_families, size=n_i),
    )
    item_class = rng.integers(1000, 3000, size=n_i)
    weighted = rng.random(n_i) < spec.pct_weighted

    names = list(PROFILES)
    probs = np.array([PROFILES[p][0] for p in names])
    profile = rng.choice(names, size=n_i, p=probs / probs.sum())
    shape = np.array([PROFILES[p][1] for p in profile])

    weekly_mean = rng.lognormal(mean=3.0, sigma=1.4, size=n_i)  # DC-level weekly units

    # --- stores -------------------------------------------------------------------------------
    store_nbr = np.arange(1, n_s + 1)
    store_city = np.array([spec.cities[i % len(spec.cities)] for i in range(n_s)])
    store_type = rng.choice(STORE_TYPES, size=n_s)
    store_weight = rng.dirichlet(np.full(n_s, 4.0)) * n_s

    # --- daily intensity (items x days) -------------------------------------------------------
    t = day_idx[None, :]
    amplitude = np.where(
        profile == "seasonal", rng.uniform(0.4, 0.8, n_i), rng.uniform(0.0, 0.08, n_i)
    )
    phase = rng.uniform(0, 2 * np.pi, n_i)
    season = 1 + amplitude[:, None] * np.sin(2 * np.pi * t / 365.25 + phase[:, None])
    weekday = WEEKDAY_FACTOR[dates.dayofweek.to_numpy()][None, :]
    week_noise = rng.gamma(shape[:, None], 1 / shape[:, None], size=(n_i, n_w))[
        :, week_of_day
    ]

    promo_week = rng.random((n_i, n_w)) < spec.promo_rate
    promo_day = promo_week[:, week_of_day]
    uplift = np.where(promo_day, rng.uniform(1.3, 1.8, n_i)[:, None], 1.0)

    lam_item_day = (weekly_mean[:, None] / 7) * season * weekday * week_noise * uplift

    # intermittent items sell on few days, with a larger size when they do
    occur_p = np.where(profile == "intermittent", rng.uniform(0.04, 0.2, n_i), 1.0)
    occurs = rng.random((n_i, n_d)) < occur_p[:, None]
    lam_item_day = np.where(occurs, lam_item_day / occur_p[:, None], 0.0)

    # new items: launched a few weeks before the default 12-week test window
    is_new = rng.random(n_i) < spec.pct_new
    launch_week = np.where(
        is_new, rng.integers(max(n_w - 22, 0), max(n_w - 14, 1), n_i), 0
    )
    lam_item_day = np.where(
        week_of_day[None, :] >= launch_week[:, None], lam_item_day, 0.0
    )

    # --- store split + sampling (items x stores x days) ---------------------------------------
    lam = lam_item_day[:, None, :] * (store_weight / n_s)[None, :, None]
    sales = rng.poisson(lam).astype(float)
    kg = rng.uniform(0.3, 1.5, size=sales.shape)
    sales = np.where(weighted[:, None, None], np.round(sales * kg, 3), sales)

    returns = (sales > 0) & (rng.random(sales.shape) < spec.return_rate)
    sales = np.where(
        returns, -rng.integers(1, 4, size=sales.shape).astype(float), sales
    )

    ii, ss, dd = np.nonzero(sales)  # zero-sale days are omitted, as in Favorita
    return pd.DataFrame(
        {
            "date": dates[dd],
            "store_nbr": store_nbr[ss],
            "item_nbr": item_nbr[ii],
            "unit_sales": sales[ii, ss, dd],
            "onpromotion": promo_day[ii, dd],
            "family": family[ii],
            "class": item_class[ii],
            "perishable": perishable[ii].astype(int),
            "city": store_city[ss],
            "type": store_type[ss],
        }
    )


def to_favorita_layout(daily: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Split denormalized daily sales into Favorita-style train / items / stores tables.

    Mirrors the Kaggle files: train has an `id` column and leaves `onpromotion` empty for the
    earliest dates; stores carries extra columns (state, cluster) that the loader ignores.
    """
    rng = np.random.default_rng(seed)
    train = daily[["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]].copy()
    train.insert(0, "id", np.arange(len(train)))
    train["date"] = train["date"].dt.strftime("%Y-%m-%d")
    early = daily["date"] < daily["date"].min() + pd.Timedelta(days=60)
    train["onpromotion"] = train["onpromotion"].astype(object).where(~early, None)

    items = (
        daily[["item_nbr", "family", "class", "perishable"]]
        .drop_duplicates("item_nbr")
        .sort_values("item_nbr")
        .reset_index(drop=True)
    )
    stores = (
        daily[["store_nbr", "city", "type"]]
        .drop_duplicates("store_nbr")
        .sort_values("store_nbr")
        .reset_index(drop=True)
    )
    stores.insert(
        2,
        "state",
        stores["city"]
        .map({"Quito": "Pichincha", "Guayaquil": "Guayas"})
        .fillna("Other"),
    )
    stores["cluster"] = rng.integers(1, 18, size=len(stores))
    return {"train": train, "items": items, "stores": stores}
