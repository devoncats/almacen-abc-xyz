from pathlib import Path

import pandas as pd
import pytest

from almacen.expert.facts import ProductFact

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rules_path() -> Path:
    return ROOT / "config" / "rules.yaml"


def make_fact(**overrides) -> ProductFact:
    base = {
        "item_nbr": 1,
        "abc": "A",
        "xyz": "X",
        "segment": "AX",
        "cv": 0.2,
        "mean_w": 100.0,
        "std_w": 20.0,
        "perishable": False,
        "is_weighted": False,
        "age_weeks": 80,
        "shelf_life_weeks": None,
    }
    base.update(overrides)
    return ProductFact(**base)


def weekly_frame(series: dict[int, list[float]]) -> pd.DataFrame:
    n = len(next(iter(series.values())))
    weeks = pd.date_range("2016-01-04", periods=n, freq="7D")
    return pd.DataFrame(
        [
            {"item_nbr": item, "week": w, "units": float(u), "promo_share": 0.0}
            for item, values in series.items()
            for w, u in zip(weeks, values)
        ]
    )
