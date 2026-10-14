"""Fact base: one ProductFact per segmented item."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from almacen.config import AppConfig


@dataclass(frozen=True)
class ProductFact:
    item_nbr: int
    abc: str
    xyz: str
    segment: str
    cv: float
    mean_w: float
    std_w: float
    perishable: bool
    is_weighted: bool
    age_weeks: int
    shelf_life_weeks: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_facts(
    seg: pd.DataFrame, items: pd.DataFrame, cfg: AppConfig
) -> list[ProductFact]:
    """Join segmentation with item attributes. `items.age_weeks` must be measured at the train cutoff."""
    df = seg.merge(items, on="item_nbr", how="left", validate="one_to_one")
    unmatched = df["perishable"].isna()
    if unmatched.to_numpy().any():
        missing = df.loc[unmatched, "item_nbr"].tolist()[:5]
        raise ValueError(f"segmented items without attributes: {missing}")
    return [
        ProductFact(
            item_nbr=int(r["item_nbr"]),
            abc=str(r["abc"]),
            xyz=str(r["xyz"]),
            segment=str(r["segment"]),
            cv=float(r["cv"]),
            mean_w=float(r["mean_w"]),
            std_w=float(r["std_w"]),
            perishable=bool(r["perishable"]),
            is_weighted=bool(r["is_weighted"]),
            age_weeks=int(r["age_weeks"]),
            shelf_life_weeks=cfg.default_shelf_life_weeks
            if bool(r["perishable"])
            else None,
        )
        for r in df.to_dict("records")
    ]
