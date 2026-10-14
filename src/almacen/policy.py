"""Translate an inferred policy into numeric inventory parameters (Silver, Pyke & Thomas, 2017)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

from almacen.config import BaselineConfig
from almacen.expert.engine import Policy
from almacen.expert.facts import ProductFact


@dataclass(frozen=True)
class PolicyParams:
    item_nbr: int
    review_type: str
    z: float
    review_weeks: int
    lead_time_weeks: int
    safety_stock: float
    reorder_point: float | None  # s (continuous review)
    order_up_to: float | None  # S (periodic review / on demand)
    order_qty: float | None  # Q (continuous review)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def z_from_service_level(sl: float) -> float:
    """Safety factor for a cycle service level; 0.5 -> 0."""
    if not 0 < sl < 1:
        raise ValueError(f"service level must be in (0, 1), got {sl}")
    return NormalDist().inv_cdf(sl)


def _round_units(x: float, weighted: bool) -> float:
    """Discrete items are stocked in whole units; weighted items keep 2 decimals."""
    return round(x, 2) if weighted else float(math.ceil(x - 1e-9))


def compute_params(
    fact: ProductFact, policy: Policy, lead_time_weeks: int
) -> PolicyParams:
    """SS = z·σ·√(L+R)·mult; s = μ·(L+R) + SS; S = μ·(L+R) + SS (capped by shelf life); Q = max(1, μ·lot_weeks).

    For continuous review R is the one-week check granularity of the simulation.
    """
    mu, sigma, L = fact.mean_w, fact.std_w, lead_time_weeks
    # demand is observed weekly, so continuous review is approximated by a weekly check: the
    # inventory position can undershoot s by up to one week of demand, hence protection over L+1
    R = 1 if policy.review_type == "continuous_sQ" else max(1, policy.review_weeks)
    z = max(0.0, z_from_service_level(policy.service_level))
    safety = z * sigma * math.sqrt(L + R) * policy.safety_factor_mult
    cap = mu * policy.max_cover_weeks if policy.max_cover_weeks else math.inf
    w = fact.is_weighted

    if policy.review_type == "continuous_sQ":
        s = min(mu * (L + R) + safety, cap)
        q = max(1.0, mu * policy.lot_weeks)
        if math.isfinite(cap):
            q = max(1.0, min(q, cap - s))
        return PolicyParams(
            item_nbr=fact.item_nbr,
            review_type=policy.review_type,
            z=round(z, 4),
            review_weeks=0,
            lead_time_weeks=L,
            safety_stock=round(min(safety, max(cap - mu * (L + R), 0.0)), 2),
            reorder_point=_round_units(s, w),
            order_up_to=None,
            order_qty=_round_units(q, w),
        )

    big_s = min(mu * (L + R) + safety, cap)
    return PolicyParams(
        item_nbr=fact.item_nbr,
        review_type=policy.review_type,
        z=round(z, 4),
        review_weeks=R,
        lead_time_weeks=L,
        safety_stock=round(min(safety, max(cap - mu * (L + R), 0.0)), 2),
        reorder_point=None,
        order_up_to=_round_units(big_s, w),
        order_qty=None,
    )


def baseline_params(
    fact: ProductFact, cfg: BaselineConfig, lead_time_weeks: int
) -> PolicyParams:
    """Uniform (R,S) policy applied to every item regardless of segment."""
    uniform = Policy(
        review_type="periodic_RS",
        service_level=cfg.service_level,
        review_weeks=cfg.review_weeks,
        lot_weeks=0,
    )
    return compute_params(fact, uniform, lead_time_weeks)
