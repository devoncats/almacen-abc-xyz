"""Decision report for the warehouse manager: adopt the expert policies, where, and what to watch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from almacen.config import AppConfig, DecisionConfig

Decision = Literal["adopt", "adopt_selectively", "review", "keep_baseline"]
DECISION_LABELS: dict[Decision, str] = {
    "adopt": "ADOPT",
    "adopt_selectively": "ADOPT SELECTIVELY",
    "review": "REVIEW",
    "keep_baseline": "KEEP BASELINE",
}
POLICY_LABELS = {
    "continuous_sQ": "continuous review (s, Q)",
    "periodic_RS": "periodic review (R, S)",
    "on_demand": "order on demand",
}


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: str


def _pct_change(new: float, old: float) -> float:
    if old:
        return (new - old) / old * 100
    return 0.0 if not new else math.inf


def judge(fill_delta_pp: float, inv_delta_pct: float, cfg: DecisionConfig) -> Verdict:
    """Service first, then stock.

    Adopt when service is kept and stock falls, or when service improves for a bounded amount of
    extra stock; a clear trade-off in either direction goes to review.
    """
    tol, cap = cfg.fill_rate_tolerance_pp, cfg.max_inventory_increase_pct
    service_kept = fill_delta_pp >= -tol
    if service_kept and inv_delta_pct <= 0:
        return Verdict(
            "adopt",
            (
                f"same or better service ({fill_delta_pp:+.2f} pp fill rate) with "
                f"{-inv_delta_pct:.1f}% less inventory"
            ),
        )
    if fill_delta_pp > 0 and inv_delta_pct <= cap:
        return Verdict(
            "adopt",
            (
                f"{fill_delta_pp:+.2f} pp fill rate for {inv_delta_pct:+.1f}% inventory, "
                f"within the {cap:.0f}% limit"
            ),
        )
    if fill_delta_pp > tol:
        return Verdict(
            "review",
            (
                f"{fill_delta_pp:+.2f} pp fill rate but {inv_delta_pct:+.1f}% inventory, "
                f"above the {cap:.0f}% limit: decide whether the service gain is worth the stock"
            ),
        )
    if inv_delta_pct < 0:
        return Verdict(
            "review",
            (
                f"{-inv_delta_pct:.1f}% less inventory at the cost of {fill_delta_pp:+.2f} pp "
                "fill rate: acceptable only if this segment's service target allows it"
            ),
        )
    return Verdict(
        "keep_baseline",
        (
            f"{fill_delta_pp:+.2f} pp fill rate and {inv_delta_pct:+.1f}% inventory: "
            "the uniform policy performs better"
        ),
    )


def segment_decisions(
    by_segment: pd.DataFrame, seg: pd.DataFrame, cfg: DecisionConfig
) -> pd.DataFrame:
    """One row per segment with the simulated deltas and its verdict."""
    volume = seg.groupby("segment")["share"].sum() * 100
    rows = []
    for r in by_segment.to_dict("records"):
        fill_pp = (r["fill_rate_expert"] - r["fill_rate_baseline"]) * 100
        inv_pct = _pct_change(r["avg_inventory_expert"], r["avg_inventory_baseline"])
        v = judge(fill_pp, inv_pct, cfg)
        rows.append(
            {
                "segment": r["segment"],
                "n_items": int(r["n_items"]),
                "volume_pct": float(volume.get(r["segment"], 0.0)),
                "fill_rate_expert": r["fill_rate_expert"],
                "fill_rate_baseline": r["fill_rate_baseline"],
                "fill_rate_delta_pp": fill_pp,
                "avg_inventory_delta_pct": inv_pct,
                "lost_units_delta": r["stockout_units_expert"]
                - r["stockout_units_baseline"],
                "decision": v.decision,
                "reason": v.reason,
            }
        )
    return pd.DataFrame(rows)


def overall_verdict(summary: pd.DataFrame, cfg: DecisionConfig) -> Verdict:
    m = summary.set_index("metric")

    def value(metric: str, policy: str) -> float:
        return float(m[policy].loc[metric])

    fill, inv = "Service level (fill rate)", "Average inventory (units)"
    return judge(
        (value(fill, "expert_system") - value(fill, "uniform_baseline")) * 100,
        _pct_change(value(inv, "expert_system"), value(inv, "uniform_baseline")),
        cfg,
    )


def final_verdict(overall: Verdict, decisions: pd.DataFrame) -> Verdict:
    """Combine the warehouse-level verdict with the per-segment ones; `reason` is the action."""
    adopt = decisions.loc[decisions["decision"] == "adopt", "segment"].tolist()
    review = decisions.loc[decisions["decision"] == "review", "segment"].tolist()
    if overall.decision == "adopt" and len(adopt) == len(decisions):
        return Verdict("adopt", "Adopt the expert-system policies for every item.")
    if adopt:
        action = (
            f"Adopt the expert-system policies in {', '.join(adopt)} and keep the uniform "
            "weekly policy in the other segments"
            + (
                f"; discuss {', '.join(review)} before switching them."
                if review
                else "."
            )
        )
        return Verdict("adopt_selectively", action)
    if review:
        return Verdict(
            "review",
            (
                f"Keep the uniform weekly policy for now and discuss {', '.join(review)}, "
                "where the expert policies trade service against inventory."
            ),
        )
    return Verdict(
        "keep_baseline",
        "Keep the uniform weekly policy; no segment improves enough to switch.",
    )


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None._"
    cells = df.astype(object).where(df.notna(), None)  # tabulate blanks only None
    return cells.to_markdown(index=False, floatfmt=".2f", missingval="-")


def _summary_for_report(summary: pd.DataFrame) -> pd.DataFrame:
    """Fill rate in percent so that small differences stay visible at two decimals."""
    out = summary.copy()
    fill = out["metric"] == "Service level (fill rate)"
    for col in ("expert_system", "uniform_baseline", "delta"):
        out.loc[fill, col] = out.loc[fill, col] * 100
    out.loc[fill, "metric"] = "Service level (fill rate, %)"
    return out.rename(columns={"delta": "delta (pp for fill rate)"})


def _seg_list(decisions: pd.DataFrame, decision: Decision) -> str:
    sub = decisions[decisions["decision"] == decision]
    if sub.empty:
        return "none"
    return (
        ", ".join(sub["segment"])
        + f" ({int(sub['n_items'].sum())} items, {sub['volume_pct'].sum():.1f}% of volume)"
    )


def _param_columns(recs: pd.DataFrame) -> pd.DataFrame:
    out = recs.assign(policy=recs["review_type"].map(POLICY_LABELS))
    out["service_level"] = out["service_level"] * 100
    cols = {
        "item_nbr": "item",
        "segment": "segment",
        "confidence": "confidence",
        "mean_w": "weekly demand",
        "policy": "policy",
        "service_level": "target SL %",
        "review_weeks": "R (weeks)",
        "reorder_point": "s",
        "order_qty": "Q",
        "order_up_to": "S",
        "safety_stock": "safety stock",
    }
    return out[list(cols)].rename(columns=cols)


def _attention(recs: pd.DataFrame, flag: str, top: int) -> tuple[int, pd.DataFrame]:
    sub = recs[recs["flags"].str.contains(flag, regex=False)]
    return len(sub), _param_columns(
        sub.sort_values("total_units", ascending=False).head(top)
    )


def build_report(
    cfg: AppConfig,
    *,
    run_id: str,
    segmenter_label: str,
    periods: dict[str, list[str]],
    summary: pd.DataFrame,
    decisions: pd.DataFrame,
    policy_by_segment: pd.DataFrame,
    recs: pd.DataFrame,
    model_report: dict[str, Any],
    model_tables: dict[str, pd.DataFrame],
    charts: list[str],
) -> tuple[str, Verdict]:
    dc = cfg.decision
    overall = overall_verdict(summary, dc)
    verdict = final_verdict(overall, decisions)

    lines = [
        "# Replenishment recommendation",
        "",
        f"- Run: `{run_id}`",
        f"- Segmentation: **{segmenter_label}**",
        f"- Scope: {cfg.dataset.city or 'all cities'}, {len(recs)} items",
        (
            f"- Training: {periods['train'][0]} to {periods['train'][1]}; "
            f"simulated test window: {periods['test'][0]} to {periods['test'][1]} "
            f"({cfg.sim.test_weeks} weeks, lead time {cfg.sim.lead_time_weeks} week(s))"
        ),
        "",
        "## Decision",
        "",
        f"**{DECISION_LABELS[verdict.decision]}.** {verdict.reason}",
        "",
        f"Warehouse-level effect of the expert policies: {overall.reason}.",
        "",
        f"- Adopt in: {_seg_list(decisions, 'adopt')}",
        f"- Review with the category manager: {_seg_list(decisions, 'review')}",
        f"- Keep uniform policy in: {_seg_list(decisions, 'keep_baseline')}",
        "",
        (
            f"Criteria (`decision` in config.yaml): adopt when the fill rate drops at most "
            f"{dc.fill_rate_tolerance_pp:.1f} pp while inventory does not grow, or when the fill "
            f"rate improves with at most {dc.max_inventory_increase_pct:.0f}% more average "
            "inventory; larger trade-offs go to review."
        ),
        "",
        "## Key figures (test window, expert system vs. uniform baseline)",
        "",
        _md(_summary_for_report(summary)),
        "",
        "## Decision by segment",
        "",
        _md(
            decisions.assign(
                decision=decisions["decision"].map(DECISION_LABELS),
                fill_rate_expert=decisions["fill_rate_expert"] * 100,
                fill_rate_baseline=decisions["fill_rate_baseline"] * 100,
            )
            .drop(columns="reason")
            .rename(
                columns={
                    "fill_rate_expert": "fill % expert",
                    "fill_rate_baseline": "fill % baseline",
                    "fill_rate_delta_pp": "fill delta pp",
                    "avg_inventory_delta_pct": "inventory delta %",
                    "lost_units_delta": "lost units delta",
                    "volume_pct": "volume %",
                }
            )
        ),
        "",
        *[
            f"- **{r['segment']}** ({DECISION_LABELS[r['decision']]}): {r['reason']}."
            for r in decisions.to_dict("records")
        ],
        "",
        "## Policy by segment",
        "",
        _md(policy_by_segment),
        "",
        "## Items that need attention",
        "",
    ]
    for flag, title, advice in (
        (
            "manual_validation",
            "Manual validation",
            (
                "erratic high-volume demand or an uncertain segment; check the order quantities "
                "before releasing them"
            ),
        ),
        (
            "low_confidence",
            "Uncertain segment (model probability < 0.5)",
            "the classifier could not decide the segment; included in manual validation",
        ),
        (
            "discontinue_candidate",
            "Discontinuation candidates",
            "low and erratic demand; evaluate delisting or make-to-order",
        ),
        (
            "provisional",
            "New items (provisional policy)",
            "less than a quarter of history; revisit when 13 weeks are available",
        ),
    ):
        n, table = _attention(recs, flag, dc.top_items)
        if (
            flag == "low_confidence"
            and n == 0
            and "mean_confidence" not in model_report
        ):
            continue  # deterministic segmenter: never uncertain
        lines += [
            f"### {title}: {n} items",
            "",
            (
                f"_{advice[0].upper() + advice[1:]}. Top {min(n, dc.top_items)} by volume; "
                "full list in `tables/recommendations.csv`._"
            ),
            "",
            _md(table),
            "",
        ]
    top = _param_columns(
        recs.sort_values("total_units", ascending=False).head(dc.top_items)
    )
    lines += [
        f"## Replenishment parameters for the top {len(top)} items by volume",
        "",
        (
            "_s = reorder point, Q = order quantity (continuous review); S = order-up-to level "
            "(periodic review). Quantities in units at the distribution center._"
        ),
        "",
        _md(top),
        "",
    ]
    if model_tables:
        metrics = model_tables["model_metrics"]
        importance = model_tables["feature_importance"].head(8)
        lines += [
            f"## Segmentation model ({segmenter_label})",
            "",
            (
                f"Validated on the last training origin ({model_report.get('validation_origin')}, "
                f"{model_report.get('validation_samples')} items), predicting the segment of the "
                f"following {cfg.sim.test_weeks} weeks. Mean confidence of the assigned segments: "
                f"{model_report.get('mean_confidence', 0):.2f}."
            ),
            "",
            _md(metrics),
            "",
            "Most informative features (gain):",
            "",
            _md(importance[["feature", "gain_pct"]]),
            "",
        ]
    lines += ["## Charts", ""]
    lines += [f"![{Path(c).stem}](charts/{c})" for c in charts]
    lines += [
        "",
        "## Notes",
        "",
        (
            "- Results come from a lost-sales simulation over the held-out weeks; "
            "they are an estimate, not a guarantee."
        ),
        (
            "- Policy parameters use historical weekly demand; recompute them each period with "
            "`almacen run`."
        ),
        "- `almacen explain --run <run dir> --item <item>` shows the rules behind any item's policy.",
        "",
    ]
    return "\n".join(lines), verdict
