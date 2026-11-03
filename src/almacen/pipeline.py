"""End-to-end orchestration: load -> segment -> infer -> parametrize -> simulate -> report."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd

from almacen import recommendation, reporting
from almacen.config import AppConfig
from almacen.data.loader import load_dataset
from almacen.expert.engine import Inference, InferenceEngine
from almacen.expert.facts import build_facts
from almacen.expert.knowledge import load_rules
from almacen.metrics import METRIC_COLUMNS, aggregate, compare
from almacen.policy import baseline_params, compute_params
from almacen.preprocessing import age_in_weeks, cap_outliers, prepare, split_train_test
from almacen.schemas import SchemaError
from almacen.segmentation import SEGMENT_ORDER, segment_matrix
from almacen.segmenters import get_segmenter
from almacen.simulation import simulate

TIMELINE_EXAMPLES = ("AX", "AZ", "CZ")


@dataclass
class RunResult:
    run_id: str
    out_dir: Path
    segments: pd.DataFrame
    inferences: list[Inference]
    comparison: pd.DataFrame
    summary: pd.DataFrame
    segmenter: str
    verdict: recommendation.Verdict


def _recommendations(
    seg: pd.DataFrame, inferences: list[Inference], params: dict
) -> pd.DataFrame:
    rows = []
    for inf in inferences:
        p = params[inf.item_nbr]
        rows.append(
            {
                "item_nbr": inf.item_nbr,
                "review_type": inf.policy.review_type,
                "service_level": inf.policy.service_level,
                "review_weeks": p.review_weeks,
                "safety_stock": p.safety_stock,
                "reorder_point": p.reorder_point,
                "order_up_to": p.order_up_to,
                "order_qty": p.order_qty,
                "flags": ";".join(inf.flags),
                "fired_rules": ">".join(f.rule_id for f in inf.trace),
            }
        )
    cols = ["item_nbr", "segment", "confidence", "total_units", "mean_w", "std_w", "cv"]
    return seg[cols].merge(pd.DataFrame(rows), on="item_nbr")


def _policy_by_segment(recs: pd.DataFrame) -> pd.DataFrame:
    flag_counts = {
        f"n_{flag}": ("flags", lambda s, f=flag: int(s.str.contains(f).sum()))
        for flag in (
            "provisional",
            "manual_validation",
            "discontinue_candidate",
            "weighted_item",
            "low_confidence",
        )
    }
    out = recs.groupby("segment").agg(
        n_items=("item_nbr", "size"),
        mean_confidence=("confidence", "mean"),
        review_type=("review_type", lambda s: s.mode().iat[0]),
        service_level=("service_level", "mean"),
        review_weeks=("review_weeks", "mean"),
        mean_weekly_demand=("mean_w", "mean"),
        mean_safety_stock=("safety_stock", "mean"),
        **flag_counts,
    )
    return out.reindex([s for s in SEGMENT_ORDER if s in out.index]).reset_index()


def _summary_table(overall_cmp: pd.DataFrame) -> pd.DataFrame:
    """The single metrics table for the report: one row per metric."""
    labels = {
        "fill_rate": "Service level (fill rate)",
        "stockout_weeks": "Stockouts (item-weeks)",
        "stockout_units": "Stockouts (lost units)",
        "avg_inventory": "Average inventory (units)",
        "days_of_cover": "Days of cover",
        "orders_placed": "Orders placed (secondary)",
    }
    row = overall_cmp.iloc[0]
    return pd.DataFrame(
        [
            {
                "metric": labels[m],
                "expert_system": row[f"{m}_expert"],
                "uniform_baseline": row[f"{m}_baseline"],
                "delta": row[f"{m}_delta"],
                "delta_pct": (row[f"{m}_delta"] / row[f"{m}_baseline"] * 100)
                if row[f"{m}_baseline"]
                else None,
            }
            for m in METRIC_COLUMNS
        ]
    )


def run(cfg: AppConfig) -> RunResult:
    t0 = time.perf_counter()
    run_id = datetime.now(UTC).astimezone().strftime("%Y%m%d-%H%M%S")
    out_dir = cfg.output_dir / run_id
    tables_dir, charts_dir = out_dir / "tables", out_dir / "charts"
    segmenter = get_segmenter(cfg)  # fail on an unknown method before loading data

    # 1-2. load and preprocess
    ds = cfg.dataset
    missing = [
        p.as_posix()
        for p in (ds.sales_path, *(j.path for j in ds.joins))
        if not p.exists()
    ]
    if missing:
        raise SchemaError(
            f"missing dataset files: {missing}. dataset.dir must contain the Favorita layout "
            f"({ds.sales_file}, {ds.items_file}, {ds.stores_file}); "
            "use `almacen generate` for synthetic data or --data-dir to point elsewhere"
        )
    loaded = load_dataset(
        ds.sales_path,
        ds.columns,
        joins=ds.joins,
        city=ds.city,
        date_from=ds.date_from,
        chunk_rows=ds.chunk_rows,
    )
    prep = prepare(loaded.sales, ds.city, cfg.new_item_weeks)
    charts_dir.mkdir(parents=True, exist_ok=True)  # only once the data loaded
    prep.report = {**loaded.report, **prep.report}

    # 3. temporal split; everything below is fitted on train only
    train, test = split_train_test(prep.weekly, cfg.sim.test_weeks)
    train_end = train["week"].max()
    raw_train_units = float(train["units"].sum())
    train, outlier_log = cap_outliers(train, cfg.outliers)  # training history only
    prep.report["outlier_weeks_capped"] = len(outlier_log)
    prep.report["outlier_items_capped"] = int(outlier_log["item_nbr"].nunique())
    prep.report["outlier_units_removed_pct"] = (
        round(
            100 * (raw_train_units - float(train["units"].sum())) / raw_train_units, 3
        )
        if raw_train_units
        else 0.0
    )
    items = prep.items.copy()
    items["age_weeks"] = age_in_weeks(items["first_week"], train_end)
    prep.report["new_items"] = int(
        (items["age_weeks"] < cfg.new_item_weeks).sum()
    )  # at train cutoff

    # 4. segmentation (classic ABC/XYZ or a learned model, same nine segments)
    segmentation = segmenter.fit_predict(train, items)
    seg = segmentation.seg
    prep.report["items_without_training_history"] = int(len(items) - len(seg))
    prep.report["train_weeks"] = int(train["week"].nunique())
    prep.report["test_weeks"] = int(test["week"].nunique())

    # 5. expert system
    facts = build_facts(seg, items, cfg)
    engine = InferenceEngine(load_rules(cfg.rules_path))
    inferences = engine.infer_all(facts)

    # 6. policy parameters
    L = cfg.sim.lead_time_weeks
    fact_by_item = {f.item_nbr: f for f in facts}
    expert_params = {
        i.item_nbr: compute_params(fact_by_item[i.item_nbr], i.policy, L)
        for i in inferences
    }
    base_params = {f.item_nbr: baseline_params(f, cfg.baseline, L) for f in facts}

    # 7. simulation
    expert_sim = simulate(test, expert_params, cfg.sim)
    base_sim = simulate(test, base_params, cfg.sim)

    # 8. metrics
    by_segment = compare(
        aggregate(expert_sim, seg, "segment"), aggregate(base_sim, seg, "segment")
    )
    by_segment = (
        by_segment.set_index("segment")
        .reindex([s for s in SEGMENT_ORDER if s in set(by_segment["segment"])])
        .reset_index()
    )
    by_abc = compare(aggregate(expert_sim, seg, "abc"), aggregate(base_sim, seg, "abc"))
    overall = compare(aggregate(expert_sim, seg, None), aggregate(base_sim, seg, None))
    summary = _summary_table(overall)

    # tables
    matrix = segment_matrix(seg)
    recs = _recommendations(seg, inferences, expert_params)
    reporting.write_table(seg, tables_dir, "segmentation")
    if not outlier_log.empty:
        reporting.write_table(
            outlier_log.sort_values("units_original", ascending=False),
            tables_dir,
            "outliers_capped",
        )
    reporting.write_table(matrix, tables_dir, "abc_xyz_matrix")
    reporting.write_table(_policy_by_segment(recs), tables_dir, "policy_by_segment")
    reporting.write_table(recs, tables_dir, "recommendations")
    reporting.write_table(
        recs.groupby("segment", group_keys=False)
        .head(2)
        .sort_values("total_units", ascending=False),
        tables_dir,
        "recommendations_sample",
    )
    reporting.write_table(summary, tables_dir, "metrics_summary")
    reporting.write_table(by_segment, tables_dir, "metrics_by_segment")
    reporting.write_table(by_abc, tables_dir, "metrics_by_abc")
    for name, table in segmentation.tables.items():
        reporting.write_table(table, tables_dir, name)
    decisions = recommendation.segment_decisions(by_segment, seg, cfg.decision)
    reporting.write_table(decisions, tables_dir, "decision_by_segment")

    # charts
    by = segmenter.label
    reporting.plot_pareto(
        seg, charts_dir / "abc_pareto.png", cfg.abc.a_cut, cfg.abc.b_cut, by
    )
    reporting.plot_matrix_heatmap(matrix, charts_dir / "abc_xyz_matrix.png", by)
    reporting.plot_cv_vs_volume(
        seg, charts_dir / "cv_vs_volume.png", cfg.xyz.x_max_cv, cfg.xyz.y_max_cv, by
    )
    segmenter.plot_diagnostics(segmentation, charts_dir)
    reporting.plot_metrics_by_segment(by_segment, charts_dir / "metrics_by_segment.png")
    examples = {}
    for label in TIMELINE_EXAMPLES:
        candidates = seg[seg["segment"] == label]
        if not candidates.empty:
            examples[label] = int(
                candidates.iloc[0]["item_nbr"]
            )  # highest volume in segment
    reporting.plot_inventory_timeline(
        {lbl: expert_sim[i] for lbl, i in examples.items()},
        {lbl: expert_params[i] for lbl, i in examples.items()},
        charts_dir / "inventory_timeline.png",
    )

    # warehouse-manager report
    periods = {
        "train": [str(train["week"].min().date()), str(train_end.date())],
        "test": [str(test["week"].min().date()), str(test["week"].max().date())],
    }
    report_md, verdict = recommendation.build_report(
        cfg,
        run_id=run_id,
        segmenter_label=segmenter.label,
        periods=periods,
        summary=summary,
        decisions=decisions,
        policy_by_segment=_policy_by_segment(recs),
        recs=recs,
        model_report=segmentation.report,
        model_tables=segmentation.tables,
        charts=sorted(p.name for p in charts_dir.glob("*.png")),
    )
    (out_dir / "recommendation.md").write_text(report_md, encoding="utf-8")

    # explanations + run summary
    reporting.write_explanations(inferences, out_dir / "explanations.jsonl")
    reporting.write_run_summary(
        cfg,
        cast(dict[str, int], prep.report),
        overall,
        out_dir / "run.json",
        extra={
            "run_id": run_id,
            "segmenter": segmenter.name,
            "segmenter_report": segmentation.report,
            "decision": {"decision": verdict.decision, "action": verdict.reason},
            "train_period": periods["train"],
            "test_period": periods["test"],
            "rules": [asdict(r) for r in engine.rules],
            "timeline_examples": examples,
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
        },
    )
    return RunResult(
        run_id=run_id,
        out_dir=out_dir,
        segments=seg,
        inferences=inferences,
        comparison=by_segment,
        summary=summary,
        segmenter=segmenter.label,
        verdict=verdict,
    )
