"""Table exports (CSV/Markdown/JSON) and PNG charts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from almacen.config import AppConfig
from almacen.expert.engine import Inference
from almacen.policy import PolicyParams
from almacen.segmentation import SEGMENT_ORDER
from almacen.simulation import SimResult

# categorical slots 1-3 of the reference palette (validated all-pairs for <= 3 series)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ABC_COLORS = {"A": BLUE, "B": ORANGE, "C": AQUA}
SCENARIO_COLORS = {"expert": BLUE, "baseline": ORANGE}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
    }
)


def _json_default(o: Any) -> Any:
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(o).date())
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serializable: {type(o)}")


def write_table(
    df: pd.DataFrame, out_dir: Path, name: str, float_digits: int = 4
) -> None:
    """Write name.csv, name.md and name.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rounded = df.round(float_digits)
    rounded.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.md").write_text(
        rounded.to_markdown(index=False) + "\n", encoding="utf-8"
    )
    records = json.loads(rounded.to_json(orient="records", date_format="iso"))
    (out_dir / f"{name}.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


def plot_pareto(
    seg: pd.DataFrame, path: Path, a_cut: float = 0.80, b_cut: float = 0.95
) -> None:
    s = seg.sort_values("total_units", ascending=False).reset_index(drop=True)
    x = (np.arange(1, len(s) + 1) / len(s)) * 100
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for cls in ("A", "B", "C"):
        m = (s["abc"] == cls).to_numpy()
        if m.any():
            ax.fill_between(
                x,
                0,
                s["cum_share"] * 100,
                where=m,
                color=ABC_COLORS[cls],
                alpha=0.18,
                step="pre",
                linewidth=0,
            )
    ax.plot(x, s["cum_share"] * 100, color=TEXT, linewidth=2)
    for cut in (a_cut, b_cut):
        ax.axhline(cut * 100, color=MUTED, linewidth=0.8, linestyle="--")
        ax.text(101, cut * 100, f"{cut:.0%}", va="center", color=MUTED, fontsize=8)
    for cls in ("A", "B", "C"):
        sub = x[(s["abc"] == cls).to_numpy()]
        if len(sub):
            ax.text(
                sub.mean(),
                8,
                f"{cls}\n{len(sub)} items",
                ha="center",
                color=TEXT,
                fontsize=8,
            )
    ax.set(
        xlim=(0, 100),
        ylim=(0, 102),
        xlabel="% of items (sorted by volume)",
        ylabel="Cumulative % of units",
        title="ABC Pareto curve (training window)",
    )
    fig.savefig(path)
    plt.close(fig)


def plot_matrix_heatmap(matrix: pd.DataFrame, path: Path) -> None:
    items = matrix.set_index("abc")[["items_X", "items_Y", "items_Z"]].to_numpy()
    vol = matrix.set_index("abc")[
        ["volume_pct_X", "volume_pct_Y", "volume_pct_Z"]
    ].to_numpy()
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    im = ax.imshow(vol, cmap="Blues", vmin=0, vmax=max(vol.max(), 1))
    ax.grid(False)
    ax.set_xticks(range(3), ["X", "Y", "Z"])
    ax.set_yticks(range(3), ["A", "B", "C"])
    for i in range(3):
        for j in range(3):
            dark = vol[i, j] > vol.max() * 0.55
            ax.text(
                j,
                i,
                f"{int(items[i, j])} items\n{vol[i, j]:.1f}% vol",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if dark else TEXT,
            )
    ax.set(
        xlabel="XYZ (demand variability)", ylabel="ABC (volume)", title="ABC/XYZ matrix"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of volume")
    fig.savefig(path)
    plt.close(fig)


def plot_cv_vs_volume(
    seg: pd.DataFrame, path: Path, x_max_cv: float = 0.5, y_max_cv: float = 1.0
) -> None:
    s = seg[np.isfinite(seg["cv"]) & (seg["total_units"] > 0)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for cls in ("A", "B", "C"):
        g = s[s["abc"] == cls]
        ax.scatter(
            g["total_units"],
            g["cv"],
            s=14,
            color=ABC_COLORS[cls],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
            label=f"Class {cls}",
        )
    for cut, lbl in ((x_max_cv, "X|Y"), (y_max_cv, "Y|Z")):
        ax.axhline(cut, color=MUTED, linewidth=0.8, linestyle="--")
        ax.text(
            s["total_units"].max(),
            cut,
            f" {lbl}",
            va="bottom",
            ha="right",
            color=MUTED,
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set(
        xlabel="Total units in training window (log scale)",
        ylabel="Coefficient of variation (weekly)",
        title="Demand variability vs. volume",
    )
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def plot_metrics_by_segment(comparison: pd.DataFrame, path: Path) -> None:
    c = comparison.set_index("segment").reindex(
        [s for s in SEGMENT_ORDER if s in set(comparison["segment"])]
    )
    panels = [
        ("fill_rate", "Fill rate (%)", 100),
        ("stockout_units", "Lost units (stockouts)", 1),
        ("avg_inventory", "Average inventory (units)", 1),
        ("days_of_cover", "Days of cover", 1),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6))
    x = np.arange(len(c))
    width = 0.38
    for ax, (metric, label, mult) in zip(axes.flat, panels):
        for k, scen in enumerate(("expert", "baseline")):
            ax.bar(
                x + (k - 0.5) * width,
                c[f"{metric}_{scen}"] * mult,
                width=width - 0.03,
                color=SCENARIO_COLORS[scen],
                label="Expert system" if scen == "expert" else "Uniform baseline",
            )
        ax.set_xticks(x, c.index)
        ax.set_title(label)
        ax.grid(axis="x", visible=False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(
        "Simulated performance by segment (test window)", fontweight="bold", color=TEXT
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.955),
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def plot_inventory_timeline(
    results: dict[str, SimResult], params: dict[str, PolicyParams], path: Path
) -> None:
    """results keyed by example segment, e.g. {"AX": ..., "AZ": ..., "CZ": ...}."""
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 2.3 * n), sharex=False, squeeze=False)
    for ax, (label, r) in zip(axes[:, 0], results.items()):
        tl, p = r.timeline, params[label]
        x = np.arange(len(tl))
        ax.bar(x, tl["demand"], color=GRID, width=0.7, label="Demand")
        ax.step(
            x,
            tl["on_hand_end"],
            where="mid",
            color=BLUE,
            linewidth=2,
            label="On hand (end of week)",
        )
        lost = tl["lost"] > 1e-9
        ax.scatter(
            x[lost],
            np.zeros(lost.sum()),
            marker="v",
            s=40,
            color=ORANGE,
            zorder=3,
            label="Stockout",
        )
        level = p.reorder_point if p.review_type == "continuous_sQ" else p.order_up_to
        ax.axhline(
            level,
            color=MUTED,
            linestyle="--",
            linewidth=0.8,
            label="Policy level (s or S)",
        )
        ax.text(
            len(tl) - 0.5,
            level,
            " s" if p.review_type == "continuous_sQ" else " S",
            va="center",
            color=MUTED,
            fontsize=8,
        )
        ax.set_title(
            f"{label} - item {r.item_nbr} ({p.review_type}, R={p.review_weeks}, L={p.lead_time_weeks})"
        )
        ax.set_xticks(
            x, [d.strftime("%m-%d") for d in pd.to_datetime(tl["week"])], fontsize=7
        )
        ax.set_ylabel("Units")
    axes[0, 0].legend(loc="upper left", ncol=4, fontsize=7, bbox_to_anchor=(0, 1.32))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_explanations(inferences: list[Inference], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for inf in inferences:
            fh.write(json.dumps(inf.to_dict(), default=_json_default) + "\n")


def write_run_summary(
    cfg: AppConfig,
    prep_report: dict[str, int],
    overall: pd.DataFrame,
    path: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    summary = {
        "config": asdict(cfg),
        "preprocessing": prep_report,
        "overall_metrics": json.loads(overall.to_json(orient="records")),
        **(extra or {}),
    }
    path.write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
