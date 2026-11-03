"""Application configuration loaded from a single YAML file."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from almacen.data.loader import DEFAULT_CHUNK_ROWS, JoinSpec


class ConfigError(ValueError):
    """Raised when the configuration file is invalid."""


@dataclass(frozen=True)
class DatasetConfig:
    """Favorita layout: a sales fact table plus item and store lookup tables in one directory."""

    dir: Path = Path("data/favorita")
    sales_file: str = "train.csv"  # date, store_nbr, item_nbr, unit_sales, onpromotion
    items_file: str = "items.csv"  # item_nbr, family, class, perishable
    stores_file: str = "stores.csv"  # store_nbr, city, type
    columns: dict[str, str] = field(
        default_factory=dict
    )  # optional rename: dataset_col -> canonical_col
    city: str | None = None  # None = all cities
    date_from: str | None = None  # drop sales before this date while reading
    chunk_rows: int = DEFAULT_CHUNK_ROWS

    @property
    def sales_path(self) -> Path:
        return self.dir / self.sales_file

    @property
    def joins(self) -> tuple[JoinSpec, ...]:
        return (
            JoinSpec(path=self.dir / self.items_file, on="item_nbr"),
            JoinSpec(path=self.dir / self.stores_file, on="store_nbr"),
        )


@dataclass(frozen=True)
class ABCConfig:
    a_cut: float = 0.80
    b_cut: float = 0.95


@dataclass(frozen=True)
class XYZConfig:
    x_max_cv: float = 0.5
    y_max_cv: float = 1.0


@dataclass(frozen=True)
class OutlierConfig:
    """Upper capping of weekly training demand (spikes only); the test window is never modified."""

    method: Literal["none", "percentile", "mad"] = "none"
    percentile: float = 0.99  # method=percentile: cap at this per-item quantile
    mad_k: float = 10.0  # method=mad: cap at median + k * 1.4826 * MAD (items with MAD = 0 are skipped)


@dataclass(frozen=True)
class SimConfig:
    test_weeks: int = 12
    lead_time_weeks: int = 1
    initial_stock: Literal["order_up_to", "zero"] = "order_up_to"


@dataclass(frozen=True)
class BaselineConfig:
    review_weeks: int = 1
    service_level: float = 0.95


@dataclass(frozen=True)
class LightGBMConfig:
    """Rolling-origin training of the segment classifier (labels = classic segment of the next
    `simulation.test_weeks` weeks)."""

    window_weeks: int = 26  # weeks of history summarized by the features
    stride_weeks: int = 4  # spacing between training origins
    min_history_weeks: int = 8  # first origin
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    class_weight: Literal["balanced"] | None = None


@dataclass(frozen=True)
class SegmenterConfig:
    method: str = "abc-xyz"  # abc-xyz | lightgbm
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)


@dataclass(frozen=True)
class DecisionConfig:
    """Criteria for the warehouse-manager recommendation (expert policies vs. uniform baseline)."""

    fill_rate_tolerance_pp: float = (
        0.5  # fill-rate loss still accepted, in percentage points
    )
    max_inventory_increase_pct: float = (
        20.0  # extra average inventory accepted for the same service
    )
    top_items: int = 15  # rows per action list in the report


@dataclass(frozen=True)
class AppConfig:
    rules_path: Path
    output_dir: Path
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    abc: ABCConfig = field(default_factory=ABCConfig)
    xyz: XYZConfig = field(default_factory=XYZConfig)
    outliers: OutlierConfig = field(default_factory=OutlierConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    new_item_weeks: int = 13
    default_shelf_life_weeks: int = 3
    seed: int = 42

    def with_dataset(self, **kwargs: Any) -> AppConfig:
        """Return a copy with dataset fields replaced (None values are ignored)."""
        changes = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, dataset=replace(self.dataset, **changes))


def _section(cls: type, raw: dict[str, Any] | None, name: str) -> Any:
    raw = raw or {}
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown keys in '{name}': {sorted(unknown)}")
    return cls(**raw)


def _dataset(raw: dict[str, Any] | None) -> DatasetConfig:
    raw = dict(raw or {})
    files = raw.pop("files", None) or {}
    unknown_files = set(files) - {"sales", "items", "stores"}
    if unknown_files:
        raise ConfigError(f"unknown keys in 'dataset.files': {sorted(unknown_files)}")
    unknown = set(raw) - {"dir", "columns", "city", "date_from", "chunk_rows"}
    if unknown:
        raise ConfigError(f"unknown keys in 'dataset': {sorted(unknown)}")

    columns = raw.get("columns") or {}
    if not isinstance(columns, dict):
        raise ConfigError(
            "'dataset.columns' must map dataset column names to canonical names"
        )
    targets = [str(v) for v in columns.values()]
    dupes = sorted({t for t in targets if targets.count(t) > 1})
    if dupes:
        raise ConfigError(
            f"several dataset columns map to the same canonical column: {dupes}"
        )

    date_from = raw.get("date_from")
    if date_from is not None:
        try:
            date_from = str(pd.Timestamp(str(date_from)).date())
        except ValueError as exc:
            raise ConfigError(f"invalid dataset.date_from '{date_from}'") from exc

    city = raw.get("city")
    defaults = DatasetConfig()
    return DatasetConfig(
        dir=Path(raw.get("dir", defaults.dir)),
        sales_file=str(files.get("sales", defaults.sales_file)),
        items_file=str(files.get("items", defaults.items_file)),
        stores_file=str(files.get("stores", defaults.stores_file)),
        columns={str(k): str(v) for k, v in columns.items()},
        city=None if city is None or str(city).lower() == "all" else str(city),
        date_from=date_from,
        chunk_rows=int(raw.get("chunk_rows", defaults.chunk_rows)),
    )


def _segmenter(raw: dict[str, Any] | None) -> SegmenterConfig:
    raw = dict(raw or {})
    lgbm = _section(LightGBMConfig, raw.pop("lightgbm", None), "segmenter.lightgbm")
    unknown = set(raw) - {"method"}
    if unknown:
        raise ConfigError(f"unknown keys in 'segmenter': {sorted(unknown)}")
    return SegmenterConfig(
        method=str(raw.get("method", SegmenterConfig.method)), lightgbm=lgbm
    )


def _validate(cfg: AppConfig) -> None:
    if cfg.dataset.chunk_rows < 1:
        raise ConfigError("dataset.chunk_rows must be >= 1")
    if not 0 < cfg.abc.a_cut < cfg.abc.b_cut < 1:
        raise ConfigError("abc cuts must satisfy 0 < a_cut < b_cut < 1")
    if not 0 < cfg.xyz.x_max_cv < cfg.xyz.y_max_cv:
        raise ConfigError("xyz thresholds must satisfy 0 < x_max_cv < y_max_cv")
    if cfg.outliers.method not in ("none", "percentile", "mad"):
        raise ConfigError("outliers.method must be 'none', 'percentile' or 'mad'")
    if not 0.5 < cfg.outliers.percentile < 1 or cfg.outliers.mad_k <= 0:
        raise ConfigError(
            "outliers.percentile must be in (0.5, 1) and outliers.mad_k > 0"
        )
    if cfg.sim.test_weeks < 1 or cfg.sim.lead_time_weeks < 0:
        raise ConfigError(
            "simulation.test_weeks must be >= 1 and simulation.lead_time_weeks >= 0"
        )
    if cfg.sim.initial_stock not in ("order_up_to", "zero"):
        raise ConfigError("simulation.initial_stock must be 'order_up_to' or 'zero'")
    if not 0 < cfg.baseline.service_level < 1:
        raise ConfigError("baseline.service_level must be in (0, 1)")
    lg = cfg.segmenter.lightgbm
    if min(lg.window_weeks, lg.stride_weeks, lg.min_history_weeks, lg.n_estimators) < 1:
        raise ConfigError(
            "segmenter.lightgbm window_weeks, stride_weeks, min_history_weeks and "
            "n_estimators must be >= 1"
        )
    if not (0 < lg.subsample <= 1 and 0 < lg.colsample_bytree <= 1):
        raise ConfigError(
            "segmenter.lightgbm subsample and colsample_bytree must be in (0, 1]"
        )
    if lg.class_weight not in (None, "balanced"):
        raise ConfigError("segmenter.lightgbm.class_weight must be null or 'balanced'")
    if cfg.decision.fill_rate_tolerance_pp < 0 or cfg.decision.top_items < 1:
        raise ConfigError(
            "decision.fill_rate_tolerance_pp must be >= 0 and decision.top_items >= 1"
        )


def load_config(path: Path) -> AppConfig:
    """Load and validate config. Relative paths resolve against the current working directory."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc

    known = {
        "dataset",
        "paths",
        "abc",
        "xyz",
        "outliers",
        "simulation",
        "baseline",
        "segmenter",
        "decision",
        "new_item_weeks",
        "default_shelf_life_weeks",
        "seed",
    }
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown top-level keys: {sorted(unknown)}")

    paths = raw.get("paths") or {}
    cfg = AppConfig(
        rules_path=Path(paths.get("rules", "config/rules.yaml")),
        output_dir=Path(paths.get("output_dir", "outputs")),
        dataset=_dataset(raw.get("dataset")),
        abc=_section(ABCConfig, raw.get("abc"), "abc"),
        xyz=_section(XYZConfig, raw.get("xyz"), "xyz"),
        outliers=_section(OutlierConfig, raw.get("outliers"), "outliers"),
        sim=_section(SimConfig, raw.get("simulation"), "simulation"),
        baseline=_section(BaselineConfig, raw.get("baseline"), "baseline"),
        segmenter=_segmenter(raw.get("segmenter")),
        decision=_section(DecisionConfig, raw.get("decision"), "decision"),
        new_item_weeks=int(raw.get("new_item_weeks", 13)),
        default_shelf_life_weeks=int(raw.get("default_shelf_life_weeks", 3)),
        seed=int(raw.get("seed", 42)),
    )
    _validate(cfg)
    return cfg
