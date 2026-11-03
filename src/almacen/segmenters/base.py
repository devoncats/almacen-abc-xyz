"""Segmenter interface: any model that assigns each item one of the nine ABC/XYZ segments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from almacen.config import AppConfig
from almacen.schemas import SEGMENT_COLUMNS

RESULT_COLUMNS = [*SEGMENT_COLUMNS, "confidence"]


@dataclass
class SegmentationResult:
    seg: pd.DataFrame  # RESULT_COLUMNS sorted by descending training volume
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)  # model diagnostics
    report: dict[str, Any] = field(
        default_factory=dict
    )  # scalar diagnostics for run.json


class Segmenter(ABC):
    """Assigns segments from the (capped) training panel only.

    `seg` must keep the historical demand statistics (total_units, share, cum_share, mean_w,
    std_w, cv) because the policy parameters are always estimated from history; only the
    abc / xyz / segment labels and `confidence` (probability of the assigned segment, 1.0 for
    deterministic methods) depend on the segmenter.
    """

    name: str  # CLI / config identifier
    label: str  # human-readable name used in chart titles and reports

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def fit_predict(
        self, weekly_train: pd.DataFrame, items: pd.DataFrame
    ) -> SegmentationResult:
        """`items` holds ITEM_COLUMNS with age_weeks measured at the train cutoff."""

    def plot_diagnostics(self, result: SegmentationResult, charts_dir: Path) -> None:
        """Write model-specific charts; deterministic segmenters have none."""
