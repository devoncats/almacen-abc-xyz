"""Classic ABC/XYZ: volume-share cuts and coefficient-of-variation thresholds on history."""

from __future__ import annotations

import pandas as pd

from almacen.segmentation import segment
from almacen.segmenters.base import RESULT_COLUMNS, SegmentationResult, Segmenter


class ClassicSegmenter(Segmenter):
    name = "abc-xyz"
    label = "classic ABC/XYZ"

    def fit_predict(
        self, weekly_train: pd.DataFrame, items: pd.DataFrame
    ) -> SegmentationResult:
        seg = segment(weekly_train, self.cfg.abc, self.cfg.xyz)
        seg["confidence"] = 1.0
        return SegmentationResult(seg=seg[RESULT_COLUMNS])
