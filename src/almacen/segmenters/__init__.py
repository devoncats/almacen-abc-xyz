"""Segmenter registry. To add a model, subclass Segmenter and register it here."""

from __future__ import annotations

from almacen.config import AppConfig, ConfigError
from almacen.segmenters.base import SegmentationResult, Segmenter
from almacen.segmenters.classic import ClassicSegmenter
from almacen.segmenters.lgbm import LightGBMSegmenter

SEGMENTERS: dict[str, type[Segmenter]] = {
    cls.name: cls for cls in (ClassicSegmenter, LightGBMSegmenter)
}


def get_segmenter(cfg: AppConfig) -> Segmenter:
    try:
        cls = SEGMENTERS[cfg.segmenter.method]
    except KeyError:
        raise ConfigError(
            f"unknown segmenter '{cfg.segmenter.method}'; available: {sorted(SEGMENTERS)}"
        ) from None
    return cls(cfg)


__all__ = ["SEGMENTERS", "SegmentationResult", "Segmenter", "get_segmenter"]
