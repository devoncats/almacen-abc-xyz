import pandas as pd
import pytest
from conftest import ROOT

from almacen.config import AppConfig, ConfigError, SegmenterConfig, SimConfig
from almacen.data.synthetic import SyntheticSpec, generate
from almacen.preprocessing import age_in_weeks, prepare, split_train_test
from almacen.schemas import RAW_COLUMNS, SchemaError, validate
from almacen.segmentation import SEGMENT_ORDER, segment
from almacen.segmenters import SEGMENTERS, get_segmenter
from almacen.segmenters.base import RESULT_COLUMNS

HISTORY = ["total_units", "share", "cum_share", "mean_w", "std_w", "cv"]


def make_cfg(method: str, test_weeks: int = 8) -> AppConfig:
    return AppConfig(
        rules_path=ROOT / "config" / "rules.yaml",
        output_dir=ROOT / "outputs",
        sim=SimConfig(test_weeks=test_weeks),
        segmenter=SegmenterConfig(method=method),
    )


@pytest.fixture(scope="module")
def panel():
    raw = generate(SyntheticSpec(n_items=60, n_stores=3, n_weeks=40, seed=7))
    prep = prepare(validate(raw, RAW_COLUMNS), city=None, new_item_weeks=13)
    train, _ = split_train_test(prep.weekly, 8)
    items = prep.items.copy()
    items["age_weeks"] = age_in_weeks(items["first_week"], train["week"].max())
    return train, items


def test_registry_and_unknown_method():
    assert {"abc-xyz", "lightgbm"} <= set(SEGMENTERS)
    with pytest.raises(ConfigError, match="unknown segmenter"):
        get_segmenter(make_cfg("random-forest"))


def test_classic_segmenter_matches_segmentation(panel):
    train, items = panel
    cfg = make_cfg("abc-xyz")
    result = get_segmenter(cfg).fit_predict(train, items)
    expected = segment(train, cfg.abc, cfg.xyz)
    pd.testing.assert_frame_equal(result.seg.drop(columns="confidence"), expected)
    assert (result.seg["confidence"] == 1.0).all()


def test_lightgbm_segmenter_output_contract(panel):
    train, items = panel
    cfg = make_cfg("lightgbm")
    result = get_segmenter(cfg).fit_predict(train, items)
    seg = result.seg
    classic = segment(train, cfg.abc, cfg.xyz)

    assert list(seg.columns) == RESULT_COLUMNS
    assert set(seg["item_nbr"]) == set(classic["item_nbr"])
    assert seg["segment"].isin(SEGMENT_ORDER).all()
    assert (seg["segment"] == seg["abc"] + seg["xyz"]).all()
    assert seg["confidence"].between(0, 1).all()
    # historical demand statistics (used by the policy parameters) are unchanged
    pd.testing.assert_frame_equal(
        seg[["item_nbr", *HISTORY]], classic[["item_nbr", *HISTORY]]
    )
    for name in (
        "model_metrics",
        "model_metrics_by_segment",
        "confusion_matrix",
        "feature_importance",
    ):
        assert name in result.tables
    assert result.report["origins"] >= 2


def test_lightgbm_is_reproducible(panel):
    train, items = panel
    a = get_segmenter(make_cfg("lightgbm")).fit_predict(train, items).seg
    b = get_segmenter(make_cfg("lightgbm")).fit_predict(train, items).seg
    pd.testing.assert_frame_equal(a, b)


def test_lightgbm_needs_enough_history(panel):
    train, items = panel
    with pytest.raises(SchemaError, match="lightgbm needs"):
        get_segmenter(make_cfg("lightgbm", test_weeks=24)).fit_predict(train, items)
