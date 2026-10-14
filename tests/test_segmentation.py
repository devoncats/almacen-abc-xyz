import numpy as np
import pytest
from conftest import weekly_frame

from almacen.config import ABCConfig, XYZConfig
from almacen.segmentation import classify_abc, classify_xyz, segment, segment_matrix


def test_abc_shares_sum_to_one_and_respect_cuts():
    # totals 700, 150, 100, 50 -> cumulative 0.70, 0.85, 0.95, 1.00
    df = weekly_frame({1: [350, 350], 2: [75, 75], 3: [50, 50], 4: [25, 25]})
    abc = classify_abc(df, ABCConfig(a_cut=0.80, b_cut=0.95)).set_index("item_nbr")
    assert abc["share"].sum() == pytest.approx(1.0)
    # item 2 crosses the 80% cut so it stays in A; item 3 crosses 95% so it stays in B
    assert abc["abc"].to_dict() == {1: "A", 2: "A", 3: "B", 4: "C"}


def test_cv_matches_hand_computation():
    df = weekly_frame({1: [10, 20, 30, 40]})
    xyz = classify_xyz(df, XYZConfig()).iloc[0]
    expected = np.std([10, 20, 30, 40], ddof=1) / 25
    assert xyz["cv"] == pytest.approx(expected)
    assert xyz["xyz"] == "Y"  # 0.516 is above the 0.5 X threshold


def test_zero_mean_is_z():
    df = weekly_frame({1: [10, 10, 10], 2: [0, 0, 0]})
    xyz = classify_xyz(df, XYZConfig()).set_index("item_nbr")
    assert xyz.loc[2, "xyz"] == "Z"
    assert xyz.loc[1, "xyz"] == "X"


def test_matrix_covers_all_cells():
    df = weekly_frame({1: [10, 10, 10], 2: [1, 0, 5]})
    m = segment_matrix(segment(df, ABCConfig(), XYZConfig()))
    assert list(m["abc"]) == ["A", "B", "C"]
    assert m["items_total"].sum() == 2
    assert m["volume_pct_total"].sum() == pytest.approx(100, abs=0.05)
