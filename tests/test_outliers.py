import numpy as np
import pytest
from conftest import weekly_frame

from almacen.config import OutlierConfig
from almacen.preprocessing import cap_outliers

SPIKY = [100, 110, 90, 105, 95, 100, 5000, 98, 102, 100]  # one spike week


def test_none_leaves_data_untouched():
    df = weekly_frame({1: SPIKY})
    capped, log = cap_outliers(df, OutlierConfig(method="none"))
    assert capped is df
    assert log.empty


def test_mad_caps_only_the_spike():
    df = weekly_frame({1: SPIKY})
    capped, log = cap_outliers(df, OutlierConfig(method="mad", mad_k=5.0))
    units = df["units"].to_numpy()
    median = np.median(units)
    cap = median + 5.0 * 1.4826 * np.median(np.abs(units - median))
    assert len(log) == 1
    assert log.iloc[0]["units_original"] == 5000
    assert capped["units"].max() == pytest.approx(cap)
    # every non-spike week is unchanged
    assert (capped["units"].to_numpy()[units != 5000] == units[units != 5000]).all()


def test_mad_skips_items_with_zero_mad():
    # intermittent item: median 0 and MAD 0 would otherwise wipe out every sale
    df = weekly_frame({1: [0, 0, 0, 0, 0, 0, 40, 0, 0, 35]})
    capped, log = cap_outliers(df, OutlierConfig(method="mad"))
    assert log.empty
    assert capped["units"].sum() == 75


def test_percentile_caps_at_item_quantile():
    df = weekly_frame({1: SPIKY, 2: [10] * 10})
    capped, log = cap_outliers(df, OutlierConfig(method="percentile", percentile=0.9))
    q = np.quantile(SPIKY, 0.9)
    item1 = capped[capped["item_nbr"] == 1]["units"]
    assert item1.max() == pytest.approx(q)
    assert set(log["item_nbr"]) == {1}  # constant item is never above its own quantile
