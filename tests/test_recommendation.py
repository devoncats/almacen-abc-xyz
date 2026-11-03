import pandas as pd
import pytest

from almacen.config import DecisionConfig
from almacen.recommendation import Verdict, final_verdict, judge

CFG = DecisionConfig(fill_rate_tolerance_pp=0.5, max_inventory_increase_pct=20.0)


@pytest.mark.parametrize(
    "fill_pp,inv_pct,expected",
    [
        (0.0, -10.0, "adopt"),  # same service, less stock
        (-0.4, -5.0, "adopt"),  # small loss within tolerance, less stock
        (1.0, 15.0, "adopt"),  # better service, bounded extra stock
        (-0.4, 5.0, "keep_baseline"),  # worse service and more stock
        (2.0, 60.0, "review"),  # better service, too much stock
        (-3.0, -40.0, "review"),  # less stock, clearly worse service
        (0.2, 60.0, "keep_baseline"),  # negligible gain, too much stock
    ],
)
def test_judge(fill_pp, inv_pct, expected):
    assert judge(fill_pp, inv_pct, CFG).decision == expected


def _decisions(*decisions):
    return pd.DataFrame(
        {"segment": [f"S{i}" for i in range(len(decisions))], "decision": decisions}
    )


@pytest.mark.parametrize(
    "overall,segments,expected",
    [
        ("adopt", ("adopt", "adopt"), "adopt"),
        ("adopt", ("adopt", "keep_baseline"), "adopt_selectively"),
        ("keep_baseline", ("adopt", "review"), "adopt_selectively"),
        ("keep_baseline", ("review", "keep_baseline"), "review"),
        ("keep_baseline", ("keep_baseline",), "keep_baseline"),
    ],
)
def test_final_verdict_is_consistent_with_segments(overall, segments, expected):
    assert (
        final_verdict(Verdict(overall, ""), _decisions(*segments)).decision == expected
    )
