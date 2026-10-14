import numpy as np
import pandas as pd
import pytest
from conftest import make_fact

from almacen.expert.engine import Policy
from almacen.metrics import item_metrics
from almacen.policy import compute_params, z_from_service_level
from almacen.simulation import simulate_item

WEEKS = pd.date_range("2017-01-02", periods=12, freq="7D")


def test_z_values():
    assert z_from_service_level(0.5) == pytest.approx(0.0)
    assert z_from_service_level(0.95) == pytest.approx(1.6449, abs=1e-4)


def test_periodic_parameters():
    p = compute_params(
        make_fact(mean_w=100, std_w=20),
        Policy("periodic_RS", 0.95, 2, 0),
        lead_time_weeks=1,
    )
    expected_ss = z_from_service_level(0.95) * 20 * np.sqrt(3)
    assert p.safety_stock == pytest.approx(expected_ss, abs=0.01)
    assert p.order_up_to == np.ceil(300 + expected_ss)


def test_shelf_life_caps_order_up_to():
    fact = make_fact(mean_w=100, std_w=50, perishable=True, shelf_life_weeks=2)
    p = compute_params(
        fact, Policy("periodic_RS", 0.99, 1, 0, max_cover_weeks=2), lead_time_weeks=1
    )
    assert p.order_up_to == 200


@pytest.mark.parametrize(
    "policy",
    [
        Policy("periodic_RS", 0.95, 1, 0),
        Policy("periodic_RS", 0.95, 4, 0),
        Policy("continuous_sQ", 0.95, 0, 2),
        Policy("on_demand", 0.5, 1, 0),
    ],
)
def test_constant_demand_never_stocks_out(policy):
    p = compute_params(make_fact(mean_w=50, std_w=0), policy, lead_time_weeks=1)
    m = item_metrics(simulate_item(np.full(12, 50.0), WEEKS, p, "order_up_to"))
    assert m.fill_rate == pytest.approx(1.0)
    assert m.stockout_weeks == 0


def test_lead_time_delays_arrivals():
    p = compute_params(
        make_fact(mean_w=10, std_w=0),
        Policy("periodic_RS", 0.5, 1, 0),
        lead_time_weeks=2,
    )
    tl = simulate_item(np.full(6, 10.0), WEEKS[:6], p, "zero").timeline
    # nothing on hand; the first order, placed at the end of week 0, arrives at the start of week 3
    assert list(tl["received"][:3]) == [0, 0, 0]
    assert tl["received"].iloc[3] == 30
    assert tl["lost"][:3].sum() == 30
