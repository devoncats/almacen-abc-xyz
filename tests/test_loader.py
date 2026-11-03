import pandas as pd
import pytest
from conftest import ROOT

from almacen.config import ConfigError, load_config
from almacen.data.loader import JoinSpec, load_dataset
from almacen.data.synthetic import SyntheticSpec, generate, to_favorita_layout
from almacen.schemas import SchemaError

IDENTITY = {}


@pytest.fixture(scope="module")
def daily():
    return generate(SyntheticSpec(n_items=40, n_stores=4, n_weeks=20, seed=3))


@pytest.fixture
def favorita_dir(tmp_path, daily):
    for name, table in to_favorita_layout(daily).items():
        table.to_csv(tmp_path / f"{name}.csv", index=False)
    return tmp_path


def joins(d):
    return (
        JoinSpec(d / "items.csv", "item_nbr"),
        JoinSpec(d / "stores.csv", "store_nbr"),
    )


def test_join_reconstructs_denormalized_sales(favorita_dir, daily):
    result = load_dataset(
        favorita_dir / "train.csv", IDENTITY, joins(favorita_dir), chunk_rows=1000
    )
    got = result.sales.sort_values(["date", "store_nbr", "item_nbr"]).reset_index(
        drop=True
    )
    want = daily.sort_values(["date", "store_nbr", "item_nbr"]).reset_index(drop=True)
    assert len(got) == len(want)
    for col in ("family", "class", "city", "type"):
        assert (got[col].astype(str).values == want[col].astype(str).values).all()
    assert (got["perishable"].values == want["perishable"].astype(bool).values).all()
    assert result.report["rows_unmatched_items"] == 0


def test_missing_onpromotion_is_false(favorita_dir):
    sales = load_dataset(
        favorita_dir / "train.csv", IDENTITY, joins(favorita_dir)
    ).sales
    early = sales["date"] < sales["date"].min() + pd.Timedelta(days=60)
    assert sales["onpromotion"].dtype == bool
    assert not sales.loc[early, "onpromotion"].any()


def test_city_and_date_filters_are_pushed_down(favorita_dir, daily):
    result = load_dataset(
        favorita_dir / "train.csv",
        IDENTITY,
        joins(favorita_dir),
        city="quito",
        date_from="2015-03-01",
        chunk_rows=500,
    )
    assert set(result.sales["city"]) == {"Quito"}
    assert result.sales["date"].min() >= pd.Timestamp("2015-03-01")
    assert result.report["rows_scanned"] == len(daily)
    assert result.report["rows_after_pushdown_filters"] < len(daily)


def test_unmatched_rows_are_dropped_and_reported(favorita_dir):
    items = pd.read_csv(favorita_dir / "items.csv")
    items.iloc[1:].to_csv(favorita_dir / "items.csv", index=False)
    result = load_dataset(favorita_dir / "train.csv", IDENTITY, joins(favorita_dir))
    assert result.report["rows_unmatched_items"] > 0
    assert items.iloc[0]["item_nbr"] not in set(result.sales["item_nbr"])


def test_duplicate_lookup_key_is_rejected(favorita_dir):
    stores = pd.read_csv(favorita_dir / "stores.csv")
    pd.concat([stores, stores.head(1)]).to_csv(favorita_dir / "stores.csv", index=False)
    with pytest.raises(SchemaError, match="not unique"):
        load_dataset(favorita_dir / "train.csv", IDENTITY, joins(favorita_dir))


def test_unknown_city_is_rejected(favorita_dir):
    with pytest.raises(SchemaError, match="no stores for city"):
        load_dataset(
            favorita_dir / "train.csv", IDENTITY, joins(favorita_dir), city="Cuenca"
        )


def test_shipped_config_uses_favorita_layout():
    cfg = load_config(ROOT / "config" / "config.yaml")
    ds = cfg.dataset
    assert ds.sales_path.name == "train.csv"
    assert [(j.path.name, j.on) for j in ds.joins] == [
        ("items.csv", "item_nbr"),
        ("stores.csv", "store_nbr"),
    ]
    assert ds.city == "Quito" and ds.date_from == "2015-08-01"


def test_data_dir_override_keeps_file_names(tmp_path):
    cfg = load_config(ROOT / "config" / "config.yaml").with_dataset(dir=tmp_path)
    assert cfg.dataset.sales_path == tmp_path / "train.csv"
    assert cfg.dataset.joins[1].path == tmp_path / "stores.csv"


@pytest.mark.parametrize(
    "bad",
    [
        "dataset:\n  files: {sales: a.csv, extra: b.csv}\n",  # unknown file role
        "dataset:\n  columns: {qty: unit_sales, q2: unit_sales}\n",  # two columns -> same target
        "dataset:\n  date_from: not-a-date\n",
        "datasets: {}\n",  # unknown top-level key
        "segmenter:\n  lightgbm: {trees: 10}\n",  # unknown model parameter
        "segmenter:\n  lightgbm: {class_weight: heavy}\n",
        "decision:\n  top_items: 0\n",
    ],
)
def test_config_rejects_invalid(tmp_path, bad):
    path = tmp_path / "bad.yaml"
    path.write_text(bad)
    with pytest.raises(ConfigError):
        load_config(path)
