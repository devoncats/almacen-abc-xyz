import json

import pytest
from conftest import ROOT

from almacen.config import AppConfig, DatasetConfig, SimConfig
from almacen.data.loader import load_dataset
from almacen.data.synthetic import SyntheticSpec, generate, to_favorita_layout
from almacen.pipeline import run
from almacen.preprocessing import prepare
from almacen.schemas import RAW_COLUMNS, SchemaError, validate


@pytest.fixture(scope="module")
def raw():
    return generate(SyntheticSpec(n_items=60, n_stores=3, n_weeks=40, seed=7))


@pytest.fixture
def data_dir(tmp_path, raw):
    for name, table in to_favorita_layout(raw).items():
        table.to_csv(tmp_path / f"{name}.csv", index=False)
    return tmp_path


def test_synthetic_has_canonical_columns(raw):
    assert set(RAW_COLUMNS) <= set(raw.columns)
    assert (raw["unit_sales"] < 0).any()  # returns are injected


def test_prepare_drops_returns_and_zero_fills(raw):
    prep = prepare(validate(raw, RAW_COLUMNS), city=None, new_item_weeks=13)
    assert (prep.weekly["units"] >= 0).all()
    assert prep.report["returns_dropped"] > 0
    span = prep.weekly.groupby("item_nbr")["week"].agg(["min", "max", "count"])
    expected = (span["max"] - span["min"]).dt.days // 7 + 1
    assert (span["count"] == expected).all()  # contiguous weekly panel


def test_column_renames_and_missing_columns(data_dir):
    ds = DatasetConfig(dir=data_dir)
    train = data_dir / "train.csv"
    text = train.read_text().replace("unit_sales", "qty", 1)
    train.write_text(text)
    assert (
        "unit_sales"
        in load_dataset(ds.sales_path, {"qty": "unit_sales"}, ds.joins).sales
    )
    with pytest.raises(SchemaError, match="missing required columns"):
        load_dataset(ds.sales_path, {}, ds.joins)


def test_end_to_end_run(tmp_path, data_dir):
    cfg = AppConfig(
        rules_path=ROOT / "config" / "rules.yaml",
        output_dir=tmp_path / "out",
        dataset=DatasetConfig(dir=data_dir),
        sim=SimConfig(test_weeks=8),
    )
    result = run(cfg)
    tables = result.out_dir / "tables"
    for name in (
        "metrics_summary",
        "metrics_by_segment",
        "abc_xyz_matrix",
        "recommendations",
    ):
        for ext in ("csv", "md", "json"):
            assert (tables / f"{name}.{ext}").exists()
    for chart in (
        "abc_pareto",
        "abc_xyz_matrix",
        "cv_vs_volume",
        "metrics_by_segment",
        "inventory_timeline",
    ):
        assert (result.out_dir / "charts" / f"{chart}.png").exists()
    lines = (result.out_dir / "explanations.jsonl").read_text().splitlines()
    assert len(lines) == len(result.segments)
    summary = json.loads((result.out_dir / "run.json").read_text())
    assert summary["overall_metrics"][0]["fill_rate_expert"] > 0
