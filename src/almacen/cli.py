"""Command-line interface: generate, run, explain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from almacen.config import ConfigError, load_config
from almacen.expert.engine import InferenceError
from almacen.expert.knowledge import RuleError
from almacen.schemas import SchemaError

app = typer.Typer(add_completion=False, help="ABC/XYZ replenishment expert system.")

USER_ERRORS = (ConfigError, SchemaError, RuleError, InferenceError)


def _fail(exc: Exception) -> NoReturn:
    typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=2)


@app.command()
def generate(
    items: Annotated[int, typer.Option(help="Number of items.")] = 300,
    stores: Annotated[int, typer.Option(help="Number of stores.")] = 5,
    weeks: Annotated[int, typer.Option(help="Number of weeks of daily history.")] = 104,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path(
        "data/synthetic"
    ),
) -> None:
    """Generate synthetic data in the Favorita layout (train.csv, items.csv, stores.csv)."""
    from almacen.data.synthetic import SyntheticSpec, to_favorita_layout
    from almacen.data.synthetic import generate as gen

    df = gen(SyntheticSpec(n_items=items, n_stores=stores, n_weeks=weeks, seed=seed))
    out.mkdir(parents=True, exist_ok=True)
    for name, table in to_favorita_layout(df, seed).items():
        table.to_csv(out / f"{name}.csv", index=False)
    typer.echo(
        f"wrote {len(df):,} sales rows ({df['item_nbr'].nunique()} items, {stores} stores) "
        f"as train.csv, items.csv, stores.csv to {out}"
    )
    typer.echo(f"run it with: almacen run --data-dir {out.as_posix()}")


@app.command()
def run(
    config: Annotated[Path, typer.Option(help="Config YAML.")] = Path(
        "config/config.yaml"
    ),
    data_dir: Annotated[
        Path | None,
        typer.Option(
            help="Override dataset.dir (directory with train.csv, items.csv, stores.csv)."
        ),
    ] = None,
    city: Annotated[
        str | None, typer.Option(help="Override DC city ('all' for every city).")
    ] = None,
    test_weeks: Annotated[
        int | None, typer.Option(help="Override number of hold-out weeks.")
    ] = None,
    outliers: Annotated[
        str | None,
        typer.Option(help="Override outlier capping: none | percentile | mad."),
    ] = None,
) -> None:
    """Run segmentation, inference, simulation and reporting."""
    from dataclasses import replace

    from almacen.pipeline import run as run_pipeline

    try:
        cfg = load_config(config)
        if data_dir is not None:
            cfg = cfg.with_dataset(dir=data_dir)
        if city is not None:
            cfg = replace(
                cfg,
                dataset=replace(
                    cfg.dataset, city=None if city.lower() == "all" else city
                ),
            )
        if test_weeks is not None:
            cfg = replace(cfg, sim=replace(cfg.sim, test_weeks=test_weeks))
        if outliers is not None:
            if outliers not in ("none", "percentile", "mad"):
                raise ConfigError("--outliers must be 'none', 'percentile' or 'mad'")
            cfg = replace(cfg, outliers=replace(cfg.outliers, method=outliers))
        result = run_pipeline(cfg)
    except USER_ERRORS as exc:
        _fail(exc)

    seg_counts = result.segments["segment"].value_counts()
    typer.echo(
        f"\nItems segmented: {len(result.segments)}  |  "
        + "  ".join(f"{s}:{seg_counts.get(s, 0)}" for s in sorted(seg_counts.index))
    )
    typer.echo("\nMetrics (test window, expert system vs. uniform baseline):\n")
    typer.echo(result.summary.round(4).to_markdown(index=False))
    typer.echo(f"\nOutputs written to {result.out_dir}")


@app.command()
def explain(
    run_dir: Annotated[Path, typer.Option("--run", help="Run output directory.")],
    item: Annotated[int, typer.Option(help="Item number.")],
) -> None:
    """Show the inferred policy and the chain of fired rules for one item."""
    path = run_dir / "explanations.jsonl"
    if not path.exists():
        _fail(FileNotFoundError(f"{path} not found"))
    with path.open(encoding="utf-8") as fh:
        record = next(
            (r for line in fh if (r := json.loads(line))["item_nbr"] == item), None
        )
    if record is None:
        _fail(KeyError(f"item {item} not found in {path}"))

    typer.secho(f"Item {item}", bold=True)
    typer.echo("Fired rules (in order):")
    for k, f in enumerate(record["fired_rules"], 1):
        typer.echo(f"  {k}. {f['rule_id']}: {f['description']}")
        typer.echo(f"     changes: {json.dumps(f['changes'])}")
    typer.echo("Final policy:")
    for key, value in record["policy"].items():
        typer.echo(f"  {key}: {value}")
    if record["flags"]:
        typer.echo(f"Flags: {', '.join(record['flags'])}")


if __name__ == "__main__":
    app()
