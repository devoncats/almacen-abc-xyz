"""Sales loader for the Favorita layout: a fact table joined with lookup tables.

train.csv is joined with items.csv on item_nbr and stores.csv on store_nbr. Optional column renames
let files with other headers follow the same layout. Large CSV fact tables are read in chunks and
filtered by store and date before they are held in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from almacen.schemas import OPTIONAL_COLUMNS, RAW_COLUMNS, SchemaError, validate

KNOWN_COLUMNS = set(RAW_COLUMNS) | set(OPTIONAL_COLUMNS)
# compact dtypes for the (potentially huge) fact table, keyed by canonical name
FACT_DTYPES = {"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"}
DEFAULT_CHUNK_ROWS = 5_000_000


@dataclass(frozen=True)
class JoinSpec:
    path: Path
    on: str  # canonical key column shared with the fact table


@dataclass
class LoadResult:
    sales: pd.DataFrame  # RAW_COLUMNS (+ optional context columns)
    report: dict[str, int] = field(default_factory=dict)


def _check_exists(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise SchemaError(f"data file not found: {path}")
    if path.suffix.lower() not in (".csv", ".gz", ".parquet"):
        raise SchemaError(
            f"unsupported data format '{path.suffix}' (use .csv, .csv.gz or .parquet)"
        )
    return path


def _is_parquet(path: Path) -> bool:
    return path.suffix.lower() == ".parquet"


def _read_table(path: Path, mapping: dict[str, str]) -> pd.DataFrame:
    """Read a small table whole, keep only known columns, rename to canonical names."""
    path = _check_exists(path)
    df = (
        pd.read_parquet(path)
        if _is_parquet(path)
        else pd.read_csv(path, low_memory=False)
    )
    df = df.rename(columns=mapping)
    return df[[c for c in df.columns if c in KNOWN_COLUMNS]]


def _read_fact(
    path: Path,
    mapping: dict[str, str],
    stores: set[int] | None,
    date_from: str | None,
    chunk_rows: int,
    report: dict[str, int],
) -> pd.DataFrame:
    """Read the fact table, filtering by store set and start date chunk by chunk."""
    path = _check_exists(path)

    def keep(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.rename(columns=mapping)
        chunk = chunk[[c for c in chunk.columns if c in KNOWN_COLUMNS]]
        report["rows_scanned"] += len(chunk)
        if stores is not None and "store_nbr" in chunk:
            chunk = chunk[chunk["store_nbr"].isin(stores)]
        if date_from is not None and "date" in chunk:
            chunk = chunk[pd.to_datetime(chunk["date"]) >= pd.Timestamp(date_from)]
        return chunk

    report["rows_scanned"] = 0
    if _is_parquet(path):
        return keep(pd.read_parquet(path))

    header = pd.read_csv(path, nrows=0).columns
    inverse = {mapping.get(c, c): c for c in header}
    usecols = [c for c in header if mapping.get(c, c) in KNOWN_COLUMNS]
    dtypes = {inverse[k]: v for k, v in FACT_DTYPES.items() if k in inverse}
    reader = pd.read_csv(
        path, usecols=usecols, dtype=dtypes, chunksize=chunk_rows, low_memory=False
    )
    parts = [keep(chunk) for chunk in reader]
    return (
        pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
    )


def load_dataset(
    data_path: Path,
    mapping: dict[str, str],
    joins: tuple[JoinSpec, ...] = (),
    city: str | None = None,
    date_from: str | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> LoadResult:
    """Load the fact table, join lookup tables (many-to-one) and validate against RAW_COLUMNS.

    When a lookup keyed by store_nbr provides `city`, the city filter is pushed down to the
    fact-table scan so only the DC's stores are kept in memory.
    """
    report: dict[str, int] = {}
    lookups: list[tuple[JoinSpec, pd.DataFrame]] = []
    for spec in joins:
        table = _read_table(spec.path, mapping)
        if spec.on not in table:
            raise SchemaError(f"join key '{spec.on}' not found in {spec.path}")
        if table[spec.on].duplicated().any():
            raise SchemaError(f"join key '{spec.on}' is not unique in {spec.path}")
        lookups.append((spec, table))

    stores: set[int] | None = None
    if city is not None:
        for spec, table in lookups:
            if spec.on == "store_nbr" and "city" in table:
                match = table[
                    table["city"].astype(str).str.casefold() == city.casefold()
                ]
                if match.empty:
                    raise SchemaError(
                        f"no stores for city '{city}'; available: {sorted(table['city'].unique())}"
                    )
                stores = set(match["store_nbr"].astype(int))

    df = _read_fact(data_path, mapping, stores, date_from, chunk_rows, report)
    report["rows_after_pushdown_filters"] = len(df)

    for spec, table in lookups:
        if spec.on not in df:
            raise SchemaError(
                f"join key '{spec.on}' not found in fact table {data_path}"
            )
        # lookup columns win over same-named fact columns; the key is kept from the fact table
        overlap = [c for c in table.columns if c in df.columns and c != spec.on]
        df = df.drop(columns=overlap)
        key_dtype = df[spec.on].dtype
        table = table.astype({spec.on: key_dtype})
        before = len(df)
        df = df.merge(table, on=spec.on, how="inner", validate="many_to_one")
        report[f"rows_unmatched_{Path(spec.path).stem}"] = before - len(df)

    return LoadResult(sales=validate(df, RAW_COLUMNS), report=report)
