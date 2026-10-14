"""Canonical column schemas and validation."""

from __future__ import annotations

import pandas as pd

RAW_COLUMNS: dict[str, str] = {  # required after column mapping
    "date": "datetime64",
    "store_nbr": "int",
    "item_nbr": "int",
    "unit_sales": "float",
    "onpromotion": "bool",
    "family": "str",
    "class": "int",
    "perishable": "bool",
    "city": "str",
    "type": "str",
}
OPTIONAL_COLUMNS: dict[str, str] = {
    "is_holiday": "bool",
    "oil_price": "float",
}  # context only

WEEKLY_COLUMNS = ["item_nbr", "week", "units", "promo_share"]  # DC-level demand
ITEM_COLUMNS = [
    "item_nbr",
    "family",
    "class",
    "perishable",
    "is_weighted",
    "first_week",
    "age_weeks",
]
SEGMENT_COLUMNS = [
    "item_nbr",
    "total_units",
    "share",
    "cum_share",
    "abc",
    "mean_w",
    "std_w",
    "cv",
    "xyz",
    "segment",
]

_TRUE = {"true", "t", "1", "yes", "y"}
_FALSE = {"false", "f", "0", "no", "n", "", "nan", "none"}


class SchemaError(ValueError):
    """Raised when input data does not satisfy the canonical schema."""


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) != 0
    # missing values count as False (Favorita leaves onpromotion empty before 2014-04)
    lowered = s.astype(object).where(s.notna(), "").astype(str).str.strip().str.lower()
    unknown = ~lowered.isin(_TRUE | _FALSE)
    if unknown.any():
        raise ValueError(
            f"unrecognized boolean values: {sorted(lowered[unknown].unique())[:5]}"
        )
    return lowered.isin(_TRUE)


def _coerce(s: pd.Series, dtype: str) -> pd.Series:
    if dtype == "datetime64":
        return pd.to_datetime(s, errors="raise")
    if dtype == "int":
        out = pd.to_numeric(s, errors="raise")
        if out.isna().any():
            raise ValueError("null values in integer column")
        return out.astype("int64")
    if dtype == "float":
        return pd.to_numeric(s, errors="raise").astype("float64")
    if dtype == "bool":
        return _to_bool(s)
    if dtype == "str":
        return s.astype(str)
    raise ValueError(f"unsupported dtype '{dtype}'")


def validate(df: pd.DataFrame, required: dict[str, str]) -> pd.DataFrame:
    """Coerce dtypes; raise SchemaError listing missing or non-convertible columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")

    out = df.copy()
    errors: list[str] = []
    for col, dtype in {
        **required,
        **{k: v for k, v in OPTIONAL_COLUMNS.items() if k in df},
    }.items():
        try:
            out[col] = _coerce(out[col], dtype)
        except (ValueError, TypeError) as exc:
            errors.append(f"{col} ({dtype}): {exc}")
    if errors:
        raise SchemaError("non-convertible columns: " + "; ".join(errors))

    nulls = [c for c in required if out[c].isna().any()]
    if nulls:
        raise SchemaError(f"null values in required columns: {nulls}")
    return out
