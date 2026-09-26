from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

_TIME_CANDIDATES = (
    "time", "date", "datetime", "timestamp", "ds", "period", "month", "year"
)
_UNIT_CANDIDATES = (
    "unit_id", "unit", "series_id", "series", "entity", "region", "lga",
    "postcode", "ticker", "symbol", "country", "currency", "id"
)


def clean_column_name(name: object) -> str:
    return str(name).replace("\ufeff", "").strip()


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [clean_column_name(c) for c in out.columns]
    return out


def infer_column(columns: Sequence[object], candidates: Sequence[str]) -> str | None:
    mapping = {clean_column_name(c).lower(): clean_column_name(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]
    return None


def infer_time_column(df: pd.DataFrame) -> str | None:
    direct = infer_column(df.columns, _TIME_CANDIDATES)
    if direct:
        return direct
    if len(df.columns) == 0:
        return None
    # A conservative fallback: if the first column parses mostly as dates, use it.
    first = clean_column_name(df.columns[0])
    parsed = pd.to_datetime(df[first], errors="coerce")
    if len(parsed) and parsed.notna().mean() >= 0.8:
        return first
    return None


def infer_unit_column(df: pd.DataFrame) -> str | None:
    return infer_column(df.columns, _UNIT_CANDIDATES)


def parse_time_series(s: pd.Series) -> pd.Series:
    """Parse date-like values while preserving numeric synthetic time indices."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    parsed = pd.to_datetime(s, errors="coerce")
    if len(parsed) == 0 or parsed.notna().mean() < 0.8:
        return s.astype(str)
    return parsed


def _repair_simple_decimal_typo(text: str) -> str:
    # The source currently contains examples such as "4..9". Do not guess a value;
    # leave malformed decimals as NaN rather than silently rewriting data.
    if ".." in text:
        return ""
    return text


def to_numeric_loose(s: pd.Series, *, percent_as_fraction: bool = True) -> pd.Series:
    """Convert common CSV numeric formats to float without crashing on dirty cells."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    text = s.astype("string").str.strip()
    is_percent = text.str.endswith("%", na=False)
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace("$", "", regex=False)
    text = text.str.replace("€", "", regex=False)
    text = text.str.replace("£", "", regex=False)
    text = text.str.replace("%", "", regex=False)
    text = text.map(lambda x: _repair_simple_decimal_typo(x) if isinstance(x, str) else x)
    text = text.replace({"": pd.NA, "-": pd.NA, "--": pd.NA, "N/A": pd.NA, "n/a": pd.NA})
    values = pd.to_numeric(text, errors="coerce")
    if percent_as_fraction and is_percent.any():
        values = values.where(~is_percent, values / 100.0)
    return values


def coerce_numeric_columns(
    df: pd.DataFrame,
    *,
    exclude: Iterable[str] = (),
    min_success_rate: float = 0.65,
) -> pd.DataFrame:
    out = df.copy()
    excluded = set(exclude)
    for col in out.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        converted = to_numeric_loose(out[col])
        original_non_null = out[col].notna().sum()
        if original_non_null == 0:
            out[col] = converted
            continue
        if converted.notna().sum() / original_non_null >= min_success_rate:
            out[col] = converted
    return out


def canonical_stem(path_or_name: str) -> str:
    stem = Path(path_or_name).stem
    stem = re.sub(r"(?i)-history$", "", stem)
    return stem.strip()


def select_by_units(paths: Sequence[str], units: Sequence[str] | None) -> list[str]:
    paths = list(paths)
    if not units:
        return paths
    wanted = {u.strip().lower() for u in units}
    return [p for p in paths if canonical_stem(p).lower() in wanted]


def safe_sort_values(values: Iterable[object]) -> list[object]:
    vals = list(pd.unique(pd.Series(list(values))))
    try:
        return sorted(vals)
    except TypeError:
        return sorted(vals, key=lambda x: str(x))


def align_low_frequency(
    target: pd.DataFrame,
    attributes: pd.DataFrame,
    *,
    target_time: str = "time",
    attribute_time: str = "time",
    mode: str = "backward",
) -> pd.DataFrame:
    """
    Align low-frequency attributes to target timestamps.

    mode="backward" is leakage-safe: each target row receives the most recent
    attribute row whose timestamp is <= the target timestamp.

    mode="same_period" matches by calendar month. This can be useful to reproduce
    conventions in existing experiments, but can leak information if a month-end
    value is treated as known earlier in that same month.
    """
    left = target.copy().sort_values(target_time)
    right = attributes.copy().sort_values(attribute_time)
    if right.empty:
        return left

    if mode == "backward":
        left[target_time] = pd.to_datetime(left[target_time], errors="raise")
        right[attribute_time] = pd.to_datetime(right[attribute_time], errors="raise")
        if attribute_time != target_time:
            right = right.rename(columns={attribute_time: target_time})
        return pd.merge_asof(left, right, on=target_time, direction="backward")

    if mode == "same_period":
        left["__period"] = pd.to_datetime(left[target_time], errors="raise").dt.to_period("M")
        right["__period"] = pd.to_datetime(right[attribute_time], errors="raise").dt.to_period("M")
        right = right.sort_values(attribute_time).drop_duplicates("__period", keep="last")
        right = right.drop(columns=[attribute_time], errors="ignore")
        out = left.merge(right, on="__period", how="left")
        return out.drop(columns="__period")

    raise ValueError("mode must be 'backward' or 'same_period'")


def prefixed_columns(df: pd.DataFrame, prefix: str, exclude: Iterable[str]) -> pd.DataFrame:
    excluded = set(exclude)
    return df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c not in excluded})


def common_target_name(columns: Sequence[str], requested: str | None = None) -> str:
    if requested:
        exact = {c.lower(): c for c in columns}
        if requested.lower() in exact:
            return exact[requested.lower()]
        raise KeyError(f"Target column {requested!r} not found. Available columns: {list(columns)}")
    candidates = ["close", "adj close", "adj. close", "new", "target", "y", "value"]
    mapping = {c.lower(): c for c in columns}
    for c in candidates:
        if c in mapping:
            return mapping[c]
    raise KeyError(f"Could not infer a target column from: {list(columns)}")


def as_float_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda s: to_numeric_loose(s) if not pd.api.types.is_numeric_dtype(s) else s)


def stable_ratio_counts(n: int, train: float, val: float, test: float) -> tuple[int, int, int]:
    if n < 1:
        return 0, 0, 0
    ratios = np.array([train, val, test], dtype=float)
    if np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("train + val + test must equal 1.0 and all ratios must be non-negative")
    raw = ratios * n
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return tuple(int(x) for x in counts)
