from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from ..core import PanelDataset
from ..hub import HubClient
from ..utils import (
    align_low_frequency,
    canonical_stem,
    clean_columns,
    coerce_numeric_columns,
    common_target_name,
    infer_time_column,
    prefixed_columns,
    select_by_units,
    to_numeric_loose,
)

STOCK_GROUPS = {
    "healthcare": "S&P 500 Health Care Index, S5HLTH",
    "health-care": "S&P 500 Health Care Index, S5HLTH",
    "it": "S&P 500 Information Technology, S5INFT",
    "information-technology": "S&P 500 Information Technology, S5INFT",
    "sp500": "S&P500",
    "s&p500": "S&P500",
}


def _market_price_frame(raw: pd.DataFrame, *, unit: str, target: str) -> pd.DataFrame:
    raw = clean_columns(raw)
    time_col = infer_time_column(raw)
    if time_col is None:
        raise ValueError(f"Could not infer market date column for unit {unit}")
    target_col = common_target_name(list(raw.columns), target)
    raw[time_col] = pd.to_datetime(raw[time_col], errors="coerce")
    raw = raw.dropna(subset=[time_col])
    raw = coerce_numeric_columns(raw, exclude=[time_col])
    out = raw.rename(columns={time_col: "time", target_col: "target"})
    out = prefixed_columns(out, "price__", exclude={"time", "target"})
    out.insert(1, "unit_id", unit)
    return out.drop_duplicates(["time", "unit_id"], keep="last")


def load_etf(
    client: HubClient,
    *,
    units: Sequence[str] | None = None,
    target: str = "Close",
) -> PanelDataset:
    prefix = "PanelTS/ETF"
    files = select_by_units(client.list_files(prefix=prefix, suffix=".csv"), units)
    if not files:
        raise FileNotFoundError("No ETF files matched the request")
    frames = [
        _market_price_frame(client.read_csv(path), unit=canonical_stem(path).upper(), target=target)
        for path in files
    ]
    return PanelDataset(
        name="etf",
        frame=pd.concat(frames, ignore_index=True),
        metadata={
            "domain": "etf",
            "source_repo": client.repo_id,
            "target_source_column": target,
            "covariate_note": "OHLC/adjusted close/change/volume fields other than the target are dynamic covariates.",
        },
    )


def _parse_transposed_stock_attributes(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert stock fundamentals stored as metrics x quarter columns into time x metrics."""
    raw = clean_columns(raw)
    if raw.empty or len(raw.columns) < 2:
        return pd.DataFrame(columns=["time"])
    metric_col = raw.columns[0]
    transposed = raw.set_index(metric_col).T
    transposed.index.name = "time"
    transposed = transposed.reset_index()
    transposed["time"] = pd.to_datetime(transposed["time"], errors="coerce")
    transposed = transposed.dropna(subset=["time"])
    # Duplicate metric labels occasionally occur in scraped statements; keep the first.
    transposed = transposed.loc[:, ~transposed.columns.duplicated()]
    for col in transposed.columns:
        if col != "time":
            transposed[col] = to_numeric_loose(transposed[col])
    return transposed.sort_values("time")


def load_stock(
    client: HubClient,
    *,
    group: str = "healthcare",
    units: Sequence[str] | None = None,
    target: str = "Close",
    include_fundamentals: bool = True,
    alignment: str = "backward",
) -> PanelDataset:
    folder = STOCK_GROUPS.get(group.lower(), group)
    root = f"PanelTS/Stock/{folder}"
    all_csv = client.list_files(prefix=root, suffix=".csv")
    price_files = [p for p in all_csv if "/price/" in p]
    attr_files = [p for p in all_csv if p not in price_files and "attribute" in p.lower()]
    if not price_files:
        raise FileNotFoundError(f"No stock price files found under {root!r}")

    # Match requested tickers against normalized price stems (e.g. A-history.csv -> A).
    if units:
        wanted = {u.lower() for u in units}
        price_files = [p for p in price_files if canonical_stem(p).lower() in wanted]
    attr_by_unit = {canonical_stem(p).lower(): p for p in attr_files}

    frames: list[pd.DataFrame] = []
    for path in price_files:
        unit = canonical_stem(path).upper()
        out = _market_price_frame(client.read_csv(path), unit=unit, target=target)
        if include_fundamentals:
            attr_path = attr_by_unit.get(unit.lower())
            if attr_path:
                attr = _parse_transposed_stock_attributes(client.read_csv(attr_path))
                attr = prefixed_columns(attr, "fundamental__", exclude={"time"})
                base = out.drop(columns="unit_id")
                base = align_low_frequency(base, attr, target_time="time", attribute_time="time", mode=alignment)
                base.insert(1, "unit_id", unit)
                out = base
        frames.append(out)

    if not frames:
        raise FileNotFoundError("No stock units matched the request")

    return PanelDataset(
        name=f"stock-{group}",
        frame=pd.concat(frames, ignore_index=True),
        metadata={
            "domain": "equity",
            "stock_group": folder,
            "source_repo": client.repo_id,
            "target_source_column": target,
            "low_frequency_alignment": alignment,
        },
    )
