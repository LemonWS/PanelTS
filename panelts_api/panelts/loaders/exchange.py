from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from ..core import PanelDataset
from ..hub import HubClient
from ..utils import (
    align_low_frequency,
    clean_columns,
    coerce_numeric_columns,
    common_target_name,
    infer_time_column,
    parse_time_series,
    prefixed_columns,
    select_by_units,
)

CURRENCY_TO_ATTRIBUTE_FILE = {
    "ARS": "Argentina",
    "AUD": "Australia",
    "BRL": "Brazil",
    "CAD": "Canada",
    "CNY": "China",
    "EUR": "Eurozone",
    "GBP": "United Kingdom",
    "IDR": "Indonesia",
    "INR": "India",
    "JPY": "Japan",
    "KRW": "South Korea",
    "MXN": "Mexico",
    "RUB": "Russia",
    "SAR": "Saudi Arabia",
    "TRY": "Turkey",
    "ZAR": "South Africa",
}


def load_exchange_rate(
    client: HubClient,
    *,
    units: Sequence[str] | None = None,
    target: str = "Close",
    include_price_covariates: bool = True,
    include_macro_covariates: bool = True,
    alignment: str = "backward",
) -> PanelDataset:
    price_prefix = "PanelTS/Exchange Rate/price"
    attr_prefix = "PanelTS/Exchange Rate/attribute"
    price_files = select_by_units(client.list_files(prefix=price_prefix, suffix=".csv"), units)
    attr_files = client.list_files(prefix=attr_prefix, suffix=".csv")
    attr_by_stem = {Path(p).stem.lower(): p for p in attr_files}
    if not price_files:
        raise FileNotFoundError("No exchange-rate price files matched the request")

    frames: list[pd.DataFrame] = []
    for path in price_files:
        unit = Path(path).stem.upper()
        raw = clean_columns(client.read_csv(path))
        time_col = infer_time_column(raw)
        if time_col is None:
            raise ValueError(f"Could not infer date column in {path}")
        target_col = common_target_name(list(raw.columns), target)
        raw[time_col] = pd.to_datetime(raw[time_col], errors="coerce")
        raw = raw.dropna(subset=[time_col])
        raw = coerce_numeric_columns(raw, exclude=[time_col])

        keep = [time_col, target_col]
        if include_price_covariates:
            keep += [c for c in raw.columns if c not in {time_col, target_col}]
        out = raw[keep].copy().rename(columns={time_col: "time", target_col: "target"})
        price_covs = [c for c in out.columns if c not in {"time", "target"}]
        out = prefixed_columns(out, "price__", exclude={"time", "target"})

        if include_macro_covariates:
            attr_name = CURRENCY_TO_ATTRIBUTE_FILE.get(unit)
            attr_path = attr_by_stem.get(attr_name.lower()) if attr_name else None
            if attr_path is not None:
                attr = clean_columns(client.read_csv(attr_path))
                attr_time = infer_time_column(attr)
                if attr_time is None:
                    raise ValueError(f"Could not infer macro date column in {attr_path}")
                attr[attr_time] = pd.to_datetime(attr[attr_time].astype(str).str.replace(" T", " ", regex=False), errors="coerce")
                attr = attr.dropna(subset=[attr_time])
                attr = coerce_numeric_columns(attr, exclude=[attr_time])
                attr = attr.rename(columns={attr_time: "time"})
                attr = prefixed_columns(attr, "macro__", exclude={"time"})
                out = align_low_frequency(out, attr, target_time="time", attribute_time="time", mode=alignment)

        out.insert(1, "unit_id", unit)
        out = out.drop_duplicates(["time", "unit_id"], keep="last")
        frames.append(out)

    return PanelDataset(
        name="exchange-rate",
        frame=pd.concat(frames, ignore_index=True),
        metadata={
            "domain": "foreign-exchange",
            "source_repo": client.repo_id,
            "target_source_column": target,
            "low_frequency_alignment": alignment,
            "alignment_warning": (
                "backward uses only attributes timestamped at or before each target time; "
                "same_period can reproduce month-level conventions but may be non-causal."
            ),
        },
    )
