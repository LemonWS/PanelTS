from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from ..core import PanelDataset
from ..hub import HubClient
from ..utils import clean_columns, coerce_numeric_columns, infer_time_column, parse_time_series, select_by_units


def _load_static(client: HubClient, level: int) -> pd.DataFrame | None:
    prefix = f"PanelTS/Covid-19/{level}/other attribute"
    files = client.list_files(prefix=prefix, suffix=".csv")
    if not files:
        return None
    # Current repository has one aggregate static-attribute CSV per granularity.
    raw = clean_columns(client.read_csv(files[0]))
    if raw.empty:
        return None
    unit_col = raw.columns[0]
    raw[unit_col] = raw[unit_col].astype(str).str.strip()
    raw = coerce_numeric_columns(raw, exclude=[unit_col])
    raw = raw.rename(columns={unit_col: "unit_id"}).set_index("unit_id")
    return raw


def load_covid(
    client: HubClient,
    *,
    level: int = 20,
    units: Sequence[str] | None = None,
    target: str = "new",
    include_static: bool = True,
) -> PanelDataset:
    if level not in {20, 79, 320}:
        raise ValueError("COVID level must be one of 20, 79, 320")

    prefix = f"PanelTS/Covid-19/{level}/daily new case"
    files = client.list_files(prefix=prefix, suffix=".csv")
    files = select_by_units(files, units)
    if not files:
        raise FileNotFoundError(f"No COVID CSV files found under {prefix!r}")

    frames: list[pd.DataFrame] = []
    for path in files:
        raw = clean_columns(client.read_csv(path))
        time_col = infer_time_column(raw)
        if time_col is None:
            raise ValueError(f"Could not infer date column in {path}")
        target_lookup = {c.lower(): c for c in raw.columns}
        if target.lower() not in target_lookup:
            raise KeyError(f"Target {target!r} not found in {path}; columns={list(raw.columns)}")
        target_col = target_lookup[target.lower()]
        raw[time_col] = parse_time_series(raw[time_col])
        raw = coerce_numeric_columns(raw, exclude=[time_col])
        unit_id = Path(path).stem
        out = raw.rename(columns={time_col: "time", target_col: "target"})
        out.insert(1, "unit_id", unit_id)
        out = out[["time", "unit_id", "target"] + [c for c in out.columns if c not in {"time", "unit_id", "target"}]]
        out = out.drop_duplicates(["time", "unit_id"], keep="last")
        frames.append(out)

    static = _load_static(client, level) if include_static else None
    if static is not None and units is not None:
        static = static.reindex([str(u) for u in units]).dropna(how="all")

    return PanelDataset(
        name=f"covid-{level}",
        frame=pd.concat(frames, ignore_index=True),
        static_covariates=static,
        metadata={
            "domain": "covid-19",
            "granularity": level,
            "source_repo": client.repo_id,
            "target_source_column": target,
            "covariate_note": "Per-unit CSV covariates are dynamic; the separate attributes CSV is static.",
        },
    )
