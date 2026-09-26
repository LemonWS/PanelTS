from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd

from ..core import PanelDataset
from ..hub import HubClient
from ..utils import clean_columns, coerce_numeric_columns, infer_time_column, infer_unit_column, parse_time_series


def _find_target_candidates(columns: Sequence[str]) -> list[str]:
    out = []
    for c in columns:
        low = c.lower().strip()
        if low in {"target", "y", "value"} or re.fullmatch(r"y[_-]?\d+", low):
            out.append(c)
    return out


def load_synthetic(
    client: HubClient,
    *,
    y: int = 5,
    x: int = 5,
    target: str | None = None,
    target_columns: Sequence[str] | None = None,
    covariate_columns: Sequence[str] | None = None,
    nrows: int | None = None,
) -> PanelDataset:
    path = f"PanelTS/Synthetic datasets/panel_y{y}_x{x}.csv"
    files = client.list_files(prefix="PanelTS/Synthetic datasets", suffix=".csv")
    if path not in files:
        available = [f.rsplit("/", 1)[-1] for f in files]
        raise FileNotFoundError(f"Synthetic file {path!r} not found. Available: {available}")

    raw = clean_columns(client.read_csv(path, nrows=nrows))
    time_col = infer_time_column(raw)
    unit_col = infer_unit_column(raw)

    # Long-form case: explicit time + unit + one target column.
    if unit_col is not None:
        if time_col is None:
            raw.insert(0, "__time", np.arange(len(raw)))
            time_col = "__time"
        candidates = _find_target_candidates(list(raw.columns))
        if target is not None:
            mapping = {c.lower(): c for c in raw.columns}
            target_col = mapping.get(target.lower())
            if target_col is None:
                raise KeyError(f"Synthetic target {target!r} not found")
        elif len(candidates) == 1:
            target_col = candidates[0]
        else:
            raise ValueError(
                "Could not uniquely infer the synthetic target column. "
                "Pass target='...' for long-form data or target_columns=[...] for wide-form data."
            )
        raw[time_col] = parse_time_series(raw[time_col])
        raw = coerce_numeric_columns(raw, exclude=[time_col, unit_col])
        keep_cov = list(covariate_columns) if covariate_columns is not None else [
            c for c in raw.columns if c not in {time_col, unit_col, target_col}
        ]
        out = raw[[time_col, unit_col, target_col] + keep_cov].rename(
            columns={time_col: "time", unit_col: "unit_id", target_col: "target"}
        )
        return PanelDataset(
            name=f"synthetic-y{y}-x{x}",
            frame=out.drop_duplicates(["time", "unit_id"], keep="last"),
            metadata={"domain": "synthetic", "source_repo": client.repo_id, "raw_file": path},
        )

    # Wide-form case: each y-column is a unit. x-columns are common dynamic covariates
    # and are broadcast to each unit. Explicit lists override inference.
    if time_col is None:
        raw.insert(0, "__time", np.arange(len(raw)))
        time_col = "__time"
    raw[time_col] = parse_time_series(raw[time_col])

    targets = list(target_columns) if target_columns is not None else _find_target_candidates(list(raw.columns))
    if target is not None and not targets:
        mapping = {c.lower(): c for c in raw.columns}
        if target.lower() in mapping:
            targets = [mapping[target.lower()]]
    if not targets:
        raise ValueError(
            "Could not infer synthetic y columns from the current file schema. "
            "Pass target_columns=['y1', ...] and optionally covariate_columns=[...]."
        )
    covs = list(covariate_columns) if covariate_columns is not None else [
        c for c in raw.columns if c != time_col and c not in targets and re.match(r"(?i)^x", c)
    ]
    raw = coerce_numeric_columns(raw, exclude=[time_col])

    frames = []
    for y_col in targets:
        part = pd.DataFrame({"time": raw[time_col], "unit_id": y_col, "target": raw[y_col]})
        for c in covs:
            part[c] = raw[c].to_numpy()
        frames.append(part)
    return PanelDataset(
        name=f"synthetic-y{y}-x{x}",
        frame=pd.concat(frames, ignore_index=True),
        metadata={
            "domain": "synthetic",
            "source_repo": client.repo_id,
            "raw_file": path,
            "schema_note": "Wide-form x columns are treated as common dynamic covariates and broadcast to y units.",
        },
    )
