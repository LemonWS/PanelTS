from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .hub import HubClient
from .loaders import load_covid, load_etf, load_exchange_rate, load_stock, load_synthetic
from .utils import canonical_stem

DATASETS = {
    "covid-20": {"loader": "covid", "description": "COVID-19, 20 regions, target=new cases + dynamic/static covariates"},
    "covid-79": {"loader": "covid", "description": "COVID-19, 79 LGAs, target=new cases + dynamic/static covariates"},
    "covid-320": {"loader": "covid", "description": "COVID-19, 320 postcodes, target=new cases + dynamic/static covariates"},
    "exchange-rate": {"loader": "exchange", "description": "FX prices + monthly macroeconomic covariates"},
    "etf": {"loader": "etf", "description": "ETF prices with market covariates"},
    "stock-healthcare": {"loader": "stock", "description": "S&P 500 Health Care stocks + quarterly fundamentals"},
    "stock-it": {"loader": "stock", "description": "S&P 500 Information Technology stocks + quarterly fundamentals"},
    "stock-sp500": {"loader": "stock", "description": "S&P500 stocks + available attributes"},
    "synthetic-y5-x5": {"loader": "synthetic", "description": "Synthetic panel, y=5, x=5"},
    "synthetic-y5-x10": {"loader": "synthetic", "description": "Synthetic panel, y=5, x=10"},
    "synthetic-y5-x20": {"loader": "synthetic", "description": "Synthetic panel, y=5, x=20"},
    "synthetic-y10-x5": {"loader": "synthetic", "description": "Synthetic panel, y=10, x=5"},
    "synthetic-y10-x10": {"loader": "synthetic", "description": "Synthetic panel, y=10, x=10"},
    "synthetic-y10-x20": {"loader": "synthetic", "description": "Synthetic panel, y=10, x=20"},
    "synthetic-y20-x5": {"loader": "synthetic", "description": "Synthetic panel, y=20, x=5"},
    "synthetic-y20-x10": {"loader": "synthetic", "description": "Synthetic panel, y=20, x=10"},
    "synthetic-y20-x20": {"loader": "synthetic", "description": "Synthetic panel, y=20, x=20"},
}


def list_datasets() -> list[dict[str, str]]:
    return [{"name": k, **v} for k, v in DATASETS.items()]


def _client_from_kwargs(
    *,
    repo_id: str,
    revision: str,
    cache_dir: str | Path | None,
    token: str | bool | None,
    local_files_only: bool,
) -> HubClient:
    return HubClient(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )


def load(
    name: str,
    *,
    repo_id: str = "Multiple-Time-Series/PanelTS",
    revision: str = "main",
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    **kwargs: Any,
):
    """Load one PanelTS subset into a canonical PanelDataset."""
    key = name.strip().lower().replace("_", "-")
    client = _client_from_kwargs(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )

    if key == "covid":
        return load_covid(client, **kwargs)
    if key.startswith("covid-"):
        level = int(key.split("-", 1)[1])
        return load_covid(client, level=level, **kwargs)
    if key in {"exchange", "exchange-rate", "fx"}:
        return load_exchange_rate(client, **kwargs)
    if key == "etf":
        return load_etf(client, **kwargs)
    if key in {"stock", "stock-healthcare", "stock-it", "stock-sp500"}:
        group = {
            "stock": "healthcare",
            "stock-healthcare": "healthcare",
            "stock-it": "it",
            "stock-sp500": "sp500",
        }[key]
        kwargs.setdefault("group", group)
        return load_stock(client, **kwargs)
    if key == "synthetic":
        return load_synthetic(client, **kwargs)
    if key.startswith("synthetic-y"):
        import re
        m = re.fullmatch(r"synthetic-y(\d+)-x(\d+)", key)
        if not m:
            raise ValueError(f"Invalid synthetic dataset name: {name}")
        kwargs.setdefault("y", int(m.group(1)))
        kwargs.setdefault("x", int(m.group(2)))
        return load_synthetic(client, **kwargs)
    raise KeyError(f"Unknown dataset {name!r}. Use panelts.list_datasets().")


def available_units(
    name: str,
    *,
    repo_id: str = "Multiple-Time-Series/PanelTS",
    revision: str = "main",
    token: str | bool | None = None,
) -> list[str]:
    client = HubClient(repo_id=repo_id, revision=revision, token=token)
    key = name.strip().lower().replace("_", "-")
    if key.startswith("covid"):
        level = 20 if key == "covid" else int(key.split("-")[1])
        files = client.list_files(prefix=f"PanelTS/Covid-19/{level}/daily new case", suffix=".csv")
        return [canonical_stem(p) for p in files]
    if key in {"exchange", "exchange-rate", "fx"}:
        files = client.list_files(prefix="PanelTS/Exchange Rate/price", suffix=".csv")
        return [canonical_stem(p).upper() for p in files]
    if key == "etf":
        files = client.list_files(prefix="PanelTS/ETF", suffix=".csv")
        return [canonical_stem(p).upper() for p in files]
    if key.startswith("stock"):
        group = {"stock-healthcare": "healthcare", "stock-it": "it", "stock-sp500": "sp500", "stock": "healthcare"}.get(key, "healthcare")
        from .loaders.market import STOCK_GROUPS
        folder = STOCK_GROUPS[group]
        files = [p for p in client.list_files(prefix=f"PanelTS/Stock/{folder}", suffix=".csv") if "/price/" in p]
        return [canonical_stem(p).upper() for p in files]
    return []
