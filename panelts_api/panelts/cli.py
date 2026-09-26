from __future__ import annotations

import argparse
import json

from .api import available_units, list_datasets, load


def main() -> None:
    parser = argparse.ArgumentParser(prog="panelts", description="PanelTS forecasting data API")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List supported logical datasets")

    units = sub.add_parser("units", help="List units/tickers/regions without downloading all data")
    units.add_argument("name")

    info = sub.add_parser("info", help="Download/load a dataset and show its canonical schema")
    info.add_argument("name")
    info.add_argument("--level", type=int)
    info.add_argument("--unit", action="append", dest="units")

    args = parser.parse_args()
    if args.command == "list":
        print(json.dumps(list_datasets(), indent=2, ensure_ascii=False))
    elif args.command == "units":
        print("\n".join(available_units(args.name)))
    elif args.command == "info":
        kwargs = {}
        if args.level is not None:
            kwargs["level"] = args.level
        if args.units:
            kwargs["units"] = args.units
        ds = load(args.name, **kwargs)
        print(json.dumps(ds.describe(), indent=2, ensure_ascii=False, default=str))
