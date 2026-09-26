from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from .utils import safe_sort_values, stable_ratio_counts


@dataclass(frozen=True)
class ForecastWindow:
    units: tuple[str, ...]
    input_times: tuple[Any, ...]
    forecast_times: tuple[Any, ...]
    context_target: np.ndarray        # [L, N]
    future_target: np.ndarray         # [H, N]
    context_covariates: np.ndarray | None = None   # [L, N, C]
    future_known_covariates: np.ndarray | None = None  # [H, N, K]
    static_covariates: np.ndarray | None = None    # [N, S]
    covariate_names: tuple[str, ...] = ()
    known_future_covariate_names: tuple[str, ...] = ()
    static_covariate_names: tuple[str, ...] = ()


@dataclass
class PanelDataset:
    """
    Canonical in-memory representation.

    frame columns:
      - time
      - unit_id
      - target
      - zero or more dynamic covariates

    static_covariates:
      - index: unit_id
      - columns: static attributes
    """

    name: str
    frame: pd.DataFrame
    static_covariates: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    known_future_covariates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required = {"time", "unit_id", "target"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"frame is missing required columns: {sorted(missing)}")

        frame = self.frame.copy()
        frame["unit_id"] = frame["unit_id"].astype(str)
        frame = frame.sort_values(["time", "unit_id"], kind="stable")
        dup = frame.duplicated(["time", "unit_id"], keep=False)
        if dup.any():
            examples = frame.loc[dup, ["time", "unit_id"]].head(5).to_dict("records")
            raise ValueError(f"Duplicate (time, unit_id) rows detected, examples: {examples}")
        self.frame = frame.reset_index(drop=True)

        if self.static_covariates is not None:
            static = self.static_covariates.copy()
            if "unit_id" in static.columns:
                static = static.set_index("unit_id")
            static.index = static.index.astype(str)
            static.index.name = "unit_id"
            static = static[~static.index.duplicated(keep="last")]
            self.static_covariates = static.sort_index()

        dynamic = set(self.dynamic_covariate_names)
        unknown = set(self.known_future_covariates) - dynamic
        if unknown:
            raise ValueError(f"known_future_covariates not in dynamic covariates: {sorted(unknown)}")

    @property
    def units(self) -> list[str]:
        return sorted(self.frame["unit_id"].unique().tolist())

    @property
    def times(self) -> list[Any]:
        return safe_sort_values(self.frame["time"])

    @property
    def dynamic_covariate_names(self) -> list[str]:
        return [c for c in self.frame.columns if c not in {"time", "unit_id", "target"}]

    @property
    def target(self) -> pd.DataFrame:
        return self.frame.pivot(index="time", columns="unit_id", values="target").sort_index()

    @property
    def covariates(self) -> pd.DataFrame:
        cols = self.dynamic_covariate_names
        if not cols:
            return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["time", "unit_id"]))
        return self.frame.set_index(["time", "unit_id"])[cols].sort_index()

    def describe(self) -> dict[str, Any]:
        target = self.target
        return {
            "name": self.name,
            "n_units": len(self.units),
            "n_timestamps": len(self.times),
            "n_rows": len(self.frame),
            "start": self.times[0] if self.times else None,
            "end": self.times[-1] if self.times else None,
            "dynamic_covariates": self.dynamic_covariate_names,
            "static_covariates": [] if self.static_covariates is None else list(self.static_covariates.columns),
            "known_future_covariates": list(self.known_future_covariates),
            "target_missing_fraction": float(target.isna().mean().mean()) if not target.empty else 0.0,
            **self.metadata,
        }

    def to_long(self, copy: bool = True) -> pd.DataFrame:
        return self.frame.copy() if copy else self.frame

    def subset(
        self,
        *,
        units: Sequence[str] | None = None,
        start: Any | None = None,
        end: Any | None = None,
        name: str | None = None,
    ) -> "PanelDataset":
        frame = self.frame
        if units is not None:
            wanted = {str(u) for u in units}
            frame = frame[frame["unit_id"].isin(wanted)]
        if start is not None:
            frame = frame[frame["time"] >= start]
        if end is not None:
            frame = frame[frame["time"] <= end]
        static = self.static_covariates
        if static is not None and units is not None:
            static = static.reindex([str(u) for u in units]).dropna(how="all")
        return PanelDataset(
            name=name or self.name,
            frame=frame.copy(),
            static_covariates=None if static is None else static.copy(),
            metadata=self.metadata.copy(),
            known_future_covariates=self.known_future_covariates,
        )

    def split(
        self,
        *,
        train: float = 0.7,
        val: float = 0.1,
        test: float = 0.2,
    ) -> "DatasetSplit":
        times = self.times
        n_train, n_val, n_test = stable_ratio_counts(len(times), train, val, test)
        train_times = times[:n_train]
        val_times = times[n_train:n_train + n_val]
        test_times = times[n_train + n_val:n_train + n_val + n_test]

        def part(ts: list[Any], label: str) -> PanelDataset:
            if not ts:
                empty = self.frame.iloc[:0].copy()
                return PanelDataset(
                    name=f"{self.name}:{label}",
                    frame=empty,
                    static_covariates=self.static_covariates,
                    metadata=self.metadata.copy(),
                    known_future_covariates=self.known_future_covariates,
                )
            mask = self.frame["time"].isin(ts)
            return PanelDataset(
                name=f"{self.name}:{label}",
                frame=self.frame.loc[mask].copy(),
                static_covariates=self.static_covariates,
                metadata=self.metadata.copy(),
                known_future_covariates=self.known_future_covariates,
            )

        return DatasetSplit(
            full=self,
            train=part(train_times, "train"),
            val=part(val_times, "val"),
            test=part(test_times, "test"),
            train_times=tuple(train_times),
            val_times=tuple(val_times),
            test_times=tuple(test_times),
        )

    def to_numpy(
        self,
        *,
        units: Sequence[str] | None = None,
        covariates: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        units = tuple(str(u) for u in (units or self.units))
        times = tuple(self.times)
        target = self.target.reindex(index=times, columns=units).to_numpy(dtype=float)
        cov_names = tuple(covariates if covariates is not None else self.dynamic_covariate_names)

        dyn = None
        if cov_names:
            idx = pd.MultiIndex.from_product([times, units], names=["time", "unit_id"])
            dyn_df = self.frame.set_index(["time", "unit_id"]).reindex(idx)
            dyn = dyn_df[list(cov_names)].to_numpy(dtype=float).reshape(len(times), len(units), len(cov_names))

        static_arr = None
        static_names: tuple[str, ...] = ()
        if self.static_covariates is not None and not self.static_covariates.empty:
            static_names = tuple(self.static_covariates.columns)
            static_arr = self.static_covariates.reindex(units).to_numpy(dtype=float)

        return {
            "times": times,
            "units": units,
            "target": target,
            "dynamic_covariates": dyn,
            "dynamic_covariate_names": cov_names,
            "static_covariates": static_arr,
            "static_covariate_names": static_names,
        }

    def to_torch(self, **kwargs: Any) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install torch support with: pip install -e '.[torch]'") from exc
        out = self.to_numpy(**kwargs)
        for key in ("target", "dynamic_covariates", "static_covariates"):
            if out.get(key) is not None:
                out[key] = torch.as_tensor(out[key], dtype=torch.float32)
        return out

    def rolling_windows(
        self,
        *,
        input_length: int,
        horizon: int,
        step: int = 1,
        units: Sequence[str] | None = None,
        covariates: Sequence[str] | None = None,
        known_future_covariates: Sequence[str] | None = None,
        forecast_start: Any | None = None,
        forecast_end: Any | None = None,
        require_complete_target: bool = True,
    ) -> Iterator[ForecastWindow]:
        if input_length <= 0 or horizon <= 0 or step <= 0:
            raise ValueError("input_length, horizon, and step must be positive integers")

        arrays = self.to_numpy(units=units, covariates=covariates)
        times = arrays["times"]
        unit_tuple = arrays["units"]
        target = arrays["target"]
        dyn = arrays["dynamic_covariates"]
        cov_names = arrays["dynamic_covariate_names"]
        static = arrays["static_covariates"]
        static_names = arrays["static_covariate_names"]

        requested_known = tuple(
            known_future_covariates if known_future_covariates is not None else self.known_future_covariates
        )
        invalid = set(requested_known) - set(cov_names)
        if invalid:
            raise KeyError(f"Unknown known-future covariates: {sorted(invalid)}")
        known_idx = [cov_names.index(c) for c in requested_known]

        total = len(times)
        last_start = total - input_length - horizon
        if last_start < 0:
            return

        for start_idx in range(0, last_start + 1, step):
            cut = start_idx + input_length
            end = cut + horizon
            forecast_times = times[cut:end]
            if forecast_start is not None and forecast_times[0] < forecast_start:
                continue
            if forecast_end is not None and forecast_times[-1] > forecast_end:
                continue
            x_y = target[start_idx:cut]
            y_y = target[cut:end]
            if require_complete_target and (np.isnan(x_y).any() or np.isnan(y_y).any()):
                continue
            future_known = None
            if dyn is not None and known_idx:
                future_known = dyn[cut:end, :, known_idx]
            yield ForecastWindow(
                units=unit_tuple,
                input_times=tuple(times[start_idx:cut]),
                forecast_times=tuple(forecast_times),
                context_target=x_y,
                future_target=y_y,
                context_covariates=None if dyn is None else dyn[start_idx:cut],
                future_known_covariates=future_known,
                static_covariates=static,
                covariate_names=tuple(cov_names),
                known_future_covariate_names=requested_known,
                static_covariate_names=tuple(static_names),
            )


@dataclass(frozen=True)
class DatasetSplit:
    full: PanelDataset
    train: PanelDataset
    val: PanelDataset
    test: PanelDataset
    train_times: tuple[Any, ...]
    val_times: tuple[Any, ...]
    test_times: tuple[Any, ...]

    def windows(
        self,
        part: str,
        *,
        input_length: int,
        horizon: int,
        step: int = 1,
        **kwargs: Any,
    ) -> Iterator[ForecastWindow]:
        """Create windows whose entire forecast horizon lies inside a split part.

        Input context is allowed to come from earlier parts, which is the usual
        forecasting evaluation setup.
        """
        mapping = {"train": self.train_times, "val": self.val_times, "test": self.test_times}
        if part not in mapping:
            raise ValueError("part must be 'train', 'val', or 'test'")
        times = mapping[part]
        if not times:
            return iter(())
        return self.full.rolling_windows(
            input_length=input_length,
            horizon=horizon,
            step=step,
            forecast_start=times[0],
            forecast_end=times[-1],
            **kwargs,
        )
