import numpy as np
import pandas as pd

from panelts.core import PanelDataset


def make_panel():
    times = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for t_idx, t in enumerate(times):
        for u_idx, unit in enumerate(["A", "B"]):
            rows.append({
                "time": t,
                "unit_id": unit,
                "target": t_idx + u_idx,
                "x1": 10 * t_idx + u_idx,
            })
    static = pd.DataFrame({"unit_id": ["A", "B"], "size": [1.0, 2.0]}).set_index("unit_id")
    return PanelDataset("toy", pd.DataFrame(rows), static_covariates=static, known_future_covariates=("x1",))


def test_split_is_temporal():
    ds = make_panel()
    split = ds.split(train=0.6, val=0.2, test=0.2)
    assert split.train.times[-1] < split.val.times[0] < split.test.times[0]
    assert len(split.train.times) == 6
    assert len(split.val.times) == 2
    assert len(split.test.times) == 2


def test_test_windows_can_use_previous_context():
    ds = make_panel()
    split = ds.split(train=0.6, val=0.2, test=0.2)
    windows = list(split.windows("test", input_length=3, horizon=1))
    assert len(windows) == 2
    assert windows[0].input_times[-1] < windows[0].forecast_times[0]
    assert windows[0].future_known_covariates.shape == (1, 2, 1)
    assert windows[0].static_covariates.shape == (2, 1)


def test_numpy_shapes():
    ds = make_panel()
    arr = ds.to_numpy()
    assert arr["target"].shape == (10, 2)
    assert arr["dynamic_covariates"].shape == (10, 2, 1)
    assert arr["static_covariates"].shape == (2, 1)
