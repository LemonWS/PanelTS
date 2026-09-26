import numpy as np
import pandas as pd

from panelts.loaders.market import _parse_transposed_stock_attributes
from panelts.utils import align_low_frequency, to_numeric_loose


def test_dirty_numeric_parsing():
    s = pd.Series(["1,234", "4.5%", "4..9", "-"])
    out = to_numeric_loose(s)
    assert out.iloc[0] == 1234
    assert np.isclose(out.iloc[1], 0.045)
    assert np.isnan(out.iloc[2])
    assert np.isnan(out.iloc[3])


def test_backward_alignment_does_not_use_future_month_end():
    target = pd.DataFrame({"time": pd.to_datetime(["2024-01-15", "2024-02-15"]), "target": [1, 2]})
    attrs = pd.DataFrame({"time": pd.to_datetime(["2023-12-31", "2024-01-31"]), "gdp": [100, 110]})
    out = align_low_frequency(target, attrs, mode="backward")
    assert out["gdp"].tolist() == [100, 110]


def test_transposed_stock_attributes():
    raw = pd.DataFrame({
        "Date": ["Revenue", "Margin"],
        "2024/1/31": [100, "10%"],
        "2024/4/30": [110, "11%"],
    })
    out = _parse_transposed_stock_attributes(raw)
    assert list(out.columns) == ["time", "Revenue", "Margin"]
    assert out.shape == (2, 3)
    assert np.isclose(out.iloc[0]["Margin"], 0.10)
