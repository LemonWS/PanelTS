# PanelTS API

A forecasting-oriented Python API for `Multiple-Time-Series/PanelTS` on Hugging Face.

The package deliberately reads files from the repository with `huggingface_hub` instead of relying on the Hugging Face Dataset Viewer. It canonicalizes targets and covariates and provides leakage-aware temporal splitting and rolling forecast windows.

## 1. Install

```bash
cd panelts_api
pip install -e .
```

Optional PyTorch support:

```bash
pip install -e ".[torch]"
```

## 2. Basic usage

```python
import panelts

print(panelts.list_datasets())

ds = panelts.load("covid-20")
print(ds.describe())

# Wide target matrix: rows=time, columns=unit
Y = ds.target

# Dynamic covariates indexed by (time, unit_id)
X = ds.covariates

# Static region attributes indexed by unit_id
S = ds.static_covariates
```

## 3. Exchange rate: target + monthly macro covariates

```python
fx = panelts.load(
    "exchange-rate",
    units=["AUD", "CAD", "CNY"],
    target="Close",
    alignment="backward",  # leakage-safe default
)

print(fx.dynamic_covariate_names)
```

The loader matches currency price files to country/region macro files. Daily prices are sorted chronologically. Low-frequency macro data are merged with `merge_asof(..., direction="backward")`, so a target date receives only the latest attribute timestamp at or before that date.

To reproduce a same-calendar-month convention explicitly:

```python
fx = panelts.load("exchange-rate", alignment="same_period")
```

`same_period` can be non-causal if a month-end macro value is assumed known earlier in that same month, so it is not the default.

## 4. COVID-19: dynamic + static covariates

```python
covid = panelts.load("covid-20")

# target = `new`
print(covid.target)

# dynamic fields from each region CSV, e.g. population/active/rate/band
print(covid.covariates)

# separate health/demographic region attributes
print(covid.static_covariates)
```

The same API supports:

```python
covid79 = panelts.load("covid-79")
covid320 = panelts.load("covid-320")
```

## 5. ETF

```python
etf = panelts.load("etf", units=["GLD", "AAAU"], target="Close")
```

Other OHLC/adjusted-close/change/volume fields are retained as dynamic covariates when parsable.

## 6. Stocks + quarterly fundamentals

```python
stocks = panelts.load(
    "stock-healthcare",
    units=["A", "ABBV", "JNJ"],
    target="Close",
    include_fundamentals=True,
    alignment="backward",
)
```

Supported logical stock groups:

- `stock-healthcare`
- `stock-it`
- `stock-sp500`

The repository stores fundamentals transposed (rows are metrics, columns are quarter-end dates). The loader rotates them into `time x metric` form and then aligns them to daily prices.

## 7. Temporal train/validation/test split

```python
split = fx.split(train=0.7, val=0.1, test=0.2)

train = split.train
val = split.val
test = split.test
```

This is chronological. There is no random shuffling.

## 8. Rolling forecast windows

```python
windows = split.windows(
    "test",
    input_length=100,
    horizon=10,
    step=1,
)

for w in windows:
    print(w.context_target.shape)      # [L, N]
    print(w.future_target.shape)       # [H, N]
    print(w.context_covariates.shape)  # [L, N, C]
    break
```

For test evaluation, the forecast horizon is restricted to the test period, but the input context is allowed to come from earlier train/validation timestamps. This avoids throwing away the first `input_length` points of the test set.

## 9. Known-future covariates

By default PanelTS does not automatically claim that observed covariates are known in the future. You can mark only genuinely known-future columns:

```python
from panelts import PanelDataset

# Example only: create a new object after deciding which columns are legitimately known.
fx.known_future_covariates = ("some_calendar_feature",)
```

Then rolling windows expose those future covariates separately in `future_known_covariates`.

## 10. NumPy / PyTorch

```python
arr = fx.to_numpy()
# arr["target"]: [T, N]
# arr["dynamic_covariates"]: [T, N, C]
# arr["static_covariates"]: [N, S] or None

pt = fx.to_torch()
```

## 11. Synthetic datasets

```python
syn = panelts.load("synthetic-y5-x5")
```

Because the large Xet-backed synthetic files are not previewable in the web UI, this loader uses schema inference. If the raw synthetic file uses non-standard column names, pass explicit columns:

```python
syn = panelts.load(
    "synthetic-y5-x5",
    target_columns=["y1", "y2", "y3", "y4", "y5"],
    covariate_columns=["x1", "x2", "x3", "x4", "x5"],
)
```

For wide synthetic files, x-columns are treated as common dynamic covariates and broadcast across y units. If the actual synthetic schema encodes unit-specific x variables, add a project-specific parser instead of silently assuming a mapping.

## 12. Cache and offline reuse

```python
fx = panelts.load(
    "exchange-rate",
    cache_dir="./hf_cache",
)

# Later, after files are cached:
fx = panelts.load(
    "exchange-rate",
    cache_dir="./hf_cache",
    local_files_only=True,
)
```

## 13. CLI

```bash
panelts list
panelts units exchange-rate
panelts info covid-20 --unit "Region 1"
```

## 14. Design choices

Canonical long-form schema:

```text
time | unit_id | target | dynamic_covariate_1 | ...
```

Static attributes live in a separate `unit_id x static_feature` table. This distinction prevents accidental duplication of static features at every timestamp while still allowing conversion to model tensors.

The package intentionally does not call `sklearn.model_selection.train_test_split` because random splitting is inappropriate for forecasting benchmarks.
