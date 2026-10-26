# Selected XGBoost–NWP pipeline

Run from the project root:

```bash
.venv/bin/python run.py
```

The root wrapper supplies the required reference directories under
`data/references/` to the unchanged, fingerprinted trainer. Invoke through this
entry point after cleanup. Defaults verify the completed evaluation without
refitting or rescoring. For a new experiment, use `--phase development --out
results/new_xgboost`, then `--phase evaluate --out results/new_xgboost`.

## Model

Historical Solcast and known solar/calendar features combine with cached archived
GFS forecasts. XGBoost learns daylight GHI residuals relative to NWP, restores the
baseline, applies the existing solar-index limits and blends with saved historical
BiLSTM–HGB predictions. The selected blend uses 50% of each component. It forecasts
144 ten-minute GHI values at midnight IST.

| Method | Daylight RMSE (W/m²) | Daylight MAE (W/m²) |
| --- | ---: | ---: |
| NWP–HGB reference | 101.68906 | 65.29546 |
| Selected XGBoost blend | **101.10356** | **64.76231** |

All scores use the same 365 days and 24,086 daylight samples. Both chronological
development checks passed. Both full-test paired confidence intervals include
zero, so the small observed gain is inconclusive. The year has been reused and
results remain exploratory. Neither error reached 50 W/m².

Final XGBoost settings: residual target, squared-error loss, histogram trees,
loss-guided growth, maximum 15 leaves, minimum child weight 40, learning rate 0.05,
L2 regularization 1, seed 42, two CPU threads and **62 selected rounds out of 300**.
The final residual tree refits on 550 complete pre-test days.

## Implementation

- `training.py`: residual fitting, chronological stopping and prediction.
- `run.py`: four candidate configurations, OOF predictions, blending, frozen guard
  and evaluation. Its original default reference paths are superseded by the root
  wrapper; keep its source unchanged when validating the original run.
- `test_pipeline.py`: nine optional backend tests. Run separately from Torch tests.

The search begins after 180 complete days, uses expanding 90-day blocks, and
purges one day. Each candidate checks at most 300 rounds on trailing inner
validation, then refits. First 214 validation dates choose the configuration and
blend; the remaining 110 check the choice. Missing NWP uses the historical
corrected forecast for the whole date and keeps it in evaluation.

## Dependencies and outputs

Install from the root `requirements.txt`; XGBoost is now a main dependency.
`data/references/historical_hybrid/` provides saved historical blend/fallback
forecasts. `data/references/nwp_hgb/` provides the frozen incumbent comparison and
hash-verified predecessor artifacts. Those files are necessary even though older
experiment output folders were removed. See the reference README for provenance.

Results remain in `results/xgboost_nwp/`, including all candidate OOF records,
`final_model.json`, `xgboost_base_pred.npz`, frozen decisions and bootstrap reports.
The final tree export does not include a deployable historical-hybrid service;
complete fresh-day inference still needs that component and feature preparation.

The dashboard and attention experiment have been deleted. The earlier paper draft
is retained as research material and needs updating before submission.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m unittest discover -s pipelines/xgboost_nwp -p test_pipeline.py
```

Keep these in separate Python processes on this macOS environment to avoid mixed
Torch/XGBoost native OpenMP conflicts. Do not add the XGBoost tests to the shared
Torch discovery directory.
