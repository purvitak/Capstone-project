# Mumbai day-ahead GHI forecasting

The selected pipeline blends **50% XGBoost-corrected archived GFS irradiance**
with **50% saved historical BiLSTM–HGB forecasts**. It issues 144 ten-minute GHI
values for Mumbai at midnight IST.

| Recorded evaluation | Value |
| --- | ---: |
| Daylight RMSE | **101.10356 W/m²** |
| Daylight MAE | **64.76231 W/m²** |
| Evaluation period | 25 July 2025–24 July 2026 |
| Days / daylight samples | 365 / 24,086 |
| NWP coverage / fallback dates | 100% / 0 |

The comparison is exploratory because the evaluation year informed earlier project
iterations. Neither error reached 50 W/m². The small gain over NWP–HGB has paired
confidence intervals including zero.

## Run

Use the existing root environment:

```bash
.venv/bin/python run.py
```

This verifies the frozen inputs, model and predictions and refreshes the report
without retraining or rescoring. `--phase all` also verifies the completed
search. Reference paths resolve automatically after cleanup.

For a fresh installation:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python run.py
```

For a separate experiment:

```bash
.venv/bin/python run.py --phase development --out results/new_xgboost
.venv/bin/python run.py --phase evaluate --out results/new_xgboost
```

Changed settings/data/code require a new output folder. The default runtime budget
is one hour; completed OOF folds resume. The source forecast cache is already included.

## Project layout

```text
run.py                         Main entry point
pipelines/xgboost_nwp/          XGBoost training, experiment runner and model tests
pv_pipeline/                   Shared data, NWP and historical-feature code
data/raw/                      Original Solcast CSV
data/prepared/                 Prepared data and solar climatology
data/external/                 Archived GFS responses and aligned forecast cache
data/references/               Required historical/NWP reference forecasts
results/xgboost_nwp/            Selected model, predictions and evaluation records
results/pipeline_audit/         Diagnostics and recommended improvements
results/verification/           Dataset audit
tests/                         Shared preparation, feature and scoring tests
docs/paper/                    Preserved earlier paper draft and source
```

The dashboard, attention experiment, other superseded runs, old archive material
and generated paper previews have been removed. Git history and the working
`.venv` remain. The deletion/hash inventory is recorded in
`results/xgboost_nwp/cleanup_2026-10-26.json`.

## Code to modify

- [XGBoost training](pipelines/xgboost_nwp/training.py): residual target,
  chronological stopping, refit and prediction.
- [Experiment runner](pipelines/xgboost_nwp/run.py): configurations, features,
  folds, blending, selection and scoring. Invoke through the root wrapper.
- [NWP data](pv_pipeline/nwp_data.py): acquisition, availability and alignment.
- [Preparation](pv_pipeline/00_prepare.py): historical features and solar proxy.

Older-named modules in `pv_pipeline/` remain because the selected pipeline imports
their feature/scoring helpers and checks their frozen source hashes. They are
support dependencies, not additional selected pipelines. Historical neural/blend
helpers also support regression tests and the cached component's provenance.
Do not edit those files and resume the original experiment: use a new output folder.

The historical half remains a BiLSTM–HGB forecast. Removing its cached predictions
would change or break the selected XGBoost **blend**. See
[reference inventory](data/references/README.md).

## Verification

Run Torch/shared and native XGBoost suites in separate processes on this macOS setup:

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m unittest discover -s pipelines/xgboost_nwp -p test_pipeline.py
```

`final_model.json` exports the new residual tree. Complete fresh-day inference
still needs the historical hybrid and input service; the saved arrays currently
support research replay. A new dashboard alone will not create that inference path.

## Reports

- [Selected results](results/xgboost_nwp/REPORT.md)
- [Pipeline details](pipelines/xgboost_nwp/README.md)
- [Audit and next experiments](results/pipeline_audit/REPORT.md)
- [Earlier paper draft](docs/paper/README.md): retained HGB-era draft; revise for
  the XGBoost selection and dashboard removal before submission.

Solcast supplies historical/reference irradiance and weather; GFS forecasts are
accessed through Open-Meteo. The target is satellite/model-derived GHI, not ground
irradiance or measured plant output. Horizon, units and scoring are unchanged.

## Inputs for a new dashboard

Load dates, actual GHI, solar proxy, daylight mask and test indices from
`data/prepared/prep.npz`. Load predicted indices from
`results/xgboost_nwp/xgboost_base_pred.npz` under `kt_base`; reconstruct GHI as
`clip(kt_base, 0, 1.3) * cs_proxy` in W/m². Each day contains 144 ten-minute
periods. Use the test indices for the saved evaluation replay. The selected
XGBoost run does not export calibrated uncertainty bounds or measured PV output.
