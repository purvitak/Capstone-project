# GFS-guided DWT-BiLSTM with cloud-regime blending

Extends the Bektaş & Altaş (2024) DWT-BiLSTM (see `../paper_reproduction/`). The paper
takes its forecast-day weather type from day-D observations; this model replaces it
with issue-time-available GFS information and blends the result with the current default.

```bash
.venv/bin/python pipelines/nwp_dwt_hybrid/run.py   # requires results/paper_reproduction/ first
.venv/bin/python -m unittest discover -s pipelines/nwp_dwt_hybrid -p test_hybrid.py
```

## Method

- **Expert (24×18 hourly input):** paper's seven dmey components of D−1 and weather
  type D−1; GFS day-D hourly clear-sky index, cloud cover, daily clearness,
  temperature, humidity and wind; clear-sky proxy; observed D−1 weather.
  ReLU BiLSTM (50 cells), paper training settings, mean of seeds 42/43/44, hourly GHI target.
- **Training data:** GFS-available development days only (2024-01-21 → 2025-07-23, 550 days).
  Expanding folds: 180 initial days, 90-day blocks, one-day purge.
- **Cloud-regime blend:** forecast daylight cloud cover < 30 %, 30–70 %, ≥ 70 % → one
  weight per regime on the expert, grid 0–1 in 0.1 steps, chosen on ranking days
  (2024-07-20 → 2025-04-04) against the default's out-of-fold forecasts.
- **Adoption rule:** hourly daylight RMSE and MAE must both beat the default on the
  ranking and guard days (2025-04-05 → 2025-07-23). Then the test year is scored once.
- **Ten-minute output:** hourly expert values are spread over six slots by the
  clear-sky shape (preserving the hourly mean, capped at 1.3 × clear sky), then blended.

## Three-model version (`--nwp-set multi --night-zero`)

```bash
.venv/bin/python pipelines/multi_nwp/fetch.py --model ecmwf_ifs025   # once
.venv/bin/python pipelines/multi_nwp/fetch.py --model icon_global    # once
.venv/bin/python pipelines/nwp_dwt_hybrid/run.py --nwp-set multi --night-zero
```

- Adds ECMWF IFS 0.25° and DWD ICON day-2 forecasts (same Open-Meteo previous-runs
  protocol as GFS) as hourly clear-sky index, cloud cover and daily clearness: 24×24 input.
- A model missing on a day is replaced by the mean of the models available that day
  (ECMWF starts 2024-03-08; test coverage GFS 365, ECMWF 358, ICON 361 days).
- Night rule, declared before scoring: the expert's output is non-negative and zero
  when the hourly clear-sky proxy is below 10 W/m².
- Regime cloud cover is the mean of the three models. Outputs: `results/multi_nwp_hybrid/`.
- With `--nwp-set gfs` (default) the input layout and cached results are unchanged.

## Outputs (`results/nwp_dwt_hybrid/`)

`REPORT.md`, `report.json`, `frozen_selection.json`, `metrics.csv`, `predictions.npz`,
`folds/` (cached fold and final forecasts; reused when settings are unchanged).
The test year influenced earlier project iterations, so results are exploratory.
