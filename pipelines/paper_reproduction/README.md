# Bektaş & Altaş (2024) reproduction on Mumbai data

Base paper: S. Ç. Bektaş and I. H. Altaş, "DWT-BILSTM-based models for day-ahead
hourly global horizontal solar irradiance forecasting", Neural Computing and
Applications 36:13243–13253 (2024), DOI 10.1007/s00521-024-09701-2.

Purpose: a same-data comparison. The paper's Model 1 and Model 2 are trained and
scored on the project's Mumbai data, test year and hourly scoring masks, beside the
current default forecast. Published Trabzon numbers remain context only.

```bash
.venv/bin/python pipelines/paper_reproduction/run.py            # full run → results/paper_reproduction/
.venv/bin/python -m unittest discover -s pipelines/paper_reproduction -p test_reproduction.py
```

Trained networks are checkpointed under `results/paper_reproduction/models/`; a
rerun with unchanged settings reuses them. Smoke runs (`--smoke`) need a separate `--out`.

## As published

| Item | Setting |
| --- | --- |
| Resolution | hourly, 24 inputs → 24 outputs |
| Wavelet | discrete Meyer (`dmey`), 6 levels: A6 + D6…D1 (Eq. 8) |
| Weather type | K = daily GHI / extraterrestrial irradiation; rainy < 0.25 < cloudy < 0.45 < sunny |
| Model 1 input | 7 DWT components of D−1, weather type D−1 and D (24×9) |
| Model 2 input | adds temperature, humidity, wind of D−1 and D (24×15) |
| Network | BiLSTM, ReLU activation, per-hour dense output, MSE, Adam, batch 32 |
| Training | ≤ 500 epochs, early stopping patience 30, cells chosen from 30/50/100 on validation MSE |
| Split | development days split chronologically 80% training / 20% validation |

## Departures (all documented in code)

- **Causal wavelet window.** Components for D−1 come from a trailing 168-day window
  ending at the last hour of D−1 (the shortest window permitting 6 dmey levels).
  Decomposing the full series, as the paper appears to, would leak later observations.
- **Eq. (4) coefficient.** The paper prints 0.33; the standard 0.033 is used.
- **dmey reconstruction.** PyWavelets' dmey is a finite approximation; components
  sum to the wavelet reconstruction, which differs from the input by about 1% RMS.
- **Best weights restored** after early stopping.
- **Test split.** The project's fixed 365-day test year replaces the paper's 70/30 split.

## Forecast-day input variants (prediction time only)

The paper derives the forecast-day weather type (and Model 2's forecast-day weather)
from that day's observations. Archived GFS forecasts start in January 2024, so each
network is trained exactly as published and the forecast-day inputs are swapped at
prediction time:

| Variant | Forecast-day weather type and weather |
| --- | --- |
| `oracle` | day-D observations (published setting; not available at issue time) |
| `gfs_raw` | archived GFS day-2 forecast (wind km/h → m/s) |
| `gfs_calibrated` | GFS after quantile mapping fitted on pre-test development days |
| `persistence` | copied from D−1 |

Only the `gfs_*` and `persistence` variants are operational forecasts. Tests check
that their inputs do not change when day-D observations are perturbed.

## Outputs (`results/paper_reproduction/`)

`REPORT.md`, `report.json` (training, calibration, weather-type agreement, paired
bootstrap of default minus paper), `metrics.csv`, `predictions.npz`, `models/`.

The test year has influenced earlier project iterations, so results are exploratory.
