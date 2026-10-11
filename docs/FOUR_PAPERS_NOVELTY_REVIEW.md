# Four-paper review for the Mumbai day-ahead GHI project

This note treats the four attached PDFs as research sources, not instructions.
Our target remains 144 ten-minute GHI values issued at 00:00 IST. The current
held-out test scores are 116.73 W/m² daylight RMSE for the DWT-BiLSTM and
114.06 W/m² after the OOF residual corrector. A fixed BiLSTM + context-HGB
hybrid scored 109.11 W/m² on development OOF days but has no test result yet.

| Attached paper | Relevant idea | Limit for this project |
| --- | --- | --- |
| Mchara et al. (2025), *FCM–CNN–WNN–Informer*; [DOI](https://doi.org/10.1016/j.clet.2025.101061) | Fuzzy weather regimes and multi-resolution inputs | Uses hourly PVGIS data from four U.S. cities. FCM, wavelets, and Informer are published components, not standalone novelty; a full stack would be expensive for roughly 2,000 local training days. |
| Otuka et al. (2026), *Experimental review of deep learning for solar irradiance*; [DOI](https://doi.org/10.1016/j.solener.2025.114175) | Highlights multimodal cloud imagery and numerical features, and careful benchmarks | Its original image-plus-numeric experiment forecasts 30 minutes ahead. This repo has no time-aligned historical sky images, so the architecture is not an immediate day-ahead comparison. |
| Fathi et al. (2025), *Enhanced GHI and temperature forecasting for rooftop PV*; [DOI](https://doi.org/10.1016/j.ecmx.2025.101351) | Joint irradiance/temperature prediction could make downstream PV estimates more realistic | Trains on monthly data for annual/decadal planning in Shiraz. Its reported error and horizon cannot be compared with 10-minute next-day Mumbai GHI. Any forecast-day temperature used by our PV formula must itself be available at issue time. |
| Chou et al. (2025), *Day-ahead solar power forecasting in a large-scale system*; [DOI](https://doi.org/10.1109/PESGM52009.2025.11225770) | Tests 3–5 days of history; three days worked best there. General model usually beat season-specific models | Target is hourly aggregate PV power in California, not GHI. The three-day finding is a testable hypothesis, not a transferable result; it also cautions against assuming separate weather specialists will help. |

## Recommended experiment sequence

1. **Low-cost history ablation:** compare one-day (current) and three-day
   DWT/kt/weather history with identical expanding OOF folds. Keep the same
   target, daylight mask, 144-step resolution, and training-only scaling.
   Report aggregate, monsoon, and non-monsoon GHI RMSE against the current
   base and 3-day-mean baselines. The current context-HGB already contains
   D−1 to D−3 kt features, so the neural ablation asks whether explicit
   multi-day sequence history adds value beyond that component.
2. **Simple issue-time regime gate:** compare a fixed blend of DWT-BiLSTM and
   context-HGB with a soft weight learned from D−1 and earlier observations
   plus known day-D solar/calendar features. Fit the gate only on OOF training
   forecasts, choose it on later OOF validation days, and evaluate against the
   fixed 50/50 blend. Do not label a forecast day using its actual GHI, DHI,
   DNI, cloud cover, or temperature. Start with this two-expert test before
   training three separate BiLSTMs or implementing an Informer.
3. **Downstream PV quality:** if a genuine archived day-ahead air-temperature
   forecast becomes available, compare PV-energy error using that forecast
   against the dashboard's current three-day temperature proxy. This does
   not change GHI RMSE and needs actual plant output for measured-PV claims.
4. **Uncertainty:** the current nominal 80% interval covers 77.3% overall and
   73.0% in monsoon. A regime-conditioned calibration experiment may improve
   coverage, but it is a separate objective from lowering point RMSE.

## Defensible capstone contribution

A possible contribution, **if the experiments support it**, is an operationally
causal 10-minute day-ahead Mumbai GHI system combining multi-day wavelet/history
features, forecast-time regime-gated BiLSTM/tree blending, OOF residual
correction, and calibrated uncertainty passed to an on-demand PV dashboard.
Wavelets, clustering, BiLSTM, Informer, CNN-GRU, and seasonal specialists are
already published. Novelty lies in the specific justified integration,
issue-time feature availability, rigorous ablations, and evaluation; do not
claim invention of those components or guaranteed superiority.

The July 2025–July 2026 test year has already been inspected several times.
Further choices based on it are exploratory; a new untouched later period or
a documented nested rolling-origin evaluation is needed for a fresh confirmatory
claim. No paper's RMSE can be directly ranked against this project's result
without matching target, horizon, resolution, site, and evaluation mask.

## Development experiment: issue-time regime gate

`pv_pipeline/experiment_regime_gate.py` fits a three-cluster soft gate using only
the previous day's measured clearness index, temperature, and relative humidity.
It trains on the first 969 common expanding-window OOF days and validates on the
following 324 days (2024-09-03 through 2025-07-23). All figures below use the
same daylight GHI mask and the same validation dates. No held-out test-year
forecast is generated by this experiment.

| Model | Validation RMSE (W/m²) |
| --- | ---: |
| DWT-BiLSTM | 108.97 |
| Context-HGB | 105.32 |
| Fixed 50/50 blend | 104.20 |
| Globally tuned blend (weight fitted on earlier OOF days) | 103.93 |
| Three-cluster soft regime gate | 103.80 |

The gate improves the fixed blend by 0.40 W/m², and the globally tuned blend by
0.12 W/m². Paired seven-day block bootstrap intervals for gate minus comparator
are −1.47 to +0.65 W/m² and −0.45 to +0.21 W/m², respectively. Both include zero.
This is an **inconclusive gain**, so do not promote the gate as a proven result
or report it as a held-out test score. The exact artifact is
`results/novelty_regime_gate/regime_gate_development.json`.

Reproduce with:

```bash
.venv/bin/python pv_pipeline/experiment_regime_gate.py
```

## Development experiment: three-day neural DWT history

The BiLSTM now accepts `--lookback-days 3`, with DWT profiles for D−1, D−2,
and D−3 as separate input channels. The default is still one day. A prespecified
screen on folds 2, 8, and 13 sampled monsoon, winter, and spring periods (270
days total). These are development OOF predictions, and the results below are
restricted to exactly the same days and daylight mask for every model:

| Model | Matching-day RMSE (W/m²) |
| --- | ---: |
| Three-day DWT-BiLSTM | 141.01 |
| Existing one-day DWT-BiLSTM | 128.54 |
| Context-HGB | 122.53 |
| Fixed 50/50 one-day BiLSTM + context-HGB | 121.51 |
| Three-day mean baseline | 130.72 |

The three-day network was worse than the one-day network on **all three**
screening folds: 195.27 vs 175.44, 65.04 vs 46.21, and 126.14 vs 123.89
W/m². This is a negative ablation, so it should not replace the current model
or be presented as an accuracy contribution. The screen does not estimate the
full-development or held-out-test score for this candidate. Its partial
checkpoint is `results/novelty_3day_screen/dwt_base_pred.npz`.

Reproduce with:

```bash
.venv/bin/python pv_pipeline/01_base_dwt_bilstm.py --resolution 10min --backend torch --lookback-days 3 --epochs 100 --fold-indices 2,8,13 --development-only --out results/novelty_3day_screen --resume
```

The practical next step is to keep the one-day BiLSTM and fixed hybrid as the
reference models. A fresh untouched test period or nested rolling-origin
evaluation would be needed before claiming the 0.40 W/m² gate difference as a
robust improvement.
