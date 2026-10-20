# Dashboard scope

User references:

- [WRLDC reports](https://wrldc.in/content/2_1_Reports.aspx): report structure/content.
- [Solcast solar forecast data](https://solcast.com/forecast-solar-irradiance-data): additional presentation reference.

Implemented content:

- Mumbai site details, date selection, and forecast issue time (00:00 IST).
- Actual versus predicted GHI across 144 ten-minute periods, in W/m².
- Optional corrected forecasts and empirical prediction intervals.
- Estimated PV power from actual and predicted GHI, in kW.
- Peak GHI, peak PV power, daily energy (kWh), and forecast-error summaries.
- Report table and CSV downloads.

## Data contract

`data/prepared/prep.npz` supplies `dates`, `ghi`, `cs_proxy`, and `mask`.
`results/ghi_search/selected_base_pred.npz` supplies the latest hybrid's `kt_base`.
`results/ghi_search/selected_stage2.npz` supplies `kt_final`, `kt_lo`, and `kt_hi`.
The original run in `results/dwt_10min_150_paper/` remains the UI's preferred
default; choose `ghi_search/selected` to view the later hybrid. The backend loader
also retains its legacy `15epochs` default for callers that omit a source path.

Convert predicted kt to GHI by multiplying by aligned `cs_proxy`. Align files by
dates and validate shapes. Base forecasts are NaN where no out-of-sample forecast
exists; corrected forecasts/intervals are available on validation/test slices.
The implemented Streamlit dashboard replays the held-out test year. It detects
saved base and corrected prediction files; only complete test days are selectable.
The corrector stores NaN outside scored daylight slots; the dashboard uses the
base forecast in those slots so the displayed corrected curve has all 144 steps.

Actual GHI is a Solcast satellite estimate, not a ground measurement. Future-day
actuals become available as the day progresses. Calculated PV is an estimate,
not measured plant output.

## PV calculation inside the dashboard

PV power is calculated on demand after the dashboard loads predicted GHI.
It is not a separate forecasting/training stage and does not change the GHI model.
The reusable Python helper is `pv_pipeline/pv_model.py`; the dashboard calls it
when users change capacity or temperature settings.

Adopt the NOCT temperature-adjusted model from Shanker & Kulkarni (2026),
DOI `10.1016/j.nxener.2026.100601`, section 3.3, equations 5–6:

`T_cell = T_air + ((NOCT - 20) / 800) × GHI`

`P_kW = max(0, capacity_kW × (GHI / 1000) × (1 - beta × (T_cell - 25)))`

Paper defaults: NOCT 45°C, beta 0.0045 per °C, capacity 100 kW. Capacity should
be adjustable. This is a horizontal-irradiance PV estimate; it does not model
tilt, shading, inverter clipping or wiring losses. Do not label it measured PV.
For complete ten-minute profiles, daily energy is `sum(P_kW) / 6` kWh.

Forecast PV must use a forecast-time temperature assumption. The helper offers
the same-clock-time median of the preceding three days; never substitute the
forecast day's observed temperature. A simulated historical reference curve may
use observed GHI and temperature, labelled separately. Missing steps make daily
energy incomplete, rather than silently counting as zero.

Flow: saved GHI prediction → dashboard selects date/settings → NOCT calculation
→ estimated PV chart and daily energy. No separate PV-processing CLI is needed.

## New daily forecasts

The current trainer saves prediction arrays, not model weights. New forecasts
require model saving/loading, consistent preprocessing, an inference entry point,
and up-to-date historical inputs.
