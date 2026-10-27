# Solar Energy Forecast Dashboard

Streamlit and Plotly interface connected to a real, read-only FastAPI backend.
It serves the final model's **365 saved daily forecasts** (25 July 2025–24 July 2026),
with 144 ten-minute period means per day. The final model is the NWP-guided
DWT-BiLSTM (GFS + ECMWF + ICON) blended with the XGBoost–NWP / historical forecast
by forecast cloud regime (`pipelines/nwp_dwt_hybrid`, `--nwp-set multi --night-zero`).
It does not fit models, change frozen outputs, or generate forecasts for new dates.

## Install and run

From the repository root:

```bash
uv pip install --python .venv/bin/python -r dashboard/requirements.txt
.venv/bin/python dashboard/run.py
```

Open **http://127.0.0.1:8501**. The launcher starts both processes. Ctrl+C stops both.
Use `--api-port 8001 --ui-port 8502` if the default ports are busy.

To run services separately, in two terminals:

```bash
.venv/bin/python -m uvicorn dashboard.backend.api:app --host 127.0.0.1 --port 8000
.venv/bin/python -m streamlit run dashboard/app.py --server.port 8501
```

The UI uses `SOLAR_API_URL=http://127.0.0.1:8000` by default. For a direct Python
integration with no separate HTTP server:

```bash
SOLAR_DATA_MODE=direct .venv/bin/python -m streamlit run dashboard/app.py
```

Both modes read the same verified outputs; direct mode is not sample mode. There
is no synthetic fallback. Missing data/service failures show a recoverable message.

## Layout and data flow

```text
dashboard/
  app.py                 Streamlit UI, navigation and session state
  charts.py              Plotly irradiance, PV, hourly and overview figures
  data_service.py        get_dates() / get_forecast(): HTTP or direct adapter
  calculations.py        Explicit PV assumptions, energy, rollups and CSV
  backend/
    api.py               Read-only FastAPI routes
    repository.py        Checked saved-forecast loading and GHI reconstruction
  assets/style.css       Navy sidebar, responsive cards and chart panels
  run.py                 Starts/stops API and UI
  requirements.txt       Optional presentation dependencies
  tests/                 Contract, calculation, API and UI checks
```

The repository's `.streamlit/config.toml` supplies the matching light/navy theme.

`app.py → data_service.get_forecast(date, capacity_kwp, performance_ratio)` calls
`GET /api/forecast/YYYY-MM-DD`. The backend checks the SHA-256 fingerprints recorded
in `results/multi_nwp_hybrid/report.json` for `predictions.npz` and
`data/prepared/prep.npz`, confirms the run was adopted and not a smoke test, checks
dates/dimensions, and serves `hybrid_10min` (GHI, W/m²) with the prepared daylight mask.
There is no import of Torch, XGBoost training, or experiment runners in the dashboard.

### Future model/output integration point

Implement the same contract in `ForecastRepository.get_forecast()` (and update
`available_dates()`) when a fresh-day input/inference service is available.
Alternatively point `SOLAR_API_URL` at an API that implements the documented contract.
The UI/charts need no model-specific changes. Fresh-day inference would need the
three NWP downloads, the D−1 history, the trained networks and the default blend's
components; the current backend truthfully serves the saved forecasts.

Routes (interactive schema at **http://127.0.0.1:8000/docs**):

- `GET /api/health`: archive readiness and available-day count.
- `GET /api/dates`: selectable ISO dates and timezone.
- `GET /api/forecast/{date}`: metadata, 144 intervals, day summary, hourly summary.
- `GET /api/forecast/{date}/csv`: optional `hour=0..23` or `interval=0..143` filter.
- Forecast/CSV routes accept paired `capacity_kwp` and `performance_ratio` parameters.

An interval contains timezone-aware `timestamp`, `interval_end`,
`ghi_forecast_wm2`, optional `pv_power_kw`, zero-based `interval_index`, local
`hour`, and `daylight`. Source metadata explicitly says `sample=false`, `live=false`.
Unavailable dates return 404, incompatible files 503, invalid parameters 422.
No raw datasets or model parameters are served.

## PV and energy rules

PV parameters start empty. No capacity, efficiency or performance ratio is assumed.
After entering BOTH DC capacity and performance ratio in the sidebar:

```text
Estimated power (kW) = capacity (kWp) × GHI (W/m²) / 1000 × performance ratio
Energy (kWh) = sum(ten-minute mean power in kW) × (10/60 hours)
```

This is an **illustrative GHI-based estimate**, using GHI as a plane-of-array proxy.
It does not model site-specific tilt, temperature, inverter clipping or measured
plant output. Energy is not calculated across missing/invalid periods. CSV omits
the PV column until the parameters are provided. GHI is always shown in W/m².
Hourly GHI/PV are means; hourly energy is a sum over six period energies.

## Interactions

Choose a saved date; use Day → Hour → 10-minute navigation. Click an hourly bar or
use the hour selector; previous/next controls respect day boundaries. The interval
inspector supplies a reliable dropdown alternative to chart selection. Back/reset
returns to the full day. Every card, chart and CSV respects the selected scope.
The day overview keeps the selected hour visible in its daily context. Refresh
reloads data through the service and never retrains the model.

## Verification

```bash
uv pip install --python .venv/bin/python httpx
.venv/bin/python -m unittest discover -s dashboard/tests
```

For the preserved modelling suites, use the separate processes documented in the
root README. `verification.json` records the dashboard checks and browser smoke test.
The National Power Portal dashboard supplied layout inspiration; no content or code
from it is copied. The old pipeline-comparison pages and model-training metrics are
deliberately absent from the product interface.

Optional browser check, against the running services:

```bash
uv pip install --python .venv/bin/python playwright
.venv/bin/python dashboard/verify_browser.py
```

The browser script uses the installed macOS Chrome by default; pass `--chrome`
with a Chromium executable path on other systems. Desktop/interval screenshots
use **test-entered** 10 kWp and 0.8 performance ratio, explicitly illustrative;
those values are never product defaults. GHI curves are the actual saved forecasts.
The mobile preview has the PV fields cleared again.
