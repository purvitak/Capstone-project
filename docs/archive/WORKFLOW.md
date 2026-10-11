> Historical document preserved from the previous layout. Some scripts and
> results mentioned below are absent. Use [current run instructions](../../README.md)
> and [dashboard scope](../DASHBOARD.md) for the reorganized project.

# Day-Ahead GHI → PV Power Forecasting for Mumbai — Final Workflow

**Version:** v4 (protocol + GBM measurements). **Written:** 1 Oct 2026.
**Paper direction (Oct 2026):** see `RESEARCH.md`. Base paper is Bektas & Altas
(2024) DWT-BiLSTM. **NeuralProphet is retired** — skip §8b / NP Kaggle notes here.
**Supersedes:** `PIPELINE_WORKFLOW_v3.md`, `PIPELINE_WORKFLOW_v2.md`, `PROJECT_PLAN.md`, `VERIFICATION_STATUS.md`.
**Dataset:** `solcast_mumbai_merged_2020_2026.csv` (2020-01-01 → 2026-07-25, 10-minute).

This file is self-contained. If a chat session is exhausted, upload this file plus
`pv_pipeline/common.py` and the CSV, and say: *"Continue from WORKFLOW.md §12."*

---

## 0. One-paragraph summary

Forecast next-day GHI for Mumbai at 10-minute resolution from Solcast history only
(no paid NWP, no archived forecasts), issued at 00:00 IST for the 144 steps of day D.
A gradient-boosting base model predicts the clear-sky index kt; a second
out-of-fold residual model applies a bounded correction; empirical quantiles give
calibrated 80% intervals; the GHI forecast is converted to simulated PV power
through a physical chain; a Streamlit dashboard replays the held-out year.
The contribution is the leak-safe protocol, the honest residual ablation, calibrated
intervals and the dashboard. **Measured outcome (§14a): +11.43% RMSE skill over a
3-day-mean baseline, significant under a moving-block bootstrap and audited against
leakage (§14b); the Stage 2 residual correction is a null result.** The earlier 2-8%
expectation was a prior guess and has been superseded by measurement.

---

## 1. Verified data facts (measured 1 Oct 2026 on the merged file)

Produced by `pv_pipeline/verify_dataset.py`; raw output in
`pv_pipeline/results/verification.json`. **13 VERIFIED, 3 CHANGED, 0 FAILED.**

| Fact | Measured value |
|---|---|
| Rows | 345,168 — exactly the expected 10-minute grid |
| Duplicates / missing timestamps / off-grid deltas | 0 / 0 / 0 |
| Complete IST days | 2,396 (2020-01-02 → 2026-07-24) |
| Partial days dropped | 2020-01-01, 2026-07-25 |
| Merge seam (2023-12-01/02) | 288 rows, continuous; no column empty or constant in only one era |
| Recovered site from zenith least-squares | lat 19.0883, lon 72.8798 — 1.4 km from nominal Mumbai |
| Solar-position accuracy | **0.388° zenith RMSE** at period **mid-points** |
| Timestamp convention test | mid-point 0.388° vs period-start 1.083° vs period-end 1.083° → mid-point confirmed |
| Azimuth convention | `(-az) % 360`, **2.20° RMSE** on daylight steps |
| kt = 1.000 pile-up | 51.8% of daytime rows exactly 1.000; **0.00% above 1.0** (right-censored at Solcast's cap) |
| kt pile-up by year | 47.2% (2021) → 60.9% (2026 partial) |
| Annual mean kt spread (all days) | 0.0845 — dominated by monsoon variability, not bias |
| Annual mean kt spread (clear-sky days only) | **0.0157** → the two source exports are homogeneous |
| `cloud_opacity` vs kt correlation | **−1.000** |
| DNI·cos(z)+DHI vs GHI | median abs. residual 0.45 W/m², p95 3.06 W/m² |
| Day-of-year climatology support | median **6 other years** per day (was ~1 on the 2023-2026 file alone) |

### Changes from v3 that matter

1. **`air_temp` is now float64**, not int64 — the 2020-2023 export carries 0.1 °C
   precision. `ghi`, `clearsky_ghi`, `zenith`, `azimuth` remain integer-valued.
   Harmless, but do not assert int dtypes in code.
2. **The day-of-year clear-sky climatology is no longer handicapped** (6 other years
   of support). v3 §3 predicted this; it is now true, and `clim_ratio` is the best
   non-leaky proxy.
3. **The monsoon seasonal split is now viable.** v3 §12 warned that monsoon `δ_max`
   would fall back to the pooled value. The validation window now holds 175 monsoon
   days, so per-season tuning is real.
4. **The test year now contains a complete monsoon.** Test = 2025-07-25 → 2026-07-24
   covers Aug-Sep 2025 *and* Jun-Jul 2026. The v3 limitation "August-September are
   not represented" **no longer applies** — delete that caveat from the write-up.
5. **One bug found and fixed in `common.py`:** the azimuth denominator used
   `cos(zenith)` where NOAA uses `sin(zenith)`, producing 79.7° RMSE. Now 2.20°.
   This matters because azimuth feeds the PV transposition model.

### Baselines on the new 365-day test year (daytime, 10-min, W/m²)

| Clear-sky proxy | Prev-day kt (all / monsoon / non) | 3-day-mean kt (all / monsoon / non) |
|---|---|---|
| `lagged` (7-day per-step median) | 123.9 / 173.6 / 85.2 | 121.5 / 173.9 / 79.1 |
| `clim` (day-of-year ±7, other years) | 123.7 / 171.5 / 86.9 | 121.4 / 171.2 / 82.1 |
| `clim_ratio` (clim × lagged ratio) | 123.1 / 173.2 / 83.8 | **120.9 / 173.7 / 78.0** |
| `solcast` — **LEAKY, reference only** | 117.7 / 171.8 / 72.6 | 114.3 / 170.6 / 65.6 |

Using Solcast's same-day clear-sky flatters the baseline by **7.2 W/m²**. Never an input.
These numbers sit within ~2 W/m² of v3 §3's measurements on the shorter file, so the
**~2% skill expectation stands**. Do not promise more.

---

## 2. Hard constraints (violating any one invalidates the result)

1. No paid NWP, no archived forecasts. Inputs are Solcast history only.
2. Forecast issued **00:00 IST** for the 144 steps of day D. Permitted inputs:
   observations up to **23:50 of D−1**, plus quantities known in advance for day D
   (astronomy, proxy clear-sky, calendar).
3. **Never** use day-D observed GHI, temperature, wind, or Solcast day-D clear-sky as an input.
4. Every learned component is fit on development data only. The test year stays untouched until the end.
5. Stage 2 trains **only** on out-of-fold Stage-1 forecasts.
6. Hyper-parameters, `δ_max`, interval quantiles and the clear-sky mode are chosen on
   out-of-fold / validation data, never on test.
7. No shuffling of days anywhere.
8. Metrics on **daylight steps only**, in W/m² (kW for PV), 10-minute native. Report
   hourly-aggregated RMSE as a secondary column for literature comparison.
9. Exclude `cloud_opacity`, `dni`, `dhi`, `gti`, `clearsky_gti` as inputs — measured
   above to be deterministic functions of the target or of kt.
10. State in the paper: targets are Solcast satellite-derived estimates (not ground
    truth); PV is a simulated reference system (no plant data).

---

## 3. Protocol

- **Day:** IST calendar day, 144 steps; `step = (hour*60 + minute) // 10` of the period **start**.
- **Solar position:** computed from `common.solar_position` at the period **mid-point**, not read from the CSV.
- **Target:** `kt = GHI / CS_proxy`, clipped to [0, 1.3]. Forecast GHI = `kt̂ × CS_proxy`.
- **Daylight mask:** `zenith < 85°` **and** `CS_proxy > 50 W/m²` — both known in advance.
- **Usable day:** the 7 preceding complete days exist.
- **Split (measured):** usable days 2,389 → development **2,023 days (2020-01-09 → 2025-07-23)**,
  1-day purge gap, test **365 days (2025-07-25 → 2026-07-24)**.
- **Validation inside development:** last 25% chronologically = 506 days, 175 of them monsoon.
- **Rolling-origin OOF over development days:** expanding window, 1-day purge,
  initial train 730 days, folds of 90 days → ~15 folds. Base predictions are OOF for
  development days; the final base model (trained on all development days) predicts the
  purge day and the test year.
- **Memory feature:** yesterday's error profile = `kt(D−1) − base forecast issued for D−1`,
  always from an out-of-sample base prediction.

### Three leak tests that must pass on every run

1. Blank all actuals of day D and later → every Stage-0 input for day D is unchanged.
2. Recompute the clear-sky climatology with all days ≥ (test start − purge) blanked →
   test-day clear-sky profiles are unchanged.
3. Perturb the actuals of test day D with random values → Stage-1 and Stage-2
   predictions for day D are unchanged.

v2's "leak self-test" asserted that test logits fall outside the train range; on this
data 0 of them do, so that assertion fails by construction. It is replaced by the three above.

---

## 4. Environment

### Local / agent environment (recommended for everything except NeuralProphet)

```bash
uv venv --python 3.11
uv pip install numpy pandas scipy scikit-learn matplotlib statsmodels pyarrow
uv pip install pvlib streamlit            # stages 4 and 5
```

Verified working: numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, statsmodels 0.15.0.
Runtime for the whole pipeline excluding NeuralProphet: **a few minutes**. The data is
345k rows; `HistGradientBoostingRegressor` is CPU-only in scikit-learn, so **a GPU
cannot help the main path.**

Agent-sandbox notes (learned the hard way):
- Writes inside `.venv` may need elevated permissions; create and install the venv
  outside the sandbox if `uv pip install` reports `Operation not permitted`.
- `~/Downloads` and `~/Desktop` may not be listable by the shell under macOS privacy
  controls even when individual files are readable. Keep the CSV inside the project folder.

### Kaggle (NeuralProphet + residual correction — §8b)

This is the novelty path. **Do not install neuralprophet into the local `.venv`.**

1. Upload a dataset containing `common.py`, `01_base_neuralprophet.py`,
   `02_corrector.py`, `03_evaluate.py`, and `results/prep.npz`.
2. New notebook: Environment **Original** (not Latest), GPU **T4**, Internet **ON**.
   Use `kaggle_neuralprophet.ipynb` or follow `KAGGLE_NEURALPROPHET.md`.
   NeuralProphet 0.9.0 needs NumPy 1.26 / Lightning <2.4 / SciPy <1.16; Latest
   breaks with `numpy.rec`. Install only the pinned stack in that file, then restart.
4. `--smoke` must print that the day-ahead pick differs from `yhat1`. Stop if it does not.
5. Full OOF run (checkpoints after every fold; `--resume` after a timeout).
6. `02_corrector.py --base results/np_base_pred.npz --prefix np_` so GBM files stay intact.
7. Download `np_results.zip` into local `pv_pipeline/results/`.

A T4 helps, but the bottleneck is one NeuralProphet refit per OOF fold plus the
dataloader. Budget a few hours; the 12-hour session is enough if `--smoke` is
under ~15 minutes. Residual correction itself is still ~1–2 minutes on CPU.

Streamlit cannot be usefully served from Kaggle — run the dashboard locally.

---

## 5. File layout

```
pv_pipeline/
├── common.py                   BUILT + TESTED (§6)
├── verify_dataset.py           BUILT + RUN (16 checks, §1)
├── 00_prepare.py               BUILT + RUN (§7, §7a)
├── 01_base_gbm.py              BUILT + RUN (§8a)
├── audit_stage1.py             BUILT + RUN (§14b leak audit)
├── 01_base_neuralprophet.py    BUILT; run on Kaggle (§8b)
├── 02_corrector.py             BUILT + RUN (§9); `--prefix np_` for the NP path
├── 03_evaluate.py              BUILT + RUN (§10); `--base/--stage2/--prefix` for NP
├── KAGGLE_NEURALPROPHET.md     Kaggle steps
├── kaggle_neuralprophet.ipynb  notebook to upload
├── pv_model.py                 BUILT + RUN (§11)
├── 04_pv_power.py              BUILT + RUN (§11)
├── dashboard/
│   ├── dash_lib.py             BUILT + TESTED (§12)  all logic, offline self-test
│   ├── app.py                  BUILT + TESTED (§12)  thin Streamlit wrapper
│   └── requirements.txt
└── results/                    all outputs (npz, csv, json, png)
```

---

## 6. `common.py` — BUILT AND TESTED

Already implemented and validated against the merged CSV. Public surface:

```python
# Loading
load_solcast(pattern) -> DataFrame      # glob or list; dedupes; IST period-start index;
                                        # adds period_end_utc, mid_utc, date, step, year,
                                        # month, doy, is_monsoon, source_file
add_time_columns(df) -> DataFrame
grid_audit(df) -> dict                  # rows, duplicates, off-grid deltas, missing timestamps
complete_days(df) -> (DataFrame, list)  # keep days with all 144 steps

# Solar position (NOAA Solar Calculator formulation)
solar_position(when_utc, lat, lon, refraction=True) -> dict
    # keys: zenith (refracted), zenith_true, azimuth (0=N clockwise), declination, eqtime
    # VALIDATED: 0.388 deg zenith RMSE at period mid-points
solcast_azimuth_to_conventional(az) -> ndarray      # (-az) % 360; VALIDATED 2.20 deg RMSE
fit_location(mid_utc, zenith_ref) -> dict           # recovers lat/lon by least squares

# Clear-sky proxies (all leak-free; row i uses only days < i, or other years)
clearsky_lagged_median(cs_by_day, window=7) -> ndarray
clearsky_climatology(cs_by_day, doy, years, half_window=7) -> (ndarray, support)
clearsky_clim_times_ratio(clim, cs_by_day, window=7) -> ndarray
kt_from(ghi, cs, clip=(0, 1.3)) -> ndarray
daylight_mask(zenith, cs_proxy) -> ndarray

# Metrics and inference
point_metrics(y, yhat, mask=None) -> dict           # rmse, mae, mbe, nrmse, n
skill(metric_model, metric_ref) -> float
per_day_rmse(y, yhat, mask) -> ndarray              # one value per day
moving_block_bootstrap_ci(diffs, block=7, n_boot=10000) -> dict
wilcoxon_days(a, b) -> dict

# Reshaping
to_day_arrays(df, zenith_col="zenith_calc") -> DayArrays   # (n_days, 144) for every field
```

Constants: `STEP_MINUTES=10`, `STEPS_PER_DAY=144`, `TZ="Asia/Kolkata"`,
`MUMBAI_LAT/LON`, `KT_CLIP=(0,1.3)`, `DAYLIGHT_ZENITH_MAX=85`, `DAYLIGHT_CS_MIN=50`,
`MONSOON_MONTHS=(6,7,8,9)`.

---

## 7. Stage 0 — `00_prepare.py`

**Purpose:** turn the CSV into leak-checked per-day arrays plus the input matrix.

**CLI:** `--csv <glob> --cs-mode {lagged,clim,clim_ratio,solcast} --out results/`

**Steps**

1. `load_solcast`, `grid_audit`, `complete_days`. Assert 0 duplicates and 0 missing.
2. Compute `zenith_calc`, `azimuth_calc` from `common.solar_position` at `mid_utc`.
   Do **not** use the CSV's zenith/azimuth for modelling (keep them only for the §1 check).
3. `to_day_arrays` → (2396, 144) matrices for ghi, cs_solcast, zenith, azimuth, air_temp, wind.
4. Build all three clear-sky proxies; select per `--cs-mode`; **print the development-data
   comparison of all three** so the choice is made on development data, never on test.
5. `kt = kt_from(ghi, cs_proxy)`, `mask = daylight_mask(zenith_calc, cs_proxy)`.
6. Mark usable days (7 prior complete days exist); compute the split indices of §3.
7. Assemble the input table, one row per (day, step):

   | Group | Features |
   |---|---|
   | Recent kt profiles | kt at the same step for D−1, D−2, D−3; ±3-step smoothed D−1 kt |
   | Daily kt means | mean kt of D−1 … D−7 (7 scalars, broadcast over steps) |
   | Clear-sky ratio | D−1 Solcast-to-proxy clear-sky ratio; proxy clear-sky/1000 at the step |
   | D−1 weather means | air_temp, relative_humidity, surface_pressure, wind_speed_10m |
   | PV proxies | per-step median of D−1…D−3 air_temp and wind_speed_10m |
   | Astronomy / calendar | zenith_calc, sin/cos day-of-year, step index |

8. Run the three leak tests of §3; abort on failure.
9. **Outputs:** `results/prep.npz` (all day arrays, mask, features, split indices),
   `results/prep.json` (split dates, cs-mode diagnostics, leak-test results, baseline table),
   `results/cs_clim_table.npy`.

**Acceptance:** printed split dates match §3; all three leak tests pass; the 3-day-mean
baseline reproduces the §1 table to within ~1 W/m².

### 7a. Day-completeness and gap policy

Day-ahead forecasting needs all 144 steps of each of the 7 preceding days, so a short
day must never be dropped silently or padded invisibly. **Measured on the merged file:
2,396 of 2,398 calendar days carry exactly 144 steps (99.92%).** The only two short days
are the file edges — 2020-01-01 (111 steps, first 33 missing) and 2026-07-25 (33 steps,
last 111 missing) — both artefacts of the UTC→IST shift, not real outages. There are no
missing whole days and no NaNs in any modelling column.

`00_prepare.py` reindexes onto a gap-free grid of whole IST days, so an absent step is
an explicit `present == 0` row rather than a shorter day. It then prints the per-day
step-count summary and records it in `prep.json` under `completeness`.

Rules, applied in this order:

1. **Classification.** Each day is `complete` (144 steps), `repaired` (every gap
   ≤ `MAX_REPAIR_GAP_STEPS = 6` steps = 1 hour, and filled) or `unrepairable`.
2. **Repair.** Smooth fields (`air_temp`, `relative_humidity`, `surface_pressure`,
   `wind_speed_10m`) are linearly interpolated **within the day only**. GHI is
   interpolated on the clear-sky index and multiplied back by the clear-sky profile, so
   the diurnal shape is preserved instead of a straight chord being cut across it.
   Interpolation never crosses a day boundary and never reaches forward into day D.
   Every repair is logged to `prep.json`.
3. **Target days.** A day with incomplete actuals **is still forecast**, because its
   inputs come from D−1 and earlier. It is scored only on the steps that exist:
   `mask = daylight & present`. A day is excluded from the scored set only if it has no
   observed daylight step at all (this is what removes 2026-07-25, whose 33 present
   steps are all night).
4. **Predecessor days.** A day is **not usable** if any of its 7 predecessors is
   `unrepairable` **or** has an undefined clear-sky proxy. The second condition matters:
   an undefined proxy makes that day's kt NaN, which `build_features` would turn into a
   silent 0. That exact failure produced a 535 W/m² baseline day (2020-01-09) in an
   earlier revision; it is now impossible.
5. **Assertions.** The run aborts unless every predecessor day of every dev/test/purge
   day carries 144 steps, a defined clear-sky proxy, and finite kt on every observed
   daylight step.
6. **Self-test.** `gap_policy_selftest()` runs on synthetic days at every invocation and
   checks that a 4-step gap is repaired to within 1% of peak GHI, a 20-step gap is
   disqualified, the 7 successors of a disqualified day are blocked, a partial target day
   is scored on present steps only, and nothing is interpolated across a day boundary.

Because `lagged` and `clim_ratio` cannot define a proxy for the first week of the record,
their development sets are 2,016 and 2,017 days against 2,023 for `clim`; the printed
comparison table shows each mode's day count so the rows stay interpretable.

---

## 8. Stage 1 — base forecaster

Output contract, identical for both implementations:
`results/base_pred.npz` with `kt_base` of shape (n_days, 144), NaN where no
out-of-sample forecast exists.

### 8a. `01_base_gbm.py` — the tested target path

- One `HistGradientBoostingRegressor` over all (day, step) rows, squared-error loss.
- Features exactly as §7 step 7. Drop rows where the mask is False.
- **Rolling-origin OOF:** initial train 730 days, folds of 90 days, 1-day purge →
  ~15 folds over development. Each fold predicts its own block only.
- Then fit a final model on **all** development days; use it for the purge day and the test year.
- Hyper-parameters (`max_iter`, `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`,
  `l2_regularization`) tuned on the §3 validation window only. Log the chosen values.
- Runtime: seconds to a couple of minutes. No GPU.

**Acceptance:** test-year RMSE is at or below the 3-day-mean baseline (120.9 W/m² with
`clim_ratio`); skill is positive overall and may be ~0 in the monsoon. A *negative*
skill means a bug, most likely a misaligned OOF block.

### 8b. `01_base_neuralprophet.py` — base-paper reproduction + residual-correction target

**BUILT.** Run on Kaggle (T4, Internet ON). See `KAGGLE_NEURALPROPHET.md`.
Writes `results/np_base_pred.npz` so it cannot overwrite the GBM `base_pred.npz`.

- `y` = kt on daylight steps (0 elsewhere); future regressor = proxy clear-sky profile;
  `n_forecasts=144`; `n_lags` 144 (default) or 432.
- **Column extraction:** in the prediction frame `yhatN` is the N-step-ahead forecast
  for that row's timestamp. For a row at step `s` of day D take column `yhat{s+1}`.
  Prefer the origin-row vector `yhat1..yhat144` at 23:50 of D−1 when it is finite.
  **Never `yhat1` on every row of day D** — that was v2's core error (nowcast RMSE 43.7
  vs day-ahead ~123 W/m²).
- `--smoke` (short series, 2 epochs) **asserts the day-ahead pick differs from `yhat1`**
  and that scrambling day-D actuals does not change the extracted forecast. Do not
  start the full job if smoke aborts.
- Same rolling-origin OOF as §8a (730 / 90 / 1-day purge). Checkpoints after every
  fold; `--resume` after a Kaggle timeout.
- Deviation from Torto et al.: they target GHI; this targets kt because GHI = kt ×
  clear-sky is multiplicative. State this in the paper.
- Then run the **same** Stage 2 corrector on these OOF forecasts:
  `python 02_corrector.py --base results/np_base_pred.npz --prefix np_`.
  That pairing is the novelty. The correction may still be a null, as it was on GBM;
  report whatever the test year says.

---

## 9. Stage 2 — `02_corrector.py` (out-of-fold residual correction)

- **Target:** `r = kt − kt̂_base`, on OOF development days only.
- **Model:** `HistGradientBoostingRegressor`.
- **Features:** `kt̂_base` at the step; daily mean of `kt̂_base` for day D; yesterday's
  error at the same step, its ±3-step mean and its daily mean; proxy clear-sky; zenith;
  step; sin/cos day-of-year; D−1 clear-sky ratio; daily mean kt of D−1.
- **Split:** chronological over OOF days — first 75% train, last 25% validation.
  Boosting iterations chosen by `staged_predict` on validation, then refit on all OOF days.
- **Bounded correction** (BRTCN's formula): `Δ = δ · tanh(Δ_raw / δ)`,
  `kt̂_final = clip(kt̂_base + Δ, 0, 1.3)`.
  The `δ_max` grid **includes 0** (no correction) and is chosen **per season**
  (monsoon = Jun-Sep) on validation RMSE in W/m². With 175 monsoon validation days
  the per-season fit is now real; keep the pooled fallback for strata under 30 days.
- **Calibrated 80% intervals:** empirical 10%/90% residual quantiles of the corrected
  forecast on validation, per stratum = monsoon flag × predicted daily-mean-kt bin
  (<0.6, 0.6-0.85, >0.85), pooled fallback for thin strata. Applied to test as `kt̂_final + q`.
- **The correction may not help. Report the null result if that is what you get.**
- **Outputs:** `results/{prefix}stage2.npz` (`kt_final`, `kt_lo`, `kt_hi`),
  `results/{prefix}stage2.json`. Default `prefix=""` keeps the GBM filenames.
  For NeuralProphet: `--base results/np_base_pred.npz --prefix np_`.

---

## 10. Stage 3 — `03_evaluate.py`

- **Methods compared:** persistence (prev-day kt), 3-day-mean kt, monthly per-step kt
  climatology (development only), Stage 1, Stage 2.
- **Metrics** overall / monsoon / non-monsoon: RMSE, MAE, MBE, nRMSE (% of mean daytime
  GHI), skill vs the 3-day baseline and vs persistence. Hourly-aggregated RMSE as a
  secondary column.
- **Paired tests on per-day RMSE** (not on 10-minute samples — they are autocorrelated):
  Stage 1 vs 3-day, Stage 2 vs Stage 1, Stage 2 vs 3-day. Report the mean difference,
  the Wilcoxon p-value, and the **moving-block bootstrap 95% CI (7-day blocks)**.
  **Treat the CI as primary.**
- **Intervals:** PICP, mean width (W/m²), Winkler/interval score, by season. Nominal 80%.
- **Lead-time curve:** RMSE by forecast lead, 1-24 h.
- **Outputs:** `results/metrics.csv`, `paired_tests.csv`, `intervals.csv`,
  `eval_summary.json`, `fig_*.png`.

---

## 11. Stage 4 — `pv_model.py` and `04_pv_power.py`

- **Reference PV ("simulated actual")** = chain(actual GHI, actual temp, actual wind).
  **Forecast PV** = chain(forecast GHI, temp/wind proxy from D−1…D−3). Comparing like
  with like isolates the GHI-forecast contribution.
- **Chain (plain numpy, unit-testable):** Erbs decomposition → Hay-Davies transposition
  (tilt 19°, surface azimuth 180°) → Faiman cell temperature (U0 = 25, U1 = 6.84) →
  DC with γ = −0.004/°C and 14% DC losses → inverter 0.96 → clip at capacity.
- Hay-Davies needs `dni_extra` (extraterrestrial DNI); v2 omitted it. Compute it.
- `crosscheck_pvlib()` compares the chain against pvlib on a sample of days.
  **pvlib installs locally** — run this check, do not defer it.
- **Metrics:** PV RMSE/MAE in kW and as % of capacity, daily-energy error, skill vs
  persistence-PV, bands propagated from the calibrated GHI interval.
- Report the **computed** capacity factor. Do not quote an expected range without a citation.

---

## 12. Stage 5 — dashboard (10-minute only)

- `dashboard/dash_lib.py` holds **all** logic: data loading, PV recompute for the
  capacity/tilt sliders, matplotlib figures, CSV export builders. Testable offline.
- `dashboard/app.py` is a thin Streamlit wrapper: date picker over the test year,
  capacity and tilt sliders.
- **Panels:** (1) GHI actual vs Stage 2 forecast with the 80% band, plus Stage 1 and the
  3-day baseline; (2) PV kW simulated reference vs forecast with band; (3) day summary
  (daily energy actual vs forecast, that day's RMSE, skill vs baseline); (4) accuracy
  tables by season with the paired-test summary; (5) interval calibration (PICP, width);
  (6) CSV export.
- **Required UI labels:** "held-out year replay", "targets are Solcast estimates",
  "PV is a simulated reference system", "10-minute resolution".
- Run locally: `streamlit run dashboard/app.py` → open `localhost:8501`.
  Streamlit Cloud needs the small `prep`/`results` npz files committed alongside the app.

---

## 13. Run order

```bash
# 0. verify the data (already done; rerun after any CSV change)
python pv_pipeline/verify_dataset.py --csv data/solcast_mumbai_merged_2020_2026.csv

# 1. prepare: print the cs-mode comparison on development data, then fix the mode
python pv_pipeline/00_prepare.py --csv data/solcast_mumbai_merged_2020_2026.csv --cs-mode clim_ratio

# 2. base model, OOF + final
python pv_pipeline/01_base_gbm.py

# 2b. audit the headline skill before believing it (§14b)
python pv_pipeline/audit_stage1.py

# 3. residual correction + calibrated intervals
python pv_pipeline/02_corrector.py

# 4. evaluation, paired tests, figures
python pv_pipeline/03_evaluate.py

# 5. PV conversion + pvlib cross-check
python pv_pipeline/04_pv_power.py --capacity 10 --tilt 19

# 6. dashboard
streamlit run pv_pipeline/dashboard/app.py

# --- Kaggle: NeuralProphet + residual correction (see KAGGLE_NEURALPROPHET.md) ---
# python 01_base_neuralprophet.py --smoke          # must PASS
# python 01_base_neuralprophet.py --resume --epochs 25
# python 02_corrector.py --base results/np_base_pred.npz --prefix np_
# python 03_evaluate.py --base results/np_base_pred.npz --stage2 results/np_stage2.npz \
#     --prefix np_ --stage1-name stage1_neuralprophet --stage2-name stage2_np_corrected
```

---

## 14. Build status (honest)

| Item | Status |
|---|---|
| `common.py` | **BUILT + TESTED.** Solar position 0.388° zenith / 2.20° azimuth RMSE against the CSV; azimuth bug found and fixed. Now also owns the single baseline definition (`shift_days`, `baseline_prev_kt`, `baseline_mean3_kt`) so Stages 0, 1 and 3 cannot drift |
| `verify_dataset.py` | **BUILT + RUN.** 16 checks: 13 VERIFIED, 3 CHANGED, 0 FAILED |
| Merged 2020-2026 CSV integrity | **VERIFIED.** This was v3's Bucket 4 item #3 and is now closed: complete grid, clean seam, clear-sky-day kt homogeneous to 0.0157 across years |
| `00_prepare.py` | **BUILT + RUN.** Completeness audit and gap policy of §7a; 2,396/2,398 days have all 144 steps; all three leak tests return exactly 0.0 |
| `01_base_gbm.py` | **BUILT + RUN.** 15 OOF folds in ~70 s on CPU. Test RMSE 107.27 W/m² |
| `audit_stage1.py` | **BUILT + RUN.** Four-part leak audit of the headline skill: all four checks pass (see §14a) |
| `01_base_neuralprophet.py` | **BUILT, not yet run.** Kaggle T4 path; `KAGGLE_NEURALPROPHET.md` + `kaggle_neuralprophet.ipynb`. Writes `np_base_pred.npz`. `--smoke` must pass before the full job |
| `02_corrector.py` | **BUILT + RUN** on GBM (null). Use `--prefix np_` after NeuralProphet so GBM files stay intact |
| NeuralProphet `yhat{s+1}` extraction | **Coded, unverified on a real fit.** That is what `--smoke` is for. Do not skip it |
| NeuralProphet runtime | **Never measured.** Extrapolate from `--smoke` wall time: (minutes/2)×25×16 / 60 hours |
| `03_evaluate.py` | **BUILT + RUN.** Metrics, paired tests, intervals, lead-time curve, 3 figures |
| `pv_model.py` + `04_pv_power.py` | **BUILT + RUN.** pvlib cross-check executed: POA agrees to **0.35%**, cell temperature to **0.08%** |
| Dashboard | **BUILT + TESTED.** `dash_lib.py` passes an offline self-test of all six panels; `streamlit run app.py` serves HTTP 200 |
| `analysis_seasonal.py` | **BUILT + RUN.** Monsoon skill inversion explained in §14c: higher in percent *and* in W/m² |
| `analysis_intervals.py` | **BUILT + RUN.** 77.3% PICP is non-transfer, not a calibration bug; §14d |

### 14a. Measured results on the held-out year (365 days, 2025-07-25 → 2026-07-24)

Daylight steps only, W/m², clear-sky mode `clim`.

| Method | RMSE | MAE | nRMSE % | Hourly RMSE | Skill vs 3-day-mean |
|---|---|---|---|---|---|
| Monthly per-step climatology | 130.50 | 91.10 | 30.44 | 122.13 | −7.74% |
| Persistence (previous day) | 123.02 | 71.09 | 28.69 | 111.98 | −1.56% |
| 3-day-mean kt (reference) | 121.12 | 74.11 | 28.25 | 112.04 | 0 |
| **Stage 1 (GBM)** | **107.27** | 67.83 | 25.02 | 99.14 | **+11.43%** |
| Stage 2 (corrected) | 108.03 | 68.04 | 25.20 | 99.89 | +10.81% |

Paired tests on per-day RMSE, moving-block bootstrap with 7-day blocks (primary):

| Comparison | Mean daily RMSE difference | 95% CI | Wilcoxon p | Verdict |
|---|---|---|---|---|
| Stage 1 vs 3-day-mean | −8.33 | [−15.13, −1.58] | 3.3e−04 | **Stage 1 better, significant** |
| Stage 1 vs persistence | −8.36 | [−14.56, −1.02] | 8.4e−07 | **Stage 1 better, significant** |
| Stage 1 vs monthly climatology | −23.86 | [−32.42, −15.53] | 7.3e−15 | **Stage 1 better, significant** |
| Stage 2 vs Stage 1 | +0.38 | [−0.87, +1.48] | 0.63 | **No significant difference** |
| Stage 2 vs 3-day-mean | −7.94 | [−14.92, −1.25] | 8.6e−04 | Stage 2 better, significant |

**The Stage 2 residual correction is an honest null result.** On validation both seasons
preferred a non-zero `δ_max` (0.05 monsoon, 0.10 non-monsoon), yet on test the corrected
forecast is 0.76 W/m² *worse* than Stage 1 and the 95% CI for the difference contains
zero. Report it as a null result; do not quietly drop it from the paper.

Intervals (nominal 80%): PICP 77.3% overall (75.5% monsoon, 78.2% non-monsoon), mean
width 205.6 W/m², PINAW 20.4%, Winkler 307.8. Slightly under-covered.

Simulated PV, 10 kW at 19° tilt: Stage 1 RMSE **0.838 kW = 8.38% of capacity**,
+13.0% skill against persistence-PV; daily-energy MAE 5.01 kWh. Computed capacity
factor **15.76%** over the test year (13,809 kWh). The 80% band propagated from the GHI
interval covers 77.3% of reference PV points.

### 14b. Why the skill is 11% and not the 2-8% originally predicted

The +11.43% figure was audited as a suspected leak before being accepted
(`audit_stage1.py`, output in `results/audit_stage1.json`). All four checks pass:

1. **Baseline identity.** Stage 0, Stage 1 and Stage 3 now all call
   `common.baseline_mean3_kt`, and all three report 121.121709669051 W/m² to the last
   digit. The earlier `np.roll` in Stage 1 differed from `shift_days` only on day
   indices 0-2, which are never in dev or test, so the reference was never wrong — but
   the duplicate definition has been removed so it cannot drift.
2. **No test row in training.** Zero overlap between the test year and the tuning fit,
   the tuning score set, any OOF fold's train or predict block, or the final fit. One
   purge day separates dev index 2030 from test index 2032.
3. **Stage-1 prediction invariance** (§3 leak test 3, at prediction level rather than
   feature level). Perturbing a test day's actuals with random values, and separately
   blanking every actual from that day onward, both leave the fitted model's prediction
   for that day **bit-identical** (max |Δ| = 0.0 over 60 daylight steps).
4. **OOF vs test consistency.** OOF development skill +12.14% against test +11.43%, a
   gap of −0.70 points. Test skill is *below* OOF skill, which is the direction you
   expect; a leak would push it above.

The explanation is simply that the 2-8% range in §15 was a prior guess carried over
from v3, not a measurement of this feature set. Two things support the larger number:
the 3-day-mean reference is a weak baseline, and the model has seven days of lagged kt
plus astronomy and seasonality. Critically, the monthly climatology baseline is *worse*
than the 3-day mean (−7.74%), and Stage 1 beats it by 23.86 W/m², so the gain is **not**
merely seasonal knowledge that a cheap climatology would also have.

### 14c. Why monsoon skill is higher than non-monsoon skill

v3 predicted ~0% skill in the monsoon. The measured numbers are the opposite
(+13.0% monsoon vs +7.1% non-monsoon). Produced by `analysis_seasonal.py`; raw
output in `results/seasonal_analysis.json`. **This is not an artefact of percent
skill flattering a large reference error.**

| Season | Days | Mean kt | kt std | Mean GHI | 3-day RMSE | Stage 1 RMSE | Skill % | Skill W/m² |
|---|---|---|---|---|---|---|---|---|
| Monsoon | 122 | 0.588 | 0.313 | 312 | 175.2 | 152.4 | +13.0% | **22.8** |
| Non-monsoon | 243 | 0.955 | 0.168 | 493 | 76.8 | 71.4 | +7.1% | **5.5** |

Three facts, all from the test year:

1. **Clear-sky index is almost twice as variable in the monsoon** (kt std 0.313 vs
   0.168). Mean kt is 0.59, far from the clear-sky pile-up at 1.0, so yesterday's kt
   profile actually carries information. Outside the monsoon, mean kt is 0.95 and
   3-day-mean persistence is already a strong forecast, leaving only 5.5 W/m² for
   the model to remove.
2. **Absolute skill is larger in the monsoon too**, not just percent skill: 22.8 vs
   5.5 W/m². Even as a fraction of mean daytime GHI the monsoon gain is larger
   (7.3% vs 1.1%). The original "percent skill against a large reference" hypothesis
   is therefore wrong as a complete explanation — it would have predicted similar
   or smaller absolute skill in the monsoon.
3. **A few extreme days do not drive it.** Trimming the 5th–95th percentiles of
   per-day RMSE differences leaves the monsoon mean essentially unchanged
   (−15.95 → −16.03 W/m²). 61% of monsoon days improve vs 56% of non-monsoon days.
   The monsoon remains *harder* in raw error (152 vs 71 W/m²); the model just has
   more lagged-kt signal to work with.

Sentence for the paper: *Monsoon RMSE remains larger than non-monsoon RMSE
(152 vs 71 W/m²); the higher skill score reflects a weaker persistence baseline
when kt is variable, not a monsoon-specialised model.*

### 14d. Interval coverage: 77.3% is the honest number

Nominal 80%; shipped PICP 77.3% overall (75.5% monsoon, 78.2% non-monsoon).
Produced by `analysis_intervals.py`; raw output in `results/interval_analysis.json`.

Diagnosis, never touching test to make a choice:

- **The quantile estimator is sound.** In-sample coverage on the validation days
  the quantiles were fitted on is **79.3%** at width 199.9 W/m² — within 0.7 points
  of nominal. This is not a calibration bug.
- **The 2.0-point drop on test is the same validation-to-test non-transfer as
  Stage 2.** An honest A→B split inside development (A = 194 days, B = 130 days)
  already shows that a pooled residual quantile under-covers B at 68.4%. The
  shipped `monsoon × predicted-kt-bin` scheme recovers **80.1% on B** with no
  inflation (`λ = 1.00`), and is the scheme selected on validation.
- **Refitting that selected scheme on all validation and applying once to test
  still yields 77.4%.** Finer bins, a zenith axis, and a conformal inflation factor
  chosen to hit 80% on B all either fail to beat `monsoon_kt3` on B or, when
  applied to test, do not close the gap. Two of six shipped strata already used
  the pooled fallback (`n < 200`).
- **Do not inflate `λ` against test coverage.** That would leak. The honest
  statement is: empirical residual quantiles, selected on validation, cover
  **77.3% of a nominal 80% interval** on the held-out year. Report the measured
  PICP; do not claim a calibrated 80%.

---

## 15. Open risks and limitations

1. ~~**Low skill ceiling.** Without any day-D forecast signal, ~2% RMSE skill overall
   and ~0% in the monsoon is the realistic outcome.~~ **Superseded by measurement
   (§14a).** Stage 1 achieves +11.43% overall (+13.0% monsoon, +7.1% non-monsoon)
   against the 3-day-mean, significant under the moving-block bootstrap and audited
   against leakage (§14b). Monsoon skill is the *higher* of the two (+13.0% vs
   +7.1%); §14c shows this is higher absolute skill as well (22.8 vs 5.5 W/m²),
   because kt is twice as variable in the monsoon and the 3-day baseline is much
   weaker there. The Stage 2 prediction did hold: the residual correction adds
   nothing, and that null result is reported in §14a. Interval PICP is 77.3% of
   a nominal 80% (§14d) — report the measured coverage, do not claim calibration.
2. **Right-censored kt.** 51.8% of daytime rows sit exactly at kt = 1.000 and none exceed
   it: Solcast caps GHI at its own clear-sky value. This is censoring, not a distributional
   problem a logit transform can fix — v2's logit step is dropped. With a proxy clear-sky
   the censoring is partly relieved because the denominator is no longer the same quantity.
3. **Solcast's 10-minute product is its noisiest** (Bright 2019: all-sky rRMSE 25.8% vs
   16.9% hourly). Report hourly-aggregated metrics alongside.
4. **Targets are satellite-derived estimates**, not ground pyranometer measurements.
5. **PV is simulated.** Real plant data (e.g. an NMIMS rooftop) would upgrade this to validated.
6. **Overlapping daily forecasts are autocorrelated.** Hence per-day aggregation plus the
   moving-block bootstrap; never a Wilcoxon over ~24k 10-minute errors.
7. **Single site, no spatial context** — no satellite imagery, neighbouring stations or
   cloud-motion field.
8. The clear-sky mode must be chosen on development data. Choosing it on the test year leaks.

---

## 16. Papers and what each supplies

- **Base (journal):** Torto et al., *e-Prime* 10, 100817 (Oct 2024) — NeuralProphet +
  clear-sky regressor for day-ahead GHI. Reproduce as a Stage-1 row. Flaws to state and
  fix: single-day test sets, mismatched error units, no persistence baseline.
- **Protocol donors:** BRTCN (*Energy and AI*, 2026) — chronological split, daylight-only
  metrics, paired daily tests, the bounded `δ·tanh(·)` correction. Chou et al.
  (*J. Cleaner Prod.*, 2026) — skill scores, interval metrics. Cortez et al. (2024) —
  correcting the forecast profile, trained out-of-sample.
- **Conference:** Roy et al. (IEEE NPSC 2022) — 15-minute block context and the
  residual/ensemble idea; its weights were tuned on the test week, so do not reuse them.
  DavCast (ACDSA 2026) — dashboard structure and interval metrics.
- **Data-limits citation:** Bright (2019) for the noisier 10-minute Solcast product and
  the India clear-sky underestimate.
- **Do not cite results from:** the Firefly CNN-kNN, HGQNN and VMD-CNN papers
  (internally inconsistent numbers or leakage risk).

---

## 17. Resuming in a new chat

Upload `WORKFLOW.md`, `pv_pipeline/common.py`, `pv_pipeline/verify_dataset.py` and
`solcast_mumbai_merged_2020_2026.csv`, then say:

> Continue from WORKFLOW.md. Every stage from `common.py` through the dashboard is
> built, run and tested; §1 holds the measured data facts, §14 the build status and
> §14a the measured results on the held-out year. Do not re-litigate the verified
> facts in §1, the audited skill number in §14b, the monsoon inversion in §14c,
> the 77.3% interval coverage in §14d, or the Stage 2 null result.

Remaining work, in order:

1. **Kaggle: NeuralProphet + residual correction** (§8b, `KAGGLE_NEURALPROPHET.md`).
   `--smoke` first; then full OOF; then `02_corrector.py --prefix np_`. Download
   `np_results.zip` into `pv_pipeline/results/`. This is the novelty row.
2. **Write-up.** §14a–§14d hold the GBM numbers; add the NeuralProphet rows after Kaggle.
3. **Real plant data** (e.g. an NMIMS rooftop) would upgrade the PV half from simulated
   to validated. Do not spend further effort inflating residual quantiles to hit 80%
   on test — that leaks, and §14d already established that validation-selected schemes
   top out at 77.3%.
