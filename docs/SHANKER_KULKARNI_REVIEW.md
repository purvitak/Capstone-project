# Shanker & Kulkarni paper: useful methods and application

Reviewed the attached 22-page paper, *Residual-enhanced hybrid deep learning
framework for day-ahead PV power forecasting with cross-site generalization*,
Next Energy 11 (2026), 100601. DOI: https://doi.org/10.1016/j.nxener.2026.100601.

## Adopted for this project

The NOCT temperature-adjusted PV conversion in section 3.3 (equations 5–6)
is directly useful for the dashboard. Implemented as the reusable
`pv_pipeline/pv_model.py` helper. Per the user's instruction, the dashboard
calculates PV after GHI is predicted; this adds no standalone PV pipeline
stage and does not replace or retrain the GHI forecaster.

Cell temperature = ambient temperature + (NOCT - 20) × GHI / 800.
PV kW = capacity × GHI / 1000 × [1 - beta × (cell temperature - 25)].
Defaults from the paper: NOCT 45°C, beta 0.0045/°C, capacity 100 kW.
Negative power is clipped to zero. This is a simulated horizontal-irradiance
proxy, not measured PV or a complete tilted-array/inverter model.
Forecast ambient temperature uses only previous-day observations (three-day
same-time median helper); historical reference PV can use observed temperature.

## Useful ideas already present or worth testing later

- Chronological scaling, direct 144-step forecasting, and validation-based
  optimization reinforce the existing approach.
- Component ablations and seasonal evaluation fit the current BiLSTM / HGB /
  blend comparison. Keep persistence as an explicit benchmark.
- SHAP-based feature selection could inform a later development-only feature
  ablation. Do not copy the paper's feature ranking: its PV target is constructed
  from GHI and temperature, whereas our target is future GHI.
- Optuna searches and a causal residual-sequence model are possible experiments,
  not evidence that the current model will improve. No new search or residual
  autoencoder was added based solely on the paper's headline gain.
- Cross-site transfer needs a second compatible dataset, which is not present
  in this project. It cannot currently be claimed.

## Priority for the current 10-minute GHI pipeline

The existing out-of-fold residual corrector already applies the paper's broad
two-stage idea in a forecast-time-safe way: on the July 2025–July 2026 test
year, daylight GHI RMSE fell from 116.73 to 114.06 W/m². That is a 2.28% drop,
not the paper's 48.8% PV-MAE claim. The paper's numerical gains cannot be
transferred between targets or datasets.

1. **Development-only feature ablation.** Rank only GHI predictors available at
   00:00 IST (past-day GHI/kt/weather, known solar geometry, clear-sky proxy,
   and any archived day-ahead weather forecasts). Use SHAP or permutation
   importance as diagnostics, then retrain and compare on the same chronological
   folds. The paper's GHI/air-temperature/cell-temperature ranking predicts a
   *constructed PV-power target* and cannot be copied for tomorrow's GHI.
2. **Small rolling-origin hyperparameter search.** Follow the paper's validation
   principle, but optimize our daylight GHI RMSE, not its PV MAE. First compare
   1- versus 3-day input history, model width, learning rate, and HGB settings.
   Keep preprocessing fitted inside each historical fold. The 100- and 150-max
   epoch runs produced identical OOF forecasts on all 1,293 days, so raising
   the epoch cap alone is not a useful search direction.
3. **Causal residual-sequence experiment.** If the first two steps plateau, train
   a separate 144-output model using only residual sequences from D−1 and
   earlier, the day-D base forecast, and known calendar/solar inputs. Compare
   it against the current HGB corrector on identical OOF validation days.
   An autoencoder that reconstructs day-D residuals from day-D residuals is
   unusable at issue time. Preserve the current HGB as the reference.

Since the July 2025–July 2026 test year has already been inspected, further
changes chosen after seeing it should be described as exploratory unless a new
untouched future test period is obtained.

## Methodological points to resolve before reproducing claims

Section 3 formulates correction from past residuals, but section 3.10.4 equation
38 reconstructs r_t from r_t, with r_t = y_t - base forecast. Section 3.11 likewise
constructs target-site residuals using actual target values before reconstruction.
As written, these later equations require unavailable future actuals at inference.
The paper does not sufficiently resolve that discrepancy. A usable adaptation
must instead predict tomorrow's errors from past out-of-sample residuals and
known inputs. This is an ambiguity in the described method, not a claim about
unavailable source code or the authors' intent.

The baseline table (Table 8) lists persistence MAE 1.0865 kW / RMSE 1.9075 kW,
which are lower than the headline Mumbai hybrid values in Table 17 (MAE 1.3864 /
RMSE 2.7345 kW). The reported 48.8% improvement is relative to the tuned MLP,
not evidence of superiority over persistence. Tables 10 and 17 also report
different hybrid values, so evaluation subsets/settings require clarification.

The paper uses overlapping sliding forecast windows and flattened errors.
Our fixed 00:00 daily forecasts and daylight GHI W/m² metrics are a different
protocol. Preserve paired daily statistics and block bootstrap rather than
copying independence-based confidence intervals over overlapping points.

The paper's power errors in kW depend on its assumed 100 kW plant and cannot
be used as numerical targets for our GHI RMSE in W/m². Adding the PV conversion
will not reduce the GHI forecasting error.
