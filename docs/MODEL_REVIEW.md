# Model review after the original 15-epoch run

Historical development record. The later selected hybrid and its test evaluation
are documented in the [latest experiment report](../results/ghi_search/REPORT.md).

Original unchanged outputs: `results/15epochs/`.

- Test native daylight RMSE: 124.21 W/m², vs 3-day mean 121.12 W/m².
- Test hourly daylight RMSE: 114.15 W/m².
- Test native daylight MAE: 82.49 W/m².
- OOF development native daylight RMSE: 124.46 W/m².

## Literature comparisons

Bektas & Altas (2024), DOI 10.1007/s00521-024-09701-2, Table 4:
Model 1 hourly RMSE is 53.20 W/m² across 24 hours and 72.83 W/m²
for daylight; Model 2 values are 50.63 and 68.89 W/m². Different
sites/data and masks prevent a controlled superiority claim on Mumbai data.
The paper uses dmey; our current wavelet is Haar with resampled coefficients,
so this is a model-family adaptation, not an exact reproduction. The paper uses
validation stopping (patience 30, max epochs 500); these are not a guarantee.

DavCast (2026), attached PDF Table VI: TFT MAE ranges from 56.90 to 87.41
W/m² for 1–4-hour-ahead hourly forecasts in Davao. This is not day-ahead
144-step RMSE. Compare like metrics and horizons.

Cortez et al. (2024), DOI 10.1007/s40313-024-01099-5, attached PDF Table 8:
PV RMSE is reported in kW (MLP 21.96, LSTM 23.19, GRU 23.66).
These numbers cannot be compared to GHI W/m².

## Training changes

Trainer version 2 standardizes channels on training data only, clips gradients,
keeps batch order chronological, and seeds each fold. Within each fold, the
last 20% of training days are used for epoch selection with a one-day purge.
The best validation GHI RMSE selects epochs; after selection, a fresh network
and scaler are fitted on all fold-training days for that epoch count.
Validation patience defaults to 15. `--epochs` is now a maximum, not a fixed count.

A diagnostic on the first 730 development days, with a chronological 80/20
split and fixed seed, gave 149.82 W/m² unscaled vs 128.87 W/m² standardized
at 15 epochs. This is one split, not a full-run or held-out improvement claim.

Use `--development-only` for comparing settings without repeatedly scoring test.
Old checkpoints cannot be resumed by version 2; use a new output directory.
Choose configuration on development data, then run the selected configuration
without `--development-only` once to generate test forecasts. Test has already
been inspected, so document this development history in any research write-up.
No paper-level accuracy or positive skill is guaranteed.


## Follow-up development experiments

The original scaled BiLSTM scored 115.2428 W/m² on 1,293 OOF development days.
We screened folds 2, 8, and 13 (zero-based) covering monsoon, winter, and
pre-monsoon. On these same 270 days:

| Candidate | RMSE W/m² |
|---|---:|
| Existing scaled DWT-BiLSTM | 128.54 |
| Context + daylight-GHI loss + mean3 residual, hidden 32 / batch 64 | 129.31 |
| Context + daylight-GHI loss, hidden 64 / batch 32 | 129.05 |
| Context HGB | 122.53 |
| Fixed 50/50 original BiLSTM + context HGB | 121.51 |

The two neural variants did not improve aggregate screening error and were not
selected. Their partial artifacts remain in `results/archive/context_residual_dev/` and
`results/archive/context_ghi_dev/`; do not compare those partial scores with full OOF.

The context HGB then ran all 15 folds, using the same chronological training days
and one-day purge as the existing BiLSTM. It adds the prepared historical
kt/weather and known solar/calendar channels to the DWT features. The fixed
50/50 blend clips each component to the existing kt bounds before averaging.

| Model | All development RMSE | Monsoon | Non-monsoon |
|---|---:|---:|---:|
| Existing DWT-BiLSTM | 115.24 | 158.77 | 83.97 |
| Context HGB | 110.21 | 153.72 | 78.41 |
| Fixed 50/50 hybrid | 109.11 | 151.71 | 78.12 |

All scores are daylight GHI W/m² over the same 1,293 development days.
The baseline is 125.22 W/m². Hybrid skill vs baseline is approximately 12.86%;
RMSE drops approximately 5.32% relative to the existing DWT-BiLSTM.
These are development-selection results, not independent test claims.
The hybrid retains a BiLSTM component but must be reported as a BiLSTM + HGB
ensemble, not a pure BiLSTM improvement or a reproduction of the base paper.

Artifacts:

- `results/context_hgb/dwt_base_pred.npz`: tree forecasts; the filename follows the
  existing pipeline convention, while its configuration explicitly says HGB.
- `results/hybrid/hybrid_base_pred.npz`: blended forecasts.
- `results/hybrid/hybrid_development.json`: component/season metrics and source provenance.

Both sources and the hybrid currently contain no test-year forecasts. The blend
script never scores the test year. All models reuse the existing prepared-data
protocol; this work did not re-audit climatology fitting inside every historical
OOF fold. In particular, development-fitted climatology is not refitted per fold,
so OOF scores are conditional on that preprocessing and do not establish a fully
rolling-origin preprocessing audit. The final held-out test remains the next
check after selecting the configuration.
