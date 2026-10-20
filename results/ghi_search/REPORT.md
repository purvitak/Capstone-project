# Direct-GHI improvement experiment

Eight configurations, 15 chronological folds, 1,293 shared development days. Selection required lower RMSE and MAE than the existing 50/50 hybrid. All values below are daylight ten-minute GHI in W/m².

Selected: **squared_error_leaves15_min100_w0.25**. Residual correction retained: **True**.

## Development selection

| Method | RMSE | MAE |
| --- | ---: | ---: |
| Existing 50/50 hybrid | 109.11261 | 69.40719 |
| Selected configuration | 108.43732 | 69.31396 |

BiLSTM weight: 0.25; direct-GHI HGB weight: 0.75. Tree settings: squared_error, 15 leaves, minimum leaf size 100, learning rate 0.05, maximum 200 iterations.

## Residual-correction validation

968 earlier fitting days / one purge day / 324 later validation days. Selected iterations: 21. Retained bounds: {'monsoon': 0.1, 'non_monsoon': 0.5}.

| Validation model | RMSE | MAE |
| --- | ---: | ---: |
| Uncorrected | 104.28840 | 65.61505 |
| Bounded correction candidate | 103.02453 | 63.99507 |

## Frozen test-year evaluation

2025-07-25 to 2026-07-24, 365 days. The configuration and correction decision were frozen before scoring.

| Test method | RMSE | MAE |
| --- | ---: | ---: |
| mean3 | 121.12171 | 74.10970 |
| dwt_bilstm | 116.72802 | 74.05424 |
| original_corrected | 114.06405 | 72.34607 |
| reference_hybrid | 109.31586 | 69.17467 |
| selected_uncorrected | 106.99613 | 69.47969 |
| selected_final | 107.41955 | 68.94594 |

## Seasonal results

| Method | Season | RMSE | MAE |
| --- | --- | ---: | ---: |
| mean3 | monsoon | 175.15588 | 122.40840 |
| mean3 | non_monsoon | 76.84236 | 47.66396 |
| dwt_bilstm | monsoon | 164.18051 | 120.17873 |
| dwt_bilstm | non_monsoon | 79.54089 | 48.79898 |
| original_corrected | monsoon | 161.25754 | 119.68706 |
| original_corrected | non_monsoon | 76.78629 | 46.42472 |
| reference_hybrid | monsoon | 155.82698 | 114.41222 |
| reference_hybrid | non_monsoon | 72.09422 | 44.40504 |
| selected_uncorrected | monsoon | 152.27836 | 113.39004 |
| selected_uncorrected | non_monsoon | 70.84984 | 45.43677 |
| selected_final | monsoon | 152.80356 | 113.90813 |
| selected_final | non_monsoon | 71.22123 | 44.32709 |

## Paired changes of selected final model

Negative differences favor the selected model. Intervals use 2,000 paired circular seven-day block resamples and recompute pooled point metrics.

| Reference | Metric | Change | 95% interval |
| --- | --- | ---: | --- |
| original_corrected | rmse | -6.64450 | [-9.16955, -3.82802] |
| original_corrected | mae | -3.40013 | [-5.23396, -1.59938] |
| reference_hybrid | rmse | -1.89631 | [-3.93223, 0.27897] |
| reference_hybrid | mae | -0.22872 | [-1.47940, 1.09920] |
| mean3 | rmse | -13.70216 | [-21.48532, -5.69705] |
| mean3 | mae | -5.16376 | [-10.34885, -0.07533] |
| selected_uncorrected | rmse | 0.42342 | [-0.90132, 1.96105] |
| selected_uncorrected | mae | -0.53375 | [-1.52713, 0.57188] |

## Interpretation

Against the previous corrected run, the RMSE reduction is 6.64450 W/m² (5.83%) and the MAE reduction is 3.40013 W/m² (4.70%). Both bootstrap intervals favor lower errors.

Against the completed reference hybrid: Both bootstrap intervals include zero; the comparison is inconclusive.

Residual correction passed both development validation guards. On the test year, its RMSE change versus the selected uncorrected model is +0.42342 W/m² and its MAE change is -0.53375 W/m². The development-frozen correction decision was not changed after seeing these test results.

Both errors at or below 20 W/m²: **False**. Nominal 80% empirical interval coverage: **78.10%**.

## Limits

- Development-derived climatology is shared across OOF folds.
- Development model and correction selection reuse validation dates.
- The existing test year has been inspected in earlier project iterations.
- Forecast horizon, units and daylight mask are unchanged. No external data was added.
- Twenty W/m² and single-digit errors were stretch targets, not promised outcomes.
- Existing saved runs and the dashboard default were preserved.

Full development candidates: `development_metrics.csv`; frozen decisions: `frozen_selection.json`; test metrics and bootstrap intervals: `evaluation.json`.
