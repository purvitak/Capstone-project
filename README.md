# Mumbai day-ahead solar forecasting

DWT-BiLSTM predicts 144 ten-minute GHI values for a day, issued at 00:00 IST.
An out-of-fold residual model corrects forecasts and estimates prediction intervals.
Published GHI is predicted `kt × cs_proxy`, in W/m².

The latest selected hybrid scores **RMSE 107.42 / MAE 68.95 W/m²** on the
recorded test year. Start with the [latest results](results/ghi_search/REPORT.md),
[documentation index](docs/README.md), and [results index](results/README.md).

## Project layout

```text
pv_pipeline/             Preparation, training, correction, and evaluation code
data/raw/                Original merged Solcast CSV and dataset notes
data/prepared/           prep.npz, prep.json, cs_clim_table.npy
results/                 Full-run predictions, metrics, and figures
results/verification/    Preserved dataset audit reports
results/archive/         Earlier neural screens and regime-gate experiments
dashboard/backend/       Saved-prediction loading and PV calculations
dashboard/frontend/      Streamlit dashboard UI
docs/README.md           Documentation index and current sources of truth
docs/DASHBOARD.md        Dashboard scope and references
docs/references/         Literature reviews
docs/archive/            Historical reports, protocol and legacy Git bundle
requirements.txt         Python dependencies
.venv/                   Current local environment
```

Historical notes describe earlier implementations. Use this README for current
commands. The unused nested checkout/environment was removed; its unique Git
history is preserved in `docs/archive/legacy-checkout.bundle`.

## Setup

Run from the project root. For your existing root environment:

```bash
cd /Users/purvitak/Documents/GitHub/Capstone-project
source .venv/bin/activate
```

For a fresh installation:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Smoke check

```bash
python pv_pipeline/01_base_dwt_bilstm.py --resolution hourly --backend torch --smoke --out results/smoke_torch
```

Smoke mode always uses two epochs and incomplete OOF coverage. Keep its outputs
separate from full training; do not resume full training from smoke checkpoints.
Smoke output directories are disposable and ignored by Git.

## Train and build reports

Prepared data is included. The following commands use `data/prepared/prep.npz`
and write to `results/` by default:

```bash
python pv_pipeline/01_base_dwt_bilstm.py --resolution 10min --backend torch --epochs 100 --resume
python pv_pipeline/02_corrector.py
python pv_pipeline/03_evaluate.py
python pv_pipeline/analysis_regimes.py --stage2 results/dwt_stage2.npz
```

100 epochs is the requested example, not a validated optimum. Epochs are now a maximum: chronological inner validation selects the stopping
epoch, then the model refits on all fold training days. Choose settings on development data. PyTorch uses CUDA if available,
otherwise CPU. HGB is available with `--backend hgb` but is not the BiLSTM model.

Changing epochs/backend requires a fresh output directory: `--resume` rejects
checkpoints with different training settings or prepared-data fingerprints. For a separate 100-epoch experiment:

```bash
python pv_pipeline/01_base_dwt_bilstm.py --resolution 10min --backend torch --epochs 100 --out results/100epochs --resume
python pv_pipeline/02_corrector.py --base results/100epochs/dwt_base_pred.npz --out results/100epochs
python pv_pipeline/03_evaluate.py --base results/100epochs/dwt_base_pred.npz --stage2 results/100epochs/dwt_stage2.npz --out results/100epochs
python pv_pipeline/analysis_regimes.py --base results/100epochs/dwt_base_pred.npz --stage2 results/100epochs/dwt_stage2.npz --out results/100epochs
```

Outputs include `dwt_base_pred.npz`, `dwt_stage2.npz`, `dwt_metrics.csv`,
`dwt_eval_summary.json`, interval/paired-test reports, and PNG figures.
DTW clustering can take additional time. There is no separate compile step.

For the hourly experiment, train with `--resolution hourly`. Pass
`results/dwt_hourly_base_pred.npz` to correction/evaluation, use
`--prefix dwt_hourly_` for those stages, and pass
`results/dwt_hourly_stage2.npz` to evaluation/regime analysis.

To compare the saved 10-minute day-ahead forecast with the hourly metrics in
Bektaş & Altaş (2024), Table 4, aggregate its test predictions to hourly GHI:

```bash
python pv_pipeline/compare_paper.py --base results/dwt_10min_150_paper/dwt_base_pred.npz --stage2 results/dwt_10min_150_paper/dwt_stage2.npz --out results/dwt_10min_150_paper
```

This writes `paper_comparison.csv` and `paper_comparison.json`. Omit `--stage2`
before running the corrector. The report labels the sites and daylight-hour
rule; it is a descriptive comparison, not an exact reproduction of the Trabzon
experiment.

## Rebuild prepared data (optional)

The merged original CSV is in `data/raw/`:

```bash
python pv_pipeline/verify_dataset.py --csv data/raw/solcast_mumbai_merged_2020_2026.csv
python pv_pipeline/00_prepare.py --csv data/raw/solcast_mumbai_merged_2020_2026.csv --cs-mode clim
```

Verification writes to `results/verification/`; preparation writes to
`data/prepared/`. Preparation uses fixed development/test boundaries in July 2025;
a different dataset may require changing that configuration.

Default paths resolve relative to the repository, independent of the working
directory. Explicit relative CLI paths resolve relative to the working directory.

## Dashboard integration

The Streamlit dashboard replays saved 10-minute test-year forecasts. The completed
150-epoch-maximum run and its corrected forecast are available, with the corrected
version selected by default. Install the dashboard dependency
if needed, then run from the project root:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m streamlit run dashboard/frontend/app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). Select
a prediction run and date, then adjust PV capacity, NOCT, and temperature loss.
The dashboard shows actual/predicted GHI and estimated PV, metrics, ten-minute
data, and a CSV download. See [dashboard scope](docs/DASHBOARD.md). It does not
produce forecasts for future dates: the trainer currently saves prediction arrays,
not model weights and an inference path.

## Trainer version 2: development-first comparison

The corrected trainer uses training-only scaling and validation-based epoch selection.
Old checkpoints must stay in their original folders. See [model review](docs/MODEL_REVIEW.md).

```bash
python pv_pipeline/01_base_dwt_bilstm.py --backend torch --resolution 10min --epochs 100 --development-only --out results/scaled_100epochs --resume
```

After choosing settings on development results, rerun the exact command without
`--development-only` to fill purge/test predictions. Then use that output folder
for correction and evaluation. Tests: `python -m unittest discover -s tests`.


## Improved development result: BiLSTM + context-tree hybrid

A fixed 50/50 ensemble scored **109.11 W/m²** vs **115.24** for the scaled BiLSTM
on the same 1,293 development forecast days. This is a hybrid result, not a pure
BiLSTM result. See [experiment details](docs/MODEL_REVIEW.md).

Development artifacts are already saved. To reproduce the context component
and fixed blend from the existing BiLSTM predictions:

```bash
python pv_pipeline/01_base_dwt_bilstm.py --backend hgb --features context --development-only --out results/context_hgb --resume
python pv_pipeline/blend_predictions.py
```

After fixing the selected configuration, generate test forecasts for both
components and rebuild the blend:

```bash
python pv_pipeline/01_base_dwt_bilstm.py --backend torch --epochs 100 --out results/scaled_100epochs --resume
python pv_pipeline/01_base_dwt_bilstm.py --backend hgb --features context --out results/context_hgb --resume
python pv_pipeline/blend_predictions.py
python pv_pipeline/02_corrector.py --base results/hybrid/hybrid_base_pred.npz --out results/hybrid --prefix hybrid_
python pv_pipeline/03_evaluate.py --base results/hybrid/hybrid_base_pred.npz --stage2 results/hybrid/hybrid_stage2.npz --out results/hybrid --prefix hybrid_ --stage1-name dwt_hgb_hybrid --stage2-name hybrid_corrected
```

The residual correction may help or hurt; compare both stages honestly.
Experimental neural options are `--features context`, `--loss ghi` (10min/torch),
and `--residual-baseline mean3`. Their screened combinations did not beat the
existing BiLSTM. `--fold-indices 2,8,13 --development-only` screens selected folds;
remove `--fold-indices` and resume to complete OOF coverage before comparison.

## Direct-GHI search for lower RMSE and MAE

The bounded experiment trains eight daylight-only context/DWT HGB models directly
on GHI: squared/absolute loss × 15/31 leaves × minimum leaf size 40/100. Each uses
learning rate 0.05 and up to 200 iterations, selected by GHI RMSE on the trailing
20% of its training days with a one-day purge. Random validation splitting is
disabled. A fresh tree then refits on all fold-training days. Predictions retain
the existing clipped clear-sky-index format for downstream compatibility.

Run development selection first (the default), then frozen test evaluation:

```bash
.venv/bin/python pv_pipeline/experiment_ghi_search.py
.venv/bin/python pv_pipeline/experiment_ghi_search.py --phase evaluate
```

`--phase all` performs both stages in sequence. `--out results/ghi_search` and
`--threads 2` are defaults. Completed folds resume automatically; changed settings,
source fingerprints, or frozen artifacts are rejected. Use a new output directory
for a different experiment. The existing saved runs and dashboard default are
preserved.

The search compares each tree alone and at BiLSTM weights 0.25, 0.50 and 0.75 on
the same 1,293 development days. A candidate must lower both RMSE and MAE relative
to the existing hybrid (109.11261 / 69.40719 W/m²); eligible candidates rank by
RMSE. Otherwise the reference is retained. Only that selection is considered for
residual correction, with a chronological 75/25 development split and one-day
purge; correction is retained only if both validation errors decrease.

Outputs include all candidate metrics in `development_metrics.csv`, per-fold
iteration metadata, `frozen_selection.json`, `test_metrics.csv`, paired seven-day
block-bootstrap RMSE/MAE intervals in `evaluation.json`, and `REPORT.md`.
`selected_base_pred.npz` and `selected_stage2.npz` use the existing prediction
array keys, but are not automatically selected by the dashboard. Test evaluation
also completes the old hybrid's test forecasts in memory without modifying its
development artifacts. Its legacy fitting behavior is preserved for comparison.

For a single model through the existing trainer:

```bash
.venv/bin/python pv_pipeline/01_base_dwt_bilstm.py --backend hgb --features context --hgb-target ghi --hgb-loss squared_error --hgb-max-iter 200 --hgb-learning-rate 0.05 --hgb-max-leaf-nodes 15 --hgb-min-samples-leaf 40 --development-only --out results/direct_ghi_single
```

`--hgb-target kt` remains the default. New HGB flags are recorded in configuration
metadata and incompatible checkpoints cannot resume. `--loss ghi` is the separate,
existing neural weighted-loss option; direct tree targets use `--hgb-target ghi`.
Tests: `.venv/bin/python -m unittest discover -s tests`.

The experiment keeps the horizon, units and daylight scoring unchanged. No error
threshold is guaranteed. Development comparisons reuse a shared development-fit
climatology and validation dates for model selection; the existing test year has
also been inspected in prior project iterations. These results are exploratory,
not a new independent confirmation.

Completed run: a 25% BiLSTM / 75% direct-GHI HGB blend was selected on development
data. Its frozen corrected test forecast scores **RMSE 107.41955 / MAE 68.94594
W/m²**, versus **114.06405 / 72.34607** for the previous corrected model. The
uncorrected selection scores **106.99613 / 69.47969**: correction improved both
development validation metrics, but traded slightly higher test RMSE for lower
MAE. The completed old hybrid scores **109.31586 / 69.17467** on the test year;
the smaller gain over that hybrid has confidence intervals including zero.
Neither error reached 20 W/m². See the [full experiment report](results/ghi_search/REPORT.md).
The new forecasts are available as `ghi_search/selected` in the dashboard run
selector; the existing preferred run remains the default.

The [paper-inspired novelty experiments](docs/references/FOUR_PAPERS_NOVELTY_REVIEW.md)
also screen a three-day DWT-BiLSTM and an issue-time regime gate. The three-day
model was worse on all three selected folds. The gate's small validation gain
over a fixed blend is inconclusive, so neither replaces the current reference
model.


## Paper-based PV calculation in the dashboard

`pv_pipeline/pv_model.py` provides the NOCT temperature-adjusted GHI-to-PV formula
from Shanker & Kulkarni (2026). The dashboard calls it after loading GHI
predictions, using adjustable system capacity and historical temperature proxies.
There is no separate PV-processing command. See [paper review](docs/references/SHANKER_KULKARNI_REVIEW.md)
and [dashboard calculation contract](docs/DASHBOARD.md).
