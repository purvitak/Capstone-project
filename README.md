# Capstone — leak-safe DWT-BiLSTM day-ahead GHI

NeuralProphet, GBM Stage 1, dashboard, and simulated PV are removed.

```text
python pv_pipeline/00_prepare.py ...
python pv_pipeline/01_base_dwt_bilstm.py --resolution hourly
python pv_pipeline/01_base_dwt_bilstm.py --resolution 10min
python pv_pipeline/02_corrector.py --base results/dwt_base_pred.npz --prefix dwt_
python pv_pipeline/analysis_regimes.py --base results/dwt_base_pred.npz
python pv_pipeline/03_evaluate.py --base results/dwt_base_pred.npz --stage2 results/dwt_stage2.npz --prefix dwt_
```

Kept: `common.py`, `verify_dataset.py`, `00_prepare.py`, `dwt_utils.py`,
`01_base_dwt_bilstm.py`, `02_corrector.py`, `03_evaluate.py`, `analysis_regimes.py`,
`results/prep.npz`. Protocol notes still in `WORKFLOW.md` (ignore NeuralProphet / GBM sections).
