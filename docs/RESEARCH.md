# Research direction (locked)

**GHI backbone:** Bektas & Altas, *DWT-BILSTM-based models for day-ahead hourly
GHI forecasting*, Neural Computing and Applications 36:13243–13253 (2024).

**Correction / analysis source:** Cortez et al., *Day-Ahead Photovoltaic Power
Forecasting Using Deep Learning with an Autoencoder-Based Correction Strategy*,
Journal of Control, Automation and Electrical Systems 35:662–676 (2024).

NeuralProphet / Torto et al. are **out**.

---

## Claim

We extend Bektas & Altas (2024) from hourly ground GHI to **leak-safe 10-minute
day-ahead GHI** on Mumbai Solcast, add a **Cortez-style correction stage** trained
only on **out-of-fold** residuals, and evaluate both stages by irradiance regime
(clear / cloudy / high intra-day variability / monsoon).

## Three extensions

1. **Resolution** — DWT-BiLSTM predicts **144** next-day 10-min values, not 24 hourly.
2. **Residual enhancement** — second model on OOF primary residuals only (`02_corrector.py`).
3. **Weather-specific evaluation** — clear / cloudy / high-variability / monsoon;
   optional soft-DTW clusters on **forecast-error** profiles (Cortez tool, eval only).

**Leak rule:** DWT and sky type use only history through **23:50 IST of D−1**.
No day-D GHI, no day-D weather type from measured GHI, no same-day Solcast
clear-sky, no NWP. Cortez’s recursive 1-step loop and test-set HPO are not used.

## Models

| Role | Method |
|---|---|
| Bektas reproduction | Hourly DWT-BiLSTM on Solcast (`--resolution hourly`) |
| Extension 1 | 144-step leak-safe DWT-BiLSTM |
| Extension 2 | OOF residual corrector on that primary (`--prefix dwt_`) |
| Non-DL baseline | LightGBM on kt (already measured) |
| Naive | Persistence, 3-day-mean kt |
| Optional ablation | `--leaky-weather-type` (Bektas forecast-day K) |

Target internally: **kt**. Published scores: **GHI W/m², daylight**, plus hourly-aggregated RMSE vs Bektas.

## Commands

```text
python pv_pipeline/01_base_dwt_bilstm.py --resolution hourly --smoke
python pv_pipeline/01_base_dwt_bilstm.py --resolution 10min --resume
python pv_pipeline/02_corrector.py --base results/dwt_base_pred.npz --prefix dwt_
python pv_pipeline/analysis_regimes.py --base results/dwt_base_pred.npz --stage2 results/dwt_stage2.npz
```
