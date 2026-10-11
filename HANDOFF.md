# Handoff: day-ahead GHI pipeline (for another model / coder)

**Read this file first.** Then `RESEARCH.md` (claim) and `WORKFLOW.md` (data protocol + measured GBM facts). Do not revive NeuralProphet.

If a chat is empty, the human should attach this file plus `pv_pipeline/common.py` and say: *Continue from HANDOFF.md. Do not use NeuralProphet.*

---

## 1. What this project is

Leak-safe **day-ahead GHI** for **Mumbai** from **Solcast history only** (no paid NWP).

- Issued at **00:00 IST** for the **144 × 10-min** steps of calendar day D.
- Inputs allowed: observations through **23:50 IST of D−1**, plus quantities known in advance (astronomy, leak-safe clear-sky **proxy**, calendar).
- Learn **kt = GHI / CS_proxy** (clip [0, 1.3]). Publish **GHÎ = kt̂ × CS_proxy** in **W/m² on daylight steps**.
- Daylight mask: `zenith < 85°` **and** `CS_proxy > 50` (both known in advance).

**Paper direction (locked):** extend Bektas & Altas (2024) DWT-BiLSTM and add a Cortez-style residual stage, without their leaks.

| Source | Role |
|---|---|
| Bektas & Altas, *Neural Computing and Applications* 36:13243–13253 (2024) | GHI backbone: DWT + BiLSTM, hourly day-ahead, sky type from K = GHI/GHI_ext |
| Cortez et al., *J. Control Autom. Electr. Syst.* 35:662–676 (2024) | Forecast-then-correct (LSTM-AE on errors); k-means + soft-DTW for **analysis** |
| Torto / NeuralProphet | **Retired. Delete/ignore any NP scripts.** |

---

## 2. Claim and novelties (research-backed)

**Claim:** leak-safe 10-min day-ahead GHI on Mumbai Solcast using Bektas’s model family, plus an **out-of-fold** residual corrector (Cortez’s idea, honest protocol), evaluated by irradiance regime.

**Novelties you may claim** (cite the gap, not “we invented OOF”):

1. **144 × 10-min DWT-BiLSTM**, 00:00 origin, DWT only on the window ending D−1. Bektas is 24 hourly only; their conclusion only mentions 1–3 h horizons. Cortez is 59 × 15-min **PV**, recursive 1-step.
2. **OOF residual enhancement** on that primary. Cortez learns errors with an LSTM-AE but uses **actual future weather** (§3.6), trains the AE on **true** daily profiles (§3.8), and tunes HPO on **test MAE** (§3.7). Bektas has no residual stage. We train Stage 2 only on expanding-window OOF residuals.
3. **Regime audit** of primary vs residual: clear / cloudy / overcast from **D−1 K**, high intra-day GHI σ, monsoon; optional DTW clusters on **error profiles**. Cortez said single-number metrics are inadequate and used DTW as an analysis tool. Monsoon is Mumbai-specific (not in either paper).
4. **kt × leak-safe CS_proxy** as the learning target. Bektas uses K only to *label* weather type and still predicts GHI. Weakest novelty bullet; keep as methodology if a reviewer objects.

**Not novelties:** OOF / walk-forward itself (Tashman; Hyndman; Wolpert stacking); DWT-BiLSTM architecture (Bektas); residual correction as an idea (Cortez); GBM +11.4% (already measured baseline).

**Do not promise:** Bektas 53.2 / 72.8 W/m² at 10-min native; Cortez +3.5%; Torto 0.067 RMSE. Compare to Bektas only after **hourly aggregation**. Stage 2 may be a **null** (GBM residual already was). Report that.

---

## 3. Hard leak rules (violating one invalidates the paper)

1. No NWP, no archived forecasts, no day-D observed GHI / temp / RH / wind / pressure.
2. No **same-day Solcast `clearsky_ghi`** as a model input (leaky vs target). CS_proxy = doy climatology and/or lagged, fit on **development** only.
3. No **forecast-day weather type from day-D GHI** except as `--leaky-weather-type` ablation.
4. **DWT** on history through D−1 only. Never `wavedec` on the concatenated train+test series.
5. No shuffling of days. Test year untouched until the final model.
6. Stage 2 trains only on **OOF** Stage-1 forecasts (development folds). Hyperparameters (`δ_max`, trees, quantiles) on the **validation slice of OOF**, never test.
7. Cortez recursion (feed predicted GHI/weather into the next 15-min step using **actual** future weather) is forbidden. Output a **direct** 24- or 144-vector from the origin.
8. Exclude `cloud_opacity`, `dni`, `dhi`, `gti` as inputs (deterministic in GHI/kt).

---

## 4. Data

CSV (not in the repo):

`/Users/purvitak/Desktop/capstone/solcast_mumbai_merged_2020_2026.csv`

- 10-min, IST after `period_end` UTC − 10 min (`common.load_solcast`).
- Site ~19.076°N, 72.878°E. Solar position at period **mid-point**.
- Complete IST days 2020-01-02 → 2026-07-24. Usable days need 7 complete predecessors.
- **Development:** 2020-01-09 → 2025-07-23. **Purge:** 2025-07-24. **Test:** 2025-07-25 → 2026-07-24 (365 d, full monsoon).
- OOF: expanding window, `INITIAL_TRAIN_DAYS=730`, `FOLD_DAYS=90`, `PURGE_DAYS=1`.
- Monsoon months: 6, 7, 8, 9.

A New Delhi NSRDB file exists (`/Users/purvitak/Downloads/New_Delhi_MERGED.csv`, 30-min 2018–2020). **Do not mix it into these scripts** unless the human asks. Bektas is Trabzon/Basel hourly, not this Delhi file.

Prepared arrays: `pv_pipeline/results/prep.npz` (already built). Keys include `kt, ghi, cs_proxy, cs_solcast, mask, X, zenith, dates, dev_idx, test_idx, val_idx, purge_idx, is_monsoon, doy`. Shapes `(n_days, 144)`.

---

## 5. Repo map

```text
capstone/
  HANDOFF.md              ← this file
  RESEARCH.md             ← short claim
  WORKFLOW.md             ← v4 protocol + measured GBM skill (ignore NP sections)
  README.md
  pv_pipeline/
    common.py             load, solar position, kt, metrics, baselines
    verify_dataset.py
    00_prepare.py         → results/prep.npz
    01_base_gbm.py        LightGBM on features X  → results/base_pred.npz
    dwt_utils.py          Haar DWT + D−1 sky type (numpy, no PyWavelets)
    01_base_dwt_bilstm.py Bektas hourly / 144-step
    02_corrector.py       OOF residual + 80% intervals  (--prefix dwt_)
    03_evaluate.py
    analysis_regimes.py   weather slices + DTW error clusters
    analysis_seasonal.py  analysis_intervals.py
    04_pv_power.py        optional simulated PV
    results/              measured artifacts; do not delete GBM files
```

`.venv` is NumPy 2 + sklearn. **No torch.** Do not install NeuralProphet here.

**Output contract for every Stage-1 model:** `kt_base` of shape `(n_days, 144)`, NaN where no OOS forecast. Hourly DWT expands each hour × 6 so Stage 2 still sees 144.

| File | Who writes it |
|---|---|
| `results/prep.npz` | `00_prepare.py` (exists) |
| `results/base_pred.npz` | GBM (exists, test RMSE ~107 vs 3-day ~121, skill +11.4%) |
| `results/dwt_hourly_base_pred.npz` | `--resolution hourly` |
| `results/dwt_base_pred.npz` | `--resolution 10min` |
| `results/dwt_hourly_leaky_base_pred.npz` | hourly + `--leaky-weather-type` |
| `results/dwt_stage2.npz` | `02_corrector.py --base results/dwt_base_pred.npz --prefix dwt_` |

---

## 6. What “OOF” means here

Out-of-fold = day D’s Stage-1 forecast comes from a model **not trained on day D**.

Expanding folds over `dev_idx`: train `dev_idx[:a-1]`, predict `dev_idx[a:a+90]`. Save those into `kt_base`. Final model on all `dev_idx` fills purge + test only.

Stage 2 (`02_corrector.py`) uses only development days where `kt_base` is finite, splits those 75/25 in time, tunes `δ_max` including **0** (null allowed). Features: base forecast, D−1 residual profile, CS_proxy, zenith, calendar — all known at issue time.

This is standard (rolling-origin CV; Wolpert stacking). Cite as **method**, not novelty.

---

## 7. DWT-BiLSTM implementation status

`dwt_utils.py` + `01_base_dwt_bilstm.py` are written.

- Sky type: Bektas thresholds on daily K = daylight ΣGHI / ΣCS_proxy. 2=sunny (K>0.45), 1=cloudy (0.25–0.45), 0=overcast.
- Leak-safe channels: DWT bands of **yesterday’s** GHI (hourly-mean if `--resolution hourly`) + `weather_type_{D-1}` + a second weather channel that **copies D−1** unless `--leaky-weather-type`.
- Haar multilevel DWT in numpy (Bektas used dmey; Haar is the leak-safe stand-in until PyWavelets `dmey` is added). Levels: 3 hourly (24 points), 6 at 10-min (144 points). Each band resampled to length T.
- Backends: `--backend torch` (real BiLSTM) or `--backend hgb` (sklearn on flattened DWT features). **`--backend hgb` must not be reported as DWT-BiLSTM in the paper.** `--backend auto` uses torch if importable.
- `--smoke`: one fold, short train, for a compile check.
- `--resume`: skip folds that already have finite `kt_base`.

**Not finished / do not assume:** a full GPU OOF of torch BiLSTM has **not** been run to completion. Hourly + 10-min smoke/HGB may be incomplete. Next coding work is: run `--resolution hourly` then `--resolution 10min` with **torch on Kaggle GPU**, then `02_corrector.py --prefix dwt_`, then `analysis_regimes.py`.

Local `.venv`: GBM and corrector and regimes run on CPU. True BiLSTM needs `pip install torch` (separate env or Kaggle). Full 144-step OOF (~15 folds) wants a **T4**, not the laptop.

---

## 8. Kaggle (torch BiLSTM only)

Do **not** install torch into project `.venv` if you want to keep NumPy 2 / GBM.

Upload as a dataset: `prep.npz`, `common.py`, `dwt_utils.py`, `01_base_dwt_bilstm.py`, `02_corrector.py`, `analysis_regimes.py`.

Notebook sketch (Original image, GPU T4, Internet on):

```python
# Cell: copy inputs
from pathlib import Path
import shutil
work = Path("/kaggle/working")
(work / "results").mkdir(exist_ok=True)
for p in Path("/kaggle/input").rglob("*"):
    if p.suffix in {".py", ".npz"}:
        dest = work / "results" / p.name if p.name == "prep.npz" else work / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dest)
        print("copied", p.name)
```

```python
# Cell: hourly reproduction (Bektas)
import os, time
os.chdir("/kaggle/working")
t0 = time.time()
!python 01_base_dwt_bilstm.py --prep results/prep.npz --out results --resolution hourly --backend torch --epochs 15 --resume
print("hours", round((time.time()-t0)/3600, 2))
```

```python
# Cell: 144-step extension
t0 = time.time()
!python 01_base_dwt_bilstm.py --prep results/prep.npz --out results --resolution 10min --backend torch --epochs 15 --resume
print("hours", round((time.time()-t0)/3600, 2))
```

```python
# Cell: Cortez-style residual (OOF)
!python 02_corrector.py --prep results/prep.npz --base results/dwt_base_pred.npz --out results --prefix dwt_
!python analysis_regimes.py --prep results/prep.npz --base results/dwt_base_pred.npz --stage2 results/dwt_stage2.npz --out results
```

Optional leaky ablation (do not use as the main result):

```python
!python 01_base_dwt_bilstm.py --prep results/prep.npz --out results --resolution hourly --leaky-weather-type --backend torch --epochs 15
```

Force one GPU if Lightning/DDP appears (`CUDA_VISIBLE_DEVICES=0`). Save `*_pred.npz` after every fold (`--resume`).

---

## 9. Evaluation rules

- Primary: daylight RMSE / MAE / MBE in W/m².
- Secondary: hourly-aggregated RMSE for Bektas tables.
- Always print 3-day-mean kt and persistence.
- Always print GBM (`base_pred.npz`) on the same test year.
- `analysis_regimes.py`: slices `clear_d1 / cloudy_d1 / overcast_d1 / high_variability / monsoon` + `residual_helped` if Stage 2 exists.
- PICP of 80% intervals from `02_corrector.py` is allowed; GBM PICP was ~77% (honest, slightly under).
- No MAPE on GHI including nights.

Measured **GBM** (do not overwrite): test RMSE **107.27** vs 3-day **121.12**, skill **+11.43%**, Stage 2 **null**.

---

## 10. Instructions for the assisting model

**Do**

- Keep kt target, 144-length `kt_base`, expanding OOF, 1-day purge.
- Prefer editing `01_base_dwt_bilstm.py` / `dwt_utils.py` to add real `dmey` DWT and a solid torch trainer (early stopping on a **validation** slice of the current train window, never test).
- Wire Kaggle notebook only if asked; keep scripts CLI-first.
- If residual RMSE does not drop, write `δ_max=0` and stop; that is a valid paper result.

**Do not**

- Reintroduce NeuralProphet, Firefly, Chu–Beasley HPO, or test-set tuning.
- Use day-D Solcast weather or same-day clear-sky.
- Compare 10-min RMSE to Bektas 53.2 without aggregating.
- Claim HGB fallback as BiLSTM.
- Mix Delhi NSRDB into `load_solcast` without a separate loader.
- Delete `pv_pipeline/results/base_pred.npz` or `prep.npz`.
- Commit unless the human asks.

**Suggested next tasks (in order)**

1. Smoke `01_base_dwt_bilstm.py --resolution hourly --smoke --backend hgb` locally to verify I/O.
2. Kaggle full hourly torch, then 10-min torch, `--resume`.
3. `02_corrector.py --prefix dwt_`.
4. `analysis_regimes.py` + `03_evaluate.py --base results/dwt_base_pred.npz --stage2 results/dwt_stage2.npz --prefix dwt_ --stage1-name dwt_bilstm --stage2-name dwt_corrected`.
5. Optional: `--leaky-weather-type` hourly ablation table.

---

## 11. One-paragraph abstract (for the paper draft)

We take the DWT-BiLSTM day-ahead GHI model of Bektas and Altas (2024) and the forecast-then-correct design of Cortez et al. (2024) and apply them under an operational leak rule on multi-year 10-minute Mumbai Solcast data. The primary model issues a 144-step forecast at 00:00 IST from wavelet features and sky type computed only up to 23:50 of the previous day. A second model is trained solely on out-of-fold residuals. Both stages are scored in W/m² on daylight steps and sliced by D−1 sky type, intra-day variability, and monsoon. A leak-safe gradient-boosted kt model is retained as a non-deep baseline. We do not use day-ahead numerical weather prediction.
