# Wang et al. (2025): day-ahead GHI clustering

Attached file: `/Users/purvitak/Documents/capstone papers/Untitled.pdf`.
Published as Yong Wang et al., *Day-ahead solar irradiance prediction based on
multi-feature perspective clustering*, Energy 320 (2025) 135216,
https://doi.org/10.1016/j.energy.2025.135216.

## What transfers to this project

The paper groups historical days into cloudy, partly cloudy, and clear-sky
regimes using time- and frequency-domain features, trains a BiLSTM per group,
then combines their predictions. This is relevant to our 144-step day-ahead GHI
forecast, especially the higher monsoon error. Our DWT-BiLSTM already extracts
frequency information from the previous day; the main new idea is **regime-aware
training or combining**, not adding another wavelet decomposition by default.

A small leak-safe development experiment should precede the paper's full
multi-feature clustering and VAE architecture:

1. Build day-level descriptors from D−1 and earlier GHI/kt, past weather,
   known solar geometry, and any archived forecast issued before 00:00 IST.
   Include both daily level and variability so clear and highly variable days
   are distinguishable. Do not use D's observed GHI, DHI, DNI, temperature,
   humidity, or a weather label derived from them.
2. Fit a simple 3-regime clustering model within each expanding training fold.
   Route the day-D forecast from issue-time descriptors only. Begin with a
   single HGB/BiLSTM plus a regime feature or a similarity-weighted historical
   forecast; train three separate BiLSTMs only if simpler variants improve.
3. Compare on the same 1,293 development OOF days against the current
   DWT-BiLSTM, context HGB/hybrid, and residual-corrected forecasts. Report
   daylight GHI RMSE overall and by monsoon/variability. Keep the July
   2025–July 2026 test period out of model selection; it has already been
   inspected, so later test gains are exploratory.

The paper's California experiment uses 1-minute data over 61 summer days,
with the last 5 days for testing. Our Mumbai test spans 365 days at 10-minute
resolution. Its W/m² scores are therefore contextual, not a target for direct
ranking. Table 5 reports an unclustered BiLSTM RMSE of 129.383 W/m²;
cluster-specific scores include 107.002 (cloudy), 144.133 (partly cloudy), and
93.546 (clear). These category scores do not by themselves show that every
regime benefits. Table 6 reports 85.765 W/m² for the final probability-weighted
method, subject to the availability issue below.

## Forecast-time availability problem in the paper

Section 2.7, equation (28), computes its probability-weighting distance from
the **true GHI at the predicted time** and each specialist forecast. That true
GHI is unavailable when issuing a day-ahead forecast. The text around Fig. 9
also describes using real and predicted GHI to estimate weights. As written,
that weighting procedure cannot be used for live day-ahead inference. A valid
adaptation must learn routing weights from information available by 00:00 IST,
or use archived day-ahead weather forecasts with their issue timestamps.

The paper is a research reference, not an instruction to alter our train/test
split or copy its reported gains. The first useful implementation is a simple
forecast-time-safe regime or similar-day ablation; the VAE weighting is lower
priority until its inputs can be made causal.
