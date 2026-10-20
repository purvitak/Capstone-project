# Documentation index

Start with the root [README](../README.md) for setup and commands.

## Current implementation and results

- [Latest evaluated model](../results/ghi_search/REPORT.md): the selected 25% BiLSTM /
  75% direct-GHI HGB blend and correction, with test RMSE 107.42 and MAE 68.95 W/m².
- [Saved-result map](../results/README.md): current forecasts, dependencies and archived experiments.
- [Dashboard](DASHBOARD.md): saved-forecast loading, issue-time inputs and illustrative PV calculations.
- [Pipeline diagram](figures/pipeline.svg) and [Mermaid source](figures/pipeline.mmd):
  the original DWT–BiLSTM/residual pipeline; the latest model adds the HGB blend
  described in the experiment report.

## Research background

- [Research-paper master](RESEARCH_PAPER_MASTER.md): dataset, original DWT–BiLSTM
  methodology, base-paper comparison and limitations. Its numerical results are
  the earlier 150-epoch-maximum run; use the latest report above for the hybrid.
- [Model review](MODEL_REVIEW.md): training changes, the original hybrid and negative neural screens.
- [Four-paper novelty review](references/FOUR_PAPERS_NOVELTY_REVIEW.md): rationale and results for history/regime ablations.
- [PV-model reference](references/SHANKER_KULKARNI_REVIEW.md): paper-based NOCT conversion and its limitations.
- [Clustering reference](references/WANG_2025_CLUSTERING_REVIEW.md): forecast-time availability concerns in regime weighting.

## Historical material

[Archive index](archive/README.md) covers the earlier research direction,
October 12 results, detailed historical protocol, and preserved legacy Git history.
These records provide context; they are not the current run instructions.
