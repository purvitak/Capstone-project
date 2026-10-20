# Saved results

All reported irradiance errors use W/m² unless explicitly normalized. Development
scores and test-year scores are different evaluations; do not compare them directly.

## Current evaluated forecasts

| Folder | Purpose |
| --- | --- |
| `ghi_search/` | Latest selected hybrid, eight-candidate development search, frozen decisions and test evaluation; start with [REPORT.md](ghi_search/REPORT.md). |
| `dwt_10min_150_paper/` | Previous DWT–BiLSTM/corrected reference, original figures and descriptive hourly comparison with Bektaş & Altaş. |
| `verification/` | Dataset quality, timestamps and clear-sky-mode audits. |

## Required reference and reproduction artifacts

| Folder | Why retained |
| --- | --- |
| `scaled_100epochs/` | Saved BiLSTM OOF component required by the blend and frozen search. |
| `context_hgb/` | Original context-tree OOF component required by the blend and frozen search. |
| `hybrid/` | Original fixed 50/50 development reference required by the frozen search. |
| `15epochs/` | Original baseline, dashboard loader's legacy default and regression-test fixture. |

Keep these paths and the files inside `ghi_search/` unchanged: the frozen search
checks source/data fingerprints and selected prediction hashes when resuming.
The eight candidate checkpoints also preserve exact development predictions and
per-fold selected iterations; they are not disposable copies.

## Archived experiments

`archive/` holds `context_ghi_dev/`, `context_residual_dev/`,
`novelty_3day_screen/`, and `novelty_regime_gate/`. They are earlier development
screens, not production forecasts or independent test results. Their arrays and
metadata are retained to support the negative/inconclusive research findings.

Original JSON provenance remains unchanged after moving these four folders;
absolute output paths recorded in old reports may describe their former locations.
The old smoke predictions were removed because they were short code checks,
not reproducible full-run research results. New smoke runs should use a separate
output directory as shown in the root README.
