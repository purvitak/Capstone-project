# Historical archive

Use the [documentation index](../README.md) and root README for current results
and commands. Historical claims below describe their original experiment dates.

- `WORKFLOW.md`: detailed earlier preparation/evaluation protocol still referenced
  by the code; some commands and paths predate the current project layout.
- `RESEARCH.md`: original research direction, superseded by the research-paper
  master and latest experiment report.
- `IMPLEMENTATION_RESULTS_2026-10-12.md`: verified original DWT–BiLSTM/corrector
  results and base-paper comparison before the later hybrid search.
- `legacy-checkout.bundle`: complete Git history of the unused nested `capstone/`
  checkout, whose only commit was `2a32b1c1ba0779c199760e91fb406ea4566e1129`.
  The bundle was verified before removing the old checkout and its unused 601 MB
  Python environment. No unique working-tree source files remained there.

To inspect the legacy source in a separate directory:

```bash
git clone docs/archive/legacy-checkout.bundle /tmp/capstone-legacy
```

The obsolete handoff note was removed; its operational instructions were already
superseded by the root README and the research-paper master.
