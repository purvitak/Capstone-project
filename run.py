"""Selected XGBoost–NWP entry point; defaults to verifying the frozen evaluation."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
ENTRY = ROOT / "pipelines" / "xgboost_nwp" / "run.py"
HISTORY = ROOT / "data" / "references" / "historical_hybrid"
INCUMBENT = ROOT / "data" / "references" / "nwp_hgb"


def arguments(argv):
    defaults = ["--history", str(HISTORY), "--incumbent", str(INCUMBENT)]
    if not any(value == "--phase" or value.startswith("--phase=") for value in argv):
        defaults += ["--phase", "evaluate"]
    return [str(ENTRY), *defaults, *argv]


def main():
    sys.argv = arguments(sys.argv[1:])
    # The unchanged fingerprinted trainer accepts the relocated reference paths.
    runpy.run_path(str(ENTRY), run_name="__main__")
    result = ROOT / "results" / "xgboost_nwp" / "REPORT.md"
    if result.exists():
        text = result.read_text()
        text = text.replace("# Separate XGBoost–NWP experiment", "# Selected XGBoost–NWP pipeline")
        text = text.replace("The existing dashboard default is unchanged.", "The dashboard has been removed; this is the selected pipeline.")
        text = text.replace("All new code is in `pipelines/xgboost_nwp/`; all experiment artifacts are here. "
                            "The prior paper, HGB results and source cache are preserved.",
                            "Main model code is in `pipelines/xgboost_nwp/`; invoke the root `run.py`. "
                            "Required historical and NWP reference inputs are in `data/references/`. "
                            "The paper draft and source cache are preserved.")
        result.write_text(text)


if __name__ == "__main__":
    main()
