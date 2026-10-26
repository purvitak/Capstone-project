"""Cleanup must preserve the selected hybrid's inputs and frozen numerical artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("project_entry", ROOT / "run.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class ProjectLayoutTests(unittest.TestCase):
    def test_entry_point_supplies_relocated_references_and_safe_default(self):
        args = entry.arguments([])
        self.assertEqual(args[args.index("--phase") + 1], "evaluate")
        self.assertEqual(Path(args[args.index("--history") + 1]), ROOT / "data/references/historical_hybrid")
        self.assertEqual(Path(args[args.index("--incumbent") + 1]), ROOT / "data/references/nwp_hgb")
        custom = entry.arguments(["--phase", "development", "--out", "results/new_run"])
        self.assertEqual(custom.count("--phase"), 1)
        self.assertEqual(custom[-2:], ["--out", "results/new_run"])

    def test_removed_runtime_components_and_dependency_stay_removed(self):
        for relative in ("dashboard", "pipelines/attention_nwp", "results/attention_nwp", "pv_pipeline/pv_model.py"):
            self.assertFalse((ROOT / relative).exists(), relative)
        requirements = (ROOT / "requirements.txt").read_text().splitlines()
        self.assertFalse(any(line.lower().startswith("streamlit") for line in requirements))
        self.assertIn("xgboost==3.2.0", requirements)

    def test_required_reference_artifacts_match_frozen_selection(self):
        source = ROOT / "data/references/nwp_hgb"
        frozen = json.loads((source / "frozen_selection.json").read_text())
        for relative, digest in frozen["artifact_hashes"].items():
            self.assertEqual(hashlib.sha256((source / relative).read_bytes()).hexdigest(), digest, relative)

    def test_selected_model_and_predictions_remain_bit_identical(self):
        source = ROOT / "results/xgboost_nwp"
        report = json.loads((source / "evaluation.json").read_text())
        for relative, digest in report["artifact_hashes"].items():
            self.assertEqual(hashlib.sha256((source / relative).read_bytes()).hexdigest(), digest, relative)


if __name__ == "__main__":
    unittest.main()
