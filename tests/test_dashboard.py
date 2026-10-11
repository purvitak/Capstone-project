"""Dashboard loading and forecast-time PV inputs."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.backend.forecast_data import (
    day_report, discover_sources, forecast_temperature_proxy, load_forecast,
)
from pv_pipeline.pv_model import PVParameters


class DashboardChecks(unittest.TestCase):
    def test_forecast_temperature_uses_prior_days_only(self):
        observed = np.repeat(np.arange(5, dtype=float)[:, None], 144, axis=1)
        proxy = forecast_temperature_proxy(observed)
        observed[3] = 999.0
        unchanged = forecast_temperature_proxy(observed)
        np.testing.assert_allclose(proxy[3], unchanged[3])
        np.testing.assert_allclose(proxy[3], 1.0)

    def test_real_saved_forecast_loads_and_reports_144_steps(self):
        base = ROOT / "results" / "15epochs" / "dwt_base_pred.npz"
        if not base.is_file():
            self.skipTest("Example forecast artifact is unavailable")
        store = load_forecast(base_path=base)
        self.assertEqual(len(store.available_test_idx), 365)
        day = int(store.available_test_idx[0])
        frame, report = day_report(store, day, PVParameters())
        self.assertEqual(len(frame), 144)
        self.assertGreater(report["scored_steps"], 0)
        self.assertTrue(np.isfinite(report["predicted_energy_kwh"]))
        self.assertGreater(report["predicted_energy_kwh"], 0)

    def test_rejects_misaligned_dates(self):
        base = ROOT / "results" / "15epochs" / "dwt_base_pred.npz"
        if not base.is_file():
            self.skipTest("Example forecast artifact is unavailable")
        with np.load(base, allow_pickle=False) as saved, tempfile.TemporaryDirectory() as tmp:
            dates = saved["dates"].copy()
            dates[0] = "1900-01-01"
            altered = Path(tmp) / "wrong_dates.npz"
            np.savez(altered, dates=dates, kt_base=saved["kt_base"])
            with self.assertRaisesRegex(ValueError, "dates differ"):
                load_forecast(base_path=altered)

    def test_corrected_run_keeps_night_slots_from_base(self):
        folder = ROOT / "results" / "dwt_10min_150_paper"
        base = folder / "dwt_base_pred.npz"
        corrected = folder / "dwt_stage2.npz"
        if not (base.is_file() and corrected.is_file()):
            self.skipTest("Corrected example artifacts are unavailable")
        store = load_forecast(base_path=base, stage2_path=corrected)
        self.assertEqual(len(store.available_test_idx), 365)
        self.assertTrue(np.isfinite(store.predicted_ghi[store.test_idx]).all())
        self.assertTrue(any("corrected" in label for label in discover_sources()))


if __name__ == "__main__":
    unittest.main()
