"""Physical units and forecast-time availability for the dashboard PV helper."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pv_pipeline"))
from pv_model import PVParameters, ghi_to_pv, historical_temperature_proxy, daily_energy_kwh


class PVModelChecks(unittest.TestCase):
    def test_paper_noct_reference_condition(self):
        power, cell = ghi_to_pv(800.0, 20.0)
        self.assertAlmostEqual(float(cell), 45.0)
        self.assertAlmostEqual(float(power), 72.8)

    def test_capacity_scaling_and_night(self):
        small, _ = ghi_to_pv([0.0, 1000.0], [np.nan, 25.0], PVParameters(capacity_kw=10))
        large, _ = ghi_to_pv([0.0, 1000.0], [np.nan, 25.0], PVParameters(capacity_kw=100))
        self.assertEqual(small[0], 0.0)
        np.testing.assert_allclose(large, 10 * small)
        self.assertAlmostEqual(large[1], 85.9375)

    def test_future_temperature_cannot_change_forecast_proxy(self):
        observed = np.repeat(np.arange(10, dtype=float)[:, None], 144, axis=1)
        before = historical_temperature_proxy(observed)
        observed[6:] = 1000
        after = historical_temperature_proxy(observed)
        np.testing.assert_array_equal(before[6], after[6])
        np.testing.assert_allclose(after[6], 4.0)
        self.assertTrue(np.isnan(before[:3]).all())

    def test_energy_units_and_missing_steps(self):
        power = np.full((2, 144), 10.0)
        power[1, 60] = np.nan
        energy = daily_energy_kwh(power)
        self.assertAlmostEqual(energy[0], 240.0)
        self.assertTrue(np.isnan(energy[1]))


if __name__ == "__main__":
    unittest.main()
