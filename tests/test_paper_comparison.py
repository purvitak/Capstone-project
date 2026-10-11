"""Verify 10-minute to hourly aggregation used for the literature table."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pv_pipeline"))

from compare_paper import hourly_metrics


class PaperComparisonChecks(unittest.TestCase):
    def test_aggregates_each_full_hour_before_scoring(self):
        actual = np.zeros((1, 144))
        predicted = np.zeros((1, 144))
        predicted[0, :6] = 6.0
        daylight = np.zeros((1, 144), dtype=bool)
        daylight[0, 0] = True
        result = hourly_metrics(actual, predicted, daylight)
        self.assertAlmostEqual(result["all24_rmse_wm2"], np.sqrt(36 / 24))
        self.assertAlmostEqual(result["daylight_rmse_wm2"], 6.0)
        self.assertEqual(result["daylight_hours"], 1)
        with self.assertRaises(ValueError):
            hourly_metrics(actual, predicted[:, :-1], daylight)


if __name__ == "__main__":
    unittest.main()
