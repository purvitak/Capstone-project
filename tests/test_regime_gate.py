"""Issue-time isolation and probability checks for the development gate."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pv_pipeline"))
import experiment_regime_gate as gate


class RegimeGateChecks(unittest.TestCase):
    def test_features_use_previous_day_only(self):
        rng = np.random.default_rng(6)
        shape = (8, 12)
        prep = {
            "mask": np.ones(shape, bool),
            "kt": rng.uniform(size=shape),
            "air_temp": rng.uniform(20, 30, size=shape),
            "rh": rng.uniform(40, 90, size=shape),
        }
        days = np.array([3, 4, 5])
        expected = gate.issue_time_features(prep, days)
        changed = {name: values.copy() for name, values in prep.items()}
        for name in ("kt", "air_temp", "rh"):
            changed[name][5:] *= 2
        np.testing.assert_array_equal(expected, gate.issue_time_features(changed, days))
        changed["kt"][2] = 0
        self.assertFalse(np.array_equal(expected, gate.issue_time_features(changed, days)))

    def test_soft_memberships_are_probabilities(self):
        x = np.array([[0., 1.], [1., 0.]])
        centers = np.array([[0., 0.], [1., 1.], [2., 2.]])
        probability = gate.soft_membership(x, centers, 0.5)
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)
        self.assertTrue((probability >= 0).all())


if __name__ == "__main__":
    unittest.main()
