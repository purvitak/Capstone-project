"""Checks for the GFS-guided DWT-BiLSTM and cloud-regime blend."""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hybrid_model as H
from hybrid_model import P


def synthetic_sources(n=P.WINDOW_DAYS + 10, seed=0):
    rng = np.random.default_rng(seed)
    profile = np.clip(np.sin((np.arange(24) - 6) / 12 * np.pi), 0, None) * 900
    ghi = profile[None, :] * rng.uniform(0.2, 1.0, (n, 1))
    cs = np.repeat(profile[None, :], n, axis=0)
    return {"ghi_h": ghi, "cs_h": cs, "met_obs": rng.normal(28, 3, (n, 24, 3)),
            "type_obs": P.weather_types(P.daily_clearness(ghi, np.arange(n) % 365 + 1, 19.0)),
            "nwp_kt": rng.uniform(0, 1, (n, 24, 3)), "nwp_cloud": rng.uniform(0, 100, (n, 24, 3)),
            "nwp_k": rng.uniform(0.2, 0.7, (n, 3)), "gfs_met": rng.normal(28, 3, (n, 24, 3)),
            "dwt": P.dwt_by_day(ghi, range(P.WINDOW_DAYS, n))}


class HybridChecks(unittest.TestCase):
    def test_inputs_ignore_day_d_observations(self):
        src = synthetic_sources()
        day = P.WINDOW_DAYS + 4
        norm = H.normalizer(src, np.arange(P.WINDOW_DAYS + 1, P.WINDOW_DAYS + 4))
        before = H.build_inputs(src, [day], norm)
        self.assertEqual(before.shape, (1, 24, len(H.feature_names(H.NWP_SETS["multi"]))))
        self.assertEqual(len(H.FEATURES), 18)
        changed = {k: v.copy() for k, v in src.items()}
        changed["ghi_h"][day:] *= 2
        changed["met_obs"][day:] += 5
        changed["type_obs"][day:] = (changed["type_obs"][day:] + 1) % 3
        changed["dwt"] = P.dwt_by_day(changed["ghi_h"], range(P.WINDOW_DAYS, len(changed["ghi_h"])))
        np.testing.assert_array_equal(before, H.build_inputs(changed, [day], norm))

    def test_regimes_and_blend(self):
        cs = np.full((3, 24), 500.0)
        cloud = np.stack([np.full(24, 10.0), np.full(24, 50.0), np.full(24, 90.0)])
        regimes = H.regime_of(cloud, cs)
        np.testing.assert_array_equal(regimes, [0, 1, 2])
        out = H.blend(np.zeros((3, 24)), np.ones((3, 24)), regimes, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(out[:, 0], [0.0, 0.5, 1.0])

    def test_weight_selection_recovers_better_expert(self):
        rng = np.random.default_rng(1)
        actual = rng.uniform(100, 800, (40, 24))
        good, bad = actual + rng.normal(0, 5, actual.shape), actual + rng.normal(0, 80, actual.shape)
        regimes = np.repeat([0, 1], 20)
        weights = H.select_weights(actual, bad, good, np.ones_like(actual, bool), regimes, np.linspace(0, 1, 11))
        self.assertGreaterEqual(min(weights[:2]), 0.9)
        self.assertEqual(weights[2], 0.0)  # no overcast days: keep the default

    def test_missing_model_uses_mean_of_available_models(self):
        values = np.array([[[1.0, 3.0, 9.0]], [[2.0, 4.0, 6.0]]])  # (days, hours, models)
        available = np.array([[True, True, False], [True, True, True]])
        out = H.fill_missing_models(values, available)
        np.testing.assert_allclose(out[0, 0], [1.0, 3.0, 2.0])
        np.testing.assert_allclose(out[1], values[1])

    def test_night_rule(self):
        cs = np.array([[0.0, 5.0, 300.0, 300.0]])
        out = H.zero_at_night(np.array([[20.0, 20.0, -4.0, 150.0]]), cs)
        np.testing.assert_allclose(out, [[0.0, 0.0, 0.0, 150.0]])

    def test_ten_minute_rebuild_preserves_hourly_mean(self):
        cs = np.tile(np.linspace(100, 700, 144), (2, 1))
        hourly = np.full((2, 24), 80.0)  # below the 1.3 × clear-sky cap in every slot
        ten = H.to_10min(hourly, cs)
        np.testing.assert_allclose(H.hourly(ten), hourly, rtol=1e-9)
        self.assertTrue(np.all(H.to_10min(np.full((2, 24), 1e4), cs) <= 1.3 * cs))


if __name__ == "__main__":
    unittest.main()
