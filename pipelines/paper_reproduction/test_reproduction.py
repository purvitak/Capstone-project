"""Checks for the Bektaş & Altaş reproduction: formulas, causality and leakage."""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "pv_pipeline"))
sys.path.insert(0, str(HERE))
import paper_model as P
from run import paired_bootstrap


def synthetic_sources(n=P.WINDOW_DAYS + 10, seed=0):
    rng = np.random.default_rng(seed)
    hours = np.arange(24)
    profile = np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None) * 900
    ghi = profile[None, :] * rng.uniform(0.2, 1.0, (n, 1))
    met = rng.normal(28, 3, (n, 24, 3))
    k = P.daily_clearness(ghi, np.arange(n) % 365 + 1, 19.0)
    types = P.weather_types(k)
    sources = {"ghi_h": ghi, "dwt": P.dwt_by_day(ghi, range(P.WINDOW_DAYS, n)),
               "type_obs": types, "met_obs": met,
               "type_gfs_raw": types[::-1].copy(), "met_gfs_raw": met[::-1].copy(),
               "type_gfs_calibrated": types[::-1].copy(), "met_gfs_calibrated": met[::-1].copy()}
    norm = {"ghi_scale": 1000.0, "met_min": np.zeros(3), "met_span": np.full(3, 50.0)}
    return sources, norm


class FormulaChecks(unittest.TestCase):
    def test_extraterrestrial_matches_reference_values(self):
        # Equator at an equinox: H0 ≈ 24/π · Isc ≈ 10.4 kWh/m² per day.
        h0 = P.extraterrestrial_daily_wh(81, 0.0)
        self.assertAlmostEqual(float(h0) / 1000, 24 / np.pi * 1.367, delta=0.15)
        # Mumbai in June exceeds Mumbai in December.
        self.assertGreater(P.extraterrestrial_daily_wh(172, 19.08), P.extraterrestrial_daily_wh(355, 19.08))

    def test_weather_type_thresholds_follow_the_paper(self):
        np.testing.assert_array_equal(P.weather_types([0.1, 0.3, 0.5]), [0, 1, 2])
        self.assertTrue(np.isnan(P.weather_types([np.nan])[0]))

    def test_mra_components_sum_to_the_wavelet_reconstruction(self):
        import pywt
        x = np.random.default_rng(1).normal(size=P.WINDOW_DAYS * 24)
        parts = P.mra(x)
        self.assertEqual(parts.shape, (len(x), P.N_COMPONENTS))
        coeffs = pywt.wavedec(x, P.WAVELET, mode="symmetric", level=P.LEVELS)
        full = pywt.waverec(coeffs, P.WAVELET, mode="symmetric")[: len(x)]
        np.testing.assert_allclose(parts.sum(axis=1), full, atol=1e-9)
        # dmey is a finite approximation of the Meyer wavelet: near, not exact, reconstruction.
        self.assertLess(np.sqrt(np.mean((full - x) ** 2)) / np.sqrt(np.mean(x ** 2)), 0.02)

    def test_quantile_map_is_monotone_and_matches_target_range(self):
        rng = np.random.default_rng(2)
        source, target = rng.normal(1, 1, 500), rng.normal(5, 2, 500)
        mapped = P.quantile_map(source, target, np.sort(source))
        self.assertTrue(np.all(np.diff(mapped) >= -1e-12))
        self.assertAlmostEqual(mapped.mean(), target.mean(), delta=0.1)


class LeakageChecks(unittest.TestCase):
    def test_wavelet_features_ignore_day_d_and_later(self):
        sources, _ = synthetic_sources()
        ghi = sources["ghi_h"]
        day = P.WINDOW_DAYS + 3
        before = P.causal_dwt(ghi, day)
        changed = ghi.copy()
        changed[day:] *= 3
        np.testing.assert_array_equal(before, P.causal_dwt(changed, day))

    def test_operational_inputs_ignore_day_d_observations(self):
        sources, norm = synthetic_sources()
        day = P.WINDOW_DAYS + 5
        for variant in ("gfs_raw", "gfs_calibrated", "persistence"):
            for model in (1, 2):
                before = P.build_inputs(sources, [day], model, variant, norm)
                changed = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sources.items()}
                changed["type_obs"][day] = (changed["type_obs"][day] + 1) % 3
                changed["met_obs"][day] += 10
                changed["ghi_h"][day] *= 2
                np.testing.assert_array_equal(before, P.build_inputs(changed, [day], model, variant, norm),
                                              err_msg=f"{variant} model {model}")

    def test_oracle_uses_day_d_type_as_published(self):
        sources, norm = synthetic_sources()
        day = P.WINDOW_DAYS + 5
        x = P.build_inputs(sources, [day], 1, "oracle", norm)
        self.assertEqual(x.shape, (1, 24, 9))
        self.assertAlmostEqual(float(x[0, 0, 8]), sources["type_obs"][day] / 2)
        self.assertEqual(P.build_inputs(sources, [day], 2, "oracle", norm).shape, (1, 24, 15))


class NetworkChecks(unittest.TestCase):
    def test_network_trains_and_predicts_24_hours(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(40, 24, 9)).astype(np.float32)
        y = rng.uniform(size=(40, 24)).astype(np.float32)
        net, info = P.train_network(x[:30], y[:30], x[30:], y[30:], hidden=8, seed=0,
                                    max_epochs=3, patience=2)
        self.assertEqual(P.predict(net, x).shape, (40, 24))
        self.assertLessEqual(info["best_epoch"], info["epochs_run"])

    def test_identical_forecasts_have_zero_bootstrap_difference(self):
        rng = np.random.default_rng(4)
        actual = rng.uniform(0, 800, (30, 24))
        pred = actual + rng.normal(0, 50, actual.shape)
        result = paired_bootstrap(actual, pred, pred, np.ones_like(actual, dtype=bool), repeats=50)
        self.assertEqual(result["rmse"]["difference"], 0.0)
        self.assertEqual(result["mae"]["ci95_hi"], 0.0)


if __name__ == "__main__":
    unittest.main()
