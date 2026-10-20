"""Physical-unit scoring, chronological isolation, selection and frozen contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pv_pipeline"))
import hgb_training as training
import experiment_ghi_search as search


class DirectGHITrainingTests(unittest.TestCase):
    def test_conversion_uses_proxy_and_existing_bounds(self):
        cs = np.array([[0., 49., 50., 100., 500., 800.]])
        ghi = np.array([[20., 20., 20., -10., 250., 2000.]])
        np.testing.assert_allclose(training.ghi_to_kt(ghi, cs),
                                   [[0., 0., 0., 0., .5, 1.3]])

    def test_chronological_purge_daylight_rows_and_refit(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(20, 4, 2)).astype(np.float32)
        cs = np.full((20, 4), 600.)
        ghi = rng.uniform(100, 500, (20, 4))
        mask = np.zeros_like(ghi, dtype=bool)
        mask[:, 1:3] = True
        # Purge is row 14, validation is 15:20. Neither may fit the first tree.
        ghi[14:] = 900
        original = training.new_model
        fitted = []
        def factory(**params):
            model = original(**params)
            old_fit = model.fit
            def record(features, target):
                fitted.append((features.copy(), target.copy(), model))
                return old_fit(features, target)
            model.fit = record
            return model
        with threadpool_limits(limits=1), patch.object(training, "new_model", side_effect=factory):
            model, info = training.fit_direct_ghi(
                x, ghi, cs, mask, val_fraction=.25, max_iter=5, min_samples_leaf=2)
        self.assertEqual(len(fitted), 2)
        np.testing.assert_array_equal(fitted[0][0], training.flat_features(x[:14]).reshape(14, 4, -1)[mask[:14]])
        np.testing.assert_array_equal(fitted[0][1], ghi[:14][mask[:14]])
        self.assertEqual(len(fitted[1][1]), 40)
        self.assertFalse(model.early_stopping)
        self.assertEqual(info["inner_train_days"], 14)
        self.assertEqual(info["inner_validation_days"], 5)
        self.assertEqual(info["refit_days"], 20)
        self.assertEqual(model.max_iter, info["selected_iterations"])
        with threadpool_limits(limits=1):
            pred = training.predict_direct_kt(model, x, cs)
        self.assertEqual(pred.shape, ghi.shape)
        self.assertTrue(np.isfinite(pred).all())

    def test_night_targets_do_not_affect_selection_or_fit(self):
        rng = np.random.default_rng(6)
        x = rng.uniform(size=(15, 4, 3)).astype(np.float32)
        cs = np.full((15, 4), 600.)
        y = rng.uniform(100, 500, cs.shape)
        mask = np.tile([False, True, True, False], (15, 1))
        changed = y.copy()
        changed[~mask] = 1e8
        outputs = []
        with threadpool_limits(limits=1):
            for target in (y, changed):
                model, info = training.fit_direct_ghi(x, target, cs, mask, max_iter=5, min_samples_leaf=2)
                outputs.append((training.predict_direct_kt(model, x, cs), info))
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        self.assertEqual(outputs[0][1], outputs[1][1])

    def test_invalid_training_alignment_and_empty_daylight_fail(self):
        x = np.ones((10, 4, 2))
        y = cs = np.ones((10, 4))
        with self.assertRaises(ValueError):
            training.fit_direct_ghi(x, y[:, :3], cs, np.ones_like(cs, bool))
        with self.assertRaises(ValueError):
            training.fit_direct_ghi(x, y, cs, np.zeros_like(cs, bool))

    def test_iteration_selection_uses_ghi_error_not_index_error(self):
        x = np.ones((20, 2, 1))
        cs = np.tile([100., 1000.], (20, 1))
        y = cs * .5
        class Tree:
            max_iter = 2
            def fit(self, x, y):
                return self
            def staged_predict(self, x):
                yield np.tile([60., 490.], len(x) // 2)
                yield np.tile([50., 520.], len(x) // 2)
        # Iteration 1: GHI RMSE 10 vs 14.14 at iteration 2. Index errors
        # would instead favor iteration 2 because its error occurs at high CS.
        with patch.object(training, "new_model", return_value=Tree()):
            _, info = training.fit_direct_ghi(x, y, cs, np.ones_like(y, bool))
        self.assertEqual(info["selected_iterations"], 1)
        self.assertAlmostEqual(info["inner_validation_rmse_wm2"], 10.)


class SearchProtocolTests(unittest.TestCase):
    def test_exactly_eight_fixed_grid_configurations(self):
        configs = dict(search.candidates())
        self.assertEqual(len(configs), 8)
        self.assertEqual({v["max_iter"] for v in configs.values()}, {200})
        self.assertEqual({v["learning_rate"] for v in configs.values()}, {.05})

    def test_selection_requires_both_metrics_and_uses_only_overall_rows(self):
        ref = {"rmse": 109.11260573423584, "mae": 69.40719016284311}
        rows = [dict(method="rmse_only", season="all", rmse=100., mae=70.),
                dict(method="mae_only", season="all", rmse=110., mae=60.),
                dict(method="clear_days", season="non_monsoon", rmse=50., mae=40.)]
        self.assertIsNone(search.choose_candidate(rows, ref))
        rows += [dict(method="both", season="all", rmse=108., mae=68.)]
        self.assertEqual(search.choose_candidate(rows, ref)["method"], "both")

    def test_source_provenance_rejects_changed_settings_or_dates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.npz"
            data = {"ghi": np.zeros((2, 144)), "dates": np.array(["2020-01-01", "2020-01-02"])}
            config = {"prep_sha256": "original"}
            search.save_predictions(path, np.zeros((2, 144)), data["dates"], config)
            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                search.load_predictions(path, data, "changed")
            data["dates"][1] = "2020-01-03"
            with self.assertRaisesRegex(ValueError, "dates differ"):
                search.load_predictions(path, data, "original")

    def test_paired_bootstrap_preserves_pairing_and_pooled_daylight_counts(self):
        n = 20
        mask = np.ones((n, 144), bool)
        mask[:10, :72] = False
        data = {"mask": mask, "ghi": np.full((n, 144), 300.),
                "cs_proxy": np.full((n, 144), 600.)}
        pred = np.full((n, 144), .55)
        ref = np.full((n, 144), .6)
        results = search.paired_bootstrap(data, pred, ref, np.arange(n), repeats=100)
        for values in results.values():
            self.assertAlmostEqual(values["difference"], -30.)
            self.assertAlmostEqual(values["ci95_lo"], -30.)
            self.assertAlmostEqual(values["ci95_hi"], -30.)
        identical = search.paired_bootstrap(data, pred, pred, np.arange(n), repeats=100)
        self.assertTrue(all(v == 0 for metric in identical.values() for v in metric.values()))

    def test_prediction_day_observations_cannot_change_forecast_or_corrector_features(self):
        base = search.load_module("feature_isolation_base", "01_base_dwt_bilstm.py")
        prep = search.load_module("feature_isolation_prep", "00_prepare.py")
        rng = np.random.default_rng(8)
        n, day = 24, 18
        cs = np.full((n, 144), 600.)
        ghi = rng.uniform(100, 500, cs.shape)
        known = {"cs_proxy": cs, "zenith": np.full_like(cs, 45),
                 "doy": np.arange(n) + 1}
        data = {**known, "ghi": ghi, "kt": ghi / cs, "cs_solcast": cs.copy(),
                "mask": np.ones_like(cs, bool), "daylight": np.ones_like(cs, bool),
                "air_temp": np.full_like(cs, 25), "rh": np.full_like(cs, 70),
                "pres": np.full_like(cs, 1000), "wind": np.ones_like(cs)}
        def context(d):
            keys = ("kt", "cs_solcast", "cs_proxy", "zenith", "mask", "air_temp", "wind", "rh", "pres", "doy")
            return prep.build_features(**{k:d[k] for k in keys})
        data["X"] = context(data)
        changed = {k:v.copy() for k,v in data.items()}
        for k in ("ghi", "kt", "cs_solcast", "air_temp", "wind", "rh", "pres"):
            changed[k][day:] *= 3
        changed["X"] = context(changed)
        np.testing.assert_array_equal(search.feature_days(data, np.array([day]), base),
                                      search.feature_days(changed, np.array([day]), base))
        predictions = np.full_like(cs, .7)
        np.testing.assert_array_equal(search.correction_features(data, predictions)[day],
                                      search.correction_features(changed, predictions)[day])

    def test_correction_rejects_rmse_improvement_that_worsens_mae(self):
        n = 40
        cs = np.full((n, 144), 600.)
        truth = np.full_like(cs, .51)
        truth[:, :14] = .1
        pred = np.full_like(cs, .5)
        data = {"ghi": truth * cs, "kt": truth, "cs_proxy": cs,
                "cs_solcast": cs, "mask": np.ones_like(cs, bool),
                "daylight": np.ones_like(cs, bool), "zenith": np.full_like(cs, 45),
                "doy": np.arange(n) + 1, "is_monsoon": np.arange(n) % 2 == 0,
                "dates": np.arange(n).astype(str)}
        class Tree:
            def fit(self, x, y):
                return self
            def staged_predict(self, x):
                yield np.full(len(x), -.01)
        with tempfile.TemporaryDirectory() as folder, patch.object(search, "correction_model", return_value=Tree()):
            result = search.validate_correction(data, pred, np.arange(1, n), Path(folder))
            saved = np.load(Path(folder) / "correction_validation.npz")
            np.testing.assert_array_equal(saved["kt_final"], pred[saved["days"]])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["bounds"], {"monsoon": 0., "non_monsoon": 0.})
        # Establish that this was a real RMSE/MAE tradeoff, not simply a bad model.
        grid = [r for r in result["delta_grid_metrics"] if r["season"] == "monsoon"]
        baseline = next(r for r in grid if r["delta"] == 0)
        tradeoff = next(r for r in grid if r["delta"] == .01)
        self.assertLess(tradeoff["rmse"], baseline["rmse"])
        self.assertGreater(tradeoff["mae"], baseline["mae"])


if __name__ == "__main__":
    unittest.main()
