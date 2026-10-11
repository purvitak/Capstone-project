"""Training provenance and validation isolation checks."""
import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
import json
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pv_pipeline"))
spec = importlib.util.spec_from_file_location("base", ROOT / "pv_pipeline" / "01_base_dwt_bilstm.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
import blend_predictions as blend


class TrainingChecks(unittest.TestCase):
    def test_blend_requires_both_complete_forecasts(self):
        a = np.array([[0.2, 0.8], [0.2, np.nan], [0.3, 0.4]])
        b = np.array([[0.6, 1.0], [0.4, 0.6], [np.nan, np.nan]])
        result = blend.blend_arrays(a, b)
        np.testing.assert_allclose(result[0], [0.4, 0.9])
        self.assertTrue(np.isnan(result[1:]).all())
        with self.assertRaises(ValueError):
            blend.blend_arrays(a, b, 1.1)

    def test_blend_rejects_misaligned_dates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.npz"
            np.savez(path, dates=np.array(["2020-01-01"]), kt_base=np.ones((1, 144)),
                     config=json.dumps({"backend": "torch", "prep_sha256": "same"}))
            with self.assertRaisesRegex(ValueError, "dates differ"):
                blend.load_source(path, np.array(["2020-01-02"]), (1, 144), "same", "torch")

    def test_baseline_preserves_unavailable_history(self):
        values = np.repeat(np.arange(1, 5, dtype=float)[:, None], 144, axis=1)
        result = base.C.baseline_mean3_kt(values)
        self.assertTrue(np.isnan(result[0]).all())
        np.testing.assert_allclose(result[1:, 0], [1.0, 1.5, 2.0])

    def test_validation_inputs_do_not_fit_selection_scaler(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(12, 4, 2)).astype(np.float32)
        x[9:] += 1000  # validation values must not change the selection scaler
        y = rng.uniform(size=(12, 4)).astype(np.float32)
        torch.manual_seed(7)
        net = base.BiLSTMNet(2, 4, hidden=3)
        original = net.predict
        observed = []
        def inspect(a, batch_size=64):
            observed.append(net.mean.copy())
            return original(a, batch_size)
        with patch.object(net, "predict", side_effect=inspect):
            info = net.fit_validated(x, y, 3, 4,
                                    (9, np.ones((3, 4)), y[9:], np.ones((3, 4), bool), 7), 2)
        for mean in observed:
            np.testing.assert_allclose(mean, x[:8].mean(axis=(0, 1), keepdims=True))
        np.testing.assert_allclose(net.mean, x.mean(axis=(0, 1), keepdims=True))
        self.assertTrue(1 <= info["selected_epochs"] <= info["epochs_checked"] <= 3)
        self.assertTrue(np.isfinite(net.predict(x)).all())

    def test_prediction_uses_fixed_training_scaler(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(6, 4, 2)).astype(np.float32)
        x[:, :, 1] = 2  # constant channel must not divide by zero
        net = base.BiLSTMNet(2, 4, hidden=3)
        net.fit(x, np.zeros((6, 4), np.float32), 1, 3)
        before = net.mean.copy()
        a = net.predict(x[:1])
        b = net.predict(np.concatenate([x[:1], x[1:] * 1000]))[:1]
        np.testing.assert_allclose(a, b, atol=1e-6)
        np.testing.assert_array_equal(net.mean, before)
        self.assertTrue(np.isfinite(b).all())

    def test_zero_weight_night_targets_do_not_change_fit(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(6, 4, 2)).astype(np.float32)
        y = rng.uniform(size=(6, 4)).astype(np.float32)
        weights = np.ones_like(y)
        weights[:, :2] = 0
        changed = y.copy()
        changed[:, :2] += 1000
        predictions = []
        for target in (y, changed):
            torch.manual_seed(3)
            net = base.BiLSTMNet(2, 4, hidden=3)
            net.fit(x, target, 2, 3, weights=weights)
            predictions.append(net.predict(x))
        np.testing.assert_allclose(*predictions, atol=1e-7)

    def test_forecast_day_actuals_do_not_change_input_features(self):
        prep_spec = importlib.util.spec_from_file_location("prepare", ROOT / "pv_pipeline" / "00_prepare.py")
        prep = importlib.util.module_from_spec(prep_spec)
        prep_spec.loader.exec_module(prep)
        rng = np.random.default_rng(8)
        n, day = 24, 18
        cs = np.full((n, 144), 600.0)
        ghi = rng.uniform(100, 500, size=(n, 144))
        mask = np.ones_like(ghi, bool)
        ctx = dict(kt=ghi / cs, cs_solcast=cs.copy(), cs_proxy=cs,
                   zenith=np.full_like(cs, 45), mask=mask,
                   air_temp=rng.uniform(20, 30, size=cs.shape),
                   wind=np.ones_like(cs), rh=np.full_like(cs, 70),
                   pres=np.full_like(cs, 1000), doy=np.arange(n) + 1)
        reference = prep.build_features(**ctx)[day]
        changed = {key: value.copy() for key, value in ctx.items()}
        for key in ("kt", "cs_solcast", "air_temp", "wind", "rh", "pres"):
            changed[key][day:] *= 3
        np.testing.assert_array_equal(reference, prep.build_features(**changed)[day])
        types = base._daily_types(ghi, cs, mask)
        x, _ = base.build_sample(ghi, cs, mask, types, day, False, False, 6)
        ghi[day:] *= 3
        types2 = base._daily_types(ghi, cs, mask)
        x2, _ = base.build_sample(ghi, cs, mask, types2, day, False, False, 6)
        np.testing.assert_array_equal(x, x2)

    def test_three_day_dwt_history_uses_only_prior_days(self):
        rng = np.random.default_rng(11)
        ghi = rng.uniform(0, 800, size=(12, 144))
        cs = np.full_like(ghi, 850.0)
        mask = np.ones_like(ghi, dtype=bool)
        day = 8
        types = base._daily_types(ghi, cs, mask)
        x, _ = base.build_sample(ghi, cs, mask, types, day, False, False, 6,
                                 lookback_days=3)
        self.assertEqual(x.shape, (144, 25))
        changed = ghi.copy()
        changed[day:] = 0.0
        other_types = base._daily_types(changed, cs, mask)
        x2, _ = base.build_sample(changed, cs, mask, other_types, day, False,
                                  False, 6, lookback_days=3)
        np.testing.assert_array_equal(x, x2)
        changed[day - 3] = 0.0
        changed_types = base._daily_types(changed, cs, mask)
        x3, _ = base.build_sample(changed, cs, mask, changed_types, day, False,
                                  False, 6, lookback_days=3)
        self.assertFalse(np.array_equal(x, x3))


if __name__ == "__main__":
    unittest.main()
