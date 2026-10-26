"""NWP chronology, physical alignment, fallback and experiment isolation checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pv_pipeline"))
import nwp_data as N
import experiment_nwp as E
import hgb_training as H
import experiment_ghi_search as L


def synthetic_input():
    dates = pd.date_range("2024-06-01", periods=4, freq="D").strftime("%Y-%m-%d").to_numpy()
    clock = np.arange(144) / 6
    cs = np.tile(np.maximum(0, 800*np.sin((clock-6)/12*np.pi)), (4, 1))
    # Restrict solar weights to the actual nominal daytime.
    cs[:, (clock<6)|(clock>=18)] = 0
    hours = pd.date_range("2024-05-31T16:00Z", "2024-06-04T20:00Z", freq="h")
    frame = pd.DataFrame({N.FIELDS[0]:np.full(len(hours), 200.),
                          N.FIELDS[1]:np.arange(len(hours), dtype=float),
                          N.FIELDS[2]:np.full(len(hours), 28.),
                          N.FIELDS[3]:np.full(len(hours), 70.),
                          N.FIELDS[4]:np.full(len(hours), 12.)}, index=hours)
    return {"dates": dates, "cs_proxy": cs}, frame


class ForecastAdapterTests(unittest.TestCase):
    def test_utc_ist_alignment_handles_pandas_microsecond_dates(self):
        data, frame = synthetic_input()
        values, audit = N.align_forecasts(data, frame)
        # The first/last local dates touch UTC hours outside this finite grid.
        # Conservatively reject those partial hours rather than renormalize them.
        np.testing.assert_array_equal(values["available"], [False, True, True, False])
        self.assertEqual(values["features"].shape, (4,144,5))
        # IST June 1 00:00 starts at UTC May 31 18:30. The weather midpoint
        # is 18:35, between source values 2 and 3, not a timestamp in 1970.
        self.assertAlmostEqual(values["features"][0,0,1], 2+35/60, places=5)
        self.assertEqual(values["availability_upper_bound_utc"][0,0], np.datetime64("2024-05-30T07:00:00"))
        self.assertLess(audit["hourly_energy_max_error_wm2"], 1e-8)

    def test_hourly_solar_energy_preserved_before_clipping(self):
        cs = np.array([0.,100.,200.,300.,200.,100., 300.,400.,500.,600.,700.,800.])
        hourly = np.repeat([150.,600.], 6)
        result, _ = N.solar_disaggregate(hourly, np.repeat([1,2],6), cs)
        np.testing.assert_allclose(result.reshape(2,6).mean(axis=1), [150.,600.])
        self.assertEqual(result[0], 0.)
        self.assertGreater(result[-1], result[6])

    def test_future_availability_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds midnight"):
            N.availability_bounds(["2024-06-02T01:00"],["2024-06-01T00:00"], offset_hours=24, buffer_hours=12)
        bounds = N.availability_bounds(["2024-06-02T01:00"],["2024-06-01T00:00"])
        self.assertLessEqual(bounds[0], np.datetime64("2024-06-01T00:00"))

    def test_missing_hours_are_not_interpolated_across_and_trigger_whole_day_fallback(self):
        data, frame = synthetic_input()
        frame.loc[pd.Timestamp("2024-06-02T06:00Z"),N.FIELDS[2]]=np.nan
        values, _ = N.align_forecasts(data,frame)
        np.testing.assert_array_equal(values["available"],[False,False,True,False])
        fallback=np.full((4,144),.4)
        forecasts=np.full((4,144),.7)
        forecasts[1]=np.nan
        result=E.fallback_predictions(forecasts,values["available"],fallback)
        np.testing.assert_array_equal(result[1],fallback[1])
        np.testing.assert_array_equal(result[2],forecasts[2])
        forecasts[2,0]=np.nan
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            E.fallback_predictions(forecasts,values["available"],fallback)

    def test_observed_target_and_weather_cannot_change_aligned_forecasts(self):
        data, frame=synthetic_input()
        data["ghi"]=np.ones((4,144))
        data["air_temp"]=np.ones((4,144))
        original,_=N.align_forecasts(data,frame)
        data["ghi"][:]=9999;data["air_temp"][:]=9999
        other,_=N.align_forecasts(data,frame)
        for key in ("features","ghi","available","availability_upper_bound_utc"):
            np.testing.assert_array_equal(original[key],other[key])

    def test_cache_checks_query_protocol_and_raw_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            raw, meta=Path(folder)/"raw.json",Path(folder)/"meta.json"
            raw.write_text('{}')
            N.write_json(meta,{"protocol":N.PROTOCOL,"query":{"models":"different"},"sha256":N.sha256(raw)})
            with self.assertRaisesRegex(ValueError,"Incompatible"):
                N.read_chunk(raw,meta,{"models":"gfs_global"})
            N.write_json(meta,{"protocol":N.PROTOCOL,"query":{},"sha256":"wrong"})
            with self.assertRaisesRegex(ValueError,"checksum"):
                N.read_chunk(raw,meta,{})

    def test_aligned_cache_edits_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cache=Path(folder)
            np.savez(cache/"forecasts.npz",values=np.ones(3))
            (cache/"forecasts.sha256").write_text("wrong")
            with self.assertRaisesRegex(ValueError,"checksum"):
                N.load_forecasts(cache,cache/"prep.npz")

    def test_bad_units_and_duplicate_timestamps_are_rejected(self):
        response={"timezone":"UTC","utc_offset_seconds":0,
                  "hourly":{"time":["2024-06-01T00:00"]},"hourly_units":{}}
        for field,unit in zip(N.FIELDS,N.UNITS):
            response["hourly"][field]=[1.]
            response["hourly_units"][field]=unit
        response["hourly_units"][N.FIELDS[0]]="kWh/m²"
        with self.assertRaisesRegex(ValueError,"units"):
            N.decode_response(json.dumps(response))
        response["hourly"]["time"]*=2
        with self.assertRaisesRegex(ValueError,"unique"):
            N.decode_response(json.dumps(response))


class ModelProtocolTests(unittest.TestCase):
    def test_forecast_day_observations_cannot_change_issued_nwp_model_predictions(self):
        base = L.load_module("nwp_isolation_base", "01_base_dwt_bilstm.py")
        prep = L.load_module("nwp_isolation_prep", "00_prepare.py")
        rng = np.random.default_rng(18)
        n, day = 24, 20
        cs = np.full((n, 144), 600.)
        ghi = rng.uniform(100, 500, cs.shape)
        data = {"cs_proxy": cs, "zenith": np.full_like(cs, 45),
                "doy": np.arange(n) + 1, "ghi": ghi, "kt": ghi/cs,
                "cs_solcast": cs.copy(), "mask": np.ones_like(cs, bool),
                "daylight": np.ones_like(cs, bool), "air_temp": np.full_like(cs, 25),
                "rh": np.full_like(cs, 70), "pres": np.full_like(cs, 1000),
                "wind": np.ones_like(cs), "dev_idx": np.arange(8, 18),
                "purge_idx": np.array(18), "test_idx": np.arange(19, n)}
        keys = ("kt", "cs_solcast", "cs_proxy", "zenith", "mask", "air_temp", "wind", "rh", "pres", "doy")
        data["X"] = prep.build_features(**{k:data[k] for k in keys})
        changed = {k:v.copy() for k,v in data.items()}
        for key in ("ghi", "kt", "cs_solcast", "air_temp", "wind", "rh", "pres"):
            changed[key][day:] *= 3
        changed["X"] = prep.build_features(**{k:changed[k] for k in keys})
        nwp = {"ghi": np.full_like(cs, 300.), "features": np.ones((n, 144, 5))}
        original = E.assemble_features(data, nwp, base)[1]
        other = E.assemble_features(changed, nwp, base)[1]
        train = data["dev_idx"]
        np.testing.assert_array_equal(original[train], other[train])
        np.testing.assert_array_equal(original[day], other[day])
        with threadpool_limits(limits=1):
            model, _ = H.fit_direct_ghi(original[train], ghi[train], cs[train],
                                        data["mask"][train], baseline_ghi=nwp["ghi"][train],
                                        max_iter=3, min_samples_leaf=2)
            issued = [H.predict_direct_kt(model, x[[day]], cs[[day]], nwp["ghi"][[day]])
                      for x in (original, other)]
        np.testing.assert_array_equal(*issued)

    def test_residual_offset_is_restored_for_selection_and_prediction(self):
        rng=np.random.default_rng(3)
        x=rng.normal(size=(20,4,2)).astype(np.float32)
        cs=np.full((20,4),600.)
        baseline=np.full_like(cs,300.)
        ghi=np.full_like(cs,350.)
        with threadpool_limits(limits=1):
            model,info=H.fit_direct_ghi(x,ghi,cs,np.ones_like(cs,bool),baseline_ghi=baseline,max_iter=3,min_samples_leaf=2)
            pred=H.predict_direct_kt(model,x,cs,baseline_ghi=baseline)*cs
        np.testing.assert_allclose(pred,350.)
        self.assertAlmostEqual(info["inner_validation_rmse_wm2"],0.)
        self.assertEqual(info["target"],"ghi_residual_to_baseline")

    def test_unchanged_default_offset_matches_explicit_zero(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(12,4,2)).astype(np.float32)
        y=rng.uniform(100,500,(12,4));cs=np.full_like(y,600.)
        predictions=[]
        with threadpool_limits(limits=1):
            for offset in (None,np.zeros_like(y)):
                model,_=H.fit_direct_ghi(x,y,cs,np.ones_like(y,bool),baseline_ghi=offset,max_iter=3,min_samples_leaf=2)
                predictions.append(H.predict_direct_kt(model,x,cs,baseline_ghi=offset))
        np.testing.assert_array_equal(*predictions)

    def test_exact_grid_and_selection_requires_both_metrics(self):
        configs=dict(E.configurations())
        self.assertEqual(len(configs),8)
        self.assertEqual({r["max_iter"] for r in configs.values()},{300})
        ref={"rmse":100.,"mae":60.}
        rows=[{"candidate":"a","weight":1.,"rmse":90.,"mae":65.}]
        self.assertIsNone(E.select_on_tuning(rows,ref))
        rows += [{"candidate":"b","weight":.5,"rmse":95.,"mae":55.}]
        self.assertEqual(E.select_on_tuning(rows,ref)["candidate"],"b")

    def test_outer_training_and_inner_selection_exclude_future_forecast_days(self):
        dates=pd.date_range("2024-01-01",periods=450,freq="D").strftime("%Y-%m-%d").to_numpy()
        data={"dev_idx":np.arange(450),"dates":dates}
        available=np.ones(450,bool);available[20:30]=False
        for train,pred in E.outer_folds(data,available):
            self.assertGreaterEqual(len(train),180)
            self.assertTrue(available[train].all())
            self.assertGreaterEqual(np.datetime64(dates[pred[0]])-np.datetime64(dates[train[-1]]),np.timedelta64(2,"D"))
            self.assertFalse(np.intersect1d(train,pred).size)

    def test_checkpoint_settings_and_frozen_artifact_edits_are_rejected(self):
        data={"dates":np.array(["2024-01-01"]),"ghi":np.zeros((1,144))}
        config={"target":"direct"}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"pred.npz"
            L.save_predictions(path,np.ones((1,144)),data["dates"],config)
            with self.assertRaisesRegex(ValueError,"settings differ"):
                E.load_checkpoint(path,data,{"target":"residual"})
            digest=N.sha256(path)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,"Frozen artifact changed"):
                E.check_hashes(Path(folder),{"pred.npz":digest})

    def test_final_export_preserves_selected_oof_and_whole_test_fallback(self):
        data = {"test_idx": np.array([3,4])}
        development = np.array([[.1],[.2],[np.nan],[np.nan],[np.nan]])
        reference = np.full((5,1), .6)
        values = E.evaluation_values(data, reference, development)
        np.testing.assert_array_equal(values[:3], development[:3])
        np.testing.assert_array_equal(values[3:], reference[3:])
        self.assertTrue(np.isnan(development[3:]).all())

    def test_partial_candidate_is_reported_unfinished_despite_training_log(self):
        data = {"dates": pd.date_range("2024-01-01", periods=450).strftime("%Y-%m-%d").to_numpy(dtype="U10"),
                "dev_idx": np.arange(450), "ghi": np.zeros((450,144))}
        nwp = {"available": np.ones(450,bool)}
        folds = E.outer_folds(data, nwp["available"])
        (name,config), (complete_name,complete_config) = list(E.configurations())[:2]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            for candidate, params, days in ((name,config,folds[0][1]),
                                            (complete_name,complete_config,np.concatenate([p for _,p in folds]))):
                (out/candidate).mkdir()
                (out/candidate/"training.json").write_text('{}')
                pred = np.full_like(data["ghi"], np.nan)
                pred[days] = .5
                L.save_predictions(out/candidate/"dwt_base_pred.npz",pred,data["dates"],
                                   E.checkpoint_config("manifest",candidate,params))
            pending = E.unfinished_candidates(out,data,nwp,"manifest")
            self.assertIn(name,pending)
            self.assertNotIn(complete_name,pending)

    def test_expired_budget_stops_without_training(self):
        budget=N.Budget(1)
        with patch.object(N.time,"monotonic",return_value=budget.deadline+1):
            with self.assertRaises(N.BudgetExpired):budget.check()


if __name__=="__main__":
    unittest.main()
