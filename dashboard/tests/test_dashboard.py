"""Forecast contract, physical units, failure behaviour and real UI state checks."""
from datetime import date
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from dashboard.backend.api import app,repository
from dashboard.backend.repository import MODEL_DIR,MODEL_LABEL,ROOT,ForecastRepository,ForecastUnavailable
from dashboard.calculations import apply_pv,csv_bytes,energy_kwh,hourly_summary,pv_parameters,select_view


class CalculationTests(unittest.TestCase):
    def frame(self,n=144):
        times=pd.date_range("2026-02-20",periods=n,freq="10min",tz="Asia/Kolkata")
        return pd.DataFrame({"timestamp":times,"interval_end":times+pd.Timedelta(minutes=10),
                             "ghi_forecast_wm2":np.full(n,1000.),"hour":times.hour,
                             "interval_index":np.arange(n),"daylight":True})

    def test_no_parameters_means_no_pv_or_energy(self):
        f=apply_pv(self.frame(),None)
        self.assertTrue(f.pv_power_kw.isna().all())
        self.assertIsNone(energy_kwh(f))
        self.assertNotIn("pv_power_kw",csv_bytes(f).decode("utf-8-sig"))

    def test_period_means_integrate_with_six_samples_per_hour(self):
        f=apply_pv(self.frame(),pv_parameters(10,.8))
        self.assertAlmostEqual(energy_kwh(f),192.)
        hours=hourly_summary(f)
        self.assertEqual(len(hours),24)
        self.assertAlmostEqual(hours[12]["pv_mean_kw"],8.)
        self.assertAlmostEqual(hours[12]["energy_kwh"],8.)
        self.assertAlmostEqual(energy_kwh(select_view(f,12,72)),8/6)

    def test_timestamp_resolution_does_not_change_energy(self):
        f=apply_pv(self.frame(),pv_parameters(10,.8))
        f["timestamp"]=f.timestamp.dt.as_unit("us")
        self.assertAlmostEqual(energy_kwh(f),192.)

    def test_gaps_duplicates_and_missing_power_do_not_become_zero_energy(self):
        f=apply_pv(self.frame(),pv_parameters(10,.8))
        self.assertIsNone(energy_kwh(f.drop(index=6)))
        missing=f.copy();missing.loc[6,"pv_power_kw"]=np.nan
        self.assertIsNone(energy_kwh(missing))
        duplicate=f.copy();duplicate.loc[6,"timestamp"]=duplicate.loc[5,"timestamp"]
        self.assertIsNone(energy_kwh(duplicate))

    def test_invalid_pv_assumptions_rejected(self):
        for capacity,ratio in [(10,None),(None,.8),(-1,.8),(10,0),(10,1.1),(float("inf"),.8),(10,float("nan"))]:
            with self.subTest(capacity=capacity,ratio=ratio),self.assertRaises(ValueError):
                pv_parameters(capacity,ratio)

    def test_drilldown_scope_and_csv_fields(self):
        f=apply_pv(self.frame(),pv_parameters(10,.8))
        self.assertEqual(len(select_view(f,23)),6)
        self.assertEqual(len(select_view(f,23,143)),1)
        with self.assertRaises(ValueError):select_view(f,12,143)
        csv=pd.read_csv(io.BytesIO(csv_bytes(select_view(f,23))))
        self.assertEqual(len(csv),6)
        self.assertIn("pv_power_kw",csv)
        self.assertIn("+05:30",csv.timestamp.iloc[0])


class RepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.repo=ForecastRepository()

    def test_all_saved_dates_have_144_finite_periods_and_midnight_issue(self):
        dates=self.repo.available_dates()
        self.assertEqual((len(dates),dates[0],dates[-1]),(365,"2025-07-25","2026-07-24"))
        for day in dates:
            p=self.repo.get_forecast(day)
            self.assertEqual(len(p["intervals"]),144)
            self.assertEqual(p["issued_at"],day+"T00:00:00+05:30")
            self.assertFalse(p["source"]["sample"])
            self.assertFalse(p["source"]["live"])
            self.assertEqual(p["intervals"][-1]["timestamp"],day+"T23:50:00+05:30")

    def test_ghi_matches_final_model_saved_forecast_exactly(self):
        p=self.repo.get_forecast("2026-02-20")
        self.assertEqual(p["source"]["model"],MODEL_LABEL)
        with np.load(ROOT/MODEL_DIR/"predictions.npz",allow_pickle=False) as f:
            i=int(np.flatnonzero(f["dates"]==p["date"])[0])
            np.testing.assert_array_equal(f["hybrid_10min"][i],[x["ghi_forecast_wm2"] for x in p["intervals"]])

    def test_saved_pv_does_not_mutate_ghi_or_previous_request(self):
        a=self.repo.get_forecast("2026-02-20")
        b=self.repo.get_forecast("2026-02-20",10,.8)
        self.assertIsNone(a["summary"]["energy_kwh"])
        expected=sum(r["ghi_forecast_wm2"]*.008/6 for r in a["intervals"])
        self.assertAlmostEqual(b["summary"]["energy_kwh"],expected)
        self.assertIsNone(self.repo.get_forecast("2026-02-20")["pv"])

    def test_missing_files_fail_without_sample_substitution(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(ForecastUnavailable):ForecastRepository(path).available_dates()

    def test_incompatible_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);(root/"data/prepared").mkdir(parents=True);(root/MODEL_DIR).mkdir(parents=True)
            (root/"data/prepared/prep.npz").write_bytes(b"invalid")
            (root/MODEL_DIR/"predictions.npz").write_bytes(b"invalid")
            (root/MODEL_DIR/"report.json").write_text(json.dumps({"prep_sha256":"wrong","frozen":{"adopted":True},
                                                                  "artifact_hashes":{"predictions.npz":"wrong"}}))
            with self.assertRaises(ForecastUnavailable):ForecastRepository(root).available_dates()


class APITests(unittest.TestCase):
    def setUp(self):self.client=TestClient(app)

    def test_health_dates_and_forecast_contract(self):
        self.assertEqual(self.client.get("/api/health").json()["available_days"],365)
        self.assertEqual(len(self.client.get("/api/dates").json()["dates"]),365)
        r=self.client.get("/api/forecast/2026-02-20")
        self.assertEqual(r.status_code,200)
        self.assertEqual(len(r.json()["intervals"]),144)
        self.assertIsNone(r.json()["summary"]["energy_kwh"])

    def test_future_dates_and_invalid_parameters_are_not_invented(self):
        self.assertEqual(self.client.get("/api/forecast/2026-10-28").status_code,404)
        self.assertEqual(self.client.get("/api/forecast/not-a-date").status_code,422)
        for query in ["capacity_kwp=10","capacity_kwp=-1&performance_ratio=.8","capacity_kwp=10&performance_ratio=nan","capacity_kwp=inf&performance_ratio=.8"]:
            with self.subTest(query=query):self.assertEqual(self.client.get("/api/forecast/2026-02-20?"+query).status_code,422)

    def test_csv_is_filtered_and_units_are_not_mixed(self):
        r=self.client.get("/api/forecast/2026-02-20/csv?hour=12&capacity_kwp=10&performance_ratio=.8")
        self.assertEqual(r.status_code,200)
        f=pd.read_csv(io.BytesIO(r.content));self.assertEqual(len(f),6)
        self.assertIn("ghi_forecast_wm2",f);self.assertIn("pv_power_kw",f)
        self.assertEqual(self.client.get("/api/forecast/2026-02-20/csv?hour=12&interval=143").status_code,422)

    def test_backend_failure_is_503(self):
        with tempfile.TemporaryDirectory() as path:
            app.dependency_overrides[repository]=lambda:ForecastRepository(path)
            try:self.assertEqual(self.client.get("/api/health").status_code,503)
            finally:app.dependency_overrides.clear()


class UITests(unittest.TestCase):
    def test_day_hour_interval_navigation_and_pv_configuration(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ,{"SOLAR_DATA_MODE":"direct"}):
            at=AppTest.from_file(str(ROOT/"dashboard/app.py"),default_timeout=20).run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state["view"],"Day")
            at.button(key="explorer_nav").click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state["view"],"Hour")
            at.selectbox(key="hour_picker").select(23).run()
            self.assertEqual(at.session_state["hour_picker"],23)
            self.assertTrue(at.button(key="next_hour").disabled)
            at.selectbox(key="inspector").select(143).run()
            self.assertEqual(at.session_state["view"],"10-minute")
            self.assertEqual(at.session_state["interval_picker"],143)
            self.assertFalse(at.exception)
            at.number_input(key="capacity_kwp").set_value(10).run()
            at.number_input(key="performance_ratio").set_value(.8).run()
            self.assertFalse(at.exception)
            at.button(key="reset").click().run()
            self.assertEqual(at.session_state["view"],"Day")

    def test_backend_unavailable_has_retry_without_traceback(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ,{"SOLAR_DATA_MODE":"api","SOLAR_API_URL":"http://127.0.0.1:1"}):
            at=AppTest.from_file(str(ROOT/"dashboard/app.py"),default_timeout=20).run()
            self.assertFalse(at.exception)
            self.assertTrue(at.warning)
            self.assertIn("Cannot reach",at.warning[0].value)


if __name__=="__main__":unittest.main()
