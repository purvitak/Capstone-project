"""Load the final model's saved forecasts. No model fitting or fake inference.

Final model: NWP-guided DWT-BiLSTM (GFS + ECMWF + ICON) blended with the
XGBoost–NWP / historical forecast by forecast cloud regime
(pipelines/nwp_dwt_hybrid, run with --nwp-set multi --night-zero).
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.calculations import apply_pv, hourly_summary, pv_parameters, summary

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = "results/multi_nwp_hybrid"
MODEL_LABEL = "NWP-guided DWT-BiLSTM hybrid (GFS + ECMWF + ICON)"


class ForecastUnavailable(RuntimeError):
    pass


class ForecastNotFound(ForecastUnavailable):
    pass


class ForecastRepository:
    def __init__(self, root=ROOT):
        self.root = Path(root)

    def _paths(self):
        return (self.root / "data/prepared/prep.npz",
                self.root / MODEL_DIR / "predictions.npz",
                self.root / MODEL_DIR / "report.json")

    def _data(self):
        paths = self._paths()
        try:
            signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
            return _load(signature)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ForecastUnavailable("The saved forecast data is unavailable or incompatible. Check the backend data files.") from exc

    def available_dates(self):
        data = self._data()
        return sorted(data)

    def get_forecast(self, forecast_date, capacity_kwp=None, performance_ratio=None):
        # Enforce a date-only request; never coerce arbitrary timestamp/path inputs.
        try:
            parsed = pd.Timestamp(forecast_date)
            if parsed.strftime("%Y-%m-%d") != str(forecast_date):
                raise ValueError
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("Use a forecast date in YYYY-MM-DD format.") from exc
        parameters = pv_parameters(capacity_kwp, performance_ratio)
        data = self._data()
        if forecast_date not in data:
            raise ForecastNotFound("No saved forecast is available for this date. Choose a date in the forecast archive.")
        ghi, mask = data[forecast_date]
        times = pd.date_range(f"{forecast_date} 00:00", periods=144, freq="10min", tz="Asia/Kolkata")
        frame = pd.DataFrame({"timestamp": times, "interval_end": times + pd.Timedelta(minutes=10),
                              "ghi_forecast_wm2": ghi, "daylight": mask,
                              "interval_index": np.arange(144), "hour": times.hour})
        frame = apply_pv(frame, parameters)
        return {"date": forecast_date, "timezone": "Asia/Kolkata", "location": "Mumbai, Maharashtra, India",
                "issued_at": times[0].isoformat(), "resolution_minutes": 10,
                "source": {"kind": "saved_forecast", "label": "Connected saved forecast",
                           "model": MODEL_LABEL,
                           "live": False, "sample": False},
                "pv": parameters, "summary": summary(frame), "hourly": hourly_summary(frame),
                "intervals": _records(frame)}


def _records(frame):
    result = []
    for row in frame.to_dict("records"):
        row["timestamp"] = row["timestamp"].isoformat()
        row["interval_end"] = row["interval_end"].isoformat()
        row["pv_power_kw"] = None if not np.isfinite(row["pv_power_kw"]) else row["pv_power_kw"]
        result.append(row)
    return result


@lru_cache(maxsize=2)
def _load(signature):
    prep_path, pred_path, report_path = [Path(item[0]) for item in signature]
    report = json.loads(report_path.read_text())
    if report.get("smoke") or not report["frozen"]["adopted"]:
        raise ValueError("The saved run is a smoke test or was not adopted.")
    if hashlib.sha256(prep_path.read_bytes()).hexdigest() != report["prep_sha256"]:
        raise ValueError("Prepared data does not match the selected run's checksum.")
    if hashlib.sha256(pred_path.read_bytes()).hexdigest() != report["artifact_hashes"]["predictions.npz"]:
        raise ValueError("Saved predictions do not match the selected run's checksum.")
    with np.load(prep_path, allow_pickle=False) as d, np.load(pred_path, allow_pickle=False) as p:
        dates, indices = d["dates"], d["test_idx"]
        if not np.array_equal(dates[indices], p["dates"]):
            raise ValueError("Prepared and forecast dates do not align.")
        if len(indices) != report["test_days"] or len(np.unique(indices)) != len(indices):
            raise ValueError("Evaluation-date coverage does not match the selected run.")
        if d["cs_proxy"].shape != (len(dates), 144) or p["hybrid_10min"].shape != (len(indices), 144):
            raise ValueError("Expected 144 ten-minute periods per day.")
        ghi = p["hybrid_10min"]
        masks = d["mask"][indices]
        if not np.isfinite(ghi).all() or np.any(ghi < 0):
            raise ValueError("Selected forecast contains invalid irradiance.")
        if masks.dtype.kind != "b":
            raise ValueError("Daylight mask must be boolean.")
        if [str(dates[indices[0]]), str(dates[indices[-1]])] != report["test_period"]:
            raise ValueError("Evaluation dates do not match the selected run.")
        return {str(dates[i]): (values.copy(), mask.copy()) for i, values, mask in zip(indices, ghi, masks)}
