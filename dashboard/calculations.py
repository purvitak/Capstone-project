"""Explicit units and period-mean calculations shared by the API and UI."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def pv_parameters(capacity_kwp=None, performance_ratio=None):
    if capacity_kwp is None and performance_ratio is None:
        return None
    if capacity_kwp is None or performance_ratio is None:
        raise ValueError("Supply both DC capacity and performance ratio, or neither.")
    capacity, ratio = float(capacity_kwp), float(performance_ratio)
    if not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("DC capacity must be a finite positive value in kWp.")
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("Performance ratio must be greater than zero and at most one.")
    return {"capacity_kwp": capacity, "performance_ratio": ratio,
            "method": "GHI-based illustrative estimate",
            "formula": "P_kW = capacity_kWp × (GHI_Wm2 / 1000) × performance_ratio",
            "assumptions": "GHI used as a plane-of-array proxy; no measured plant-power validation."}


def apply_pv(frame, parameters):
    result = frame.copy()
    result["pv_power_kw"] = (result["ghi_forecast_wm2"] / 1000
                              * parameters["capacity_kwp"] * parameters["performance_ratio"]
                              if parameters else np.nan)
    return result


def energy_kwh(frame):
    """Integrate complete, contiguous ten-minute period means, never skip gaps."""
    if frame.empty or "pv_power_kw" not in frame:
        return None
    values = frame["pv_power_kw"].to_numpy(dtype=float)
    times = pd.DatetimeIndex(frame["timestamp"])
    if not np.isfinite(values).all() or np.any(values < 0):
        return None
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        return None
    if len(times) > 1 and not ((times[1:] - times[:-1]) == pd.Timedelta(minutes=10)).all():
        return None
    return float(values.sum() / 6)


def summary(frame):
    if frame.empty:
        return {"peak_ghi_wm2": None, "peak_pv_kw": None, "energy_kwh": None,
                "peak_time": None, "daytime_mean_ghi_wm2": None, "n_intervals": 0}
    ghi = frame["ghi_forecast_wm2"]
    peak = frame.loc[ghi.idxmax()]
    pv = frame["pv_power_kw"]
    daytime = frame.loc[frame["daylight"], "ghi_forecast_wm2"]
    return {"peak_ghi_wm2": float(ghi.max()),
            "peak_pv_kw": float(pv.max()) if pv.notna().all() else None,
            "energy_kwh": energy_kwh(frame),
            "peak_time": pd.Timestamp(peak["timestamp"]).strftime("%H:%M"),
            "daytime_mean_ghi_wm2": float(daytime.mean()) if len(daytime) else None,
            "n_intervals": len(frame)}


def hourly_summary(frame):
    rows = []
    for hour, group in frame.groupby("hour", sort=True):
        complete = len(group) == 6 and group["interval_index"].nunique() == 6
        rows.append({"hour": int(hour), "label": f"{hour:02d}:00",
                     "ghi_mean_wm2": float(group["ghi_forecast_wm2"].mean()) if complete else None,
                     "pv_mean_kw": float(group["pv_power_kw"].mean())
                         if complete and group["pv_power_kw"].notna().all() else None,
                     "energy_kwh": energy_kwh(group) if complete else None,
                     "complete": complete})
    return rows


def select_view(frame, hour=None, interval=None):
    if interval is not None:
        if not 0 <= interval < 144:
            raise ValueError("Interval must be between 0 and 143.")
        if hour is not None and interval // 6 != hour:
            raise ValueError("The selected interval does not belong to the selected hour.")
        return frame.loc[frame.interval_index == interval].copy()
    if hour is not None:
        if not 0 <= hour <= 23:
            raise ValueError("Hour must be between 0 and 23.")
        return frame.loc[frame.hour == hour].copy()
    return frame.copy()


def csv_bytes(frame):
    fields = ["timestamp", "interval_end", "ghi_forecast_wm2"]
    if frame["pv_power_kw"].notna().all():
        fields.append("pv_power_kw")
    fields += ["hour", "interval_index", "daylight"]
    return frame[fields].to_csv(index=False, float_format="%.6f").encode("utf-8-sig")
