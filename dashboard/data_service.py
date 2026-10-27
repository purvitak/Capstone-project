"""Stable UI data interface: HTTP by default, explicit direct-Python mode optional."""
from __future__ import annotations

import os
from datetime import date
from functools import lru_cache

import pandas as pd
import requests

from dashboard.backend.repository import ForecastRepository, ForecastUnavailable


class DataServiceError(RuntimeError):
    pass


def mode():
    value = os.environ.get("SOLAR_DATA_MODE", "api")
    if value not in ("api", "direct"):
        raise DataServiceError("SOLAR_DATA_MODE must be api or direct.")
    return value


@lru_cache(maxsize=1)
def direct_repository():
    root = os.environ.get("SOLAR_PROJECT_ROOT")
    return ForecastRepository(root) if root else ForecastRepository()


def _request(path, params=None):
    url = os.environ.get("SOLAR_API_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        response = requests.get(url + path, params=params, timeout=(3, 20))
        if not response.ok:
            try:
                detail = response.json().get("detail", "Forecast request failed.")
            except ValueError:
                detail = "Forecast request failed."
            raise DataServiceError(str(detail))
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise DataServiceError("Cannot reach the forecast service. Start the backend or retry shortly.") from exc


def get_dates():
    try:
        return direct_repository().available_dates() if mode() == "direct" else _request("/api/dates")["dates"]
    except ForecastUnavailable as exc:
        raise DataServiceError(str(exc)) from exc


def get_forecast(forecast_date, capacity_kwp=None, performance_ratio=None):
    requested = forecast_date.isoformat() if isinstance(forecast_date, date) else str(forecast_date)
    try:
        if mode() == "direct":
            payload = direct_repository().get_forecast(requested, capacity_kwp, performance_ratio)
        else:
            params = {"capacity_kwp": capacity_kwp, "performance_ratio": performance_ratio}
            payload = _request(f"/api/forecast/{requested}", {k:v for k,v in params.items() if v is not None})
        frame = pd.DataFrame(payload["intervals"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
        frame["interval_end"] = pd.to_datetime(frame["interval_end"], utc=True).dt.tz_convert("Asia/Kolkata")
        frame["pv_power_kw"] = pd.to_numeric(frame["pv_power_kw"], errors="coerce")
        return payload, frame
    except (ForecastUnavailable, ValueError, KeyError, TypeError) as exc:
        raise DataServiceError("Forecast data could not be loaded. Check the selected date and service configuration.") from exc
