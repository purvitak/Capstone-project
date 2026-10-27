"""Read-only FastAPI service for the selected forecast archive."""
from datetime import date
from functools import lru_cache
import os
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import Response

from dashboard.backend.repository import ROOT, ForecastNotFound, ForecastRepository, ForecastUnavailable
from dashboard.calculations import csv_bytes, select_view

app = FastAPI(title="Mumbai Solar Forecast API", version="1.0.0",
              description="Read-only saved ten-minute GHI forecasts. No training or live inference.")


@lru_cache(maxsize=1)
def repository():
    return ForecastRepository(Path(os.environ.get("SOLAR_PROJECT_ROOT", ROOT)))


def fetch(repo, requested_date, capacity, ratio):
    try:
        return repo.get_forecast(requested_date.isoformat(), capacity, ratio)
    except ForecastNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except ForecastUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/health")
def health(repo: ForecastRepository = Depends(repository)):
    try:
        dates = repo.available_dates()
        return {"status": "ok", "source": "saved_forecast", "sample": False, "live": False,
                "available_days": len(dates)}
    except ForecastUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/dates")
def dates(repo: ForecastRepository = Depends(repository)):
    try:
        available = repo.available_dates()
        return {"dates": available, "timezone": "Asia/Kolkata", "source": "saved_forecast"}
    except ForecastUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


Capacity = Annotated[float | None, Query(gt=0, allow_inf_nan=False)]
Ratio = Annotated[float | None, Query(gt=0, le=1, allow_inf_nan=False)]


@app.get("/api/forecast/{forecast_date}")
def forecast(forecast_date: date, capacity_kwp: Capacity = None, performance_ratio: Ratio = None,
             repo: ForecastRepository = Depends(repository)):
    return fetch(repo, forecast_date, capacity_kwp, performance_ratio)


@app.get("/api/forecast/{forecast_date}/csv")
def forecast_csv(forecast_date: date, capacity_kwp: Capacity = None, performance_ratio: Ratio = None,
                 hour: Annotated[int | None, Query(ge=0, le=23)] = None,
                 interval: Annotated[int | None, Query(ge=0, le=143)] = None,
                 repo: ForecastRepository = Depends(repository)):
    payload = fetch(repo, forecast_date, capacity_kwp, performance_ratio)
    frame = pd.DataFrame(payload["intervals"])
    try:
        frame = select_view(frame, hour, interval)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(csv_bytes(frame), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="solar-{forecast_date}.csv"'})
