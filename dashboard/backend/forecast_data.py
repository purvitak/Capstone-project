"""Load saved day-ahead forecasts for the historical dashboard replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from pv_pipeline.common import KT_CLIP, PREP_PATH, RESULTS_DIR, STEPS_PER_DAY
from pv_pipeline.pv_model import PVParameters, daily_energy_kwh, ghi_to_pv


@dataclass(frozen=True)
class ForecastStore:
    dates: np.ndarray
    actual_ghi: np.ndarray
    cs_proxy: np.ndarray
    daylight_mask: np.ndarray
    actual_air_temp: np.ndarray
    forecast_air_temp: np.ndarray
    predicted_ghi: np.ndarray
    lower_ghi: np.ndarray | None
    upper_ghi: np.ndarray | None
    test_idx: np.ndarray
    source: Path

    @property
    def available_test_idx(self) -> np.ndarray:
        idx = self.test_idx
        valid = np.isfinite(self.predicted_ghi[idx]).all(axis=1)
        return idx[valid]


def forecast_temperature_proxy(temperature: np.ndarray, lookback: int = 3) -> np.ndarray:
    """Use only earlier calendar days, ignoring missing temperature readings."""
    values = np.asarray(temperature, dtype=float)
    if values.ndim != 2 or values.shape[1] != STEPS_PER_DAY:
        raise ValueError("Expected 144 temperatures per day")
    result = np.full_like(values, np.nan)
    for day in range(1, len(values)):
        history = values[max(0, day - lookback):day]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            result[day] = np.nanmedian(history, axis=0)
    return result


def _prediction_to_ghi(kt: np.ndarray, cs: np.ndarray) -> np.ndarray:
    if kt.shape != cs.shape:
        raise ValueError(f"Prediction shape {kt.shape} does not match prepared data {cs.shape}")
    return np.clip(kt, *KT_CLIP) * cs


def load_forecast(prep_path: Path | str = PREP_PATH,
                  base_path: Path | str = RESULTS_DIR / "15epochs" / "dwt_base_pred.npz",
                  stage2_path: Path | str | None = None) -> ForecastStore:
    with np.load(prep_path, allow_pickle=False) as prep, np.load(base_path, allow_pickle=False) as base:
        dates = prep["dates"]
        if not np.array_equal(dates, base["dates"]):
            raise ValueError("Forecast and prepared-data dates differ")
        config = str(base["config"]) if "config" in base.files else ""
        if '"resolution": "hourly"' in config or '"smoke": true' in config:
            raise ValueError("Dashboard requires a full 10-minute forecast, not hourly or smoke output")
        cs = prep["cs_proxy"]
        kt_base = base["kt_base"]
        predicted = _prediction_to_ghi(kt_base, cs)
        lower = upper = None
        if stage2_path is not None:
            with np.load(stage2_path, allow_pickle=False) as stage2:
                if "dates" not in stage2.files or not np.array_equal(dates, stage2["dates"]):
                    raise ValueError("Corrected forecast and prepared-data dates differ")
                final_kt = stage2["kt_final"]
                if final_kt.shape != kt_base.shape:
                    raise ValueError("Corrected forecast shape differs from base forecast")
                if not np.isfinite(final_kt[prep["test_idx"]][prep["mask"][prep["test_idx"]]]).all():
                    raise ValueError("Corrected forecast is missing daylight test predictions")
                valid = np.isfinite(final_kt)
                # Stage 2 deliberately leaves night slots NaN; keep the base
                # forecast there so every test day still has 144 display steps.
                predicted = _prediction_to_ghi(np.where(valid, final_kt, kt_base), cs)
                lower = _prediction_to_ghi(np.where(valid, stage2["kt_lo"], kt_base), cs)
                upper = _prediction_to_ghi(np.where(valid, stage2["kt_hi"], kt_base), cs)
        return ForecastStore(
            dates=dates, actual_ghi=prep["ghi"], cs_proxy=cs,
            daylight_mask=prep["mask"].astype(bool),
            actual_air_temp=prep["air_temp"],
            forecast_air_temp=forecast_temperature_proxy(prep["air_temp"]),
            predicted_ghi=predicted, lower_ghi=lower, upper_ghi=upper,
            test_idx=prep["test_idx"], source=Path(stage2_path or base_path),
        )


def discover_sources(results_dir: Path | str = RESULTS_DIR) -> dict[str, tuple[Path, Path | None]]:
    """List runs with at least one complete held-out 10-minute test day."""
    root = Path(results_dir)
    found = {}
    with np.load(PREP_PATH, allow_pickle=False) as prep:
        test_idx = prep["test_idx"]
        expected_shape = prep["ghi"].shape
        daylight_test = prep["mask"][test_idx]
    for base_path in sorted(root.glob("**/*base_pred.npz")):
        if "smoke" in base_path.parts:
            continue
        try:
            with np.load(base_path, allow_pickle=False) as base:
                config = str(base["config"]) if "config" in base.files else ""
                if '"resolution": "hourly"' in config or '"smoke": true' in config:
                    continue
                predictions = base["kt_base"]
                if predictions.shape != expected_shape or not np.isfinite(predictions[test_idx]).all(axis=1).any():
                    continue
        except (OSError, KeyError, ValueError):
            continue
        stage2_path = base_path.with_name(base_path.name.replace("base_pred.npz", "stage2.npz"))
        label = str(base_path.relative_to(root).parent / base_path.stem.replace("_base_pred", ""))
        found[label + " · base"] = (base_path, None)
        if stage2_path.is_file():
            try:
                with np.load(stage2_path, allow_pickle=False) as corrected:
                    final = corrected["kt_final"]
                    ok = final.shape == expected_shape and np.isfinite(final[test_idx][daylight_test]).all()
                if ok:
                    found[label + " · corrected"] = (base_path, stage2_path)
            except (OSError, KeyError, ValueError):
                pass
    return found


def day_report(store: ForecastStore, day_idx: int, parameters: PVParameters) -> tuple[pd.DataFrame, dict]:
    if day_idx not in store.available_test_idx:
        raise ValueError("Selected day has no complete held-out 10-minute forecast")
    date = str(store.dates[day_idx])
    times = pd.date_range(date, periods=STEPS_PER_DAY, freq="10min", tz="Asia/Kolkata")
    actual = store.actual_ghi[day_idx]
    predicted = store.predicted_ghi[day_idx]
    actual_pv, _ = ghi_to_pv(actual, store.actual_air_temp[day_idx], parameters)
    forecast_pv, _ = ghi_to_pv(predicted, store.forecast_air_temp[day_idx], parameters)
    frame = pd.DataFrame({
        "Time (IST)": times.strftime("%H:%M"),
        "Actual GHI (W/m²)": actual,
        "Predicted GHI (W/m²)": predicted,
        "Clear-sky proxy (W/m²)": store.cs_proxy[day_idx],
        "Daylight for scoring": store.daylight_mask[day_idx],
        "Forecast air temperature (°C)": store.forecast_air_temp[day_idx],
        "Observed air temperature (°C)": store.actual_air_temp[day_idx],
        "Estimated PV from actual GHI (kW)": actual_pv,
        "Estimated PV from predicted GHI (kW)": forecast_pv,
    })
    if store.lower_ghi is not None:
        frame["Predicted GHI lower (W/m²)"] = store.lower_ghi[day_idx]
        frame["Predicted GHI upper (W/m²)"] = store.upper_ghi[day_idx]
    mask = store.daylight_mask[day_idx] & np.isfinite(actual) & np.isfinite(predicted)
    error = predicted[mask] - actual[mask]
    metrics = {
        "date": date,
        "peak_predicted_ghi_wm2": float(np.nanmax(predicted)),
        "peak_predicted_pv_kw": float(np.nanmax(forecast_pv)),
        "predicted_energy_kwh": float(daily_energy_kwh(forecast_pv)),
        "actual_estimated_energy_kwh": float(daily_energy_kwh(actual_pv)),
        "daylight_rmse_wm2": float(np.sqrt(np.mean(error ** 2))) if error.size else float("nan"),
        "daylight_mae_wm2": float(np.mean(np.abs(error))) if error.size else float("nan"),
        "scored_steps": int(mask.sum()),
    }
    return frame, metrics
