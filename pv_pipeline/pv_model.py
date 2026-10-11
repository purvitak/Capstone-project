"""Temperature-adjusted GHI-to-PV proxy from Shanker & Kulkarni (2026), §3.3.

DOI: 10.1016/j.nxener.2026.100601, equations 5–6.
This is simulated PV using horizontal irradiance, not a tilted-array/inverter model.
Forecast temperature must be historical or independently forecast, never day-D actuals.
"""
from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class PVParameters:
    capacity_kw: float = 100.0
    noct_c: float = 45.0
    temperature_coefficient: float = 0.0045  # positive loss magnitude, per °C
    reference_temperature_c: float = 25.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("PV parameters must be finite")
        if self.capacity_kw <= 0 or self.noct_c < 20 or self.temperature_coefficient < 0:
            raise ValueError("Require capacity > 0, NOCT >= 20°C, temperature coefficient >= 0")


def ghi_to_pv(ghi, air_temperature, parameters=PVParameters()):
    """Return estimated power in kW and cell temperature in °C.

    Only negative power is clipped, matching the paper. Rated capacity is not
    an AC inverter limit. Unknown inputs stay NaN, except zero GHI implies zero power.
    """
    irradiance, ambient = np.broadcast_arrays(np.asarray(ghi, dtype=float),
                                              np.asarray(air_temperature, dtype=float))
    irradiance = np.maximum(irradiance, 0)
    cell = ambient + (parameters.noct_c - 20.0) * irradiance / 800.0
    efficiency = 1 - parameters.temperature_coefficient * (cell - parameters.reference_temperature_c)
    power = parameters.capacity_kw * irradiance / 1000.0 * efficiency
    power = np.where(irradiance == 0, 0.0, np.maximum(power, 0.0))
    return power, cell


def historical_temperature_proxy(temperature, lookback=3):
    """Median at the same clock time over D-1...D-lookback; no future filling."""
    values = np.asarray(temperature, dtype=float)
    if values.ndim != 2 or lookback < 1:
        raise ValueError("Expected a day-by-step temperature matrix and positive lookback")
    proxy = np.full_like(values, np.nan)
    for day in range(lookback, len(values)):
        proxy[day] = np.median(values[day - lookback:day], axis=0)
    return proxy


def daily_energy_kwh(power, step_minutes=10):
    """Integrate period-mean kW. Missing steps yield NaN daily energy."""
    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive")
    return np.sum(np.asarray(power, dtype=float), axis=-1) * step_minutes / 60.0
