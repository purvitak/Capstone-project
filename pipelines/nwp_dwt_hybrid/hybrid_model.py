"""GFS-guided DWT-BiLSTM and cloud-regime blending.

Extends Bektaş & Altaş (2024): the published forecast-day weather type comes from
day-D observations. Here it is replaced by continuous, issue-time-available GFS
information for day D, and the result is blended with the current default using
weights that depend on the forecast cloud regime.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "paper_reproduction"))
import paper_model as P

CLEAR_SKY_MIN = 10.0  # W/m²; hourly clear-sky proxy below this is treated as night
REGIME_EDGES = (30.0, 70.0)  # forecast daylight cloud cover, %
REGIMES = ("clear", "partly_cloudy", "overcast")
NWP_SETS = {"gfs": ("gfs",), "multi": ("gfs", "ecmwf_ifs025", "icon_global")}


def feature_names(models=("gfs",)) -> tuple:
    """With GFS alone this is the original 18-feature layout."""
    return tuple([f"dwt_{c}" for c in ("A6", "D6", "D5", "D4", "D3", "D2", "D1")] + ["type_d1"]
                 + [f"{m}_kt" for m in models] + [f"{m}_cloud" for m in models]
                 + [f"{m}_daily_k" for m in models]
                 + ["clear_sky", "gfs_temp", "gfs_rh", "gfs_wind", "obs_temp_d1", "obs_rh_d1", "obs_wind_d1"])


FEATURES = feature_names()


def hourly(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, float).reshape(len(a), 24, 6).mean(axis=2)


def gfs_clear_sky_index(gfs_ghi_h: np.ndarray, cs_h: np.ndarray) -> np.ndarray:
    ratio = np.divide(gfs_ghi_h, cs_h, out=np.zeros_like(gfs_ghi_h), where=cs_h > CLEAR_SKY_MIN)
    return np.clip(ratio, 0.0, 1.3)


def fill_missing_models(values: np.ndarray, available: np.ndarray) -> np.ndarray:
    """Replace an unavailable model's day by the mean of the models available that day.

    values: (n_days, ..., n_models); available: (n_days, n_models).
    """
    out = np.array(values, dtype=float)
    shape = (len(out),) + (1,) * (out.ndim - 2) + (out.shape[-1],)
    mask = np.asarray(available, bool).reshape(shape)
    count = mask.sum(axis=-1, keepdims=True)
    mean = np.divide(np.where(mask, out, 0).sum(axis=-1, keepdims=True), count,
                     out=np.full(out.shape[:-1] + (1,), np.nan), where=count > 0)
    return np.where(mask, out, mean)


def build_inputs(src: dict, days, norm: dict) -> np.ndarray:
    """(n, 24, 15 + 3 × models). Day-D values come only from NWP or the clear-sky proxy."""
    lo, span, scale = norm["met_min"], norm["met_span"], norm["ghi_scale"]
    rows = []
    for d in days:
        d = int(d)
        rows.append(np.concatenate([
            src["dwt"][d] / scale,
            np.full((24, 1), src["type_obs"][d - 1] / 2.0),
            src["nwp_kt"][d],
            src["nwp_cloud"][d] / 100.0,
            np.repeat(src["nwp_k"][d][None, :], 24, axis=0),
            src["cs_h"][d][:, None] / scale,
            (src["gfs_met"][d] - lo) / span,
            (src["met_obs"][d - 1] - lo) / span,
        ], axis=1))
    x = np.stack(rows).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite expert inputs")
    return x


def normalizer(src: dict, fit) -> dict:
    met = src["met_obs"][np.asarray(fit) - 1].reshape(-1, 3)
    lo, hi = met.min(axis=0), met.max(axis=0)
    return {"ghi_scale": float(np.max(src["ghi_h"][fit])), "met_min": lo,
            "met_span": np.where(hi - lo > 0, hi - lo, 1.0)}


def zero_at_night(hourly_values: np.ndarray, cs_h: np.ndarray) -> np.ndarray:
    """No negative irradiance, and none when the known clear-sky proxy is below CLEAR_SKY_MIN."""
    return np.where(cs_h > CLEAR_SKY_MIN, np.maximum(hourly_values, 0.0), 0.0)


def regime_of(cloud_h: np.ndarray, cs_h: np.ndarray) -> np.ndarray:
    """0 clear, 1 partly cloudy, 2 overcast, from forecast daylight cloud cover
    (hourly, or hourly × models, which are averaged)."""
    if cloud_h.ndim == 3:
        cloud_h = cloud_h.mean(axis=-1)
    daylight = cs_h > 50.0
    mean = np.where(daylight, cloud_h, 0).sum(axis=1) / np.maximum(daylight.sum(axis=1), 1)
    return np.digitize(mean, REGIME_EDGES)


def blend(default_h, expert_h, regimes, weights) -> np.ndarray:
    w = np.asarray(weights, float)[regimes][:, None]
    return (1 - w) * default_h + w * expert_h


def select_weights(actual, default_h, expert_h, mask, regimes, grid) -> list[float]:
    """Per-regime weight minimizing pooled daylight squared error. Regimes partition
    days, so minimizing each regime's error minimizes the pooled RMSE."""
    weights = []
    for r in range(len(REGIMES)):
        sel = regimes == r
        if not sel.any():
            weights.append(0.0)
            continue
        m = mask[sel]
        sse = [np.sum(((1 - w) * default_h[sel] + w * expert_h[sel] - actual[sel])[m] ** 2) for w in grid]
        weights.append(float(grid[int(np.argmin(sse))]))
    return weights


def to_10min(hourly_values: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """Spread hourly means over six slots by the clear-sky shape, preserving the mean."""
    groups = cs.reshape(len(cs), 24, 6)
    totals = groups.sum(axis=2, keepdims=True)
    factor = np.divide(6 * groups, totals, out=np.zeros_like(groups), where=totals > 0)
    out = (np.maximum(hourly_values, 0)[..., None] * factor).reshape(cs.shape)
    return np.minimum(out, 1.3 * cs)


__all__ = ["P", "FEATURES", "NWP_SETS", "REGIMES", "build_inputs", "feature_names",
           "fill_missing_models", "normalizer", "regime_of", "blend", "select_weights",
           "to_10min", "zero_at_night", "hourly", "gfs_clear_sky_index"]
