"""Leak-safe discrete wavelet features and Bektas-style sky type.

DWT is always computed on a finite window that ends at the last step of D−1.
Never call these helpers on a series that already contains day-D GHI.
"""

from __future__ import annotations

import numpy as np

# Bektas & Altas (2024) §2.1
K_SUNNY = 0.45
K_CLOUDY = 0.25


def sky_type_from_k(k: float) -> int:
    """0 = rainy / overcast, 1 = cloudy, 2 = sunny."""
    if not np.isfinite(k):
        return 0
    if k > K_SUNNY:
        return 2
    if k > K_CLOUDY:
        return 1
    return 0


def daily_clearness(ghi_row: np.ndarray, cs_row: np.ndarray, mask_row: np.ndarray) -> float:
    m = mask_row.astype(bool) & np.isfinite(ghi_row) & np.isfinite(cs_row) & (cs_row > 0)
    if not m.any():
        return np.nan
    return float(np.nansum(ghi_row[m]) / np.nansum(cs_row[m]))


def _haar_forward_once(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(a) % 2:
        a = np.concatenate([a, a[-1:]])
    even, odd = a[0::2], a[1::2]
    approx = (even + odd) / np.sqrt(2.0)
    detail = (even - odd) / np.sqrt(2.0)
    return approx, detail


def _upsample_to(band: np.ndarray, n: int) -> np.ndarray:
    if len(band) == 0:
        return np.zeros(n, dtype=float)
    idx = np.linspace(0, len(band) - 1, n)
    return np.interp(idx, np.arange(len(band)), band)


def multilevel_haar(x: np.ndarray, levels: int) -> list[np.ndarray]:
    """Return [A_L, D_L, D_{L-1}, ..., D_1], each resampled to len(x)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    a = np.where(np.isfinite(x), x, 0.0)
    details = []
    max_levels = 0
    tmp = a.copy()
    while len(tmp) >= 2:
        tmp, _ = _haar_forward_once(tmp)
        max_levels += 1
    levels = max(1, min(int(levels), max_levels))
    a = np.where(np.isfinite(x), x, 0.0)
    for _ in range(levels):
        a, d = _haar_forward_once(a)
        details.append(d)
    bands = [_upsample_to(a, n)] + [_upsample_to(d, n) for d in reversed(details)]
    return bands


def dwt_matrix(ghi_window: np.ndarray, levels: int = 6) -> np.ndarray:
    """(T, n_bands) DWT components of a single historical window."""
    bands = multilevel_haar(ghi_window, levels)
    return np.stack(bands, axis=1)


def hourly_mean(day_144: np.ndarray) -> np.ndarray:
    """(144,) -> (24,) mean of each six 10-minute steps."""
    x = np.asarray(day_144, dtype=float).reshape(24, 6)
    with np.errstate(all="ignore"):
        return np.nanmean(x, axis=1)


def expand_hourly_to_10min(day_24: np.ndarray) -> np.ndarray:
    """Repeat each hourly value across six 10-minute slots."""
    return np.repeat(np.asarray(day_24, dtype=float), 6)
