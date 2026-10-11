"""Shared utilities: loading, solar position, clear-sky proxies, metrics, bootstrap.

Nothing here depends on a model. Every function is deliberately plain numpy or
pandas so it can be unit-tested and so it runs identically on a laptop, in an
agent sandbox, or in a Kaggle notebook.

Timestamp convention (verified in verify_dataset.py): Solcast stamps the
*period end* in UTC and each record covers the preceding `period` (PT10M).
Period start = period_end - 10 min, then converted to IST. Solar position is
evaluated at the period *mid-point*, which is what matches the CSV's own
zenith/azimuth columns most closely.
"""

from __future__ import annotations

import glob as _glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREP_PATH = PROJECT_ROOT / "data" / "prepared" / "prep.npz"
RESULTS_DIR = PROJECT_ROOT / "results"

STEP_MINUTES = 10
STEPS_PER_HOUR = 6
STEPS_PER_DAY = 144
TZ = "Asia/Kolkata"

# Recovered from the data by fit_location(); see verify_dataset.py.
MUMBAI_LAT = 19.0760
MUMBAI_LON = 72.8777

KT_CLIP = (0.0, 1.3)
DAYLIGHT_ZENITH_MAX = 85.0
DAYLIGHT_CS_MIN = 50.0
MONSOON_MONTHS = (6, 7, 8, 9)


# ==========================================================================
# Loading
# ==========================================================================

def load_solcast(pattern: str | list[str]) -> pd.DataFrame:
    """Load and concatenate Solcast CSVs, de-duplicate, index by IST period start.

    Accepts a glob pattern or an explicit list so that the 2020-2023 and
    2023-2026 exports can be merged in one call.
    """
    paths = sorted(_glob.glob(pattern)) if isinstance(pattern, str) else list(pattern)
    if not paths:
        raise FileNotFoundError(f"no CSV matched {pattern!r}")

    frames = []
    for p in paths:
        part = pd.read_csv(p)
        part["source_file"] = p.split("/")[-1]
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    end_utc = pd.to_datetime(df["period_end"], utc=True)
    df["period_end_utc"] = end_utc
    df["timestamp"] = (end_utc - pd.Timedelta(minutes=STEP_MINUTES)).dt.tz_convert(TZ)
    # Mid-point is what solar position is evaluated at.
    df["mid_utc"] = end_utc - pd.Timedelta(minutes=STEP_MINUTES / 2)

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset="timestamp", keep="first")
        .reset_index(drop=True)
    )
    return add_time_columns(df)


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df = df.copy()
    df["date"] = ts.dt.normalize()
    df["step"] = (ts.dt.hour * 60 + ts.dt.minute) // STEP_MINUTES
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["doy"] = ts.dt.dayofyear
    df["is_monsoon"] = df["month"].isin(MONSOON_MONTHS)
    return df


def grid_audit(df: pd.DataFrame) -> dict:
    """Duplicates, off-grid deltas and timestamps missing from the 10-min grid."""
    ts = pd.DatetimeIndex(df["timestamp"])
    expected = pd.date_range(ts.min(), ts.max(), freq=f"{STEP_MINUTES}min", tz=TZ)
    deltas = ts.to_series().diff().dropna()
    missing = expected.difference(ts)
    return {
        "rows": int(len(df)),
        "start": str(ts.min()),
        "end": str(ts.max()),
        "expected_grid_rows": int(len(expected)),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "off_grid_deltas": int((deltas != pd.Timedelta(minutes=STEP_MINUTES)).sum()),
        "missing_timestamps": int(len(missing)),
        "missing_examples": [str(t) for t in missing[:10]],
    }


def complete_days(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep only IST days holding all 144 steps."""
    sizes = df.groupby("date").size()
    bad = sizes[sizes != STEPS_PER_DAY].index
    kept = df[~df["date"].isin(bad)].reset_index(drop=True)
    return kept, [str(pd.Timestamp(d).date()) for d in bad]


# ==========================================================================
# Solar position (NOAA Solar Calculator equations)
# ==========================================================================

def solar_position(
    when_utc: pd.DatetimeIndex | pd.Series,
    lat: float = MUMBAI_LAT,
    lon: float = MUMBAI_LON,
    refraction: bool = True,
) -> dict[str, np.ndarray]:
    """Apparent solar zenith and azimuth, degrees.

    Implements the NOAA Solar Calculator formulation (a Meeus-derived
    low-precision algorithm, accurate to well under 0.1 deg for this purpose).
    Azimuth is returned in the conventional frame: 0 = North, 90 = East.
    Validated against the CSV's own columns in verify_dataset.py.
    """
    t = pd.DatetimeIndex(when_utc)
    jd = t.to_julian_date().to_numpy(dtype=float)
    jc = (jd - 2451545.0) / 36525.0

    # Geometric mean longitude and anomaly of the sun.
    L0 = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0)
    M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    Mr = np.deg2rad(M)
    C = (
        np.sin(Mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + np.sin(2 * Mr) * (0.019993 - 0.000101 * jc)
        + np.sin(3 * Mr) * 0.000289
    )
    true_long = L0 + C
    omega = 125.04 - 1934.136 * jc
    app_long = true_long - 0.00569 - 0.00478 * np.sin(np.deg2rad(omega))

    # Obliquity of the ecliptic.
    eps0 = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    eps = eps0 + 0.00256 * np.cos(np.deg2rad(omega))

    decl = np.rad2deg(
        np.arcsin(np.sin(np.deg2rad(eps)) * np.sin(np.deg2rad(app_long)))
    )

    # Equation of time, minutes.
    y = np.tan(np.deg2rad(eps / 2.0)) ** 2
    eqtime = 4.0 * np.rad2deg(
        y * np.sin(2 * np.deg2rad(L0))
        - 2 * e * np.sin(Mr)
        + 4 * e * y * np.sin(Mr) * np.cos(2 * np.deg2rad(L0))
        - 0.5 * y * y * np.sin(4 * np.deg2rad(L0))
        - 1.25 * e * e * np.sin(2 * Mr)
    )

    minutes = (
        t.hour.to_numpy() * 60.0
        + t.minute.to_numpy()
        + t.second.to_numpy() / 60.0
    )
    true_solar_time = np.mod(minutes + eqtime + 4.0 * lon, 1440.0)
    hour_angle = true_solar_time / 4.0 - 180.0

    lat_r, decl_r, ha_r = np.deg2rad(lat), np.deg2rad(decl), np.deg2rad(hour_angle)
    cos_z = np.sin(lat_r) * np.sin(decl_r) + np.cos(lat_r) * np.cos(decl_r) * np.cos(ha_r)
    cos_z = np.clip(cos_z, -1.0, 1.0)
    zenith_true = np.rad2deg(np.arccos(cos_z))

    # Azimuth uses the true (unrefracted) zenith, per the NOAA formulation.
    # 0 = North, measured clockwise.
    zr = np.deg2rad(zenith_true)
    denom = np.cos(lat_r) * np.sin(zr)
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    cos_az = (np.sin(lat_r) * np.cos(zr) - np.sin(decl_r)) / denom
    az = np.rad2deg(np.arccos(np.clip(cos_az, -1.0, 1.0)))
    azimuth = np.where(hour_angle > 0.0, np.mod(az + 180.0, 360.0), np.mod(540.0 - az, 360.0))

    zenith = zenith_true - _refraction_correction(90.0 - zenith_true) if refraction else zenith_true

    return {
        "zenith": zenith,
        "zenith_true": zenith_true,
        "azimuth": azimuth,
        "declination": decl,
        "eqtime": eqtime,
    }


def _refraction_correction(elev_deg: np.ndarray) -> np.ndarray:
    """Atmospheric refraction in degrees, NOAA piecewise approximation."""
    e = np.asarray(elev_deg, dtype=float)
    te = np.tan(np.deg2rad(np.where(np.abs(e) < 1e-6, 1e-6, e)))
    r = np.where(
        e > 85.0,
        0.0,
        np.where(
            e > 5.0,
            58.1 / te - 0.07 / te**3 + 0.000086 / te**5,
            np.where(
                e > -0.575,
                1735.0 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711))),
                -20.772 / te,
            ),
        ),
    )
    return r / 3600.0


def solcast_azimuth_to_conventional(az: np.ndarray) -> np.ndarray:
    """Solcast uses 0 = N with East negative; convert to 0 = N clockwise."""
    return np.mod(-np.asarray(az, dtype=float), 360.0)


def fit_location(
    mid_utc: pd.Series, zenith_ref: np.ndarray, lat0: float = 19.0, lon0: float = 72.9
) -> dict:
    """Recover the site coordinates by minimizing zenith RMSE against the CSV.

    A successful fit landing on Mumbai is independent evidence that both the
    timestamp convention and the solar-position code are right.
    """
    from scipy.optimize import minimize

    mask = np.isfinite(zenith_ref)
    t = pd.DatetimeIndex(mid_utc)[mask]
    ref = np.asarray(zenith_ref, dtype=float)[mask]

    def rmse(p):
        z = solar_position(t, p[0], p[1])["zenith"]
        return float(np.sqrt(np.mean((z - ref) ** 2)))

    res = minimize(rmse, x0=[lat0, lon0], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400})
    return {"lat": float(res.x[0]), "lon": float(res.x[1]),
            "zenith_rmse_deg": float(res.fun), "converged": bool(res.success)}


# ==========================================================================
# Clear-sky proxies  (never use Solcast's same-day clear-sky as a model input)
# ==========================================================================

def clearsky_lagged_median(cs_by_day: np.ndarray, window: int = 7) -> np.ndarray:
    """Per-step median of the previous `window` days' clear-sky profile.

    cs_by_day is (n_days, 144). Row i of the output uses days i-window .. i-1
    only, so it is available before day i begins. Rows without a full window
    are NaN.
    """
    n = len(cs_by_day)
    out = np.full_like(cs_by_day, np.nan, dtype=float)
    for i in range(window, n):
        out[i] = np.median(cs_by_day[i - window : i], axis=0)
    return out


def clearsky_climatology(
    cs_by_day: np.ndarray, doy: np.ndarray, years: np.ndarray, half_window: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Day-of-year climatology built from *other years* only.

    For each target day, average the per-step clear-sky profile of days within
    +/- `half_window` day-of-year across all other calendar years. Using other
    years keeps it leak-free with respect to the target day and makes it
    available for a future date. Returns (profiles, n_years_used).
    """
    n = len(cs_by_day)
    out = np.full_like(cs_by_day, np.nan, dtype=float)
    support = np.zeros(n, dtype=int)

    for i in range(n):
        d = doy[i]
        # Circular day-of-year distance.
        dist = np.minimum(np.abs(doy - d), 366 - np.abs(doy - d))
        sel = (dist <= half_window) & (years != years[i])
        if sel.any():
            out[i] = np.nanmean(cs_by_day[sel], axis=0)
            support[i] = int(len(np.unique(years[sel])))
    return out, support


def clearsky_clim_times_ratio(
    clim: np.ndarray, cs_by_day: np.ndarray, window: int = 7
) -> np.ndarray:
    """Climatology rescaled by the recent observed-to-climatology ratio."""
    n = len(cs_by_day)
    out = np.full_like(clim, np.nan, dtype=float)
    for i in range(window, n):
        past_obs = np.nansum(cs_by_day[i - window : i])
        past_clim = np.nansum(clim[i - window : i])
        if past_clim > 0:
            out[i] = clim[i] * (past_obs / past_clim)
    return out


def shift_days(a: np.ndarray, k: int) -> np.ndarray:
    """out[i] = a[i - k], NaN for the first k rows.

    Deliberately not np.roll: roll wraps the end of the record onto the start, which
    would fabricate a predecessor for the first days. Calendar contiguity is enforced
    separately by the usable-day mask.
    """
    out = np.full_like(a, np.nan, dtype=float)
    if k < len(a):
        out[k:] = a[:-k] if k else a
    return out


def baseline_prev_kt(kt: np.ndarray) -> np.ndarray:
    """Persistence: yesterday's kt profile. The single reference baseline definition."""
    return shift_days(np.nan_to_num(kt, nan=0.0), 1)


def baseline_mean3_kt(kt: np.ndarray, n: int = 3) -> np.ndarray:
    """Mean kt profile of the previous `n` days. The single reference definition,
    shared by Stage 0's comparison table, Stage 1's acceptance check and Stage 3."""
    ktf = np.nan_to_num(kt, nan=0.0)
    lags = np.stack([shift_days(ktf, k) for k in range(1, n + 1)])
    count = np.isfinite(lags).sum(axis=0)
    return np.divide(np.nansum(lags, axis=0), count,
                     out=np.full_like(ktf, np.nan, dtype=float), where=count > 0)


def kt_from(ghi: np.ndarray, cs: np.ndarray, clip: tuple = KT_CLIP) -> np.ndarray:
    """Clear-sky index, undefined where the clear-sky term is negligible."""
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(cs > DAYLIGHT_CS_MIN, ghi / cs, np.nan)
    return np.clip(kt, clip[0], clip[1])


def daylight_mask(zenith: np.ndarray, cs_proxy: np.ndarray) -> np.ndarray:
    """Both conditions are known in advance, so the mask is leak-free."""
    return (zenith < DAYLIGHT_ZENITH_MAX) & (cs_proxy > DAYLIGHT_CS_MIN)


# ==========================================================================
# Metrics
# ==========================================================================

def point_metrics(y: np.ndarray, yhat: np.ndarray, mask: np.ndarray | None = None) -> dict:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    if mask is not None:
        ok &= mask
    y, yhat = y[ok], yhat[ok]
    if y.size == 0:
        return {k: float("nan") for k in ("rmse", "mae", "mbe", "nrmse", "n")}
    err = yhat - y
    rmse = float(np.sqrt(np.mean(err**2)))
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(err))),
        "mbe": float(np.mean(err)),
        "nrmse": float(rmse / np.mean(y) * 100) if np.mean(y) else float("nan"),
        "n": int(y.size),
    }


def skill(metric_model: float, metric_ref: float) -> float:
    """Fraction of the reference error removed. Positive means better."""
    if not np.isfinite(metric_ref) or metric_ref == 0:
        return float("nan")
    return float(1.0 - metric_model / metric_ref)


def per_day_rmse(y: np.ndarray, yhat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """One RMSE per day, for paired daily tests. NaN where a day has no mask."""
    err2 = np.where(mask, (yhat - y) ** 2, np.nan)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.nanmean(err2, axis=1))


def moving_block_bootstrap_ci(
    diffs: np.ndarray, block: int = 7, n_boot: int = 10000, seed: int = 42, alpha: float = 0.05
) -> dict:
    """CI for the mean of an autocorrelated daily series.

    Resampling whole blocks of consecutive days preserves the short-range
    dependence that an i.i.d. bootstrap would destroy, which otherwise makes
    the interval far too narrow.
    """
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2 * block:
        return {"mean": float(np.mean(d)) if n else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "n_days": int(n)}

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(d.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n_days": int(n),
        "block_days": block,
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def wilcoxon_days(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank on per-day errors (not on 10-minute samples)."""
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return {"stat": float("nan"), "p": float("nan"), "n_days": int(ok.sum())}
    res = stats.wilcoxon(a[ok], b[ok], zero_method="zsplit")
    return {"stat": float(res.statistic), "p": float(res.pvalue), "n_days": int(ok.sum())}


# ==========================================================================
# Day-matrix helpers
# ==========================================================================

@dataclass
class DayArrays:
    """Everything reshaped to (n_days, 144), aligned on `dates`."""

    dates: pd.DatetimeIndex
    ghi: np.ndarray
    cs_solcast: np.ndarray
    zenith: np.ndarray
    azimuth: np.ndarray
    air_temp: np.ndarray
    wind_speed: np.ndarray
    doy: np.ndarray
    year: np.ndarray
    is_monsoon: np.ndarray

    def __len__(self) -> int:
        return len(self.dates)


def to_day_arrays(df: pd.DataFrame, zenith_col: str = "zenith_calc") -> DayArrays:
    def pivot(col: str) -> np.ndarray:
        return (
            df.pivot_table(index="date", columns="step", values=col, aggfunc="first")
            .reindex(columns=range(STEPS_PER_DAY))
            .to_numpy(dtype=float)
        )

    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    per_day = df.drop_duplicates("date").set_index("date").loc[dates]
    return DayArrays(
        dates=dates,
        ghi=pivot("ghi"),
        cs_solcast=pivot("clearsky_ghi"),
        zenith=pivot(zenith_col),
        azimuth=pivot("azimuth_calc") if "azimuth_calc" in df.columns else pivot("azimuth"),
        air_temp=pivot("air_temp"),
        wind_speed=pivot("wind_speed_10m"),
        doy=per_day["doy"].to_numpy(),
        year=per_day["year"].to_numpy(),
        is_monsoon=per_day["is_monsoon"].to_numpy(),
    )
