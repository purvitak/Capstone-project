"""Stage 0 — turn the merged Solcast CSV into leak-checked per-day arrays and features.

Implements WORKFLOW.md section 7, including the day-completeness policy of section 7a.
Outputs data/prepared/prep.npz, prep.json and cs_clim_table.npy.

Two deliberate deviations from common.clearsky_climatology, both to make leak test 2
pass by construction:
  * the day-of-year clear-sky climatology is built from development days only, so no
    test-year day can ever contribute to it;
  * a day's own calendar year is still excluded, as in common.py.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

DEV_END = "2025-07-23"      # last development day (section 3)
TEST_START = "2025-07-25"   # first test day; 2025-07-24 is the purge day
PRIOR_DAYS = 7              # a usable day needs this many contiguous preceding days

# --- day-completeness policy (section 7a) ---------------------------------
MAX_REPAIR_GAP_STEPS = 6    # 6 x 10 min = 1 hour; longer gaps disqualify the day
SMOOTH_FIELDS = ("air_temp", "relative_humidity", "surface_pressure", "wind_speed_10m")
STATUS = {0: "complete", 1: "repaired", 2: "unrepairable"}

# The first few rows of every lagged array are NaN by construction; those slices are
# excluded by the usable-day mask, so their reductions are expected to be empty.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")

RAW_COLS = ("ghi", "clearsky_ghi", "air_temp", "relative_humidity",
            "surface_pressure", "wind_speed_10m")


# ==========================================================================
# Full-grid reshaping
# ==========================================================================

def reindex_full_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Place the data on a gap-free 10-minute grid spanning whole IST days.

    Rows absent from the CSV become all-NaN with `present == 0`, so a missing step is
    represented explicitly instead of silently shortening a day. Solar position is then
    computable for every slot because the timestamps exist regardless of the data.
    """
    ts = pd.DatetimeIndex(df["timestamp"])
    start = ts.min().normalize()
    end = ts.max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(minutes=C.STEP_MINUTES)
    full = pd.date_range(start, end, freq=f"{C.STEP_MINUTES}min", tz=C.TZ, name="timestamp")

    out = df.set_index("timestamp").reindex(full)
    out["present"] = out["present"].fillna(0.0)
    out = out.reset_index()
    out["mid_utc"] = (out["timestamp"] + pd.Timedelta(minutes=C.STEP_MINUTES / 2)).dt.tz_convert("UTC")
    return C.add_time_columns(out)


def day_matrices(df: pd.DataFrame, cols) -> tuple[pd.DatetimeIndex, dict]:
    """Reshape a gap-free grid to (n_days, 144) by plain reshape — no pivot needed."""
    dates = pd.DatetimeIndex(df["date"].unique())
    n = len(dates)
    assert len(df) == n * C.STEPS_PER_DAY, (len(df), n)
    return dates, {c: df[c].to_numpy(dtype=float).reshape(n, C.STEPS_PER_DAY) for c in cols}


# ==========================================================================
# Completeness audit and gap repair (section 7a)
# ==========================================================================

def missing_runs(present_row: np.ndarray) -> list[tuple[int, int]]:
    """Runs of consecutive absent steps as (start_step, length)."""
    gaps = np.flatnonzero(~present_row)
    if gaps.size == 0:
        return []
    splits = np.split(gaps, np.flatnonzero(np.diff(gaps) > 1) + 1)
    return [(int(g[0]), int(g.size)) for g in splits]


def completeness_audit(dates: pd.DatetimeIndex, present: np.ndarray) -> dict:
    """Per-day step counts, and for every short day its gaps and longest run."""
    counts = present.sum(axis=1).astype(int)
    short = np.flatnonzero(counts != C.STEPS_PER_DAY)
    detail = []
    for i in short:
        runs = missing_runs(present[i])
        miss = np.flatnonzero(~present[i]).tolist()
        detail.append({
            "date": str(dates[i].date()),
            "n_steps": int(counts[i]),
            "n_missing": int(C.STEPS_PER_DAY - counts[i]),
            "missing_steps": miss[:50] + (["..."] if len(miss) > 50 else []),
            "gap_runs": [{"start_step": s, "length": l} for s, l in runs],
            "longest_gap_steps": max((l for _, l in runs), default=0),
        })
    return {
        "n_days": int(len(dates)),
        "n_days_with_144_steps": int((counts == C.STEPS_PER_DAY).sum()),
        "n_days_short": int(len(short)),
        "max_repair_gap_steps": MAX_REPAIR_GAP_STEPS,
        "short_days": detail,
    }


def classify_and_repair(mats: dict, present: np.ndarray) -> tuple[np.ndarray, list]:
    """Classify every day and repair the repairable ones, in place.

    A day is `complete` (0) when all 144 steps are present, `repaired` (1) when every
    gap is at most MAX_REPAIR_GAP_STEPS and has been filled, and `unrepairable` (2)
    otherwise. Interpolation is strictly within one day, so nothing ever crosses a day
    boundary or reaches forward into the forecast target day.

    Smooth fields are linearly interpolated. GHI is interpolated on the clear-sky index
    and multiplied back by the clear-sky profile, which preserves the diurnal shape
    instead of cutting a straight chord across it.
    """
    status = np.zeros(len(present), dtype=int)
    log = []
    for i in np.flatnonzero(~present.all(axis=1)):
        runs = missing_runs(present[i])
        longest = max(l for _, l in runs)
        if longest > MAX_REPAIR_GAP_STEPS:
            status[i] = 2
            log.append({"day_index": int(i), "action": "disqualified",
                        "longest_gap_steps": longest, "n_missing": int((~present[i]).sum())})
            continue

        ok = np.flatnonzero(present[i])
        bad = np.flatnonzero(~present[i])
        for f in SMOOTH_FIELDS:
            mats[f][i, bad] = np.interp(bad, ok, mats[f][i, ok])
        # Clear-sky is deterministic and smooth; interpolate it first, then scale GHI.
        cs = mats["clearsky_ghi"][i]
        cs[bad] = np.interp(bad, ok, cs[ok])
        kt = np.where(cs > C.DAYLIGHT_CS_MIN, mats["ghi"][i] / np.where(cs > 0, cs, 1), 0.0)
        kt[bad] = np.interp(bad, ok, kt[ok])
        mats["ghi"][i, bad] = np.where(cs[bad] > C.DAYLIGHT_CS_MIN, kt[bad] * cs[bad], 0.0)

        status[i] = 1
        log.append({"day_index": int(i), "action": "repaired",
                    "longest_gap_steps": longest, "n_missing": int(len(bad)),
                    "gap_runs": [{"start_step": s, "length": l} for s, l in runs]})
    return status, log


def gap_policy_selftest() -> dict:
    """End-to-end check of the policy on synthetic days, so the real run does not have
    to be taken on trust. Punches a short gap, a long gap, and a gap on a target day."""
    n = 24
    step = np.arange(C.STEPS_PER_DAY)
    cs = np.clip(900 * np.sin(np.pi * (step - 36) / 72), 0, None)
    mats = {c: np.tile(cs if c in ("ghi", "clearsky_ghi") else np.full(144, 25.0), (n, 1))
            for c in RAW_COLS}
    mats["ghi"] = mats["ghi"] * 0.8
    truth = mats["ghi"].copy()
    present = np.ones((n, C.STEPS_PER_DAY), bool)

    present[10, 70:74] = False      # 4-step gap  -> repairable
    present[12, 60:80] = False      # 20-step gap -> unrepairable
    present[20, 65:69] = False      # 4-step gap on a would-be target day
    for i, sl in ((10, slice(70, 74)), (12, slice(60, 80)), (20, slice(65, 69))):
        for c in RAW_COLS:
            mats[c][i, sl] = np.nan

    status, _ = classify_and_repair(mats, present)
    # A repaired step cannot be exact: the clear-sky profile is itself missing on that
    # step and is interpolated, so a short chord replaces a curve. Budget it relatively.
    repaired_err = float(np.max(np.abs(mats["ghi"][10, 70:74] - truth[10, 70:74])))
    rel_err_pct = 100.0 * repaired_err / float(np.nanmax(truth))

    # Predecessor eligibility: 7 days following the unrepairable day 12 are blocked.
    eligible = np.array([
        i >= PRIOR_DAYS and all(status[i - k] != 2 for k in range(1, PRIOR_DAYS + 1))
        for i in range(n)
    ])
    # Scoring on a partial day must count only the steps that exist.
    daylight = cs > C.DAYLIGHT_CS_MIN
    scored = C.point_metrics(np.where(present[20], truth[20], np.nan), mats["ghi"][20],
                             daylight)["n"]
    expect = int((daylight & present[20]).sum())

    checks = {
        "short_gap_repaired": status[10] == 1,
        "long_gap_disqualified": status[12] == 2,
        "repaired_ghi_within_1pct_of_peak": rel_err_pct < 1.0,
        "successors_of_bad_day_blocked": not eligible[13:20].any(),
        "day_after_window_eligible_again": bool(eligible[20]),
        "partial_target_scored_on_present_steps_only": scored == expect,
        "no_interpolation_across_day_boundary": bool(np.isnan(mats["ghi"][12, 60:80]).all()),
    }
    checks = {k: bool(v) for k, v in checks.items()}
    checks["all_passed"] = all(checks.values())
    checks["repaired_ghi_max_err_wm2"] = round(repaired_err, 3)
    checks["repaired_ghi_max_err_pct_of_peak"] = round(rel_err_pct, 3)
    return checks


# ==========================================================================
# Clear-sky climatology
# ==========================================================================

shift_days = C.shift_days   # single definition, shared with Stage 1 and Stage 3


def all_predecessors(ok: np.ndarray, k: int) -> np.ndarray:
    """out[i] = all(ok[i-k] .. ok[i-1]). The day itself is deliberately excluded:
    an incomplete day can still be a forecast target, it just cannot be a predecessor."""
    n = len(ok)
    out = np.zeros(n, bool)
    out[k:] = np.logical_and.reduce([ok[k - j: n - j] for j in range(1, k + 1)])
    return out


def clim_table_from_allowed(
    cs_by_day: np.ndarray,
    doy: np.ndarray,
    years: np.ndarray,
    allowed: np.ndarray,
    half_window: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Day-of-year clear-sky climatology drawn only from `allowed` days of other years.

    Returns (per_day_profiles, support_years, doy_table) where doy_table is
    (367, 144) indexed by day-of-year and averaged over every allowed year, for export.
    """
    n = len(cs_by_day)
    prof = np.full_like(cs_by_day, np.nan, dtype=float)
    support = np.zeros(n, dtype=int)
    for i in range(n):
        dist = np.abs(doy - doy[i])
        dist = np.minimum(dist, 366 - dist)
        sel = (dist <= half_window) & (years != years[i]) & allowed
        if sel.any():
            prof[i] = np.nanmean(cs_by_day[sel], axis=0)
            support[i] = int(len(np.unique(years[sel])))

    table = np.full((367, C.STEPS_PER_DAY), np.nan)
    for d in range(1, 367):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, 366 - dist)
        sel = (dist <= half_window) & allowed
        if sel.any():
            table[d] = np.nanmean(cs_by_day[sel], axis=0)
    return prof, support, table


# ==========================================================================
# Features
# ==========================================================================

FEATURE_NAMES = [
    "kt_d1", "kt_d2", "kt_d3", "kt_d1_smooth",
    "ktmean_d1", "ktmean_d2", "ktmean_d3", "ktmean_d4", "ktmean_d5", "ktmean_d6", "ktmean_d7",
    "cs_ratio_d1", "cs_proxy_k",
    "temp_d1", "rh_d1", "pres_d1", "wind_d1",
    "temp_med3", "wind_med3",
    "zenith", "doy_sin", "doy_cos", "step",
]


def build_features(
    kt: np.ndarray,
    cs_solcast: np.ndarray,
    cs_proxy: np.ndarray,
    zenith: np.ndarray,
    mask: np.ndarray,
    air_temp: np.ndarray,
    wind: np.ndarray,
    rh: np.ndarray,
    pres: np.ndarray,
    doy: np.ndarray,
) -> np.ndarray:
    """(n_days, 144, F). Every column is either a lag of D-1 or earlier, or a day-D
    quantity that is deterministic in advance (astronomy, proxy clear-sky, calendar)."""
    n = len(kt)
    ktf = np.nan_to_num(kt, nan=0.0)

    kt_lags = [shift_days(ktf, k) for k in (1, 2, 3)]

    # +/- 3-step smoothing of yesterday's kt profile, edges handled by reflection.
    pad = np.pad(kt_lags[0], ((0, 0), (3, 3)), mode="edge")
    kt_smooth = np.lib.stride_tricks.sliding_window_view(pad, 7, axis=1).mean(axis=2)

    # Daytime daily mean kt per day, then lagged.
    with np.errstate(invalid="ignore"):
        daily_kt = np.nanmean(np.where(mask, kt, np.nan), axis=1)
    daily_kt = np.nan_to_num(daily_kt, nan=0.0)
    kt_day_lags = [shift_days(daily_kt[:, None], k)[:, 0] for k in range(1, 8)]

    # Yesterday's Solcast-to-proxy clear-sky ratio, one scalar per day.
    with np.errstate(invalid="ignore"):
        ratio = np.nansum(cs_solcast, axis=1) / np.where(
            np.nansum(cs_proxy, axis=1) > 0, np.nansum(cs_proxy, axis=1), np.nan
        )
    ratio_d1 = shift_days(np.nan_to_num(ratio, nan=1.0)[:, None], 1)[:, 0]

    def daily_mean_lag1(a: np.ndarray) -> np.ndarray:
        m = np.nanmean(a, axis=1)
        return shift_days(m[:, None], 1)[:, 0]

    temp_med3 = np.nanmedian(np.stack([shift_days(air_temp, k) for k in (1, 2, 3)]), axis=0)
    wind_med3 = np.nanmedian(np.stack([shift_days(wind, k) for k in (1, 2, 3)]), axis=0)

    step = np.tile(np.arange(C.STEPS_PER_DAY, dtype=float), (n, 1))
    ang = 2 * np.pi * doy / 365.25

    def bcast(v):
        return np.repeat(np.asarray(v, float)[:, None], C.STEPS_PER_DAY, axis=1)

    cols = [
        *kt_lags, kt_smooth,
        *[bcast(v) for v in kt_day_lags],
        bcast(ratio_d1), cs_proxy / 1000.0,
        bcast(daily_mean_lag1(air_temp)), bcast(daily_mean_lag1(rh)),
        bcast(daily_mean_lag1(pres)), bcast(daily_mean_lag1(wind)),
        temp_med3, wind_med3,
        zenith, bcast(np.sin(ang)), bcast(np.cos(ang)), step,
    ]
    assert len(cols) == len(FEATURE_NAMES), (len(cols), len(FEATURE_NAMES))
    return np.stack(cols, axis=-1)


# ==========================================================================
# Baselines (used both for the cs-mode choice and the acceptance check)
# ==========================================================================

def baseline_table(kt, cs_proxy, ghi, mask, idx) -> dict:
    """Prev-day and 3-day-mean kt baselines, scored in W/m² on the given day indices.

    Both come from common.py so Stage 1's acceptance check and Stage 3's evaluation
    compare against exactly the same reference.
    """
    out = {}
    for name, khat in (("prev_day", C.baseline_prev_kt(kt)),
                       ("mean3", C.baseline_mean3_kt(kt))):
        out[name] = C.point_metrics(ghi[idx], (khat * cs_proxy)[idx], mask[idx])["rmse"]
    return out


# ==========================================================================
# Leak tests
# ==========================================================================

def leak_tests(ctx: dict, aux: dict, test_idx: np.ndarray, rng_seed: int = 0) -> dict:
    """The three tests of WORKFLOW.md section 3. Each returns pass/fail plus the
    largest absolute feature change observed, which should be exactly 0.

    `ctx` is the build_features keyword bundle; `aux` carries what test 2 needs to
    rebuild the climatology.
    """
    rng = np.random.default_rng(rng_seed)
    i = int(test_idx[len(test_idx) // 2])
    keys = ("kt", "cs_solcast", "air_temp", "wind", "rh", "pres")

    def features_for(day_arrays: dict, row: int) -> np.ndarray:
        return build_features(**day_arrays)[row]

    base = features_for(ctx, i)

    # 1. Blank every actual from day i onward.
    blanked = dict(ctx)
    for k in keys:
        a = ctx[k].copy()
        a[i:] = np.nan
        blanked[k] = a
    d1 = np.nanmax(np.abs(np.nan_to_num(features_for(blanked, i) - base)))

    # 2. Rebuild the climatology with every day from (test start - purge) onward blanked.
    allowed, cs = aux["allowed_clim"], aux["cs_raw"]
    cs_cut = cs.copy()
    cs_cut[aux["purge_idx"]:] = np.nan
    p_full, _, _ = clim_table_from_allowed(cs, aux["doy"], aux["years"], allowed)
    p_cut, _, _ = clim_table_from_allowed(cs_cut, aux["doy"], aux["years"], allowed)
    d2 = np.nanmax(np.abs(p_full[test_idx] - p_cut[test_idx]))

    # 3. Replace day i's actuals with random values.
    noisy = dict(ctx)
    for k in keys:
        a = ctx[k].copy()
        a[i] = rng.uniform(0, np.nanmax(ctx[k]), size=C.STEPS_PER_DAY)
        noisy[k] = a
    d3 = np.nanmax(np.abs(np.nan_to_num(features_for(noisy, i) - base)))

    return {
        "day_tested": str(pd.Timestamp(aux["dates"][i]).date()),
        "test1_blank_future_max_feature_delta": float(d1),
        "test2_clim_cutoff_max_delta_wm2": float(np.nan_to_num(d2)),
        "test3_perturb_day_max_feature_delta": float(d3),
        "all_passed": bool(d1 == 0 and np.nan_to_num(d2) == 0 and d3 == 0),
    }


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="glob or path to the merged Solcast CSV")
    ap.add_argument("--cs-mode", default="clim",
                    choices=["lagged", "clim", "clim_ratio", "solcast"])
    ap.add_argument("--out", default=str(C.PREP_PATH.parent))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. load, audit, place on a gap-free grid --------------------------
    df = C.load_solcast(args.csv)
    df["present"] = 1.0
    audit = C.grid_audit(df)
    assert audit["duplicate_timestamps"] == 0, audit
    assert audit["missing_timestamps"] == 0, audit
    df = reindex_full_grid(df)
    print(f"[load] {audit['rows']:,} rows  {audit['start']} -> {audit['end']}")

    # --- 2. solar position at period mid-points ----------------------------
    sp = C.solar_position(df["mid_utc"])
    df["zenith_calc"] = sp["zenith"]
    df["azimuth_calc"] = sp["azimuth"]

    # --- 3. day matrices ---------------------------------------------------
    cols = (*RAW_COLS, "present", "zenith_calc", "azimuth_calc")
    dates, mats = day_matrices(df, cols)
    present = mats["present"] == 1.0
    per_day = df.drop_duplicates("date")
    doy = per_day["doy"].to_numpy()
    years = per_day["year"].to_numpy()
    is_monsoon = per_day["is_monsoon"].to_numpy().astype(bool)
    print(f"[days] {len(dates)} calendar days  {dates[0].date()} -> {dates[-1].date()}")

    # --- 4. completeness audit and gap policy (section 7a) -----------------
    comp = completeness_audit(dates, present)
    print(f"[steps] {comp['n_days_with_144_steps']}/{comp['n_days']} days have all 144 "
          f"steps ({100*comp['n_days_with_144_steps']/comp['n_days']:.2f}%)")
    for sd in comp["short_days"]:
        print(f"[steps]   SHORT {sd['date']}: {sd['n_steps']}/144 steps, "
              f"{sd['n_missing']} missing, longest gap {sd['longest_gap_steps']} steps, "
              f"runs {[(r['start_step'], r['length']) for r in sd['gap_runs']]}")

    selftest = gap_policy_selftest()
    print(f"[gap-policy] selftest: {selftest}")
    if not selftest["all_passed"]:
        raise SystemExit("ABORT: the gap policy self-test failed.")

    status, repair_log = classify_and_repair(mats, present)
    for r in repair_log:
        r["date"] = str(dates[r["day_index"]].date())
        print(f"[gap-policy] {r['date']}: {r['action']} "
              f"(longest gap {r['longest_gap_steps']} steps, {r['n_missing']} missing)")
    n_rep, n_bad = int((status == 1).sum()), int((status == 2).sum())
    print(f"[gap-policy] complete {int((status==0).sum())} | repaired {n_rep} | "
          f"unrepairable {n_bad} (max repairable gap = {MAX_REPAIR_GAP_STEPS} steps)")
    for name, m in zip(cols, (mats[c] for c in cols)):
        if name in RAW_COLS:
            assert not np.isnan(m[status != 2]).any(), f"{name} still has NaN on a usable day"

    ghi, cs_solcast = mats["ghi"], mats["clearsky_ghi"]
    air_temp, wind = mats["air_temp"], mats["wind_speed_10m"]
    rh, pres = mats["relative_humidity"], mats["surface_pressure"]
    zenith, azimuth = mats["zenith_calc"], mats["azimuth_calc"]

    # --- 5. eligibility: predecessors must be complete or repaired ---------
    eligible = all_predecessors(status != 2, PRIOR_DAYS)
    # Whole calendar days are never missing from the grid by construction, but assert it.
    assert (dates[PRIOR_DAYS:] - dates[:-PRIOR_DAYS] == pd.Timedelta(days=PRIOR_DAYS)).all()

    dev_end = pd.Timestamp(DEV_END, tz=C.TZ)
    test_start = pd.Timestamp(TEST_START, tz=C.TZ)
    purge_idx = int(np.searchsorted(dates, test_start - pd.Timedelta(days=1)))
    is_dev_clim = eligible & (dates <= dev_end)

    # --- 6. clear-sky proxies ---------------------------------------------
    cs_lagged = C.clearsky_lagged_median(cs_solcast)
    cs_clim, clim_support, clim_table = clim_table_from_allowed(
        cs_solcast, doy, years, is_dev_clim
    )
    cs_ratio = C.clearsky_clim_times_ratio(cs_clim, cs_solcast)
    proxies = {"lagged": cs_lagged, "clim": cs_clim,
               "clim_ratio": cs_ratio, "solcast": cs_solcast}

    # A proxy that is undefined on a predecessor would make that day's kt NaN, which
    # build_features turns into a silent 0 -- the failure mode that produced a 535 W/m2
    # baseline day in an earlier revision. Require the proxy on the day and its
    # PRIOR_DAYS predecessors, per mode.
    def mode_usable(cs: np.ndarray) -> np.ndarray:
        ok = np.isfinite(cs).all(axis=1)
        daylight_any = (C.daylight_mask(zenith, cs) & present).any(axis=1)
        return eligible & ok & all_predecessors(ok, PRIOR_DAYS) & daylight_any

    usable_by_mode = {k: mode_usable(v) for k, v in proxies.items()}
    usable = usable_by_mode[args.cs_mode]

    is_dev = usable & (dates <= dev_end)
    is_test = usable & (dates >= test_start)
    dev_idx, test_idx = np.flatnonzero(is_dev), np.flatnonzero(is_test)
    print(f"[split] usable {int(usable.sum())} | dev {len(dev_idx)} "
          f"({dates[dev_idx[0]].date()} -> {dates[dev_idx[-1]].date()}) | "
          f"test {len(test_idx)} ({dates[test_idx[0]].date()} -> {dates[test_idx[-1]].date()})")
    val_cut = dev_idx[int(len(dev_idx) * 0.75)]
    val_idx = dev_idx[dev_idx >= val_cut]
    print(f"[split] validation {len(val_idx)} days, {int(is_monsoon[val_idx].sum())} monsoon")
    print(f"[clim]  day-of-year support: median {int(np.median(clim_support[usable]))} other years")

    # Every day that feeds a model must carry a full 144-step profile and a defined
    # clear-sky proxy, so that no lag feature is a nan_to_num stand-in for 0.
    needed = np.unique(np.concatenate([
        np.concatenate([dev_idx, test_idx, [purge_idx]]) - k for k in range(1, PRIOR_DAYS + 1)
    ]))
    bad = needed[status[needed] == 2]
    assert bad.size == 0, (
        f"ABORT: {bad.size} predecessor day(s) lack 144 steps and were not repaired: "
        f"{[str(dates[i].date()) for i in bad[:10]]}"
    )
    sel_cs = proxies[args.cs_mode]
    assert np.isfinite(sel_cs[needed]).all(), (
        "ABORT: the clear-sky proxy is undefined on a predecessor day: "
        f"{[str(dates[i].date()) for i in needed[~np.isfinite(sel_cs[needed]).all(axis=1)]][:10]}"
    )
    kt_chk = C.kt_from(ghi, sel_cs)
    dl_chk = C.daylight_mask(zenith, sel_cs) & present
    assert np.isfinite(kt_chk[needed][dl_chk[needed]]).all(), \
        "ABORT: kt is NaN on an observed daylight step of a predecessor day."
    print(f"[steps] ASSERTED: all {len(needed)} predecessor days carry 144 steps "
          f"(complete or repaired within {MAX_REPAIR_GAP_STEPS} steps), a defined "
          f"clear-sky proxy and finite kt on every observed daylight step")

    # --- 7. cs-mode comparison, decided on development data ---------------
    comparison = {}
    for name, cs in proxies.items():
        kt_n = C.kt_from(ghi, cs)
        mask_n = C.daylight_mask(zenith, cs) & present
        u = usable_by_mode[name]
        d_i = np.flatnonzero(u & (dates <= dev_end))
        t_i = np.flatnonzero(u & (dates >= test_start))
        comparison[name] = {
            "n_dev_days": len(d_i), "n_test_days": len(t_i),
            "dev": baseline_table(kt_n, cs, ghi, mask_n, d_i),
            "test": baseline_table(kt_n, cs, ghi, mask_n, t_i),
        }
    print("\n[cs-mode] baseline RMSE W/m2  (choose on dev; test shown only to match "
          "WORKFLOW section 1)")
    print(f"{'mode':<12}{'dev days':>9}{'dev prev':>10}{'dev mean3':>11}"
          f"{'test prev':>11}{'test mean3':>12}")
    for name, r in comparison.items():
        flag = "  <- LEAKY" if name == "solcast" else ""
        print(f"{name:<12}{r['n_dev_days']:>9}{r['dev']['prev_day']:>10.1f}"
              f"{r['dev']['mean3']:>11.1f}{r['test']['prev_day']:>11.1f}"
              f"{r['test']['mean3']:>12.1f}{flag}")
    print(f"[cs-mode] selected: {args.cs_mode}\n")

    # --- 8. target, mask, features ----------------------------------------
    cs_proxy = proxies[args.cs_mode]
    kt = C.kt_from(ghi, cs_proxy)
    daylight = C.daylight_mask(zenith, cs_proxy)
    mask = daylight & present   # score and train on observed steps only

    ctx = dict(
        kt=kt, cs_solcast=cs_solcast, cs_proxy=cs_proxy, zenith=zenith, mask=mask,
        air_temp=air_temp, wind=wind, rh=rh, pres=pres, doy=doy,
    )
    X = build_features(**ctx)
    print(f"[features] X {X.shape} ({len(FEATURE_NAMES)} columns), "
          f"{int(mask[usable].sum()):,} daylight rows on usable days")

    # --- 9. leak tests -----------------------------------------------------
    aux = dict(allowed_clim=is_dev_clim, cs_raw=cs_solcast, purge_idx=purge_idx,
               doy=doy, years=years, dates=dates)
    lt = leak_tests(ctx, aux, test_idx)
    for k, v in lt.items():
        print(f"[leak] {k}: {v}")
    if not lt["all_passed"]:
        raise SystemExit("ABORT: a leak test failed; do not proceed to Stage 1.")

    # --- 10. outputs -------------------------------------------------------
    np.savez_compressed(
        out / "prep.npz",
        dates=np.array([str(d.date()) for d in dates]), X=X.astype(np.float32),
        feature_names=np.array(FEATURE_NAMES), kt=kt, ghi=ghi, cs_proxy=cs_proxy,
        cs_solcast=cs_solcast, mask=mask, daylight=daylight, present=present,
        day_status=status, zenith=zenith, azimuth=azimuth,
        air_temp=air_temp, wind_speed=wind, rh=rh, pres=pres,
        doy=doy, year=years, is_monsoon=is_monsoon,
        usable=usable, dev_idx=dev_idx, test_idx=test_idx, val_idx=val_idx,
        purge_idx=purge_idx,
    )
    np.save(out / "cs_clim_table.npy", clim_table)
    meta = {
        "cs_mode": args.cs_mode,
        "grid_audit": audit,
        "completeness": {
            **comp,
            "policy": {
                "max_repair_gap_steps": MAX_REPAIR_GAP_STEPS,
                "smooth_fields_interpolated": list(SMOOTH_FIELDS),
                "ghi_repair": "linear interpolation of kt, rescaled by the clear-sky profile",
                "never_across_day_boundary": True,
                "target_day_rule": "a day with incomplete actuals is still forecast; "
                                   "it is scored only on steps that exist",
                "predecessor_rule": f"a day is unusable if any of its {PRIOR_DAYS} "
                                    "predecessors is unrepairable or has an undefined "
                                    "clear-sky proxy",
            },
            "n_repaired": n_rep,
            "n_unrepairable": n_bad,
            "n_complete": int((status == 0).sum()),
            "repair_log": repair_log,
            "selftest": selftest,
        },
        "n_days": len(dates),
        "n_usable_days": int(usable.sum()),
        "dev": {"n": len(dev_idx), "start": str(dates[dev_idx[0]].date()),
                "end": str(dates[dev_idx[-1]].date())},
        "test": {"n": len(test_idx), "start": str(dates[test_idx[0]].date()),
                 "end": str(dates[test_idx[-1]].date())},
        "validation": {"n": len(val_idx), "n_monsoon": int(is_monsoon[val_idx].sum()),
                       "start": str(dates[val_idx[0]].date())},
        "purge_day": str(dates[purge_idx].date()),
        "clim_support_median_other_years": int(np.median(clim_support[usable])),
        "cs_mode_baselines_rmse_wm2": comparison,
        "leak_tests": lt,
        "feature_names": FEATURE_NAMES,
    }
    (out / "prep.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[done] wrote {out/'prep.npz'}, {out/'prep.json'}, {out/'cs_clim_table.npy'}")


if __name__ == "__main__":
    main()
