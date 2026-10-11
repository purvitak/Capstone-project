"""Stage 2 — out-of-fold residual correction and calibrated 80% intervals.

Implements WORKFLOW.md section 9. The correction may well add nothing; if so this
script says so plainly rather than hiding it. Everything is fitted on out-of-fold
development forecasts only, and delta_max, the boosting iteration count and the
interval quantiles are all chosen on the validation slice of those OOF days.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import common as C

# Night rows and days without an OOF base forecast are all-NaN by design.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# The grid includes 0, i.e. "apply no correction at all", so a null result can win.
DELTA_GRID = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
MIN_STRATUM_DAYS = 30
KT_BINS = (0.6, 0.85)       # predicted daily-mean-kt bin edges
NOMINAL = 0.80


def kt_bin(v: np.ndarray) -> np.ndarray:
    return np.digitize(v, KT_BINS)


def build_corrector_features(kt, kt_base, cs_proxy, cs_solcast, zenith, mask, doy):
    """Features of WORKFLOW section 9. Every term is known at issue time: the base
    forecast for day D, and realised quantities from D-1 and earlier."""
    n = C.STEPS_PER_DAY
    with np.errstate(invalid="ignore"):
        base_daily = np.nanmean(np.where(mask, kt_base, np.nan), axis=1)
        kt_daily = np.nanmean(np.where(mask, kt, np.nan), axis=1)

    # Yesterday's error profile: realised kt(D-1) minus the base forecast issued for
    # D-1. kt_base is out-of-sample everywhere it is finite, so this never peeks.
    err = kt - kt_base
    err_d1 = C.shift_days(err, 1)
    pad = np.pad(np.nan_to_num(err_d1, nan=0.0), ((0, 0), (3, 3)), mode="edge")
    err_d1_smooth = np.lib.stride_tricks.sliding_window_view(pad, 7, axis=1).mean(axis=2)
    with np.errstate(invalid="ignore"):
        err_d1_daily = np.nanmean(np.where(C.shift_days(mask.astype(float), 1) > 0,
                                           err_d1, np.nan), axis=1)

    with np.errstate(invalid="ignore"):
        ratio = np.nansum(cs_solcast, axis=1) / np.where(
            np.nansum(cs_proxy, axis=1) > 0, np.nansum(cs_proxy, axis=1), np.nan)
    ratio_d1 = C.shift_days(ratio[:, None], 1)[:, 0]
    kt_daily_d1 = C.shift_days(kt_daily[:, None], 1)[:, 0]

    step = np.tile(np.arange(n, dtype=float), (len(kt), 1))
    ang = 2 * np.pi * doy / 365.25

    def b(v):
        return np.repeat(np.asarray(v, float)[:, None], n, axis=1)

    cols = [kt_base, b(base_daily), err_d1, err_d1_smooth, b(err_d1_daily),
            cs_proxy / 1000.0, zenith, step, b(np.sin(ang)), b(np.cos(ang)),
            b(ratio_d1), b(kt_daily_d1)]
    names = ["kt_base", "kt_base_daily", "err_d1", "err_d1_smooth", "err_d1_daily",
             "cs_proxy_k", "zenith", "step", "doy_sin", "doy_cos",
             "cs_ratio_d1", "kt_daily_d1"]
    return np.stack(cols, axis=-1), names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="results/prep.npz")
    ap.add_argument("--base", default="results/base_pred.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--prefix", default="",
                    help="Output filename prefix so DWT-BiLSTM does not overwrite GBM "
                         "(use dwt_ to write dwt_stage2.npz / dwt_stage2.json)")
    args = ap.parse_args()
    out = Path(args.out)
    prefix = args.prefix
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.prep, allow_pickle=True)
    kt, mask, ghi, cs = d["kt"], d["mask"], d["ghi"], d["cs_proxy"]
    dev_idx, test_idx = d["dev_idx"], d["test_idx"]
    is_monsoon, dates = d["is_monsoon"].astype(bool), d["dates"]
    kt_base = np.load(args.base, allow_pickle=True)["kt_base"]

    Xc, names = build_corrector_features(kt, kt_base, cs, d["cs_solcast"],
                                         d["zenith"], mask, d["doy"])

    def rmse(days, khat):
        return C.point_metrics(ghi[days], np.clip(khat, *C.KT_CLIP) * cs[days],
                               mask[days])["rmse"]

    # --- out-of-fold development days, split chronologically 75/25 --------
    oof = dev_idx[np.isfinite(kt_base[dev_idx]).any(axis=1)]
    cut = int(len(oof) * 0.75)
    tr_days, va_days = oof[:cut], oof[cut:]
    print(f"[stage2] OOF days {len(oof)}  train {len(tr_days)} "
          f"({dates[tr_days[0]]} -> {dates[tr_days[-1]]})  "
          f"validation {len(va_days)} ({dates[va_days[0]]} -> {dates[va_days[-1]]}, "
          f"{int(is_monsoon[va_days].sum())} monsoon)")

    def rows(days):
        m = mask[days] & np.isfinite(kt[days]) & np.isfinite(kt_base[days])
        return Xc[days][m], (kt - kt_base)[days][m], m

    Xtr, ytr, _ = rows(tr_days)
    Xva, yva, mva = rows(va_days)

    # --- boosting iterations chosen on validation -------------------------
    model = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=100, l2_regularization=1.0, early_stopping=False,
        random_state=42).fit(Xtr, ytr)
    stage_rmse = [float(np.sqrt(np.mean((p - yva) ** 2))) for p in model.staged_predict(Xva)]
    best_iter = int(np.argmin(stage_rmse)) + 1
    print(f"[stage2] best boosting iterations on validation: {best_iter} "
          f"(kt-RMSE {min(stage_rmse):.5f} vs {stage_rmse[-1]:.5f} at 500)")

    # Validation predictions at the chosen iteration, for the delta and quantile fits.
    raw_va = np.full((len(va_days), C.STEPS_PER_DAY), np.nan)
    raw_va[mva] = list(model.staged_predict(Xva))[best_iter - 1]

    # --- bounded correction, delta_max chosen per season on validation ----
    def apply_delta(base, raw, delta):
        if delta == 0:
            return base.copy()
        return np.clip(base + delta * np.tanh(raw / delta), *C.KT_CLIP)

    season_va = {"monsoon": is_monsoon[va_days], "non_monsoon": ~is_monsoon[va_days]}
    pooled_scores = [rmse(va_days, apply_delta(kt_base[va_days], raw_va, dl))
                     for dl in DELTA_GRID]
    pooled_delta = DELTA_GRID[int(np.argmin(pooled_scores))]
    print(f"[stage2] pooled delta_max grid: "
          f"{[f'{dl}:{s:.2f}' for dl, s in zip(DELTA_GRID, pooled_scores)]}")

    delta_by_season, delta_detail = {}, {}
    for name, sel in season_va.items():
        days = va_days[sel]
        if len(days) < MIN_STRATUM_DAYS:
            delta_by_season[name] = pooled_delta
            delta_detail[name] = {"n_days": int(len(days)), "source": "pooled fallback"}
            continue
        sc = [rmse(days, apply_delta(kt_base[days], raw_va[sel], dl)) for dl in DELTA_GRID]
        delta_by_season[name] = DELTA_GRID[int(np.argmin(sc))]
        delta_detail[name] = {"n_days": int(len(days)), "source": "fitted",
                              "grid": dict(zip(map(str, DELTA_GRID), sc)),
                              "best_rmse": float(min(sc)),
                              "rmse_at_delta0": float(sc[0])}
        print(f"[stage2] {name}: delta_max {delta_by_season[name]} "
              f"(RMSE {min(sc):.3f} vs {sc[0]:.3f} with no correction, "
              f"{len(days)} days)")

    # --- refit on all OOF days, then apply to the test year ---------------
    Xall, yall, _ = rows(oof)
    final = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=best_iter, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=100, l2_regularization=1.0, early_stopping=False,
        random_state=42).fit(Xall, yall)
    print(f"[stage2] refit on all {len(oof)} OOF days ({len(yall):,} rows)")

    n_days = kt.shape[0]
    kt_final = np.full_like(kt_base, np.nan)
    raw_all = np.full_like(kt_base, np.nan)
    for days in (va_days, test_idx):
        Xp, _, mp = rows(days)
        blk = np.full((len(days), C.STEPS_PER_DAY), np.nan)
        blk[mp] = final.predict(Xp) if days is not va_days else raw_va[mp]
        raw_all[days] = blk
        dl = np.where(is_monsoon[days], delta_by_season["monsoon"],
                      delta_by_season["non_monsoon"])[:, None]
        kt_final[days] = np.where(dl == 0, kt_base[days],
                                  np.clip(kt_base[days] + dl * np.tanh(blk / np.where(dl == 0, 1, dl)),
                                          *C.KT_CLIP))

    # --- calibrated 80% intervals from validation residuals ---------------
    lo_q, hi_q = (1 - NOMINAL) / 2, 1 - (1 - NOMINAL) / 2
    res_va = kt[va_days] - kt_final[va_days]
    with np.errstate(invalid="ignore"):
        pred_daily_va = np.nanmean(np.where(mask[va_days], kt_final[va_days], np.nan), axis=1)
    strat_va = list(zip(is_monsoon[va_days].astype(int), kt_bin(pred_daily_va)))
    flat = res_va[mask[va_days] & np.isfinite(res_va)]
    pooled_q = (float(np.quantile(flat, lo_q)), float(np.quantile(flat, hi_q)))

    quantiles, qdetail = {}, {}
    for key in sorted(set(strat_va)):
        sel = np.array([s == key for s in strat_va])
        r = res_va[sel][mask[va_days][sel] & np.isfinite(res_va[sel])]
        if sel.sum() < MIN_STRATUM_DAYS or r.size < 100:
            quantiles[key] = pooled_q
            qdetail[str(key)] = {"n_days": int(sel.sum()), "source": "pooled fallback"}
        else:
            quantiles[key] = (float(np.quantile(r, lo_q)), float(np.quantile(r, hi_q)))
            qdetail[str(key)] = {"n_days": int(sel.sum()), "source": "fitted",
                                 "q_lo": quantiles[key][0], "q_hi": quantiles[key][1]}

    kt_lo, kt_hi = np.full_like(kt_final, np.nan), np.full_like(kt_final, np.nan)
    for days in (va_days, test_idx):
        with np.errstate(invalid="ignore"):
            pd_ = np.nanmean(np.where(mask[days], kt_final[days], np.nan), axis=1)
        for j, day in enumerate(days):
            key = (int(is_monsoon[day]), int(kt_bin(pd_[j:j + 1])[0]))
            q = quantiles.get(key, pooled_q)
            kt_lo[day] = np.clip(kt_final[day] + q[0], *C.KT_CLIP)
            kt_hi[day] = np.clip(kt_final[day] + q[1], *C.KT_CLIP)

    # --- honest verdict ----------------------------------------------------
    r1 = rmse(test_idx, kt_base[test_idx])
    r2 = rmse(test_idx, kt_final[test_idx])
    mean3 = C.baseline_mean3_kt(kt)
    rb = rmse(test_idx, mean3[test_idx])
    cov = float(np.nanmean(((kt >= kt_lo) & (kt <= kt_hi))[test_idx][mask[test_idx]]))
    width = float(np.nanmean(((kt_hi - kt_lo) * cs)[test_idx][mask[test_idx]]))

    print(f"\n[test] Stage 1 {r1:.2f} -> Stage 2 {r2:.2f} W/m2 "
          f"(change {r2 - r1:+.2f}, {100*(r1-r2)/r1:+.2f}%)")
    print(f"[test] skill vs 3-day-mean: Stage 1 {100*C.skill(r1,rb):+.2f}%  "
          f"Stage 2 {100*C.skill(r2,rb):+.2f}%")
    print(f"[test] 80% interval: PICP {100*cov:.1f}% (nominal 80%), "
          f"mean width {width:.1f} W/m2")
    verdict = ("the correction HELPS" if r2 < r1 - 0.05 else
               "the correction is a NULL RESULT - it does not improve on Stage 1"
               if r2 <= r1 + 0.05 else "the correction HURTS")
    print(f"[test] VERDICT: {verdict}")
    if all(v == 0.0 for v in delta_by_season.values()):
        print("[test] delta_max was chosen as 0 for every season on validation, i.e. "
              "the validation data itself preferred no correction.")

    np.savez_compressed(out / f"{prefix}stage2.npz", kt_final=kt_final, kt_lo=kt_lo, kt_hi=kt_hi,
                        raw_correction=raw_all, dates=dates)
    (out / f"{prefix}stage2.json").write_text(json.dumps({
        "feature_names": names, "best_iterations": best_iter,
        "val_kt_rmse_at_best": min(stage_rmse), "val_kt_rmse_at_max_iter": stage_rmse[-1],
        "delta_grid": DELTA_GRID, "delta_max_by_season": delta_by_season,
        "delta_detail": delta_detail, "pooled_delta_max": pooled_delta,
        "pooled_delta_grid_rmse": dict(zip(map(str, DELTA_GRID), pooled_scores)),
        "interval_nominal": NOMINAL, "pooled_quantiles": pooled_q,
        "quantiles_by_stratum": qdetail,
        "test_stage1_rmse": r1, "test_stage2_rmse": r2, "test_mean3_rmse": rb,
        "test_stage1_skill": C.skill(r1, rb), "test_stage2_skill": C.skill(r2, rb),
        "test_picp": cov, "test_mean_width_wm2": width, "verdict": verdict,
        "n_oof_days": len(oof), "n_val_days": len(va_days),
    }, indent=2, default=float))
    print(f"[done] wrote {out / (prefix + 'stage2.npz')}, {out / (prefix + 'stage2.json')}")


if __name__ == "__main__":
    main()
