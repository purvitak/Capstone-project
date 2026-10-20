"""Stage 3 — evaluation, paired significance tests, interval calibration, figures.

Implements docs/archive/WORKFLOW.md section 10. The moving-block bootstrap CI is treated as the
primary inferential statement, because overlapping daily forecasts are autocorrelated
and a test over ~24k 10-minute errors would be badly over-confident.

A monthly per-step kt climatology (fitted on development days only) is included as a
baseline on purpose: it reveals how much of the gradient-boosting skill is merely
seasonal knowledge that a cheap climatology also has.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")

NOMINAL = 0.80
ALPHA = 1 - NOMINAL


def monthly_climatology_kt(kt, mask, months, dev_idx, all_idx):
    """Per-month, per-step mean kt learned on development days only."""
    out = np.full((len(kt), C.STEPS_PER_DAY), np.nan)
    table = {}
    for m in range(1, 13):
        sel = dev_idx[months[dev_idx] == m]
        if sel.size:
            table[m] = np.nanmean(np.where(mask[sel], kt[sel], np.nan), axis=0)
    for i in all_idx:
        prof = table.get(int(months[i]))
        if prof is not None:
            out[i] = np.nan_to_num(prof, nan=0.0)
    return out, table


def winkler(y, lo, hi, alpha=ALPHA):
    """Interval score: width, plus a penalty proportional to any miss."""
    w = hi - lo
    return np.where(y < lo, w + 2 / alpha * (lo - y),
                    np.where(y > hi, w + 2 / alpha * (y - hi), w))


def hourly_rmse(y, yhat, mask):
    """RMSE of hourly-aggregated GHI, the resolution most of the literature reports."""
    n_days, n = y.shape
    h = n // C.STEPS_PER_HOUR
    def agg(a):
        a = np.where(mask, a, np.nan)
        return np.nanmean(a.reshape(n_days, h, C.STEPS_PER_HOUR), axis=2)
    ya, pa = agg(y), agg(yhat)
    ok = np.isfinite(ya) & np.isfinite(pa)
    return float(np.sqrt(np.mean((pa[ok] - ya[ok]) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=str(C.PREP_PATH))
    ap.add_argument("--out", default=str(C.RESULTS_DIR))
    ap.add_argument("--base", default=str(C.RESULTS_DIR / "dwt_base_pred.npz"))
    ap.add_argument("--stage2", default=str(C.RESULTS_DIR / "dwt_stage2.npz"))
    ap.add_argument("--prefix", default="dwt_",
                    help="Prefix for output csv/json/png (use dwt_ for the DWT-BiLSTM path)")
    ap.add_argument("--stage1-name", default="dwt_bilstm")
    ap.add_argument("--stage2-name", default="stage2_corrected")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    d = np.load(args.prep, allow_pickle=True)
    kt, mask, ghi, cs = d["kt"], d["mask"], d["ghi"], d["cs_proxy"]
    dev_idx, test_idx = d["dev_idx"], d["test_idx"]
    is_monsoon, dates = d["is_monsoon"].astype(bool), d["dates"]
    kt_base = np.load(args.base, allow_pickle=True)["kt_base"]
    s2 = np.load(args.stage2, allow_pickle=True)
    kt_final, kt_lo, kt_hi = s2["kt_final"], s2["kt_lo"], s2["kt_hi"]
    months = pd.to_datetime(pd.Series(dates)).dt.month.to_numpy()

    clim_kt, _ = monthly_climatology_kt(kt, mask, months, dev_idx,
                                        np.concatenate([dev_idx, test_idx]))

    methods = {
        "persistence_prev_day": C.baseline_prev_kt(kt),
        "mean3": C.baseline_mean3_kt(kt),
        "monthly_climatology": clim_kt,
        args.stage1_name: kt_base,
        args.stage2_name: kt_final,
    }
    ghi_hat = {k: np.clip(v, *C.KT_CLIP) * cs for k, v in methods.items()}

    # ---------------- metrics table ---------------------------------------
    rows = []
    for season, sel in (("all", np.ones(len(test_idx), bool)),
                        ("monsoon", is_monsoon[test_idx]),
                        ("non_monsoon", ~is_monsoon[test_idx])):
        days = test_idx[sel]
        ref3 = C.point_metrics(ghi[days], ghi_hat["mean3"][days], mask[days])["rmse"]
        refp = C.point_metrics(ghi[days], ghi_hat["persistence_prev_day"][days],
                               mask[days])["rmse"]
        for name, gh in ghi_hat.items():
            m = C.point_metrics(ghi[days], gh[days], mask[days])
            rows.append({
                "method": name, "season": season, "n_days": int(len(days)), **m,
                "rmse_hourly": hourly_rmse(ghi[days], gh[days], mask[days]),
                "skill_vs_mean3_pct": 100 * C.skill(m["rmse"], ref3),
                "skill_vs_persistence_pct": 100 * C.skill(m["rmse"], refp),
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / f"{prefix}metrics.csv", index=False)
    print("\n=== Test-year metrics (daylight steps, W/m2) ===")
    print(metrics[metrics.season == "all"][
        ["method", "rmse", "mae", "mbe", "nrmse", "rmse_hourly",
         "skill_vs_mean3_pct", "skill_vs_persistence_pct"]].to_string(index=False))

    # ---------------- paired tests on per-day RMSE ------------------------
    pdr = {k: C.per_day_rmse(ghi[test_idx], v[test_idx], mask[test_idx])
           for k, v in ghi_hat.items()}
    pairs = [(args.stage1_name, "mean3"), (args.stage2_name, args.stage1_name),
             (args.stage2_name, "mean3"), (args.stage1_name, "monthly_climatology"),
             (args.stage1_name, "persistence_prev_day")]
    prows = []
    for a, b in pairs:
        diff = pdr[a] - pdr[b]          # negative means `a` is better
        boot = C.moving_block_bootstrap_ci(diff, block=7)
        wil = C.wilcoxon_days(pdr[a], pdr[b])
        prows.append({
            "model": a, "reference": b,
            "mean_daily_rmse_diff": boot["mean"],
            "ci95_lo": boot["lo"], "ci95_hi": boot["hi"],
            "ci_excludes_zero": boot["excludes_zero"],
            "wilcoxon_p": wil["p"], "n_days": wil["n_days"],
            "better": a if boot["mean"] < 0 else b,
            "significant": bool(boot["excludes_zero"]),
        })
    paired = pd.DataFrame(prows)
    paired.to_csv(out / f"{prefix}paired_tests.csv", index=False)
    print("\n=== Paired tests on per-day RMSE (moving-block bootstrap, 7-day blocks) ===")
    print("negative difference = the model beats its reference; CI is primary")
    print(paired.to_string(index=False))

    # ---------------- interval calibration --------------------------------
    irows = []
    for season, sel in (("all", np.ones(len(test_idx), bool)),
                        ("monsoon", is_monsoon[test_idx]),
                        ("non_monsoon", ~is_monsoon[test_idx])):
        days = test_idx[sel]
        m = mask[days] & np.isfinite(kt[days]) & np.isfinite(kt_lo[days])
        y, lo, hi = (ghi[days][m], (kt_lo[days] * cs[days])[m], (kt_hi[days] * cs[days])[m])
        irows.append({
            "season": season, "nominal_pct": 100 * NOMINAL,
            "picp_pct": 100 * float(np.mean((y >= lo) & (y <= hi))),
            "mean_width_wm2": float(np.mean(hi - lo)),
            "pinaw_pct": 100 * float(np.mean(hi - lo) / (y.max() - y.min())),
            "winkler_score": float(np.mean(winkler(y, lo, hi))),
            "n": int(m.sum()),
        })
    intervals = pd.DataFrame(irows)
    intervals.to_csv(out / f"{prefix}intervals.csv", index=False)
    print("\n=== 80% prediction intervals on the test year ===")
    print(intervals.to_string(index=False))

    # ---------------- lead-time curve -------------------------------------
    lead_h = np.arange(C.STEPS_PER_DAY) // C.STEPS_PER_HOUR + 1
    lrows = []
    for h in range(1, 25):
        cols = lead_h == h
        sub = np.zeros_like(mask); sub[:, cols] = True
        mm = mask & sub
        if not mm[test_idx].any():
            continue
        r = {"lead_hour": h}
        for name, gh in ghi_hat.items():
            r[name] = C.point_metrics(ghi[test_idx], gh[test_idx], mm[test_idx])["rmse"]
        lrows.append(r)
    lead = pd.DataFrame(lrows)
    lead.to_csv(out / f"{prefix}lead_time.csv", index=False)

    # ---------------- figures ---------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    s1, s2n = args.stage1_name, args.stage2_name
    for name in ("mean3", "monthly_climatology", s1, s2n):
        ax.plot(lead["lead_hour"], lead[name], marker="o", ms=3, label=name)
    ax.set_xlabel("forecast lead (hours from the 00:00 IST origin)")
    ax.set_ylabel("RMSE (W/m$^2$)")
    ax.set_title("Test-year RMSE by lead time (daylight steps)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(out / f"{prefix}fig_lead_time.png", dpi=140); plt.close(fig)

    sub = metrics[metrics.method.isin(["mean3", "monthly_climatology", s1, s2n])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    seasons = ["all", "monsoon", "non_monsoon"]
    w, names = 0.2, sub.method.unique()
    for j, nm in enumerate(names):
        vals = [float(sub[(sub.method == nm) & (sub.season == s)]["rmse"].iloc[0])
                for s in seasons]
        ax.bar(np.arange(3) + j * w, vals, w, label=nm)
    ax.set_xticks(np.arange(3) + 1.5 * w); ax.set_xticklabels(seasons)
    ax.set_ylabel("RMSE (W/m$^2$)"); ax.set_title("Test-year RMSE by season")
    ax.legend(); ax.grid(alpha=.3, axis="y"); fig.tight_layout()
    fig.savefig(out / f"{prefix}fig_rmse_by_season.png", dpi=140); plt.close(fig)

    # A clear day and a monsoon day, actual vs forecast with the 80% band.
    dm = np.nanmean(np.where(mask[test_idx], kt[test_idx], np.nan), axis=1)
    picks = [("clear", test_idx[int(np.nanargmax(dm))]),
             ("cloudy", test_idx[int(np.nanargmin(dm))])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for axx, (lab, i) in zip(axes, picks):
        t = np.arange(C.STEPS_PER_DAY) / C.STEPS_PER_HOUR
        m = mask[i]
        axx.fill_between(t[m], (kt_lo[i] * cs[i])[m], (kt_hi[i] * cs[i])[m],
                         alpha=.25, label="80% interval")
        axx.plot(t[m], ghi[i][m], "k-", lw=1.8, label="actual")
        axx.plot(t[m], ghi_hat[s1][i][m], label="Stage 1")
        axx.plot(t[m], ghi_hat["mean3"][i][m], "--", label="3-day mean")
        axx.set_title(f"{lab}: {dates[i]}"); axx.set_xlabel("hour (IST)")
        axx.set_ylabel("GHI (W/m$^2$)"); axx.legend(fontsize=7); axx.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / f"{prefix}fig_example_days.png", dpi=140); plt.close(fig)

    summary = {
        "test_days": int(len(test_idx)),
        "metrics_all": metrics[metrics.season == "all"].to_dict("records"),
        "paired_tests": paired.to_dict("records"),
        "intervals": intervals.to_dict("records"),
        "headline": {
            "stage1_rmse": float(metrics.query(f"method=='{s1}' and season=='all'")["rmse"].iloc[0]),
            "stage2_rmse": float(metrics.query(f"method=='{s2n}' and season=='all'")["rmse"].iloc[0]),
            "mean3_rmse": float(metrics.query("method=='mean3' and season=='all'")["rmse"].iloc[0]),
            "monthly_clim_rmse": float(metrics.query("method=='monthly_climatology' and season=='all'")["rmse"].iloc[0]),
        },
    }
    (out / f"{prefix}eval_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] wrote {prefix}metrics.csv, {prefix}paired_tests.csv, "
          f"{prefix}intervals.csv, {prefix}lead_time.csv, "
          f"{prefix}eval_summary.json and 3 figures to {out}/")


if __name__ == "__main__":
    main()
