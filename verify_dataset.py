"""Re-verify every data claim in PIPELINE_WORKFLOW_v3.md against the merged CSV.

Each check prints VERIFIED / CHANGED / FAILED so that VERIFICATION_STATUS.md can
be updated from real output rather than from memory. Nothing here fits a
forecasting model; it only establishes what the data actually is.

Run:  python pv_pipeline/verify_dataset.py --csv /path/to/merged.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pv_pipeline import common as K  # noqa: E402

RESULTS: list[dict] = []


def check(name: str, status: str, detail: str, **extra) -> None:
    RESULTS.append({"check": name, "status": status, "detail": detail, **extra})
    print(f"  [{status:<8}] {name}: {detail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="merged Solcast CSV (glob allowed)")
    ap.add_argument("--out", default="pv_pipeline/results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    print("\n" + "=" * 72 + "\n1. LOAD AND GRID INTEGRITY\n" + "=" * 72)
    df = K.load_solcast(args.csv)
    audit = K.grid_audit(df)
    for k, v in audit.items():
        if k != "missing_examples":
            print(f"    {k:<22} {v}")

    check(
        "grid_complete",
        "VERIFIED" if audit["missing_timestamps"] == 0 and audit["duplicate_timestamps"] == 0
        else "FAILED",
        f"{audit['rows']:,} rows, {audit['missing_timestamps']} missing, "
        f"{audit['duplicate_timestamps']} duplicate, "
        f"{audit['off_grid_deltas']} off-grid deltas",
        **{k: v for k, v in audit.items() if k != "missing_examples"},
    )

    kept, dropped = K.complete_days(df)
    n_days = kept["date"].nunique()
    check(
        "complete_ist_days", "VERIFIED",
        f"{n_days} complete days {kept['date'].min().date()} -> {kept['date'].max().date()}; "
        f"dropped {len(dropped)} partial ({dropped})",
        n_days=n_days, dropped=dropped,
    )

    # ======================================================================
    print("\n" + "=" * 72 + "\n2. MERGE SEAM AND PER-ERA CONSISTENCY\n" + "=" * 72)
    era = np.where(kept["timestamp"] < pd.Timestamp("2023-12-02", tz=K.TZ), "2020_2023", "2023_2026")
    kept = kept.assign(era=era)
    print(f"    rows per era: {kept['era'].value_counts().to_dict()}")

    # A column present in one era but entirely absent/constant in the other is a
    # merge artifact, not physics.
    suspicious = []
    for col in kept.select_dtypes(include=[np.number]).columns:
        if col in ("step", "year", "month", "doy"):
            continue
        a = kept.loc[kept["era"] == "2020_2023", col]
        b = kept.loc[kept["era"] == "2023_2026", col]
        if a.notna().sum() == 0 or b.notna().sum() == 0:
            suspicious.append({"column": col, "issue": "all-NaN in one era"})
        elif a.nunique() <= 1 and b.nunique() > 1:
            suspicious.append({"column": col, "issue": "constant in 2020-2023 only"})
    check(
        "no_merge_artifacts", "VERIFIED" if not suspicious else "FAILED",
        "no column is empty or constant in only one era" if not suspicious else str(suspicious),
        suspicious=suspicious,
    )

    # Precision/dtype differs across eras in the source exports.
    dtypes = {c: str(kept[c].dtype) for c in ("ghi", "clearsky_ghi", "air_temp", "zenith", "azimuth")}
    int_like = {
        c: bool(np.allclose(kept[c].dropna() % 1, 0))
        for c in ("ghi", "clearsky_ghi", "air_temp", "zenith", "azimuth")
    }
    check(
        "dtypes_after_merge",
        "CHANGED" if not all(int_like.values()) else "VERIFIED",
        f"dtypes={dtypes}; integer-valued={int_like} "
        f"(v3 Bucket 1 claimed int64 for all five on the 2023-2026 file alone)",
        dtypes=dtypes, integer_valued=int_like,
    )

    seam = kept[(kept["timestamp"] >= pd.Timestamp("2023-12-01", tz=K.TZ))
                & (kept["timestamp"] < pd.Timestamp("2023-12-03", tz=K.TZ))]
    check(
        "merge_seam_continuous",
        "VERIFIED" if len(seam) == 2 * K.STEPS_PER_DAY else "FAILED",
        f"{len(seam)} rows across the 2023-12-01/02 seam (expected {2 * K.STEPS_PER_DAY})",
    )

    # ======================================================================
    print("\n" + "=" * 72 + "\n3. SOLAR POSITION AND TIMESTAMP CONVENTION\n" + "=" * 72)
    fit = K.fit_location(kept["mid_utc"], kept["zenith"].to_numpy(dtype=float))
    print(f"    recovered site: lat={fit['lat']:.4f}  lon={fit['lon']:.4f}  "
          f"zenith RMSE={fit['zenith_rmse_deg']:.3f} deg")
    dist_km = np.hypot(fit["lat"] - K.MUMBAI_LAT, fit["lon"] - K.MUMBAI_LON) * 111.0
    check(
        "solar_position_and_site",
        "VERIFIED" if fit["zenith_rmse_deg"] < 1.0 and dist_km < 50 else "FAILED",
        f"least-squares fit lands {dist_km:.1f} km from nominal Mumbai with "
        f"{fit['zenith_rmse_deg']:.3f} deg zenith RMSE at period mid-points",
        **fit, km_from_mumbai=round(float(dist_km), 1),
    )

    sp = K.solar_position(kept["mid_utc"], fit["lat"], fit["lon"])
    kept["zenith_calc"] = sp["zenith"]
    kept["azimuth_calc"] = sp["azimuth"]
    az_ref = K.solcast_azimuth_to_conventional(kept["azimuth"].to_numpy(dtype=float))
    day = kept["zenith_calc"] < 85
    az_err = np.abs(((kept["azimuth_calc"] - az_ref + 180) % 360) - 180)[day]
    check(
        "azimuth_convention", "VERIFIED" if np.sqrt(np.mean(az_err**2)) < 5 else "FAILED",
        f"(-az) %% 360 reproduces Solcast azimuth to {np.sqrt(np.mean(az_err**2)):.2f} deg "
        f"RMSE on daylight steps",
        azimuth_rmse_deg=round(float(np.sqrt(np.mean(az_err**2))), 3),
    )

    # Alternative conventions must fit worse, or our timestamp reading is luck.
    alts = {
        "period_start": kept["timestamp"].dt.tz_convert("UTC"),
        "period_mid (used)": kept["mid_utc"],
        "period_end": kept["period_end_utc"],
    }
    conv = {}
    for label, t in alts.items():
        z = K.solar_position(t, fit["lat"], fit["lon"])["zenith"]
        conv[label] = round(float(np.sqrt(np.mean((z - kept["zenith"]) ** 2))), 3)
    best = min(conv, key=conv.get)
    print(f"    zenith RMSE by assumed stamp: {conv}")
    check(
        "timestamp_convention", "VERIFIED" if "mid" in best else "CHANGED",
        f"best fit is {best} ({conv[best]} deg); v3 assumed mid-point",
        rmse_by_convention=conv,
    )

    # ======================================================================
    print("\n" + "=" * 72 + "\n4. kt STRUCTURE AND THE 1.0 PILE-UP\n" + "=" * 72)
    d = kept[kept["clearsky_ghi"] > K.DAYLIGHT_CS_MIN].copy()
    d["kt_raw"] = d["ghi"] / d["clearsky_ghi"]
    rows = []
    for yr, g in d.groupby("year"):
        rows.append({
            "year": int(yr),
            "daytime_rows": int(len(g)),
            "pct_exactly_1": round(float((g["kt_raw"] == 1.0).mean() * 100), 2),
            "pct_above_1": round(float((g["kt_raw"] > 1.0).mean() * 100), 2),
            "mean_kt": round(float(g["kt_raw"].mean()), 4),
        })
    kt_tbl = pd.DataFrame(rows).set_index("year")
    print(kt_tbl.to_string())
    kt_tbl.to_csv(out / "kt_by_year.csv")

    overall_pile = float((d["kt_raw"] == 1.0).mean() * 100)
    check(
        "kt_pileup_at_1",
        "VERIFIED" if abs(overall_pile - 53.8) < 15 else "CHANGED",
        f"{overall_pile:.1f}% of daytime rows are exactly 1.000, "
        f"{(d['kt_raw'] > 1.0).mean() * 100:.2f}% exceed it "
        f"(v3 measured 53.8% / 0.00% on 2023-2026 only)",
        pct_exactly_1=round(overall_pile, 2),
        pct_above_1=round(float((d["kt_raw"] > 1.0).mean() * 100), 4),
    )

    spread = kt_tbl["mean_kt"].max() - kt_tbl["mean_kt"].min()
    clear_days = d.groupby("date")["cloud_opacity"].mean()
    clear_ids = clear_days[clear_days < 5].index
    clear_kt = d[d["date"].isin(clear_ids)].groupby("year")["kt_raw"].mean().round(4)
    print(f"    annual mean kt spread: {spread:.4f}")
    print(f"    clear-sky-day mean kt by year: {clear_kt.to_dict()}")
    check(
        "annual_kt_homogeneity",
        "VERIFIED" if (clear_kt.max() - clear_kt.min()) < 0.05 else "CHANGED",
        f"all-day kt spread {spread:.4f} across years, but clear-sky-day kt spread "
        f"{clear_kt.max() - clear_kt.min():.4f} - v3 warned the 0.03 all-day "
        f"threshold confuses monsoon variability with bias",
        annual_mean_kt=kt_tbl["mean_kt"].to_dict(),
        clear_day_mean_kt=clear_kt.to_dict(),
        n_clear_days=int(len(clear_ids)),
    )

    # ======================================================================
    print("\n" + "=" * 72 + "\n5. REDUNDANT COLUMNS (must not be model inputs)\n" + "=" * 72)
    corr = float(np.corrcoef(d["cloud_opacity"], d["kt_raw"])[0, 1])
    recon = (
        d["dni"] * np.cos(np.deg2rad(d["zenith_calc"])) + d["dhi"] - d["ghi"]
    ).abs()
    check(
        "cloud_opacity_not_independent", "VERIFIED" if corr < -0.9 else "CHANGED",
        f"corr(cloud_opacity, kt) = {corr:.3f}", corr_cloud_kt=round(corr, 4),
    )
    check(
        "ghi_reconstructible_from_dni_dhi",
        "VERIFIED" if recon.median() < 5 else "CHANGED",
        f"|DNI*cos(z)+DHI-GHI| median {recon.median():.2f} W/m2, "
        f"p95 {recon.quantile(0.95):.2f} - dni/dhi/cloud_opacity stay out of the inputs",
        median_abs_resid=round(float(recon.median()), 3),
    )

    # ======================================================================
    print("\n" + "=" * 72 + "\n6. CLEAR-SKY PROXY SUPPORT AND SPLIT\n" + "=" * 72)
    arr = K.to_day_arrays(kept)
    lagged = K.clearsky_lagged_median(arr.cs_solcast, window=7)
    clim, support = K.clearsky_climatology(arr.cs_solcast, arr.doy, arr.year)
    clim_ratio = K.clearsky_clim_times_ratio(clim, arr.cs_solcast)

    print(f"    day-of-year climatology support: median {int(np.median(support))} other years "
          f"(min {support.min()}, max {support.max()})")
    check(
        "climatology_support",
        "CHANGED" if np.median(support) >= 4 else "VERIFIED",
        f"median {int(np.median(support))} other years per day vs ~1 on the "
        f"2023-2026 file; v3 §3 said mode B was handicapped and should improve",
        median_other_years=int(np.median(support)),
    )

    usable = np.arange(len(arr)) >= 7
    n_usable = int(usable.sum())
    test_n = 365
    test_start_i = n_usable - test_n
    purge = 1
    idx = np.flatnonzero(usable)
    test_i = idx[test_start_i:]
    dev_i = idx[: test_start_i - purge]
    print(f"    usable days {n_usable}; dev {len(dev_i)} "
          f"({arr.dates[dev_i[0]].date()} -> {arr.dates[dev_i[-1]].date()}); "
          f"test {len(test_i)} ({arr.dates[test_i[0]].date()} -> {arr.dates[test_i[-1]].date()})")

    val_i = dev_i[int(len(dev_i) * 0.75):]
    val_monsoon = int(arr.is_monsoon[val_i].sum())
    check(
        "validation_monsoon_coverage",
        "CHANGED" if val_monsoon >= 30 else "VERIFIED",
        f"last 25% of development days = {len(val_i)} days with {val_monsoon} monsoon days "
        f"(v3 §12 risk: on the single file the monsoon delta_max fell back to pooled)",
        validation_days=len(val_i), validation_monsoon_days=val_monsoon,
    )

    # ======================================================================
    print("\n" + "=" * 72 + "\n7. LEAK-SAFE BASELINES ON THE TEST YEAR\n" + "=" * 72)
    results = {}
    for mode, cs_proxy in (("lagged", lagged), ("clim", clim), ("clim_ratio", clim_ratio),
                           ("solcast_LEAKY", arr.cs_solcast)):
        kt_obs = K.kt_from(arr.ghi, arr.cs_solcast)
        mask = K.daylight_mask(arr.zenith, np.nan_to_num(cs_proxy))

        prev_kt = np.vstack([np.full((1, K.STEPS_PER_DAY), np.nan), kt_obs[:-1]])
        mean3 = np.full_like(kt_obs, np.nan)
        for i in range(3, len(kt_obs)):
            mean3[i] = np.nanmean(kt_obs[i - 3 : i], axis=0)

        row = {}
        for label, kt_hat in (("prev_day_kt", prev_kt), ("mean3_kt", mean3)):
            ghi_hat = np.nan_to_num(kt_hat) * np.nan_to_num(cs_proxy)
            for season, sel in (("all", np.ones(len(arr), bool)),
                                ("monsoon", arr.is_monsoon),
                                ("non_monsoon", ~arr.is_monsoon)):
                keep = np.zeros(len(arr), bool)
                keep[test_i] = True
                keep &= sel
                m = point = K.point_metrics(arr.ghi[keep], ghi_hat[keep], mask[keep])
                row[f"{label}_{season}_rmse"] = round(point["rmse"], 1)
        results[mode] = row
        print(f"    cs_mode={mode:<14} "
              f"prev-day RMSE all/mon/non = {row['prev_day_kt_all_rmse']:.1f}/"
              f"{row['prev_day_kt_monsoon_rmse']:.1f}/{row['prev_day_kt_non_monsoon_rmse']:.1f}   "
              f"3-day-mean = {row['mean3_kt_all_rmse']:.1f}/"
              f"{row['mean3_kt_monsoon_rmse']:.1f}/{row['mean3_kt_non_monsoon_rmse']:.1f}")

    pd.DataFrame(results).T.to_csv(out / "baselines_by_cs_mode.csv")
    leaky_gain = (results["lagged"]["mean3_kt_all_rmse"]
                  - results["solcast_LEAKY"]["mean3_kt_all_rmse"])
    check(
        "solcast_cs_is_leaky_advantage", "VERIFIED" if leaky_gain > 0 else "CHANGED",
        f"using Solcast same-day clear-sky lowers 3-day-baseline RMSE by "
        f"{leaky_gain:.1f} W/m2 - it must never be an input",
        rmse_advantage_wm2=round(float(leaky_gain), 2),
    )

    best_dev_mode = min(("lagged", "clim", "clim_ratio"),
                        key=lambda m: results[m]["mean3_kt_all_rmse"])
    check(
        "cs_mode_choice", "VERIFIED",
        f"best non-leaky proxy on this comparison is '{best_dev_mode}' "
        f"(select it on development data in 00_prepare.py, never on the test year)",
        ranking={m: results[m]["mean3_kt_all_rmse"] for m in ("lagged", "clim", "clim_ratio")},
    )

    # ======================================================================
    print("\n" + "=" * 72 + "\nSUMMARY\n" + "=" * 72)
    tally: dict[str, int] = {}
    for r in RESULTS:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    print(f"    {tally}")
    for r in RESULTS:
        if r["status"] != "VERIFIED":
            print(f"    {r['status']}: {r['check']}")

    payload = {
        "csv": args.csv,
        "generated": str(pd.Timestamp.now(tz=K.TZ)),
        "tally": tally,
        "checks": RESULTS,
        "split": {
            "usable_days": n_usable,
            "dev_days": len(dev_i),
            "dev_start": str(arr.dates[dev_i[0]].date()),
            "dev_end": str(arr.dates[dev_i[-1]].date()),
            "purge_days": purge,
            "test_days": len(test_i),
            "test_start": str(arr.dates[test_i[0]].date()),
            "test_end": str(arr.dates[test_i[-1]].date()),
        },
        "baselines_by_cs_mode": results,
        "recovered_site": fit,
    }
    with open(out / "verification.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n    saved {out / 'verification.json'}")
    print(f"    saved {out / 'kt_by_year.csv'}, {out / 'baselines_by_cs_mode.csv'}")


if __name__ == "__main__":
    main()
