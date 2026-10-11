"""Regime-wise GHI skill and optional DTW clusters on forecast-error profiles.

Clear / cloudy / high intra-day variability / monsoon, plus whether residual
correction helped. Sky type uses D−1 clearness (leak-safe).

  python analysis_regimes.py --base results/dwt_base_pred.npz --stage2 results/dwt_stage2.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
from dwt_utils import daily_clearness, sky_type_from_k


def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    a = np.nan_to_num(a, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)
    n, m = len(a), len(b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def _kmeans_dtw(rows: np.ndarray, k: int, rng: np.random.Generator, iters: int = 8):
    n = len(rows)
    if n < k:
        return np.zeros(n, dtype=int), []
    cents = rows[rng.choice(n, size=k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        for i, row in enumerate(rows):
            labels[i] = int(np.argmin([_dtw(row, c) for c in cents]))
        for j in range(k):
            mem = rows[labels == j]
            if len(mem):
                cents[j] = mem.mean(axis=0)
    return labels, cents


def _rmse(ghi, hat, mask, days):
    return C.point_metrics(ghi[days], hat[days], mask[days])["rmse"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=str(C.PREP_PATH))
    ap.add_argument("--base", default=str(C.RESULTS_DIR / "dwt_base_pred.npz"))
    ap.add_argument("--stage2", default="")
    ap.add_argument("--gbm", default=str(C.RESULTS_DIR / "base_pred.npz"))
    ap.add_argument("--out", default=str(C.RESULTS_DIR))
    ap.add_argument("--clusters", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.prep, allow_pickle=True)
    kt, mask, ghi, cs = d["kt"], d["mask"], d["ghi"], d["cs_proxy"]
    test_idx = d["test_idx"]
    monsoon = d["is_monsoon"].astype(bool)
    dates = d["dates"]

    kt_base = np.load(args.base, allow_pickle=True)["kt_base"]
    mean3 = C.baseline_mean3_kt(kt)
    methods = {
        "mean3": np.clip(mean3, *C.KT_CLIP) * cs,
        "dwt_primary": np.clip(kt_base, *C.KT_CLIP) * cs,
    }
    if Path(args.gbm).is_file():
        methods["gbm"] = np.clip(np.load(args.gbm, allow_pickle=True)["kt_base"],
                                 *C.KT_CLIP) * cs
    stage2_path = args.stage2 or str(Path(args.base).with_name(
        Path(args.base).name.replace("_base_pred.npz", "_stage2.npz").replace(
            "base_pred.npz", "stage2.npz")))
    # prefer explicit --stage2; also try dwt_stage2.npz next to prefix files
    candidates = [args.stage2] if args.stage2 else []
    stem = Path(args.base).name
    if stem.endswith("_base_pred.npz"):
        candidates.append(str(Path(args.base).with_name(stem.replace("_base_pred.npz", "_stage2.npz"))))
    candidates.append(str(C.RESULTS_DIR / "dwt_stage2.npz"))
    for p in candidates:
        if p and Path(p).is_file():
            z = np.load(p, allow_pickle=True)
            key = "kt_final" if "kt_final" in z.files else "kt_base"
            methods["dwt_corrected"] = np.clip(z[key], *C.KT_CLIP) * cs
            break

    # D−1 sky type and intra-day variability on the test year
    types_d1 = np.full(len(test_idx), -1, dtype=int)
    high_var = np.zeros(len(test_idx), dtype=bool)
    sigmas = []
    for j, i in enumerate(test_idx):
        types_d1[j] = sky_type_from_k(daily_clearness(ghi[i - 1], cs[i - 1], mask[i - 1])) if i > 0 else 0
        m = mask[i]
        sigmas.append(float(np.nanstd(ghi[i][m])) if m.any() else 0.0)
    sigmas = np.asarray(sigmas)
    high_var = sigmas >= np.nanmedian(sigmas)

    slices = {
        "all": np.ones(len(test_idx), dtype=bool),
        "monsoon": monsoon[test_idx],
        "non_monsoon": ~monsoon[test_idx],
        "clear_d1": types_d1 == 2,
        "cloudy_d1": types_d1 == 1,
        "overcast_d1": types_d1 == 0,
        "high_variability": high_var,
        "low_variability": ~high_var,
    }

    rows = []
    for sname, sel in slices.items():
        days = test_idx[sel]
        if len(days) == 0:
            continue
        rec = {"regime": sname, "n_days": int(len(days))}
        for mname, hat in methods.items():
            rec[f"rmse_{mname}"] = _rmse(ghi, hat, mask, days)
        if "dwt_primary" in methods and "dwt_corrected" in methods:
            rec["residual_delta_wm2"] = rec["rmse_dwt_corrected"] - rec["rmse_dwt_primary"]
            rec["residual_helped"] = bool(rec["residual_delta_wm2"] < -0.05)
        rows.append(rec)
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    table.to_csv(out / "regime_metrics.csv", index=False)

    # soft-style DTW k-means on primary error profiles (daylight-masked)
    err = (methods["dwt_primary"] - ghi)[test_idx]
    err = np.where(mask[test_idx], err, 0.0)
    labels, _ = _kmeans_dtw(err, k=args.clusters, rng=np.random.default_rng(0))
    crows = []
    for c in range(args.clusters):
        sel = labels == c
        if not sel.any():
            continue
        days = test_idx[sel]
        crows.append({
            "cluster": c, "n_days": int(sel.sum()),
            "rmse_dwt": _rmse(ghi, methods["dwt_primary"], mask, days),
            "rmse_mean3": _rmse(ghi, methods["mean3"], mask, days),
            "frac_monsoon": float(monsoon[days].mean()),
            "mean_abs_err": float(np.mean(np.abs(err[sel]))),
        })
    clus = pd.DataFrame(crows)
    print("\n=== DTW clusters on DWT-BiLSTM error profiles ===")
    print(clus.round(3).to_string(index=False))
    clus.to_csv(out / "regime_dtw_clusters.csv", index=False)
    np.savez_compressed(out / "regime_dtw_labels.npz", labels=labels, dates=dates[test_idx])

    summary = {
        "regimes": rows,
        "clusters": crows,
        "note": "sky type is D-1 K (leak-safe); residual_helped is RMSE drop > 0.05 W/m2",
    }
    (out / "regime_analysis.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] wrote {out/'regime_metrics.csv'} {out/'regime_dtw_clusters.csv'}")


if __name__ == "__main__":
    main()
