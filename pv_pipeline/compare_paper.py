"""Report hourly metrics for a saved 10-minute GHI forecast beside Bektaş & Altaş.

The literature values use a different site and dataset. This script places metrics
side by side for context; it does not claim an exact reproduction or fair ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import common as C


PAPER_URL = "https://link.springer.com/article/10.1007/s00521-024-09701-2"
PAPER_ROWS = [
    ("Bektaş & Altaş (2024), Model 1", 53.20, 72.83, 27.67, 53.20),
    ("Bektaş & Altaş (2024), Model 2", 50.63, 68.89, 28.46, 48.31),
]


def hourly_metrics(actual_10min: np.ndarray, predicted_10min: np.ndarray,
                   daylight_10min: np.ndarray) -> dict:
    """Compare full-hour mean GHI; daylight means any scored 10-minute slot."""
    if (actual_10min.shape != predicted_10min.shape
            or actual_10min.shape != daylight_10min.shape
            or actual_10min.ndim != 2 or actual_10min.shape[1] != C.STEPS_PER_DAY):
        raise ValueError("Expected equal day-by-144 arrays")
    n = len(actual_10min)
    actual_hourly = actual_10min.reshape(n, 24, 6).mean(axis=2)
    predicted_hourly = predicted_10min.reshape(n, 24, 6).mean(axis=2)
    daylight_hourly = daylight_10min.reshape(n, 24, 6).any(axis=2)
    all_hours = C.point_metrics(actual_hourly, predicted_hourly)
    daylight = C.point_metrics(actual_hourly, predicted_hourly, daylight_hourly)
    return {
        "all24_rmse_wm2": all_hours["rmse"],
        "daylight_rmse_wm2": daylight["rmse"],
        "all24_mae_wm2": all_hours["mae"],
        "daylight_mae_wm2": daylight["mae"],
        "all24_hours": all_hours["n"],
        "daylight_hours": daylight["n"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep", default=str(C.PREP_PATH))
    ap.add_argument("--base", required=True, help="Complete 10-minute base prediction NPZ")
    ap.add_argument("--stage2", help="Optional corrected prediction NPZ; night slots use base forecast")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()

    with np.load(args.prep, allow_pickle=False) as prep, np.load(args.base, allow_pickle=False) as base:
        dates = prep["dates"]
        if not np.array_equal(dates, base["dates"]):
            raise ValueError("Prediction and prepared-data dates differ")
        config = json.loads(str(base["config"]))
        if config.get("resolution") != "10min" or config.get("smoke") or config.get("leaky"):
            raise ValueError("Expected a non-leaky, full 10-minute run")
        test = prep["test_idx"]
        kt_pred = base["kt_base"]
        if kt_pred.shape != prep["ghi"].shape or not np.isfinite(kt_pred[test]).all():
            raise ValueError("Missing or misaligned held-out test predictions")
        actual = prep["ghi"][test]
        cs = prep["cs_proxy"][test]
        mask = prep["mask"][test]
        model_ghi = np.clip(kt_pred[test], *C.KT_CLIP) * cs
        baseline_ghi = np.clip(C.baseline_mean3_kt(prep["kt"])[test], *C.KT_CLIP) * cs
        rows = []
        forecasts = [("Mumbai DWT-BiLSTM 10min→hourly", model_ghi)]
        if args.stage2:
            with np.load(args.stage2, allow_pickle=False) as corrected:
                if not np.array_equal(dates, corrected["dates"]):
                    raise ValueError("Corrected prediction dates differ")
                kt_final = corrected["kt_final"]
                if kt_final.shape != kt_pred.shape or not np.isfinite(kt_final[test][mask]).all():
                    raise ValueError("Missing or misaligned corrected daylight predictions")
                # The corrector intentionally leaves night slots NaN. Retain the
                # original base forecast there for all-24-hour aggregation.
                merged = np.where(np.isfinite(kt_final[test]), kt_final[test], kt_pred[test])
                corrected_ghi = np.clip(merged, *C.KT_CLIP) * cs
                forecasts.append(("Mumbai corrected 10min→hourly", corrected_ghi))
        forecasts.append(("Mumbai prior-3-day baseline", baseline_ghi))
        for name, predicted in forecasts:
            rows.append({"method": name, "site": "Mumbai", "source": "this project",
                         "n_test_days": len(test), **hourly_metrics(actual, predicted, mask)})

        for name, all_rmse, day_rmse, all_mae, day_mae in PAPER_ROWS:
            rows.append({"method": name, "site": "Trabzon", "source": PAPER_URL,
                         "n_test_days": None, "all24_rmse_wm2": all_rmse,
                         "daylight_rmse_wm2": day_rmse, "all24_mae_wm2": all_mae,
                         "daylight_mae_wm2": day_mae,
                         "all24_hours": None, "daylight_hours": None})

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "paper_comparison.csv", index=False)
        report = {
            "prediction_file": str(Path(args.base).resolve()),
            "corrected_prediction_file": str(Path(args.stage2).resolve()) if args.stage2 else None,
            "test_period": [str(dates[test[0]]), str(dates[test[-1]])],
            "native_resolution_minutes": 10,
            "hourly_aggregation": "mean of all six 10-minute GHI values in each hour",
            "daylight_hour_rule": "any of the six 10-minute slots passes the local zenith/clear-sky mask",
            "paper_url": PAPER_URL,
            "paper_reference": "Bektaş & Altaş (2024), Table 4, Trabzon BILSTM",
            "comparability": "Different sites, datasets, forecast inputs, native resolutions, and possibly daylight definitions; descriptive comparison only.",
            "model_native_test_daylight_rmse_wm2": C.point_metrics(actual, model_ghi, mask)["rmse"],
            "corrected_native_test_daylight_rmse_wm2": (
                C.point_metrics(actual, corrected_ghi, mask)["rmse"] if args.stage2 else None),
            "rows": rows,
        }
        (out / "paper_comparison.json").write_text(json.dumps(report, indent=2))
        print(pd.DataFrame(rows)[["method", "all24_rmse_wm2", "daylight_rmse_wm2"]].to_string(index=False))
        print(f"[done] {out / 'paper_comparison.csv'}")


if __name__ == "__main__":
    main()
