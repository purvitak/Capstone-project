"""Matched reproduction of Bektaş & Altaş (2024) on the Mumbai data.

.venv/bin/python pipelines/paper_reproduction/run.py

Trains the paper's Model 1 and Model 2 on development days with the published
(observed) forecast-day inputs, selects the cell count on a chronological
validation split, then scores the held-out year under four forecast-day input
variants beside the current default and D−1 persistence. Everything is scored on
local hourly means, as in the paper. The test year has been seen by earlier
project iterations, so results are exploratory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "pv_pipeline"))
sys.path.insert(0, str(HERE))
import common as C
import nwp_data as N
import paper_model as P

DEFAULT_OUT = ROOT / "results" / "paper_reproduction"
DEFAULT_PRED = ROOT / "results" / "xgboost_nwp" / "xgboost_base_pred.npz"
VAL_FRACTION = 0.2
TRABZON_MEAN_GHI = 134.0  # paper Table 2, all hours
PUBLISHED = {  # paper Table 4, Trabzon BiLSTM (W/m²)
    1: {"all24_rmse": 53.20, "all24_mae": 27.67, "daylight_rmse": 72.83, "daylight_mae": 53.20},
    2: {"all24_rmse": 50.63, "all24_mae": 28.46, "daylight_rmse": 68.89, "daylight_mae": 48.31},
}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hourly(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, float).reshape(len(a), 24, 6).mean(axis=2)


# ==========================================================================
# Data
# ==========================================================================

def load(prep_path: Path, cache: Path) -> tuple[dict, dict]:
    keys = ("dates", "ghi", "cs_proxy", "mask", "air_temp", "rh", "wind_speed",
            "doy", "is_monsoon", "dev_idx", "test_idx")
    with np.load(prep_path, allow_pickle=False) as source:
        data = {k: source[k] for k in keys}
    nwp, _ = N.load_forecasts(cache, prep_path)
    if not np.array_equal(nwp["dates"], data["dates"]):
        raise ValueError("Forecast cache dates differ from prepared data")
    return data, nwp


def build_sources(data: dict, nwp: dict) -> tuple[dict, dict]:
    """Hourly observed and GFS-derived inputs; GFS calibration uses development days only."""
    ghi_h = hourly(data["ghi"])
    met_obs = np.stack([hourly(data["air_temp"]), hourly(data["rh"]),
                        hourly(data["wind_speed"])], axis=-1)
    k_obs = P.daily_clearness(ghi_h, data["doy"], C.MUMBAI_LAT)

    features = nwp["features"].astype(float)
    available = nwp["available"]
    gfs_ghi = hourly(features[..., 0])
    # NWP wind is km/h; observed wind is m/s.
    met_gfs = np.stack([hourly(features[..., 2]), hourly(features[..., 3]),
                        hourly(features[..., 4]) / 3.6], axis=-1)
    gfs_ghi[~available] = np.nan
    met_gfs[~available] = np.nan
    k_gfs = P.daily_clearness(gfs_ghi, data["doy"], C.MUMBAI_LAT)

    dev = data["dev_idx"]
    cal = dev[available[dev] & np.isfinite(k_obs[dev]) & np.isfinite(met_obs[dev]).all(axis=(1, 2))]
    k_gfs_cal = P.quantile_map(k_gfs[cal], k_obs[cal], k_gfs)
    met_gfs_cal = np.stack([P.quantile_map(met_gfs[cal][..., v].ravel(), met_obs[cal][..., v].ravel(),
                                           met_gfs[..., v]) for v in range(3)], axis=-1)

    first = P.WINDOW_DAYS + 1  # day 0 is missing, so every window must start after it
    filled = P.forward_fill_days(ghi_h)
    sources = {
        "ghi_h": ghi_h, "dwt": P.dwt_by_day(filled, range(first, len(ghi_h))),
        "type_obs": P.weather_types(k_obs), "met_obs": met_obs, "k_obs": k_obs,
        "type_gfs_raw": P.weather_types(k_gfs), "met_gfs_raw": met_gfs, "k_gfs_raw": k_gfs,
        "type_gfs_calibrated": P.weather_types(k_gfs_cal), "met_gfs_calibrated": met_gfs_cal,
        "k_gfs_calibrated": k_gfs_cal, "first_day": first,
    }
    calibration = {"days": int(len(cal)), "first": str(data["dates"][cal[0]]),
                   "last": str(data["dates"][cal[-1]]),
                   "development_k_bias_raw": float(np.mean(k_gfs[cal] - k_obs[cal])),
                   "development_k_bias_calibrated": float(np.mean(k_gfs_cal[cal] - k_obs[cal]))}
    return sources, calibration


def training_days(data: dict, sources: dict) -> np.ndarray:
    dev = data["dev_idx"]
    ok = (dev >= sources["first_day"]) & np.isfinite(sources["ghi_h"][dev]).all(axis=1)
    for key in ("type_obs",):
        ok &= np.isfinite(sources[key][dev]) & np.isfinite(sources[key][dev - 1])
    for key in ("met_obs",):
        ok &= np.isfinite(sources[key][dev]).all(axis=(1, 2)) & np.isfinite(sources[key][dev - 1]).all(axis=(1, 2))
    return dev[ok]


def split(days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chronological 80/20 sub-train/validation with a one-day purge (paper Section 3.1)."""
    cut = int(len(days) * (1 - VAL_FRACTION))
    return days[:cut - 1], days[cut:]


def normalizer(sources: dict, fit: np.ndarray) -> dict:
    met = np.concatenate([sources["met_obs"][fit], sources["met_obs"][fit - 1]]).reshape(-1, 3)
    lo, hi = met.min(axis=0), met.max(axis=0)
    return {"ghi_scale": float(np.max(sources["ghi_h"][fit])),
            "met_min": lo, "met_span": np.where(hi - lo > 0, hi - lo, 1.0)}


# ==========================================================================
# Training with resumable checkpoints
# ==========================================================================

def fit_model(model: int, sources: dict, fit, val, norm, args, contract: dict):
    import torch

    x_fit = P.build_inputs(sources, fit, model, "oracle", norm)
    x_val = P.build_inputs(sources, val, model, "oracle", norm)
    y_fit = (sources["ghi_h"][fit] / norm["ghi_scale"]).astype(np.float32)
    y_val = (sources["ghi_h"][val] / norm["ghi_scale"]).astype(np.float32)
    folder = args.out / "models"
    folder.mkdir(parents=True, exist_ok=True)
    candidates = []
    for hidden in args.cells:
        stem = folder / f"model{model}_cells{hidden}"
        config = {**contract, "model": model, "hidden": hidden}
        meta = stem.with_suffix(".json")
        if meta.exists() and json.loads(meta.read_text()).get("config") == config:
            info = json.loads(meta.read_text())["training"]
            net = P.make_network(x_fit.shape[-1], hidden)
            net.load_state_dict(torch.load(stem.with_suffix(".pt"), weights_only=True))
            print(f"[model {model}] cells {hidden}: reused checkpoint "
                  f"(validation MSE {info['best_validation_mse']:.5f})", flush=True)
        else:
            started = time.time()
            net, info = P.train_network(x_fit, y_fit, x_val, y_val, hidden, args.seed,
                                        max_epochs=args.max_epochs, patience=args.patience)
            info["seconds"] = time.time() - started
            torch.save(net.state_dict(), stem.with_suffix(".pt"))
            meta.write_text(json.dumps({"config": config, "training": info}, indent=2))
            print(f"[model {model}] cells {hidden}: best epoch {info['best_epoch']}/"
                  f"{info['epochs_run']}, validation MSE {info['best_validation_mse']:.5f} "
                  f"({info['seconds']:.0f}s)", flush=True)
        candidates.append((info["best_validation_mse"], hidden, net, info))
    best = min(candidates, key=lambda c: c[0])
    pred_val = P.predict(best[2], x_val) * norm["ghi_scale"]
    summary = {"selected_cells": best[1],
               "candidates": [{k: v for k, v in c[3].items() if k != "validation_mse_history"}
                              for c in candidates],
               "validation_hourly_all24": C.point_metrics(sources["ghi_h"][val], pred_val)}
    return best[2], summary


# ==========================================================================
# Scoring
# ==========================================================================

def scopes(daylight: np.ndarray, monsoon: np.ndarray) -> dict:
    every = np.ones_like(daylight, dtype=bool)
    return {"all24": every, "daylight": daylight,
            "daylight_monsoon": daylight & monsoon[:, None],
            "daylight_non_monsoon": daylight & ~monsoon[:, None]}


def paired_bootstrap(actual, pred, reference, mask, *, repeats=2000, seed=42) -> dict:
    """Pred minus reference; paired circular seven-day blocks recompute pooled errors."""
    counts = mask.sum(axis=1)
    stats = []
    for values in (pred, reference):
        err = np.where(mask, values - actual, 0.0)
        if not np.isfinite(err).all():
            raise ValueError("Paired comparison requires finite predictions")
        stats.append((np.sum(err ** 2, axis=1), np.sum(np.abs(err), axis=1)))

    def difference(idx):
        n = counts[idx].sum(axis=-1)
        rmse = [np.sqrt(sq[idx].sum(axis=-1) / n) for sq, _ in stats]
        mae = [ab[idx].sum(axis=-1) / n for _, ab in stats]
        return rmse[0] - rmse[1], mae[0] - mae[1]

    days = len(actual)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, days, (repeats, int(np.ceil(days / 7))))
    idx = ((starts[..., None] + np.arange(7)) % days).reshape(repeats, -1)[:, :days]
    samples = difference(idx)
    observed = difference(np.arange(days))
    return {name: {"difference": float(point), "ci95_lo": float(np.quantile(values, 0.025)),
                   "ci95_hi": float(np.quantile(values, 0.975))}
            for name, point, values in zip(("rmse", "mae"), observed, samples)}


def default_forecast(path: Path, data: dict, test: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved["dates"], data["dates"]):
            raise ValueError("Default prediction dates differ from prepared data")
        config = json.loads(str(saved["config"]))
        if config.get("leaky") or config.get("smoke") or config.get("resolution") != "10min":
            raise ValueError("Expected the non-leaky full 10-minute default forecast")
        kt = saved["kt_base"][test]
    return hourly(np.clip(kt, *C.KT_CLIP) * data["cs_proxy"][test])


def type_agreement(observed: np.ndarray, assumed: np.ndarray) -> dict:
    confusion = np.zeros((3, 3), dtype=int)
    for o, a in zip(observed.astype(int), assumed.astype(int)):
        confusion[o, a] += 1
    return {"accuracy": float(np.trace(confusion) / confusion.sum()),
            "confusion_rows_observed_cols_assumed": confusion.tolist(),
            "labels": ["rainy", "cloudy", "sunny"]}


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", type=Path, default=C.PREP_PATH)
    ap.add_argument("--nwp-cache", type=Path, default=N.DEFAULT_CACHE)
    ap.add_argument("--default-pred", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--models", type=int, nargs="+", default=[1, 2], choices=(1, 2))
    ap.add_argument("--cells", type=int, nargs="+", default=[30, 50, 100])
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="Few epochs; use a separate --out")
    args = ap.parse_args()
    if args.smoke:
        args.max_epochs, args.patience = min(args.max_epochs, 3), 2
        if args.out == DEFAULT_OUT:
            ap.error("--smoke requires a separate --out directory")

    import torch
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    data, nwp = load(args.prep, args.nwp_cache)
    sources, calibration = build_sources(data, nwp)
    days = training_days(data, sources)
    fit, val = split(days)
    norm = normalizer(sources, fit)
    test = data["test_idx"]
    dates = data["dates"]
    print(f"[data] train {dates[fit[0]]}..{dates[fit[-1]]} ({len(fit)}d), validation "
          f"{dates[val[0]]}..{dates[val[-1]]} ({len(val)}d), test {dates[test[0]]}..{dates[test[-1]]} "
          f"({len(test)}d)", flush=True)

    contract = {"version": 1, "prep_sha256": sha256(args.prep), "wavelet": P.WAVELET,
                "levels": P.LEVELS, "window_days": P.WINDOW_DAYS, "fit": [int(fit[0]), int(fit[-1]), len(fit)],
                "validation": [int(val[0]), int(val[-1]), len(val)], "max_epochs": args.max_epochs,
                "patience": args.patience, "batch_size": 32, "learning_rate": 1e-3,
                "seed": args.seed, "smoke": args.smoke}

    actual = sources["ghi_h"][test]
    daylight = data["mask"][test].reshape(len(test), 24, 6).any(axis=2)
    masks = scopes(daylight, data["is_monsoon"][test].astype(bool))
    if not np.isfinite(actual).all():
        raise ValueError("Test-year observations are incomplete")

    forecasts = {"default_xgboost_blend": default_forecast(args.default_pred, data, test),
                 "persistence_d1": sources["ghi_h"][test - 1]}
    training = {}
    for model in args.models:
        net, training[model] = fit_model(model, sources, fit, val, norm, args, contract)
        for variant in P.VARIANTS:
            x = P.build_inputs(sources, test, model, variant, norm)
            forecasts[f"paper_model{model}_{variant}"] = P.predict(net, x) * norm["ghi_scale"]

    rows = []
    for name, pred in forecasts.items():
        for scope, mask in masks.items():
            m = C.point_metrics(actual, pred, mask)
            rows.append({"method": name, "scope": scope, **m,
                         "nmae": m["mae"] / float(np.mean(actual[mask])) * 100})
    metrics = pd.DataFrame(rows)
    comparisons = {f"default_minus_{name}": {scope: paired_bootstrap(actual, forecasts["default_xgboost_blend"],
                                                                    pred, masks[scope])
                                             for scope in ("all24", "daylight")}
                   for name, pred in forecasts.items() if name.startswith("paper_")}
    agreement = {variant: type_agreement(sources["type_obs"][test], sources[f"type_{key}"][test])
                 for variant, key in (("gfs_raw", "gfs_raw"), ("gfs_calibrated", "gfs_calibrated"))}
    agreement["persistence"] = type_agreement(sources["type_obs"][test], sources["type_obs"][test - 1])

    metrics.to_csv(args.out / "metrics.csv", index=False)
    np.savez_compressed(args.out / "predictions.npz", dates=dates[test], actual_hourly=actual,
                        daylight_hourly=daylight, **forecasts)
    report = {
        "paper": "Bektaş & Altaş (2024), Neural Computing and Applications 36:13243-13253, "
                 "DOI 10.1007/s00521-024-09701-2",
        "contract": contract, "variants": P.VARIANT_NOTES, "gfs_calibration": calibration,
        "normalization": {"ghi_scale_wm2": norm["ghi_scale"], "met_min": norm["met_min"].tolist(),
                          "met_span": norm["met_span"].tolist()},
        "training": training, "test_weather_type_agreement": agreement,
        "test_observed_type_counts": np.bincount(sources["type_obs"][test].astype(int), minlength=3).tolist(),
        "paired_bootstrap_default_minus_paper": comparisons,
        "published_trabzon": PUBLISHED, "trabzon_mean_ghi_all_hours": TRABZON_MEAN_GHI,
        "mumbai_test_mean_ghi_all_hours": float(actual.mean()),
        "seconds": time.time() - started,
        "limitations": [
            "The test year influenced earlier project iterations; this comparison is exploratory.",
            "Reference GHI is Solcast satellite/model-derived, not a ground pyranometer.",
            "Wavelet components use a trailing 168-day window ending at D−1, not the full series.",
            "GFS forecasts begin in January 2024, so the network is trained with observed forecast-day "
            "inputs (as published) and GFS-derived inputs are substituted only at prediction time.",
            "Published Trabzon numbers come from a different site, data source and split; they are context only.",
        ],
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    write_markdown(args.out, metrics, report, args.models)
    print(metrics[metrics.scope.isin(["all24", "daylight"])]
          .pivot(index="method", columns="scope", values=["rmse", "mae"]).round(2).to_string())
    print(f"[done] {args.out} ({time.time() - started:.0f}s)")


def write_markdown(out: Path, metrics: pd.DataFrame, report: dict, models) -> None:
    def cell(method, scope, key):
        row = metrics[(metrics.method == method) & (metrics.scope == scope)]
        return f"{row[key].iloc[0]:.2f}"

    names = ["default_xgboost_blend"] + [f"paper_model{m}_{v}" for m in models for v in P.VARIANTS] + ["persistence_d1"]
    lines = ["# Bektaş & Altaş (2024) reproduction on Mumbai data", "",
             "Local hourly means over the held-out year. All methods share the same days, "
             f"observations and scoring masks. Networks used {report['contract']['fit'][2]} training "
             f"and {report['contract']['validation'][2]} validation days.", "",
             "| Method | All-24 RMSE | All-24 MAE | All-24 nRMSE % | Daylight RMSE | Daylight MAE | Monsoon daylight RMSE | Non-monsoon daylight RMSE |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in names:
        lines.append(f"| {name} | {cell(name, 'all24', 'rmse')} | {cell(name, 'all24', 'mae')} | "
                     f"{cell(name, 'all24', 'nrmse')} | {cell(name, 'daylight', 'rmse')} | "
                     f"{cell(name, 'daylight', 'mae')} | {cell(name, 'daylight_monsoon', 'rmse')} | "
                     f"{cell(name, 'daylight_non_monsoon', 'rmse')} |")
    lines += ["", "Variants: " + "; ".join(f"`{k}` = {v}" for k, v in report["variants"].items()) + ".", "",
              "## Default minus paper (negative favours the default)", "",
              "| Paper variant | Scope | ΔRMSE [95% CI] | ΔMAE [95% CI] |", "| --- | --- | ---: | ---: |"]
    for key, by_scope in report["paired_bootstrap_default_minus_paper"].items():
        for scope, d in by_scope.items():
            r, a = d["rmse"], d["mae"]
            lines.append(f"| {key.removeprefix('default_minus_')} | {scope} | {r['difference']:.2f} "
                         f"[{r['ci95_lo']:.2f}, {r['ci95_hi']:.2f}] | {a['difference']:.2f} "
                         f"[{a['ci95_lo']:.2f}, {a['ci95_hi']:.2f}] |")
    lines += ["", "## Training", ""]
    for model, info in report["training"].items():
        cands = ", ".join(f"{c['hidden']} cells: MSE {c['best_validation_mse']:.5f} "
                          f"(epoch {c['best_epoch']}/{c['epochs_run']})" for c in info["candidates"])
        lines.append(f"- Model {model}: selected {info['selected_cells']} cells. {cands}.")
    lines += ["", "## Forecast-day weather type agreement on the test year", ""]
    for variant, info in report["test_weather_type_agreement"].items():
        lines.append(f"- {variant}: {info['accuracy'] * 100:.1f}% "
                     f"(rows observed, columns assumed: {info['confusion_rows_observed_cols_assumed']})")
    lines += ["", "## Published Trabzon values (context only)", "",
              "| Model | All-24 RMSE | All-24 MAE | All-24 nRMSE % | Daylight RMSE | Daylight MAE |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for model, v in report["published_trabzon"].items():
        lines.append(f"| {model} | {v['all24_rmse']:.2f} | {v['all24_mae']:.2f} | "
                     f"{100 * v['all24_rmse'] / report['trabzon_mean_ghi_all_hours']:.1f} | "
                     f"{v['daylight_rmse']:.2f} | {v['daylight_mae']:.2f} |")
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in report["limitations"]]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
