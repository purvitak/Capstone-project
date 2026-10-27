"""GFS-guided DWT-BiLSTM + cloud-regime blend with the current default.

.venv/bin/python pipelines/nwp_dwt_hybrid/run.py

1. Development: expanding chronological folds over GFS-available development days
   (180 initial days, 90-day blocks, one-day purge) give out-of-fold expert forecasts.
2. Selection: per-regime blend weights with the default's out-of-fold forecasts on
   ranking days (before 2025-04-05); guard days (2025-04-05..2025-07-23) check them.
   The blend is adopted only if hourly daylight RMSE and MAE both improve on both.
3. Test: refit on all GFS development days, apply frozen weights, score once beside
   the default and the Bektaş & Altaş reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
import hybrid_model as H
from hybrid_model import P

_spec = importlib.util.spec_from_file_location("paper_run", HERE.parent / "paper_reproduction" / "run.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

DEFAULT_OUT = ROOT / "results" / "nwp_dwt_hybrid"
DEFAULT_TEST = ROOT / "results" / "xgboost_nwp" / "xgboost_base_pred.npz"
DEFAULT_DEV = ROOT / "results" / "xgboost_nwp" / "trial_development.npz"
PAPER_PRED = ROOT / "results" / "paper_reproduction" / "predictions.npz"
INITIAL_DAYS, BLOCK_DAYS = 180, 90
GUARD_FIRST = "2025-04-05"
GRID = np.round(np.arange(0.0, 1.0001, 0.1), 2)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_extra(models, prep: Path) -> dict:
    sys.path.insert(0, str(HERE.parent / "multi_nwp"))
    import fetch
    return {m: fetch.load(m, prep) for m in models if m != "gfs"}


def sources(data: dict, forecasts: dict, models, days_needed) -> tuple[dict, dict]:
    """forecasts maps model name to an aligned cache; GFS defines the usable days.

    A model missing on a day is replaced by the mean of the models available that day.
    """
    ghi_h, cs_h = H.hourly(data["ghi"]), H.hourly(data["cs_proxy"])
    gfs = forecasts["gfs"]
    f = gfs["features"].astype(float)
    gfs_met = np.stack([H.hourly(f[..., 2]), H.hourly(f[..., 3]), H.hourly(f[..., 4]) / 3.6], axis=-1)
    gfs_met[~gfs["available"]] = np.nan
    kt, cloud, k, available = [], [], [], []
    for m in models:
        f = forecasts[m]["features"].astype(float)
        ok = forecasts[m]["available"] & gfs["available"]
        ghi_m, cloud_m = H.hourly(f[..., 0]), H.hourly(f[..., 1])
        ghi_m[~ok], cloud_m[~ok] = np.nan, np.nan
        kt.append(H.gfs_clear_sky_index(ghi_m, cs_h))
        cloud.append(cloud_m)
        k.append(P.daily_clearness(ghi_m, data["doy"], C.MUMBAI_LAT))
        available.append(ok)
    available = np.stack(available, axis=-1)
    met_obs = np.stack([H.hourly(data["air_temp"]), H.hourly(data["rh"]),
                        H.hourly(data["wind_speed"])], axis=-1)
    src = {"ghi_h": ghi_h, "cs_h": cs_h, "met_obs": met_obs,
           "type_obs": P.weather_types(P.daily_clearness(ghi_h, data["doy"], C.MUMBAI_LAT)),
           "nwp_kt": H.fill_missing_models(np.stack(kt, axis=-1), available),
           "nwp_cloud": H.fill_missing_models(np.stack(cloud, axis=-1), available),
           "nwp_k": H.fill_missing_models(np.stack(k, axis=-1), available),
           "gfs_met": gfs_met, "dwt": P.dwt_by_day(P.forward_fill_days(ghi_h), days_needed)}
    coverage = {m: int(available[:, j].sum()) for j, m in enumerate(models)}
    return src, {"available_days": coverage, "available": available}


def fit_predict(src, train_days, pred_days, args, cache: Path, tag: str):
    config = {"version": 1, "tag": tag, "train": [int(train_days[0]), int(train_days[-1]), len(train_days)],
              "pred": [int(pred_days[0]), int(pred_days[-1]), len(pred_days)], "hidden": args.hidden,
              "seeds": args.seeds, "max_epochs": args.max_epochs, "patience": args.patience,
              "prep_sha256": args.prep_hash, "smoke": args.smoke}
    if args.nwp_set != "gfs":  # GFS-only runs keep their original cache contract
        config["nwp_set"] = list(H.NWP_SETS[args.nwp_set])
    path = cache / f"{tag}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if json.loads(str(saved["config"])) == config:
                print(f"[{tag}] reused", flush=True)
                return saved["pred"], json.loads(str(saved["info"]))
    fit, val = R.split(np.asarray(train_days))
    norm = H.normalizer(src, fit)
    scale = norm["ghi_scale"]
    xf, xv, xp = (H.build_inputs(src, d, norm) for d in (fit, val, pred_days))
    yf, yv = ((src["ghi_h"][d] / scale).astype(np.float32) for d in (fit, val))
    preds, infos = [], []
    started = time.time()
    for seed in args.seeds:
        net, info = P.train_network(xf, yf, xv, yv, args.hidden, seed,
                                    max_epochs=args.max_epochs, patience=args.patience)
        preds.append(P.predict(net, xp) * scale)
        infos.append({k: v for k, v in info.items() if k != "validation_mse_history"})
    pred = np.mean(preds, axis=0)
    print(f"[{tag}] {len(fit)} fit / {len(val)} validation days, best epochs "
          f"{[i['best_epoch'] for i in infos]} ({time.time() - started:.0f}s)", flush=True)
    np.savez_compressed(path, pred=pred, config=json.dumps(config), info=json.dumps(infos))
    return pred, infos


def daylight_metrics(actual, pred, mask) -> dict:
    return C.point_metrics(actual, pred, mask)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", type=Path, default=C.PREP_PATH)
    ap.add_argument("--nwp-cache", type=Path, default=N.DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=None,
                    help="Default: results/nwp_dwt_hybrid (gfs) or results/multi_nwp_hybrid (multi)")
    ap.add_argument("--nwp-set", choices=sorted(H.NWP_SETS), default="gfs")
    ap.add_argument("--night-zero", action="store_true",
                    help="Expert output is non-negative and zero when the clear-sky proxy is night")
    ap.add_argument("--hidden", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="Few epochs; use a separate --out")
    args = ap.parse_args()
    if args.out is None:
        if args.smoke:
            ap.error("--smoke requires a separate --out directory")
        args.out = DEFAULT_OUT if args.nwp_set == "gfs" else ROOT / "results" / "multi_nwp_hybrid"
    if args.smoke:
        args.max_epochs, args.patience, args.seeds = 3, 2, args.seeds[:1]
    import torch
    torch.set_num_threads(args.threads)
    started = time.time()
    args.prep_hash = sha256(args.prep)
    (args.out / "folds").mkdir(parents=True, exist_ok=True)

    data, nwp = R.load(args.prep, args.nwp_cache)
    dates, dev, test = data["dates"], data["dev_idx"], data["test_idx"]
    gfs_dev = dev[nwp["available"][dev]]
    if not (np.diff(gfs_dev) == 1).all() or not nwp["available"][test].all():
        raise ValueError("Expected contiguous GFS development days and complete test coverage")
    models = H.NWP_SETS[args.nwp_set]
    forecasts_nwp = {"gfs": nwp, **load_extra(models, args.prep)}
    src, coverage = sources(data, forecasts_nwp, models, np.concatenate([gfs_dev, test]))
    nwp_coverage = {"development": {m: int(coverage["available"][gfs_dev, j].sum()) for j, m in enumerate(models)},
                    "test": {m: int(coverage["available"][test, j].sum()) for j, m in enumerate(models)},
                    "missing_model_rule": "mean of the models available that day"}
    print(f"[nwp] {nwp_coverage}", flush=True)

    def finish(pred, days):
        return H.zero_at_night(pred, src["cs_h"][days]) if args.night_zero else pred

    # 1. Development out-of-fold expert forecasts.
    expert = np.full((len(dates), 24), np.nan)
    folds = []
    for start in range(INITIAL_DAYS, len(gfs_dev), BLOCK_DAYS):
        pred_days = gfs_dev[start:start + BLOCK_DAYS]
        train_days = gfs_dev[:start - 1]
        tag = f"fold_{dates[pred_days[0]]}"
        pred, info = fit_predict(src, train_days, pred_days, args, args.out / "folds", tag)
        expert[pred_days] = finish(pred, pred_days)
        folds.append({"train_days": len(train_days), "first": str(dates[pred_days[0]]),
                      "last": str(dates[pred_days[-1]]), "seeds": info})

    # 2. Blend selection on development days only.
    with np.load(DEFAULT_DEV, allow_pickle=False) as saved:
        default_dev = H.hourly(np.clip(saved["kt_base"], *C.KT_CLIP) * data["cs_proxy"])
    oof = gfs_dev[np.isfinite(expert[gfs_dev]).all(axis=1) & np.isfinite(default_dev[gfs_dev]).all(axis=1)]
    rank = oof[dates[oof] < GUARD_FIRST]
    guard = oof[dates[oof] >= GUARD_FIRST]
    daylight_all = data["mask"].reshape(len(dates), 24, 6).any(axis=2)
    regimes_all = H.regime_of(src["nwp_cloud"], src["cs_h"])
    actual_all = src["ghi_h"]
    weights = H.select_weights(actual_all[rank], default_dev[rank], expert[rank], daylight_all[rank],
                               regimes_all[rank], GRID)
    global_w = H.select_weights(actual_all[rank], default_dev[rank], expert[rank], daylight_all[rank],
                                np.zeros(len(rank), dtype=int), GRID)[0]
    development = {}
    for name, days in (("ranking", rank), ("guard", guard)):
        row = {}
        for label, pred in (("default", default_dev[days]), ("expert", expert[days]),
                            ("regime_blend", H.blend(default_dev[days], expert[days], regimes_all[days], weights)),
                            ("global_blend", H.blend(default_dev[days], expert[days], np.zeros(len(days), int),
                                                     [global_w]))):
            row[label] = daylight_metrics(actual_all[days], pred, daylight_all[days])
        row["days"] = [str(dates[days[0]]), str(dates[days[-1]]), len(days)]
        row["regime_days"] = np.bincount(regimes_all[days], minlength=3).tolist()
        development[name] = row
    adopted = all(development[s]["regime_blend"][m] < development[s]["default"][m]
                  for s in ("ranking", "guard") for m in ("rmse", "mae"))
    print(f"[select] regime weights {dict(zip(H.REGIMES, weights))}, global {global_w}; adopted={adopted}")
    for s in ("ranking", "guard"):
        d = development[s]
        print(f"[select] {s}: default {d['default']['rmse']:.2f}/{d['default']['mae']:.2f}  "
              f"expert {d['expert']['rmse']:.2f}/{d['expert']['mae']:.2f}  "
              f"regime {d['regime_blend']['rmse']:.2f}/{d['regime_blend']['mae']:.2f}  "
              f"global {d['global_blend']['rmse']:.2f}/{d['global_blend']['mae']:.2f}")
    frozen = {"regime_weights": dict(zip(H.REGIMES, weights)), "global_weight": global_w,
              "regime_edges_cloud_pct": H.REGIME_EDGES, "adopted": adopted, "development": development,
              "test_scored_after_freeze": True}
    (args.out / "frozen_selection.json").write_text(json.dumps(frozen, indent=2, default=float))

    # 3. Final fit on all GFS development days; score the test year once.
    expert_test, final_info = fit_predict(src, gfs_dev, test, args, args.out / "folds", "final")
    expert_test = finish(expert_test, test)
    actual = actual_all[test]
    daylight = daylight_all[test]
    regimes = regimes_all[test]
    default_h = R.default_forecast(DEFAULT_TEST, data, test)
    hybrid_h = H.blend(default_h, expert_test, regimes, weights)
    forecasts = {"hybrid_regime_blend": hybrid_h, "nwp_dwt_bilstm_expert": expert_test,
                 "hybrid_global_blend": H.blend(default_h, expert_test, np.zeros(len(test), int), [global_w]),
                 "default_xgboost_blend": default_h}
    with np.load(PAPER_PRED, allow_pickle=False) as paper:
        if not np.array_equal(paper["dates"], dates[test]) or not np.allclose(paper["actual_hourly"], actual):
            raise ValueError("Paper reproduction predictions are not aligned with this test year")
        for key in paper.files:
            if key.startswith("paper_") or key == "persistence_d1":
                forecasts[key] = paper[key]

    masks = R.scopes(daylight, data["is_monsoon"][test].astype(bool))
    rows = []
    for name, pred in forecasts.items():
        for scope, mask in masks.items():
            m = C.point_metrics(actual, pred, mask)
            rows.append({"method": name, "scope": scope, **m,
                         "nmae": m["mae"] / float(np.mean(actual[mask])) * 100})
    metrics = pd.DataFrame(rows)

    # Ten-minute view under the project's primary mask.
    cs, ghi10, mask10 = data["cs_proxy"][test], data["ghi"][test], data["mask"][test]
    with np.load(DEFAULT_TEST, allow_pickle=False) as saved:
        default10 = np.clip(saved["kt_base"][test], *C.KT_CLIP) * cs
    w10 = np.asarray(weights)[regimes][:, None]
    hybrid10 = (1 - w10) * default10 + w10 * H.to_10min(expert_test, cs)
    ten_minute = {"default_xgboost_blend": C.point_metrics(ghi10, default10, mask10),
                  "hybrid_regime_blend": C.point_metrics(ghi10, hybrid10, mask10)}
    for season, sel in (("monsoon", data["is_monsoon"][test].astype(bool)),
                        ("non_monsoon", ~data["is_monsoon"][test].astype(bool))):
        for name, pred in (("default_xgboost_blend", default10), ("hybrid_regime_blend", hybrid10)):
            ten_minute[f"{name}_{season}"] = C.point_metrics(ghi10[sel], pred[sel], mask10[sel])

    comparisons = {}
    for name, pred in forecasts.items():
        if name == "hybrid_regime_blend":
            continue
        comparisons[f"hybrid_minus_{name}"] = {scope: R.paired_bootstrap(actual, hybrid_h, pred, masks[scope])
                                              for scope in ("all24", "daylight")}

    metrics.to_csv(args.out / "metrics.csv", index=False)
    np.savez_compressed(args.out / "predictions.npz", dates=dates[test], actual_hourly=actual,
                        daylight_hourly=daylight, regimes=regimes, hybrid_10min=hybrid10,
                        **{k: v for k, v in forecasts.items() if not k.startswith("paper_")})
    report = {"description": __doc__, "nwp_set": list(models), "night_zero": args.night_zero,
              "nwp_coverage": nwp_coverage, "features": H.feature_names(models),
              "frozen": frozen, "folds": folds,
              "final_training": final_info, "test_regime_days": np.bincount(regimes, minlength=3).tolist(),
              "ten_minute": ten_minute, "paired_bootstrap": comparisons,
              "seconds": time.time() - started, "smoke": args.smoke,
              "limitations": [
                  "The test year influenced earlier project iterations; results are exploratory.",
                  "The expert trains only on GFS-available days (January 2024 onward): 550 development days.",
                  "Blend weights are selected on 369 development days that include one monsoon season.",
                  "Reference GHI is Solcast satellite/model-derived, not a ground pyranometer.",
              ],
              # Read by the dashboard to verify the served forecasts.
              "prep_sha256": args.prep_hash, "test_days": int(len(test)),
              "test_period": [str(dates[test[0]]), str(dates[test[-1]])],
              "artifact_hashes": {"predictions.npz": sha256(args.out / "predictions.npz")}}
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    write_markdown(args.out, metrics, report)
    view = metrics[metrics.scope.isin(["all24", "daylight"])].pivot(index="method", columns="scope",
                                                                    values=["rmse", "mae"])
    print(view.round(2).to_string())
    print("[10-min daylight]", {k: (round(v["rmse"], 2), round(v["mae"], 2)) for k, v in ten_minute.items()})
    print(f"[done] {args.out} ({time.time() - started:.0f}s)")


def write_markdown(out: Path, metrics: pd.DataFrame, report: dict) -> None:
    def cell(method, scope, key):
        row = metrics[(metrics.method == method) & (metrics.scope == scope)]
        return f"{row[key].iloc[0]:.2f}"

    frozen = report["frozen"]
    order = ["hybrid_regime_blend", "hybrid_global_blend", "nwp_dwt_bilstm_expert", "default_xgboost_blend"]
    order += [m for m in metrics.method.unique() if m.startswith("paper_")] + ["persistence_d1"]
    lines = [f"# NWP-guided DWT-BiLSTM with cloud-regime blending ({', '.join(report['nwp_set'])})", "",
             f"Night rule applied to the expert: {report['night_zero']}. NWP coverage: {report['nwp_coverage']}.", "",
             f"Blend adopted on development data: **{frozen['adopted']}**. Regime weights on the new "
             f"expert: {frozen['regime_weights']} (cloud edges {frozen['regime_edges_cloud_pct']} %); "
             f"global ablation weight {frozen['global_weight']}.", "",
             "## Development (hourly daylight RMSE / MAE, W/m²)", "",
             "| Split | Days | Default | Expert | Regime blend | Global blend |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for split, d in frozen["development"].items():
        lines.append(f"| {split} | {d['days'][0]}..{d['days'][1]} ({d['days'][2]}) | " + " | ".join(
            f"{d[k]['rmse']:.2f} / {d[k]['mae']:.2f}" for k in ("default", "expert", "regime_blend", "global_blend")) + " |")
    lines += ["", "## Test year (hourly, W/m²)", "",
              "| Method | All-24 RMSE | All-24 MAE | All-24 nRMSE % | Daylight RMSE | Daylight MAE | Monsoon daylight RMSE / MAE | Non-monsoon daylight RMSE / MAE |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in order:
        lines.append(f"| {name} | {cell(name, 'all24', 'rmse')} | {cell(name, 'all24', 'mae')} | "
                     f"{cell(name, 'all24', 'nrmse')} | {cell(name, 'daylight', 'rmse')} | {cell(name, 'daylight', 'mae')} | "
                     f"{cell(name, 'daylight_monsoon', 'rmse')} / {cell(name, 'daylight_monsoon', 'mae')} | "
                     f"{cell(name, 'daylight_non_monsoon', 'rmse')} / {cell(name, 'daylight_non_monsoon', 'mae')} |")
    lines += ["", "## Ten-minute daylight (project primary mask)", "", "| Forecast | RMSE | MAE |", "| --- | ---: | ---: |"]
    for name, m in report["ten_minute"].items():
        lines.append(f"| {name} | {m['rmse']:.2f} | {m['mae']:.2f} |")
    lines += ["", "## Hybrid minus comparator (negative favours the hybrid)", "",
              "| Comparator | Scope | ΔRMSE [95% CI] | ΔMAE [95% CI] |", "| --- | --- | ---: | ---: |"]
    for key, by_scope in report["paired_bootstrap"].items():
        for scope, d in by_scope.items():
            r, a = d["rmse"], d["mae"]
            lines.append(f"| {key.removeprefix('hybrid_minus_')} | {scope} | {r['difference']:.2f} "
                         f"[{r['ci95_lo']:.2f}, {r['ci95_hi']:.2f}] | {a['difference']:.2f} "
                         f"[{a['ci95_lo']:.2f}, {a['ci95_hi']:.2f}] |")
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in report["limitations"]]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
