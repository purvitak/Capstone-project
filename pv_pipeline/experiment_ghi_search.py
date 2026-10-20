"""Bounded direct-GHI search, development-only selection, then frozen evaluation.

Default phase never fits or scores the test year. Each fold is checkpointed.
Existing prediction artifacts and dashboard selections are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import common as C
from hgb_training import fit_direct_ghi, new_model, predict_direct_kt

HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Atomic report/checkpoint metadata replacement."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def save_predictions(path, values, dates, config):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, kt_base=values, dates=dates,
                        config=json.dumps(config, sort_keys=True))
    temporary.replace(path)


def load_predictions(path, data, prep_hash):
    with np.load(path, allow_pickle=False) as saved:
        config = json.loads(str(saved["config"]))
        if config.get("prep_sha256") != prep_hash:
            raise ValueError(f"{path}: prepared-data fingerprint differs")
        if config.get("leaky") or config.get("smoke"):
            raise ValueError(f"{path}: leaky/smoke source is not allowed")
        if not np.array_equal(saved["dates"], data["dates"]):
            raise ValueError(f"{path}: dates differ")
        pred = saved["kt_base"].copy()
        if pred.shape != data["ghi"].shape:
            raise ValueError(f"{path}: prediction shape differs")
    return pred, config


def feature_days(data, days, base):
    types = base._daily_types(data["ghi"], data["cs_proxy"], data["mask"])
    x, _, returned = base.pack_days(data["ghi"], data["cs_proxy"], data["mask"],
                                    types, days, False, False, 6)
    if not np.array_equal(returned, days):
        raise ValueError("Feature generation dropped forecast days")
    # The prepared context contains historical weather/index and known solar data.
    return np.concatenate((x, data["X"][days]), axis=-1)


def metrics(data, pred, days):
    if not np.isfinite(pred[days]).all():
        raise ValueError("Every compared model must cover exactly the same days")
    return C.point_metrics(data["ghi"][days],
                           np.clip(pred[days], *C.KT_CLIP) * data["cs_proxy"][days],
                           data["mask"][days])


def seasonal_rows(data, pred, days, method, **extra):
    rows = []
    for season, selected in (("all", np.ones(len(days), bool)),
                             ("monsoon", data["is_monsoon"][days]),
                             ("non_monsoon", ~data["is_monsoon"][days])):
        idx = days[selected]
        if len(idx):
            rows.append({"method": method, "season": season, "n_days": len(idx),
                         **metrics(data, pred, idx), **extra})
    return rows


def blend(a, b, weight):
    return weight * np.clip(a, *C.KT_CLIP) + (1 - weight) * np.clip(b, *C.KT_CLIP)


def choose_candidate(rows, reference):
    """Both pooled development errors must strictly beat the full-precision reference."""
    eligible = [r for r in rows if r["season"] == "all"
                and r["rmse"] < reference["rmse"] and r["mae"] < reference["mae"]]
    return min(eligible, key=lambda r: (r["rmse"], r["mae"], r["method"])) if eligible else None


def candidates():
    for loss, leaves, samples in itertools.product(
            ("squared_error", "absolute_error"), (15, 31), (40, 100)):
        yield f"{loss}_leaves{leaves}_min{samples}", {
            "loss": loss, "max_leaf_nodes": leaves, "min_samples_leaf": samples,
            "max_iter": 200, "learning_rate": 0.05, "seed": 42}


def develop(args, data, base, manifest):
    out = args.out
    if (out / "frozen_selection.json").exists():
        print("[development] Configuration already frozen; development search skipped", flush=True)
        return
    dev = data["dev_idx"]
    days = dev[base.INITIAL_TRAIN_DAYS:]
    dwt, _ = load_predictions(args.dwt, data, manifest["prep_sha256"])
    reference, _ = load_predictions(args.reference, data, manifest["prep_sha256"])
    reference_metrics = metrics(data, reference, days)
    xdev = feature_days(data, dev, base)
    all_rows = seasonal_rows(data, reference, days, "reference_hybrid")
    for name, params in candidates():
        folder = out / name
        folder.mkdir(exist_ok=True)
        ckpt = folder / "dwt_base_pred.npz"
        report = folder / "training.json"
        config = {"trainer_version": 5, "backend": "hgb", "resolution": "10min",
                  "features": "context", "target": "ghi", "leaky": False,
                  "smoke": False, "prep_sha256": manifest["prep_sha256"],
                  "params": params, "validation": "chronological_20pct_one_day_purge",
                  "daylight_training": True, "experiment_manifest_sha256": fingerprint(out / "manifest.json")}
        if ckpt.exists():
            pred, old = load_predictions(ckpt, data, manifest["prep_sha256"])
            if old != config or not report.exists():
                raise ValueError(f"{folder}: checkpoint configuration/metadata differs")
            folds = json.loads(report.read_text())["folds"]
        else:
            pred, folds = np.full_like(data["ghi"], np.nan), []
        for a in range(base.INITIAL_TRAIN_DAYS, len(dev), base.FOLD_DAYS):
            b = min(a + base.FOLD_DAYS, len(dev))
            train, predict = dev[:a - base.PURGE_DAYS], dev[a:b]
            if np.isfinite(pred[predict]).all():
                continue
            started = time.monotonic()
            model, info = fit_direct_ghi(
                xdev[:a - base.PURGE_DAYS], data["ghi"][train],
                data["cs_proxy"][train], data["mask"][train], **params)
            pred[predict] = predict_direct_kt(model, xdev[a:b], data["cs_proxy"][predict])
            cut = int(len(train) * 0.8)
            info.update(first_train_date=str(data["dates"][train[0]]),
                        last_train_date=str(data["dates"][train[-1]]),
                        inner_last_fit_date=str(data["dates"][train[cut - 2]]),
                        inner_first_validation_date=str(data["dates"][train[cut]]))
            folds = [f for f in folds if f["first"] != str(data["dates"][predict[0]])]
            folds.append({"first": str(data["dates"][predict[0]]),
                          "last": str(data["dates"][predict[-1]]), "training": info,
                          "seconds": time.monotonic() - started})
            # Metadata precedes array replacement, so an interrupted array save is
            # rerun, while every completed array checkpoint has iteration metadata.
            write_json(report, {"config": config, "folds": folds})
            save_predictions(ckpt, pred, data["dates"], config)
            score = metrics(data, pred, predict)
            print(f"[{name}] {a}:{b} iter={info['selected_iterations']} "
                  f"RMSE={score['rmse']:.3f} MAE={score['mae']:.3f} "
                  f"({time.monotonic()-started:.1f}s)", flush=True)
        for weight in (0.0, 0.25, 0.5, 0.75):
            values = pred if weight == 0 else blend(dwt, pred, weight)
            all_rows += seasonal_rows(data, values, days, f"{name}_w{weight:g}",
                                      candidate=name, bilstm_weight=weight)
        pd.DataFrame(all_rows).to_csv(out / "development_metrics.csv", index=False)
    selected = choose_candidate(all_rows, reference_metrics)
    if selected is None:
        selected = {"method": "reference_hybrid", "candidate": None,
                    "bilstm_weight": 0.5, **reference_metrics}
        values = reference.copy()
    else:
        tree, _ = load_predictions(out / selected["candidate"] / "dwt_base_pred.npz",
                                    data, manifest["prep_sha256"])
        values = tree if selected["bilstm_weight"] == 0 else blend(dwt, tree, selected["bilstm_weight"])
    # Freeze a development-only base; test rows must still be absent.
    values[data["test_idx"]] = np.nan
    values[int(data["purge_idx"])]=np.nan
    selected_config = {"model": "selected_ghi_search", "backend": "hgb_hybrid",
                       "resolution": "10min", "leaky": False, "smoke": False,
                       "prep_sha256": manifest["prep_sha256"], "selection": selected,
                       "manifest_sha256": fingerprint(out / "manifest.json")}
    save_predictions(out / "selected_development.npz", values, data["dates"], selected_config)
    correction = validate_correction(data, values, days, out)
    freeze = {"selection": selected, "reference_development": reference_metrics,
              "correction": correction, "selection_rule": "lower_RMSE_and_lower_MAE_than_reference",
              "development_days": len(days), "first": str(data["dates"][days[0]]),
              "last": str(data["dates"][days[-1]]), "test_scored": False,
              "manifest_sha256": fingerprint(out / "manifest.json"),
              "selected_predictions_sha256": fingerprint(out / "selected_development.npz"),
              "correction_validation_sha256": fingerprint(out / "correction_validation.npz"),
              "candidate_predictions_sha256": (
                  fingerprint(out / selected["candidate"] / "dwt_base_pred.npz")
                  if selected["candidate"] else None),
              "limitations": ["Development-derived climatology is shared across OOF folds.",
                              "Development model and correction selection reuse validation dates.",
                              "The existing test year has been inspected in earlier project iterations."]}
    write_json(out / "frozen_selection.json", freeze)
    print(f"[frozen] {selected['method']}: RMSE {selected['rmse']:.5f}, "
          f"MAE {selected['mae']:.5f}; correction retained={correction['accepted']}", flush=True)


def correction_features(data, pred):
    module = load_module("residual_corrector", "02_corrector.py")
    # Day-D feature summaries use the known solar mask, not observed availability.
    return module.build_corrector_features(
        data["kt"], pred, data["cs_proxy"], data["cs_solcast"],
        data["zenith"], data["daylight"], data["doy"])[0]


def apply_correction(pred, raw, monsoon, bounds):
    delta = np.where(monsoon, bounds["monsoon"], bounds["non_monsoon"])[:, None]
    return np.clip(pred + delta * np.tanh(raw / np.where(delta == 0, 1, delta)), *C.KT_CLIP)


def correction_model(iterations):
    return new_model(max_iter=iterations, max_leaf_nodes=31, min_samples_leaf=100,
                     l2_regularization=1.0)


def validate_correction(data, pred, days, out):
    x = correction_features(data, pred)
    cut = int(len(days) * 0.75)
    train, val = days[:cut - 1], days[cut:]  # one-day purge
    mask = data["mask"]
    residual = data["kt"] - pred
    model = correction_model(500).fit(x[train][mask[train]], residual[train][mask[train]])
    raw = np.zeros_like(pred)
    base_score = metrics(data, pred, val)
    best, best_raw = (np.inf, np.inf, 0), None
    cs, truth = data["cs_proxy"][val][mask[val]], data["ghi"][val][mask[val]]
    for iteration, r in enumerate(model.staged_predict(x[val][mask[val]]), 1):
        error = np.clip(pred[val][mask[val]] + r, *C.KT_CLIP) * cs - truth
        score = (float(np.sqrt(np.mean(error ** 2))), float(np.mean(abs(error))), iteration)
        if score[:2] < best[:2]:
            best, best_raw = score, r.copy()
    raw_val = np.zeros_like(pred[val])
    raw_val[mask[val]] = best_raw
    raw[val] = raw_val
    bounds, grid = {}, []
    for season, sel in (("monsoon", data["is_monsoon"][val]),
                        ("non_monsoon", ~data["is_monsoon"][val])):
        selected_days = val[sel]
        baseline = metrics(data, pred, selected_days)
        eligible = []
        for delta in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
            corrected = pred.copy()
            corrected[selected_days] = apply_correction(
                pred[selected_days], raw[selected_days], data["is_monsoon"][selected_days],
                {"monsoon": delta, "non_monsoon": delta})
            score = metrics(data, corrected, selected_days)
            grid.append({"season": season, "delta": delta, **score})
            if delta and score["rmse"] < baseline["rmse"] and score["mae"] < baseline["mae"]:
                eligible.append((score["rmse"], score["mae"], delta))
        bounds[season] = min(eligible)[2] if eligible else 0.0
    corrected = pred.copy()
    corrected[val] = apply_correction(pred[val], raw[val], data["is_monsoon"][val], bounds)
    score = metrics(data, corrected, val)
    accepted = score["rmse"] < base_score["rmse"] and score["mae"] < base_score["mae"]
    if not accepted:
        bounds = {"monsoon": 0.0, "non_monsoon": 0.0}
        corrected[val] = pred[val]
    # Empirical intervals are diagnostic: validation has also selected settings.
    errors = (data["kt"] - corrected)[val][mask[val]]
    quantiles = np.quantile(errors, [0.1, 0.9]).tolist()
    np.savez_compressed(out / "correction_validation.npz", days=val, kt_final=corrected[val])
    return {"accepted": bool(accepted), "selected_iterations": best[2],
            "bounds": bounds, "quantiles": quantiles,
            "train_days": len(train), "purge_days": 1, "validation_days": len(val),
            "last_train_date": str(data["dates"][train[-1]]),
            "first_validation_date": str(data["dates"][val[0]]),
            "base_validation": base_score, "candidate_validation": score,
            "delta_grid_metrics": grid,
            "validation_season_metrics": seasonal_rows(data, corrected, val, "selected_correction")}


def paired_bootstrap(data, pred, reference, days, *, repeats=2000, seed=42):
    """Paired circular seven-day blocks; recompute pooled RMSE and MAE.

    Resample days together to preserve paired errors and within-week dependence.
    Sum-of-errors statistics keep unequal daylight sample counts correct.
    """
    mask = data["mask"][days]
    actual, cs = data["ghi"][days], data["cs_proxy"][days]
    counts = mask.sum(axis=1)
    stats = []
    for values in (pred, reference):
        err = np.where(mask, np.clip(values[days], *C.KT_CLIP) * cs - actual, 0)
        if not np.isfinite(err).all():
            raise ValueError("Paired comparison requires finite predictions")
        stats.append((np.sum(err ** 2, axis=1), np.sum(abs(err), axis=1)))
    def difference(idx):
        n = counts[idx].sum(axis=-1)
        rmses = [np.sqrt(sq[idx].sum(axis=-1) / n) for sq, _ in stats]
        maes = [ab[idx].sum(axis=-1) / n for _, ab in stats]
        return rmses[0] - rmses[1], maes[0] - maes[1]
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(days), (repeats, int(np.ceil(len(days) / 7))))
    idx = ((starts[..., None] + np.arange(7)) % len(days)).reshape(repeats, -1)[:, :len(days)]
    samples = difference(idx)
    observed = difference(np.arange(len(days)))
    return {name: {"difference": float(point), "ci95_lo": float(np.quantile(values, 0.025)),
                   "ci95_hi": float(np.quantile(values, 0.975))}
            for name, point, values in zip(("rmse", "mae"), observed, samples)}


def evaluate(args, data, base, manifest):
    out = args.out
    frozen_path = out / "frozen_selection.json"
    if not frozen_path.exists():
        raise ValueError("Run development and freeze selection before evaluation")
    frozen = json.loads(frozen_path.read_text())
    checks = {"manifest.json": "manifest_sha256", "selected_development.npz": "selected_predictions_sha256",
              "correction_validation.npz": "correction_validation_sha256"}
    for file, key in checks.items():
        if fingerprint(out / file) != frozen[key]:
            raise ValueError(f"Frozen artifact changed: {file}")
    chosen = frozen["selection"]
    if chosen["candidate"] and fingerprint(out / chosen["candidate"] / "dwt_base_pred.npz") != frozen["candidate_predictions_sha256"]:
        raise ValueError("Frozen candidate checkpoint changed")
    frozen_hash = fingerprint(frozen_path)
    final_report = out / "evaluation.json"
    if final_report.exists():
        completed = json.loads(final_report.read_text())
        if completed["frozen_selection_sha256"] != frozen_hash:
            raise ValueError("Evaluation belongs to a different frozen selection")
        write_summary(out, frozen, completed)
        print(f"[evaluation] Already complete: {final_report}", flush=True)
        return
    dev, test = data["dev_idx"], data["test_idx"]
    need = np.concatenate(([int(data["purge_idx"])], test))
    oof_dwt, _ = load_predictions(args.dwt, data, manifest["prep_sha256"])
    test_dwt, _ = load_predictions(args.test_dwt, data, manifest["prep_sha256"])
    days = dev[base.INITIAL_TRAIN_DAYS:]
    if not np.array_equal(oof_dwt[days], test_dwt[days]):
        raise ValueError("Saved test BiLSTM does not match development component")
    dwt = oof_dwt.copy()
    dwt[need] = test_dwt[need]
    xtrain, xpred = feature_days(data, dev, base), feature_days(data, need, base)
    # Complete the existing reference hybrid without changing its saved OOF files.
    context, _ = load_predictions(args.context, data, manifest["prep_sha256"])
    legacy_target = np.divide(data["ghi"][dev], data["cs_proxy"][dev],
                              out=np.zeros_like(data["ghi"][dev]),
                              where=data["cs_proxy"][dev] > C.DAYLIGHT_CS_MIN)
    legacy_target = np.clip(np.nan_to_num(legacy_target), *C.KT_CLIP).astype(np.float32)
    legacy = base.fit_hgb(xtrain, legacy_target)
    context[need] = base.predict_hgb(legacy, xpred)
    reference = blend(dwt, context, 0.5)
    pred, selected_config = load_predictions(out / "selected_development.npz", data, manifest["prep_sha256"])
    final_training = {}
    if chosen["candidate"]:
        params = dict(candidates())[chosen["candidate"]]
        model, final_training = fit_direct_ghi(xtrain, data["ghi"][dev],
                                             data["cs_proxy"][dev], data["mask"][dev], **params)
        tree = predict_direct_kt(model, xpred, data["cs_proxy"][need])
        pred[need] = tree if chosen["bilstm_weight"] == 0 else blend(dwt[need], tree, chosen["bilstm_weight"])
    else:
        pred[need] = reference[need]
    selected_config.update(frozen_selection_sha256=frozen_hash, final_training=final_training)
    save_predictions(out / "selected_base_pred.npz", pred, data["dates"], selected_config)
    corrected = pred.copy()
    correction = frozen["correction"]
    if correction["accepted"]:
        x = correction_features(data, pred)
        mask = data["mask"]
        residual = data["kt"] - pred
        model = correction_model(correction["selected_iterations"]).fit(x[days][mask[days]], residual[days][mask[days]])
        raw = np.zeros_like(pred[test])
        raw[mask[test]] = model.predict(x[test][mask[test]])
        corrected[test] = apply_correction(pred[test], raw, data["is_monsoon"][test], correction["bounds"])
    val = np.load(out / "correction_validation.npz", allow_pickle=False)
    corrected[val["days"]] = val["kt_final"]
    lo, hi = [np.clip(corrected + q, *C.KT_CLIP) for q in correction["quantiles"]]
    np.savez_compressed(out / "selected_stage2.npz", dates=data["dates"],
                        kt_final=corrected, kt_lo=lo, kt_hi=hi,
                        config=json.dumps(selected_config, sort_keys=True))
    with np.load(args.original_corrected, allow_pickle=False) as old:
        if not np.array_equal(old["dates"], data["dates"]):
            raise ValueError("Original corrected prediction dates differ")
        original = np.where(np.isfinite(old["kt_final"]), old["kt_final"], test_dwt)
    methods = {"mean3": C.baseline_mean3_kt(data["kt"]), "dwt_bilstm": dwt,
               "original_corrected": original, "reference_hybrid": reference,
               "selected_uncorrected": pred, "selected_final": corrected}
    rows = [row for name, values in methods.items() for row in seasonal_rows(data, values, test, name)]
    comparisons = {}
    for name in ("original_corrected", "reference_hybrid", "mean3", "selected_uncorrected"):
        comparisons[name] = paired_bootstrap(data, corrected, methods[name], test)
    m = data["mask"][test]
    covered = (data["kt"][test] >= lo[test]) & (data["kt"][test] <= hi[test])
    report = {"frozen_selection_sha256": frozen_hash, "test_days": len(test),
              "test_period": [str(data["dates"][test[0]]), str(data["dates"][test[-1]])],
              "selection": chosen, "correction_retained": correction["accepted"],
              "final_training": final_training, "metrics": rows,
              "paired_seven_day_block_bootstrap": comparisons,
              "bootstrap_repeats": 2000, "bootstrap_seed": 42,
              "interval_nominal_pct": 80, "interval_coverage_pct": float(100 * covered[m].mean()),
              "limitations": frozen["limitations"]}
    pd.DataFrame(rows).to_csv(out / "test_metrics.csv", index=False)
    write_json(final_report, report)
    print(pd.DataFrame(rows).query("season == 'all'")[["method", "rmse", "mae"]].to_string(index=False), flush=True)
    write_summary(out, frozen, report)


def write_summary(out, frozen, evaluation):
    rows = [r for r in evaluation["metrics"] if r["season"] == "all"]
    selection = frozen["selection"]
    reference = frozen["reference_development"]
    text = ["# Direct-GHI improvement experiment", "",
            "Eight configurations, 15 chronological folds, 1,293 shared development days. "
            "Selection required lower RMSE and MAE than the existing 50/50 hybrid. "
            "All values below are daylight ten-minute GHI in W/m².", "",
            f"Selected: **{selection['method']}**. "
            f"Residual correction retained: **{evaluation['correction_retained']}**.", "",
            "## Development selection", "", "| Method | RMSE | MAE |",
            "| --- | ---: | ---: |",
            f"| Existing 50/50 hybrid | {reference['rmse']:.5f} | {reference['mae']:.5f} |",
            f"| Selected configuration | {selection['rmse']:.5f} | {selection['mae']:.5f} |", ""]
    if selection["candidate"]:
        params = dict(candidates())[selection["candidate"]]
        text += [f"BiLSTM weight: {selection['bilstm_weight']:.2f}; direct-GHI HGB weight: "
                 f"{1-selection['bilstm_weight']:.2f}. Tree settings: {params['loss']}, "
                 f"{params['max_leaf_nodes']} leaves, minimum leaf size {params['min_samples_leaf']}, "
                 "learning rate 0.05, maximum 200 iterations.", ""]
    correction = frozen["correction"]
    before, after = correction["base_validation"], correction["candidate_validation"]
    text += ["## Residual-correction validation", "",
             f"{correction['train_days']} earlier fitting days / one purge day / "
             f"{correction['validation_days']} later validation days. "
             f"Selected iterations: {correction['selected_iterations']}. "
             f"Retained bounds: {correction['bounds']}.", "",
             "| Validation model | RMSE | MAE |", "| --- | ---: | ---: |",
             f"| Uncorrected | {before['rmse']:.5f} | {before['mae']:.5f} |",
             f"| Bounded correction candidate | {after['rmse']:.5f} | {after['mae']:.5f} |", "",
             "## Frozen test-year evaluation", "",
             f"{evaluation['test_period'][0]} to {evaluation['test_period'][1]}, "
             f"{evaluation['test_days']} days. The configuration and correction decision were frozen before scoring.", "",
             "| Test method | RMSE | MAE |", "| --- | ---: | ---: |"]
    text += [f"| {r['method']} | {r['rmse']:.5f} | {r['mae']:.5f} |" for r in rows]
    text += ["", "## Seasonal results", "", "| Method | Season | RMSE | MAE |",
             "| --- | --- | ---: | ---: |"]
    text += [f"| {r['method']} | {r['season']} | {r['rmse']:.5f} | {r['mae']:.5f} |"
             for r in evaluation["metrics"] if r["season"] != "all"]
    text += ["", "## Paired changes of selected final model", "",
             "Negative differences favor the selected model. Intervals use 2,000 paired "
             "circular seven-day block resamples and recompute pooled point metrics.", "",
             "| Reference | Metric | Change | 95% interval |", "| --- | --- | ---: | --- |"]
    for name, comparisons in evaluation["paired_seven_day_block_bootstrap"].items():
        for metric, result in comparisons.items():
            text.append(f"| {name} | {metric} | {result['difference']:.5f} | "
                        f"[{result['ci95_lo']:.5f}, {result['ci95_hi']:.5f}] |")
    final = next(r for r in rows if r["method"] == "selected_final")
    original = next(r for r in rows if r["method"] == "original_corrected")
    uncorrected = next(r for r in rows if r["method"] == "selected_uncorrected")
    def describe_intervals(reference):
        intervals = evaluation["paired_seven_day_block_bootstrap"][reference].values()
        if all(v["ci95_hi"] < 0 for v in intervals):
            return "Both bootstrap intervals favor lower errors."
        if all(v["ci95_lo"] <= 0 <= v["ci95_hi"] for v in intervals):
            return "Both bootstrap intervals include zero; the comparison is inconclusive."
        return "The bootstrap intervals give mixed evidence; inspect each metric separately."
    text += ["", "## Interpretation", "",
             f"Against the previous corrected run, the RMSE reduction is "
             f"{original['rmse']-final['rmse']:.5f} W/m² "
             f"({100*(1-final['rmse']/original['rmse']):.2f}%) and the MAE reduction is "
             f"{original['mae']-final['mae']:.5f} W/m² "
             f"({100*(1-final['mae']/original['mae']):.2f}%). "
             f"{describe_intervals('original_corrected')}", "",
             "Against the completed reference hybrid: " + describe_intervals("reference_hybrid"), "",
             ("Residual correction passed both development validation guards. "
              if correction["accepted"] else "Residual correction was rejected on development validation. ") +
             f"On the test year, "
             f"its RMSE change versus the selected uncorrected model is "
             f"{final['rmse']-uncorrected['rmse']:+.5f} W/m² and its MAE change is "
             f"{final['mae']-uncorrected['mae']:+.5f} W/m². "
             "The development-frozen correction decision was not changed after seeing these test results."]
    target_met = final["rmse"] <= 20 and final["mae"] <= 20
    text += ["", f"Both errors at or below 20 W/m²: **{target_met}**. "
             f"Nominal 80% empirical interval coverage: **{evaluation['interval_coverage_pct']:.2f}%**."]
    text += ["", "## Limits", ""] + [f"- {note}" for note in frozen["limitations"]]
    text += ["- Forecast horizon, units and daylight mask are unchanged. No external data was added.",
             "- Twenty W/m² and single-digit errors were stretch targets, not promised outcomes.",
             "- Existing saved runs and the dashboard default were preserved.", "",
             "Full development candidates: `development_metrics.csv`; frozen decisions: "
             "`frozen_selection.json`; test metrics and bootstrap intervals: `evaluation.json`.", ""]
    (out / "REPORT.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", type=Path, default=C.PREP_PATH)
    parser.add_argument("--out", type=Path, default=C.RESULTS_DIR / "ghi_search")
    parser.add_argument("--phase", choices=("development", "evaluate", "all"), default="development")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dwt", type=Path, default=C.RESULTS_DIR / "scaled_100epochs/dwt_base_pred.npz")
    parser.add_argument("--context", type=Path, default=C.RESULTS_DIR / "context_hgb/dwt_base_pred.npz")
    parser.add_argument("--reference", type=Path, default=C.RESULTS_DIR / "hybrid/hybrid_base_pred.npz")
    parser.add_argument("--test-dwt", type=Path, default=C.RESULTS_DIR / "dwt_10min_150_paper/dwt_base_pred.npz")
    parser.add_argument("--original-corrected", type=Path, default=C.RESULTS_DIR / "dwt_10min_150_paper/dwt_stage2.npz")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment_version": 1, "prep_sha256": fingerprint(args.prep),
                "sources": {k: {"path": str(getattr(args, k).resolve()),
                                  "sha256": fingerprint(getattr(args, k))}
                            for k in ("dwt", "context", "reference", "test_dwt", "original_corrected")},
                "candidates": dict(candidates()), "bilstm_weights": [0, 0.25, 0.5, 0.75],
                "selection": "strictly_lower_RMSE_and_MAE_than_reference", "threads": args.threads}
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Experiment configuration/sources differ; use a fresh output directory")
    elif args.phase == "evaluate":
        raise ValueError("No frozen development experiment exists in this output directory")
    else:
        write_json(manifest_path, manifest)
    with np.load(args.prep, allow_pickle=False) as saved:
        data = {k: saved[k] for k in saved.files}
    base = load_module("ghi_base", "01_base_dwt_bilstm.py")
    with threadpool_limits(limits=args.threads):
        if args.phase in ("development", "all"):
            develop(args, data, base, manifest)
        if args.phase in ("evaluate", "all"):
            evaluate(args, data, base, manifest)


if __name__ == "__main__":
    main()
