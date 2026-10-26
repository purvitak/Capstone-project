"""Bounded NWP-assisted day-ahead experiment with an untouched later guard.

The saved reference already saw these development dates in earlier research.
The new tuning/guard split is chronological; the existing test year is exploratory.
No existing result files or dashboard defaults are modified.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import common as C
import experiment_ghi_search as legacy
from hgb_training import fit_direct_ghi, ghi_to_kt, predict_direct_kt
import nwp_data as N

REFERENCE_DIR = C.RESULTS_DIR / "ghi_search"
PARAMS = {"loss": "squared_error", "learning_rate": .05, "max_iter": 300, "seed": 42}
WEIGHTS = (.25, .5, .75, 1.)
LIMITATIONS = [
    "The existing test year was inspected in earlier project development; results are exploratory.",
    "The saved reference previously used all 324 development-validation dates.",
    "The clear-sky climatology is shared development preprocessing, not refitted in each OOF fold.",
    "NWP availability is a conservative inferred bound, not a recorded publication timestamp.",
    "Hourly NWP is disaggregated to ten minutes; this does not create observed cloud detail."]


def configurations():
    for target, leaves, samples in itertools.product(("direct", "residual"), (15, 31), (40, 100)):
        yield f"{target}_leaves{leaves}_min{samples}", {
            "target": target, **PARAMS, "max_leaf_nodes": leaves, "min_samples_leaf": samples}


def checkpoint_config(manifest_hash, name, config):
    return {"experiment_version": 1, "backend": "hgb", "resolution": "10min",
            "leaky": False, "smoke": False, "name": name, "parameters": config,
            "experiment_manifest_sha256": manifest_hash}


def load_checkpoint(path, data, expected):
    with np.load(path, allow_pickle=False) as saved:
        if json.loads(str(saved["config"])) != expected:
            raise ValueError(f"Checkpoint settings differ: {path}")
        if not np.array_equal(saved["dates"], data["dates"]) or saved["kt_base"].shape != data["ghi"].shape:
            raise ValueError("Checkpoint dates/shape differ")
        return saved["kt_base"].copy()


def fallback_predictions(values, available, fallback):
    """Whole-day fallback, never drop difficult/missing forecast days from scoring."""
    if values.shape != fallback.shape or available.shape != values.shape[:1]:
        raise ValueError("Fallback arrays must align")
    result = fallback.copy()
    if not np.isfinite(values[available]).all():
        raise ValueError("Available forecast days contain nonfinite predictions")
    result[available] = values[available]
    return result


def passes_both(candidate, reference):
    return candidate["rmse"] < reference["rmse"] and candidate["mae"] < reference["mae"]


def evaluation_values(data, reference, development):
    """Preserve selected OOF predictions when adding frozen test predictions."""
    values = development.copy()
    values[data["test_idx"]] = reference[data["test_idx"]]
    return values


def unfinished_candidates(out, data, nwp, manifest_hash):
    """A training log alone does not prove that every OOF fold completed."""
    forecast_days = np.concatenate([days for _, days in outer_folds(data, nwp["available"])])
    pending = []
    for name, config in configurations():
        path = out / name / "dwt_base_pred.npz"
        if not path.exists():
            pending.append(name)
            continue
        values = load_checkpoint(path, data, checkpoint_config(manifest_hash, name, config))
        if not np.isfinite(values[forecast_days]).all():
            pending.append(name)
    return pending


def select_on_tuning(rows, reference):
    eligible = [r for r in rows if passes_both(r, reference)]
    return min(eligible, key=lambda r: (r["rmse"], r["mae"], r["candidate"], r["weight"])) if eligible else None


def assemble_features(data, nwp, base):
    days = np.concatenate((data["dev_idx"], [int(data["purge_idx"])], data["test_idx"]))
    history = np.zeros(data["ghi"].shape + (32,), dtype=np.float32)
    block = legacy.feature_days(data, days, base)
    if block.shape[-1] != history.shape[-1]:
        raise ValueError("Historical feature contract changed")
    history[days] = block
    index = np.divide(nwp["ghi"], data["cs_proxy"], out=np.zeros_like(nwp["ghi"]),
                      where=data["cs_proxy"] > C.DAYLIGHT_CS_MIN)
    enhanced = np.concatenate((history, nwp["features"], index[..., None]), axis=-1).astype(np.float32)
    return history, enhanced


def load_reference(data, prep_hash, folder):
    base, _ = legacy.load_predictions(folder / "selected_development.npz", data, prep_hash)
    reference = base.copy()
    with np.load(folder / "correction_validation.npz", allow_pickle=False) as val:
        days = val["days"].copy()
        if days.shape != (324,) or np.any(np.diff(days) <= 0) or not np.isin(days, data["dev_idx"]).all():
            raise ValueError("Expected the original 324 chronological development validation days")
        reference[days] = val["kt_final"]
    if not np.isfinite(reference[days]).all():
        raise ValueError("Reference does not cover all validation slots")
    return base, reference, days


def outer_folds(data, available):
    dev = data["dev_idx"]
    eligible = dev[available[dev]]
    if len(eligible) < 181:
        raise ValueError("Need at least 180 complete forecast-input fitting days plus prediction days")
    dates = data["dates"].astype("datetime64[D]")
    first = dates[eligible[179]] + np.timedelta64(2, "D")
    span = dev[dates[dev] >= first]
    folds = []
    for start in range(0, len(span), 90):
        predict = span[start:start + 90]
        train = eligible[dates[eligible] < dates[predict[0]] - np.timedelta64(1, "D")]
        if len(train) < 180:
            raise ValueError("Outer chronological purge violated initial training requirement")
        folds.append((train, predict))
    return folds


def fit_candidate(name, config, data, nwp, x, fallback, folds, out, manifest_hash, budget):
    folder = out / name
    folder.mkdir(exist_ok=True)
    path, log_path = folder / "dwt_base_pred.npz", folder / "training.json"
    contract = checkpoint_config(manifest_hash, name, config)
    if path.exists():
        pred = load_checkpoint(path, data, contract)
        if not log_path.exists():
            raise ValueError("Checkpoint missing training provenance")
        log = json.loads(log_path.read_text())["folds"]
    else:
        pred, log = np.full_like(data["ghi"], np.nan), []
    params = {k:v for k,v in config.items() if k != "target"}
    for train, predict in folds:
        if np.isfinite(pred[predict]).all():
            continue
        budget.check(reserve=90)
        started = time.monotonic()
        good = predict[nwp["available"][predict]]
        block = fallback[predict].copy()
        info = {"selected_iterations": None}
        if len(good):
            offset = nwp["ghi"][train] if config["target"] == "residual" else None
            model, info = fit_direct_ghi(x[train], data["ghi"][train], data["cs_proxy"][train],
                                         data["mask"][train], baseline_ghi=offset, **params)
            predicted = predict_direct_kt(model, x[good], data["cs_proxy"][good],
                                          baseline_ghi=nwp["ghi"][good] if offset is not None else None)
            block[np.isin(predict, good)] = predicted
        pred[predict] = block
        cut = int(len(train) * .8)
        entry = {"first": str(data["dates"][predict[0]]), "last": str(data["dates"][predict[-1]]),
                 "first_train_date": str(data["dates"][train[0]]),
                 "last_train_date": str(data["dates"][train[-1]]),
                 "inner_last_fit_date": str(data["dates"][train[cut - 2]]),
                 "inner_first_validation_date": str(data["dates"][train[cut]]),
                 "train_days": len(train), "forecast_days": len(good),
                 "fallback_days": len(predict) - len(good), "training": info,
                 "seconds": time.monotonic() - started}
        log = [r for r in log if r["first"] != entry["first"]] + [entry]
        N.write_json(log_path, {"config": contract, "folds": log})
        legacy.save_predictions(path, pred, data["dates"], contract)
        print(f"[{name}] {entry['first']}..{entry['last']}: "
              f"iterations={info['selected_iterations']}, {entry['seconds']:.1f}s", flush=True)
    return pred


def check_hashes(out, expected):
    for filename, digest in expected.items():
        if N.sha256(out / filename) != digest:
            raise ValueError(f"Frozen artifact changed: {filename}")


def develop(args, data, nwp, base, manifest, budget):
    out = args.out
    if (out / "frozen_selection.json").exists():
        print("[development] Already frozen; search skipped", flush=True)
        return
    historical, reference, validation = load_reference(data, manifest["prep_sha256"], args.reference)
    tuning, guard = validation[:214], validation[214:]
    folds = outer_folds(data, nwp["available"])
    forecast_days = np.concatenate([p for _,p in folds])
    if not np.isin(validation, forecast_days).all():
        raise ValueError("NWP archive/warmup does not cover the complete 324-day development comparison")
    history, enhanced = assemble_features(data, nwp, base)
    manifest_hash = N.sha256(out / "manifest.json")
    baseline_config = {"target": "history_only", **PARAMS, "max_leaf_nodes": 15, "min_samples_leaf": 100}
    history_pred = fit_candidate("history_only", baseline_config, data, nwp, history, reference,
                                 folds, out, manifest_hash, budget)
    trials = {}
    for name, config in configurations():
        trials[name] = fit_candidate(name, config, data, nwp, enhanced, reference,
                                     folds, out, manifest_hash, budget)
    budget.check(reserve=75)
    tuning_path = out / "tuning_selection.json"
    if tuning_path.exists():
        chosen = json.loads(tuning_path.read_text())
        if chosen["manifest_sha256"] != manifest_hash:
            raise ValueError("Tuning selection belongs to a different manifest")
        check_hashes(out, chosen["candidate_hashes"])
    else:
        rows = []
        reference_score = legacy.metrics(data, reference, tuning)
        for name, pred in trials.items():
            for weight in WEIGHTS:
                combined = legacy.blend(pred, historical, weight)
                # An absent NWP day falls back to the corrected reference,
                # independently of ensemble weight.
                combined[~nwp["available"]] = reference[~nwp["available"]]
                rows.append({"candidate": name, "weight": weight,
                             **legacy.metrics(data, combined, tuning)})
        pd.DataFrame(rows).to_csv(out / "tuning_metrics.csv", index=False)
        chosen = {"selection": select_on_tuning(rows, reference_score),
                  "reference_tuning": reference_score, "tuning_days": tuning.tolist(),
                  "guard_days": guard.tolist(), "manifest_sha256": manifest_hash,
                  "candidate_hashes": {f"{name}/dwt_base_pred.npz": N.sha256(out / name / "dwt_base_pred.npz")
                                       for name in trials}, "test_scored": False}
        # This write precedes every guard score, so guard results cannot rank trials.
        N.write_json(tuning_path, chosen)
    selection = chosen["selection"]
    values = reference.copy()
    guard_reference = legacy.metrics(data, reference, guard)
    accepted = False
    guard_score = guard_reference.copy()
    if selection is not None:
        trial = legacy.blend(trials[selection["candidate"]], historical, selection["weight"])
        trial[~nwp["available"]] = reference[~nwp["available"]]
        guard_score = legacy.metrics(data, trial, guard)
        accepted = passes_both(guard_score, guard_reference)
        if accepted:
            values[forecast_days] = trial[forecast_days]
    values[data["test_idx"]] = np.nan
    values[int(data["purge_idx"])]=np.nan
    config = {"model": "nwp_assisted_hgb" if accepted else "retained_reference",
              "resolution": "10min", "leaky": False, "smoke": False,
              "prep_sha256": manifest["prep_sha256"], "manifest_sha256": manifest_hash,
              "selected_on_tuning": selection, "guard_accepted": bool(accepted)}
    legacy.save_predictions(out / "selected_development.npz", values, data["dates"], config)
    raw = fallback_predictions(ghi_to_kt(nwp["ghi"], data["cs_proxy"]), nwp["available"], reference)
    baseline_rows = []
    for name, pred in (("current_corrected_reference", reference), ("raw_nwp", raw), ("history_only", history_pred)):
        for split, days in (("tuning", tuning), ("guard", guard)):
            baseline_rows.extend(legacy.seasonal_rows(data, pred, days, name, split=split))
    pd.DataFrame(baseline_rows).to_csv(out / "development_baselines.csv", index=False)
    N.write_json(out / "frozen_selection.json", {
        "selection": selection, "accepted": bool(accepted),
        "decision": ("promote_nwp_candidate" if accepted else
                     "retain_reference_no_tuning_improvement" if selection is None else "retain_reference_guard_failed"),
        "guard_candidate": guard_score, "guard_reference": guard_reference,
        "guard_first": str(data["dates"][guard[0]]), "guard_last": str(data["dates"][guard[-1]]),
        "guard_days": len(guard), "test_scored": False, "manifest_sha256": manifest_hash,
        "artifact_hashes": {"selected_development.npz": N.sha256(out / "selected_development.npz"),
                            "tuning_selection.json": N.sha256(tuning_path),
                            "history_only/dwt_base_pred.npz": N.sha256(out / "history_only/dwt_base_pred.npz"),
                            **chosen["candidate_hashes"]},
        "limitations": LIMITATIONS})
    print(f"[frozen] {selection}; decision={'promote' if accepted else 'retain reference'}", flush=True)


def load_test_reference(args, data, prep_hash):
    with np.load(args.reference / "selected_base_pred.npz", allow_pickle=False) as b, \
            np.load(args.reference / "selected_stage2.npz", allow_pickle=False) as s:
        if not (np.array_equal(data["dates"], b["dates"]) and np.array_equal(data["dates"], s["dates"])):
            raise ValueError("Test reference dates differ")
        for source in (b, s):
            config = json.loads(str(source["config"]))
            if config.get("prep_sha256") != prep_hash or config.get("leaky") or config.get("smoke"):
                raise ValueError("Test reference provenance differs")
        return b["kt_base"].copy(), np.where(np.isfinite(s["kt_final"]), s["kt_final"], b["kt_base"])


def evaluate(args, data, nwp, base, manifest, budget):
    out = args.out
    freeze_path = out / "frozen_selection.json"
    frozen = json.loads(freeze_path.read_text())
    if frozen["manifest_sha256"] != N.sha256(out / "manifest.json"):
        raise ValueError("Frozen manifest changed")
    check_hashes(out, frozen["artifact_hashes"])
    frozen_hash = N.sha256(freeze_path)
    if (out / "evaluation.json").exists():
        report = json.loads((out / "evaluation.json").read_text())
        if report["frozen_selection_sha256"] != frozen_hash:
            raise ValueError("Evaluation belongs to another frozen selection")
        if report["prediction_sha256"] != N.sha256(out / "selected_base_pred.npz"):
            raise ValueError("Evaluated prediction checksum differs")
        write_report(out, frozen, report)
        print("[evaluation] Already complete; no test rescoring", flush=True)
        return
    budget.check(reserve=70)
    historical, reference = load_test_reference(args, data, manifest["prep_sha256"])
    test = data["test_idx"]
    if len(test) != 365 or data["mask"][test].sum() != 24086:
        raise ValueError("Expected the unchanged 365-day / 24,086-point evaluation")
    history, enhanced = assemble_features(data, nwp, base)
    train = data["dev_idx"][nwp["available"][data["dev_idx"]]]
    predict = test[nwp["available"][test]]
    development, _ = legacy.load_predictions(out / "selected_development.npz", data, manifest["prep_sha256"])
    values = evaluation_values(data, reference, development)
    final_training = None
    if frozen["accepted"]:
        selected = frozen["selection"]
        config = dict(configurations())[selected["candidate"]]
        offset = nwp["ghi"][train] if config["target"] == "residual" else None
        model, final_training = fit_direct_ghi(
            enhanced[train], data["ghi"][train], data["cs_proxy"][train], data["mask"][train],
            baseline_ghi=offset, **{k:v for k,v in config.items() if k != "target"})
        if len(predict):
            block = predict_direct_kt(model, enhanced[predict], data["cs_proxy"][predict],
                                      baseline_ghi=nwp["ghi"][predict] if offset is not None else None)
            values[predict] = legacy.blend(block, historical[predict], selected["weight"])
    budget.check(reserve=30)
    baseline, history_info = fit_direct_ghi(history[train], data["ghi"][train], data["cs_proxy"][train],
                                            data["mask"][train], **PARAMS, max_leaf_nodes=15, min_samples_leaf=100)
    history_values = reference.copy()
    if len(predict):
        history_values[predict] = predict_direct_kt(baseline, history[predict], data["cs_proxy"][predict])
    raw = fallback_predictions(ghi_to_kt(nwp["ghi"], data["cs_proxy"]), nwp["available"], reference)
    methods = {"current_corrected_reference": reference, "raw_nwp_with_fallback": raw,
               "history_only_matched_training": history_values,
               "nwp_selected_final": values, "mean3": C.baseline_mean3_kt(data["kt"])}
    rows = []
    for name, pred in methods.items():
        for row in legacy.seasonal_rows(data, pred, test, name):
            idx = test if row["season"] == "all" else test[(data["is_monsoon"][test] if row["season"] == "monsoon" else ~data["is_monsoon"][test])]
            ref_score = legacy.metrics(data, reference, idx)["rmse"]
            mean_score = legacy.metrics(data, methods["mean3"], idx)["rmse"]
            rows.append({**row, "skill_vs_reference_pct": 100*(1-row["rmse"]/ref_score),
                         "skill_vs_mean3_pct": 100*(1-row["rmse"]/mean_score)})
    pairs = {name: legacy.paired_bootstrap(data, values, pred, test)
             for name,pred in methods.items() if name != "nwp_selected_final"}
    lead_rows = []
    for hour in range(24):
        mask = data["mask"][test].copy()
        mask[:, :hour*6] = False
        mask[:, (hour+1)*6:] = False
        if not mask.any():
            continue
        for name, pred in methods.items():
            lead_rows.append({"method": name, "lead_hour": hour+1,
                              **C.point_metrics(data["ghi"][test], np.clip(pred[test],*C.KT_CLIP)*data["cs_proxy"][test],mask)})
    selected_metrics = next(r for r in rows if r["method"] == "nwp_selected_final" and r["season"] == "all")
    report = {"frozen_selection_sha256": frozen_hash, "decision": frozen["decision"],
              "test_period": [str(data["dates"][test[0]]),str(data["dates"][test[-1]])],
              "test_days": len(test), "daylight_samples": int(data["mask"][test].sum()),
              "forecast_input_days": len(predict), "fallback_days": len(test)-len(predict),
              "fallback_dates": data["dates"][test[~nwp["available"][test]]].tolist(),
              "coverage_pct": 100*len(predict)/len(test), "metrics": rows,
              "final_training": final_training, "history_only_training": history_info,
              "target_both_below_50": bool(selected_metrics["rmse"] < 50 and selected_metrics["mae"] < 50),
              "paired_seven_day_block_bootstrap": pairs, "bootstrap_repeats":2000,"bootstrap_seed":42,
              "elapsed_seconds_this_invocation": budget.elapsed, "limitations": LIMITATIONS}
    output_config = {"resolution":"10min","model":"nwp_assisted_hgb" if frozen["accepted"] else "retained_reference","smoke":False,"leaky":False,
                     "prep_sha256":manifest["prep_sha256"],"frozen_selection_sha256":frozen_hash,
                     "decision":frozen["decision"], "parameters":frozen["selection"],
                     "model_parameters":dict(configurations())[frozen["selection"]["candidate"]] if frozen["accepted"] else None,
                     "final_training":final_training, "weather_cache_sha256":manifest["weather_cache_sha256"]}
    legacy.save_predictions(out / "selected_base_pred.npz", values, data["dates"], output_config)
    report["prediction_sha256"] = N.sha256(out / "selected_base_pred.npz")
    pd.DataFrame(rows).to_csv(out / "test_metrics.csv",index=False)
    pd.DataFrame(lead_rows).to_csv(out / "lead_time.csv",index=False)
    N.write_json(out / "evaluation.json",report)
    write_report(out,frozen,report)
    print(pd.DataFrame(rows).query("season == 'all'")[["method","rmse","mae"]].to_string(index=False),flush=True)


def write_report(out, frozen, report):
    rows = [r for r in report["metrics"] if r["season"] == "all"]
    tuning = json.loads((out / "tuning_selection.json").read_text())
    selected = frozen["selection"]
    lines = ["# NWP-assisted irradiance optimization", "",
             "Native ten-minute daylight GHI in W/m²; midnight IST issue time is unchanged.", "",
             f"Decision: **{report['decision']}**. Target: both errors below 50 W/m². "
             f"Target reached: **{report['target_both_below_50']}**.", "",
             "## Development selection", "",
             "Eight HGB configurations; first 214 validation days select the candidate and "
             "blend, then the remaining 110 days check that one choice. Both errors must "
             "improve in both splits. The saved reference previously saw these dates.", "",
             (f"Selected candidate: `{selected['candidate']}`, blended at new-model weight "
              f"{selected['weight']:.2f} with the saved uncorrected hybrid." if selected else "No eligible new candidate."), "",
             "| Comparison | RMSE | MAE |", "| --- | ---: | ---: |",
             f"| Tuning reference | {tuning['reference_tuning']['rmse']:.5f} | {tuning['reference_tuning']['mae']:.5f} |"]
    if selected:
        lines.append(f"| Tuning candidate | {selected['rmse']:.5f} | {selected['mae']:.5f} |")
    for name,key in (("Later guard reference","guard_reference"),("Later guard candidate","guard_candidate")):
        value=frozen[key]
        lines.append(f"| {name} | {value['rmse']:.5f} | {value['mae']:.5f} |")
    lines += ["", "## Frozen test-year results", "",
              f"{report['test_period'][0]} through {report['test_period'][1]}: "
              f"{report['test_days']} days / {report['daylight_samples']} scored daylight samples.", "",
              f"Forecast input coverage: **{report['coverage_pct']:.2f}%**; "
              f"fallback days: **{report['fallback_days']}**. Missing-input days use the saved "
              "corrected reference and remain in all reported full-period metrics.", "",
              "| Method | RMSE | MAE | RMSE skill vs current reference |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {r['method']} | {r['rmse']:.5f} | {r['mae']:.5f} | {r['skill_vs_reference_pct']:.2f}% |" for r in rows]
    lines += ["", "## Seasonal performance", "", "| Method | Season | RMSE | MAE |", "| --- | --- | ---: | ---: |"]
    lines += [f"| {r['method']} | {r['season']} | {r['rmse']:.5f} | {r['mae']:.5f} |" for r in report['metrics'] if r['season']!='all']
    lines += ["", "## Paired seven-day block bootstrap", "", "Negative change favors the selected pipeline. "
              "Intervals recompute pooled errors over 2,000 paired circular block resamples.", "",
              "| Reference | Metric | Change | 95% interval |", "| --- | --- | ---: | --- |"]
    for name,comparisons in report['paired_seven_day_block_bootstrap'].items():
        for metric,result in comparisons.items():
            lines.append(f"| {name} | {metric} | {result['difference']:.5f} | [{result['ci95_lo']:.5f}, {result['ci95_hi']:.5f}] |")
    final = next(r for r in rows if r["method"] == "nwp_selected_final")
    lines += ["", f"Remaining gap to 50: **{max(0,final['rmse']-50):.2f} W/m² RMSE** and "
              f"**{max(0,final['mae']-50):.2f} W/m² MAE**. All eight candidates and their OOF folds completed.",
              "", "## Inputs and limits", "",
              "Open-Meteo GFS Global fixed previous-day-2 forecasts: 48-hour offsets plus "
              "a conservative 12-hour availability buffer. Availability is inferred, not a "
              "record of exact publication. Hourly GHI is clear-sky-weighted into ten-minute "
              "values with energy preserved before the existing kt clipping. Other fields "
              "are interpolated to period midpoints. Original API responses, queries and "
              "checksums are cached separately from the unchanged prepared data.", "",
              "Source: https://open-meteo.com/en/docs/previous-runs-api; attribution: Open-Meteo / NOAA GFS (CC BY 4.0).", ""]
    lines += [f"- {value}" for value in LIMITATIONS]
    lines += ["- No new residual-correction stage or uncertainty-calibration claim was added.",
              "- Existing frozen results and dashboard defaults are unchanged.", "",
              "See `tuning_metrics.csv`, `development_baselines.csv`, `frozen_selection.json`, "
              "`test_metrics.csv`, `lead_time.csv`, and `evaluation.json` for full-precision evidence. "
              "`verification.json` records the regression and artifact audit for the completed implementation.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("acquire","development","evaluate","all"), default="development")
    parser.add_argument("--prep",type=Path,default=C.PREP_PATH)
    parser.add_argument("--cache",type=Path,default=N.DEFAULT_CACHE)
    parser.add_argument("--reference",type=Path,default=REFERENCE_DIR)
    parser.add_argument("--out",type=Path,default=C.RESULTS_DIR / "nwp_optimization")
    parser.add_argument("--max-seconds",type=float,default=3600)
    parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args()
    if args.threads<1 or args.max_seconds<=0:
        parser.error("Threads and runtime budget must be positive")
    budget=N.Budget(args.max_seconds)
    args.out.mkdir(parents=True,exist_ok=True)
    try:
        if args.phase in ("acquire","all"):
            nwp,weather_manifest=N.prepare(args.cache,args.prep,budget)
        else:
            nwp,weather_manifest=N.load_forecasts(args.cache,args.prep)
        if args.phase=="acquire":
            N.write_json(args.out / "acquisition.json",weather_manifest)
            return
        with np.load(args.prep,allow_pickle=False) as saved:
            data={k:saved[k] for k in saved.files}
        if not np.array_equal(data["dates"],nwp["dates"]) or nwp["features"].shape[:2]!=data["ghi"].shape:
            raise ValueError("Forecast cache dates/shape differ from prepared data")
        sources=("selected_development.npz","correction_validation.npz","selected_base_pred.npz","selected_stage2.npz","frozen_selection.json")
        manifest={"experiment_version":1,"prep_sha256":N.sha256(args.prep),
                  "weather_cache_sha256":N.sha256(args.cache / "forecasts.npz"),
                  "weather_manifest_sha256":N.sha256(args.cache / "manifest.json"),
                  "sources":{name:N.sha256(args.reference / name) for name in sources},
                  "configurations":dict(configurations()),"weights":list(WEIGHTS),
                  "initial_complete_training_days":180,"fold_calendar_days":90,
                  "outer_purge_calendar_days":1,"tuning_days":214,"guard_days":110,
                  "threads":args.threads,"protocol":N.PROTOCOL}
        path=args.out / "manifest.json"
        if path.exists():
            if json.loads(path.read_text())!=manifest:
                raise ValueError("Experiment sources/configuration differ; use a fresh output folder")
        elif args.phase=="evaluate":
            raise ValueError("No frozen development experiment exists")
        else:
            N.write_json(path,manifest)
        base=legacy.load_module("nwp_base","01_base_dwt_bilstm.py")
        with threadpool_limits(limits=args.threads):
            if args.phase in ("development","all"):
                develop(args,data,nwp,base,manifest,budget)
            if args.phase in ("evaluate","all"):
                evaluate(args,data,nwp,base,manifest,budget)
        N.write_json(args.out / "run_status.json",{"phase":args.phase,"complete":True,"elapsed_seconds":budget.elapsed})
    except N.BudgetExpired as exc:
        pending = (unfinished_candidates(args.out, data, nwp, N.sha256(args.out / "manifest.json"))
                   if "data" in locals() and (args.out / "manifest.json").exists()
                   else [name for name,_ in configurations()])
        N.write_json(args.out / "run_status.json",{"phase":args.phase,"complete":False,"elapsed_seconds":budget.elapsed,
                                                  "reason":str(exc),"resume_supported":True,
                                                  "unfinished_candidates":pending})
        print(f"[incomplete] {exc}",flush=True)
        raise SystemExit(2)


if __name__=="__main__":
    main()
