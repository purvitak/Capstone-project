"""Isolated residual-XGBoost experiment; reuse history/NWP, never retrain old models.

python pipelines/xgboost_nwp/run.py --phase all
Selection is development-only. The already-exposed test year is exploratory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "pv_pipeline"))
sys.path.insert(0, str(HERE))
import common as C
import experiment_ghi_search as L
import experiment_nwp as E
import nwp_data as N
from training import fit_residual, predict_kt

WEIGHTS = (.25, .5, .75, 1.)
MAX_ROUNDS = 300
LIMITATIONS = E.LIMITATIONS + [
    "Both the NWP-HGB incumbent and this experiment reuse the same development ranking/guard dates.",
    "XGBoost min_child_weight is a Hessian threshold; this is an approximate complexity match to HGB minimum leaf size.",
    "A failed development guard retains the incumbent even if the later exploratory trial test score is better."]


def configurations(threads=2):
    for leaves in (15, 31):
        for child in (40, 100):
            yield f"residual_leaves{leaves}_child{child}", {
                "objective":"reg:squarederror", "tree_method":"hist", "grow_policy":"lossguide",
                "max_depth":0, "max_leaves":leaves, "min_child_weight":child, "max_bin":255,
                "eta":.05, "subsample":1., "colsample_bytree":1., "lambda":1., "alpha":0.,
                "gamma":0., "seed":42, "nthread":threads, "device":"cpu", "verbosity":0}


def contract(manifest_hash, name, params):
    return {"experiment_version":1, "backend":"xgboost", "target":"ghi_residual_to_nwp",
            "resolution":"10min", "leaky":False, "smoke":False, "name":name,
            "parameters":params, "max_rounds":MAX_ROUNDS, "manifest_sha256":manifest_hash}


def combine(trial, historical, available, fallback, weight):
    values = L.blend(trial, historical, weight)
    values[~available] = fallback[~available]
    return values


def rank_tuning(rows, reference):
    """Freeze one trial even if ineligible; never disguise a failed trial as incumbent."""
    eligible = [r for r in rows if E.passes_both(r, reference)]
    chosen = min(eligible or rows, key=lambda r:(r["rmse"], r["mae"], r["candidate"], r["weight"]))
    return chosen, bool(eligible)


def get_features(data, nwp):
    base = L.load_module("xgboost_historical_features", "01_base_dwt_bilstm.py")
    return E.assemble_features(data, nwp, base)[1]


def fit_candidate(args, data, nwp, features, fallback, folds, name, params, manifest_hash, budget):
    folder = args.out/name
    folder.mkdir(exist_ok=True)
    path, logpath = folder/"oof.npz", folder/"training.json"
    expected = contract(manifest_hash, name, params)
    if path.exists():
        values = E.load_checkpoint(path, data, expected)
        logged = json.loads(logpath.read_text())
        if logged["config"] != expected:
            raise ValueError("Training log/checkpoint contract differs")
        log = logged["folds"]
    else:
        values, log = np.full_like(data["ghi"], np.nan), []
    for train, days in folds:
        if np.isfinite(values[days]).all():
            continue
        budget.check(reserve=30)
        started = time.monotonic()
        good = days[nwp["available"][days]]
        block = fallback[days].copy()
        info = None
        if len(good):
            model, info = fit_residual(features[train], data["ghi"][train], data["cs_proxy"][train],
                                       data["mask"][train], nwp["ghi"][train], params, MAX_ROUNDS, budget)
            block[np.isin(days, good)] = predict_kt(model, features[good], data["cs_proxy"][good], nwp["ghi"][good])
        values[days] = block
        cut = int(len(train)*.8)
        entry = {"first":str(data["dates"][days[0]]), "last":str(data["dates"][days[-1]]),
                 "first_train_date":str(data["dates"][train[0]]), "last_train_date":str(data["dates"][train[-1]]),
                 "inner_last_fit_date":str(data["dates"][train[cut-2]]),
                 "inner_first_validation_date":str(data["dates"][train[cut]]),
                 "train_days":len(train), "forecast_days":len(good), "fallback_days":len(days)-len(good),
                 "training":info, "seconds":time.monotonic()-started}
        log = [r for r in log if r["first"] != entry["first"]]+[entry]
        N.write_json(logpath, {"config":expected,"folds":log})
        L.save_predictions(path, values, data["dates"], expected)
        print(f"[{name}] {entry['first']}..{entry['last']}: "
              f"iterations={info['selected_iterations'] if info else None}, {entry['seconds']:.1f}s", flush=True)
    return values


def develop(args, data, nwp, manifest, budget):
    if (args.out/"frozen_selection.json").exists():
        print("[development] Already frozen; no new selection", flush=True)
        return
    historical, fallback, validation = E.load_reference(data, manifest["prep_sha256"], args.history)
    incumbent, _ = L.load_predictions(args.incumbent/"selected_development.npz", data, manifest["prep_sha256"])
    tuning, guard = validation[:214], validation[214:]
    folds = E.outer_folds(data, nwp["available"])
    days = np.concatenate([p for _,p in folds])
    if not np.isin(validation, days).all():
        raise ValueError("Incomplete development comparison coverage")
    features = get_features(data, nwp)
    manifest_hash = N.sha256(args.out/"manifest.json")
    trials = {name:fit_candidate(args, data, nwp, features, fallback, folds, name, params, manifest_hash, budget)
              for name,params in configurations(args.threads)}
    if any(not np.isfinite(values[days]).all() for values in trials.values()):
        raise ValueError("Incomplete candidates cannot enter selection")
    budget.check(reserve=25)
    path = args.out/"tuning_selection.json"
    hashes = {f"{name}/oof.npz":N.sha256(args.out/name/"oof.npz") for name in trials}
    if path.exists():
        tuning_choice = json.loads(path.read_text())
        if tuning_choice["manifest_sha256"] != manifest_hash or tuning_choice["candidate_hashes"] != hashes:
            raise ValueError("Frozen tuning inputs differ")
    else:
        rows = []
        for name, values in trials.items():
            for weight in WEIGHTS:
                blend = combine(values, historical, nwp["available"], fallback, weight)
                rows.append({"candidate":name,"weight":weight,**L.metrics(data, blend, tuning)})
        pd.DataFrame(rows).to_csv(args.out/"tuning_metrics.csv",index=False)
        ref = L.metrics(data, incumbent, tuning)
        selected, eligible = rank_tuning(rows, ref)
        tuning_choice = {"selection":selected, "eligible_vs_incumbent":eligible,
                         "incumbent_tuning":ref,"tuning_days":tuning.tolist(),"guard_days":guard.tolist(),
                         "manifest_sha256":manifest_hash,"candidate_hashes":hashes,"test_scored":False}
        # This file is written before any guard or test score is computed.
        N.write_json(path,tuning_choice)
    selected = tuning_choice["selection"]
    blend = combine(trials[selected["candidate"]], historical, nwp["available"], fallback, selected["weight"])
    guard_trial, guard_incumbent = L.metrics(data, blend, guard), L.metrics(data, incumbent, guard)
    accepted = tuning_choice["eligible_vs_incumbent"] and E.passes_both(guard_trial, guard_incumbent)
    blend[data["test_idx"]] = np.nan
    blend[int(data["purge_idx"])]=np.nan
    metadata = {"model":"xgboost_nwp_trial", "backend":"xgboost", "resolution":"10min", "leaky":False,
                "smoke":False,"prep_sha256":manifest["prep_sha256"],"manifest_sha256":manifest_hash,
                "parameters":selected,"passed_development_checks":bool(accepted)}
    L.save_predictions(args.out/"trial_development.npz",blend,data["dates"],metadata)
    baseline_rows=[]
    for name,values in (("nwp_hgb_incumbent",incumbent),("historical_corrected",fallback),("xgboost_trial",blend)):
        for split,idx in (("tuning",tuning),("guard",guard)):
            baseline_rows.extend(L.seasonal_rows(data,values,idx,name,split=split))
    pd.DataFrame(baseline_rows).to_csv(args.out/"development_metrics.csv",index=False)
    N.write_json(args.out/"frozen_selection.json",{
        "selection":selected,"eligible_on_tuning":tuning_choice["eligible_vs_incumbent"],
        "passed_development_checks":bool(accepted),
        "decision":"xgboost_passed_development_checks" if accepted else "retain_nwp_hgb_incumbent",
        "guard_trial":guard_trial,"guard_incumbent":guard_incumbent,
        "guard_first":str(data["dates"][guard[0]]),"guard_last":str(data["dates"][guard[-1]]),
        "manifest_sha256":manifest_hash,"test_scored":False,
        "artifact_hashes":{**hashes,"tuning_selection.json":N.sha256(path),
                           "trial_development.npz":N.sha256(args.out/"trial_development.npz")},
        "limitations":LIMITATIONS})
    print(f"[frozen] {selected}; passed development checks={accepted}",flush=True)


def evaluate(args,data,nwp,manifest,budget):
    out=args.out
    frozen=json.loads((out/"frozen_selection.json").read_text())
    frozen_hash=N.sha256(out/"frozen_selection.json")
    if frozen["manifest_sha256"]!=N.sha256(out/"manifest.json"):
        raise ValueError("Frozen manifest changed")
    E.check_hashes(out,frozen["artifact_hashes"])
    if (out/"evaluation.json").exists():
        report=json.loads((out/"evaluation.json").read_text())
        if report["frozen_selection_sha256"]!=frozen_hash:
            raise ValueError("Evaluation belongs to different frozen settings")
        E.check_hashes(out,report["artifact_hashes"])
        write_report(out,frozen,report)
        print("[evaluation] Already complete; no test rescoring",flush=True)
        return
    budget.check(reserve=30)
    test=data["test_idx"]
    if len(test)!=365 or data["mask"][test].sum()!=24086:
        raise ValueError("Expected unchanged 365-day / 24,086-sample evaluation")
    historical,fallback=E.load_test_reference(argparse.Namespace(reference=args.history),data,manifest["prep_sha256"])
    incumbent,_=L.load_predictions(args.incumbent/"selected_base_pred.npz",data,manifest["prep_sha256"])
    values,_=L.load_predictions(out/"trial_development.npz",data,manifest["prep_sha256"])
    values=E.evaluation_values(data,fallback,values)
    selected=frozen["selection"]
    features=get_features(data,nwp)
    train=data["dev_idx"][nwp["available"][data["dev_idx"]]]
    good=test[nwp["available"][test]]
    params=dict(configurations(args.threads))[selected["candidate"]]
    model,info=fit_residual(features[train],data["ghi"][train],data["cs_proxy"][train],data["mask"][train],
                             nwp["ghi"][train],params,MAX_ROUNDS,budget)
    if len(good):
        block=predict_kt(model,features[good],data["cs_proxy"][good],nwp["ghi"][good])
        values[good]=L.blend(block,historical[good],selected["weight"])
    model.save_model(out/"final_model.tmp.json")
    (out/"final_model.tmp.json").replace(out/"final_model.json")
    N.write_json(out/"final_training.json",info)
    metadata={"model":"xgboost_nwp_trial","backend":"xgboost","resolution":"10min","leaky":False,"smoke":False,
              "prep_sha256":manifest["prep_sha256"],"weather_cache_sha256":manifest["weather_cache_sha256"],
              "frozen_selection_sha256":frozen_hash,"xgboost_version":xgb.__version__,
              "parameters":selected,"final_training":info,"decision":frozen["decision"]}
    L.save_predictions(out/"xgboost_base_pred.npz",values,data["dates"],metadata)
    raw=E.fallback_predictions(E.ghi_to_kt(nwp["ghi"],data["cs_proxy"]),nwp["available"],fallback)
    methods={"nwp_hgb_incumbent":incumbent,"historical_corrected":fallback,
             "raw_nwp":raw,"xgboost_trial":values,"mean3":C.baseline_mean3_kt(data["kt"])}
    rows=[]
    for name,pred in methods.items():
        for row in L.seasonal_rows(data,pred,test,name):
            idx=test if row["season"]=="all" else test[data["is_monsoon"][test] if row["season"]=="monsoon" else ~data["is_monsoon"][test]]
            rows.append({**row,"rmse_skill_vs_incumbent_pct":100*(1-row["rmse"]/L.metrics(data,incumbent,idx)["rmse"])})
    budget.check(reserve=5)
    paired={name:L.paired_bootstrap(data,values,pred,test) for name,pred in methods.items() if name!="xgboost_trial"}
    leads=[]
    for hour in range(24):
        mask=data["mask"][test].copy();mask[:,:hour*6]=False;mask[:,(hour+1)*6:]=False
        if mask.any():
            for name,pred in methods.items():
                leads.append({"method":name,"lead_hour":hour+1,
                              **C.point_metrics(data["ghi"][test],np.clip(pred[test],*C.KT_CLIP)*data["cs_proxy"][test],mask)})
    trial=next(r for r in rows if r["method"]=="xgboost_trial" and r["season"]=="all")
    pd.DataFrame(rows).to_csv(out/"test_metrics.csv",index=False)
    pd.DataFrame(leads).to_csv(out/"lead_time.csv",index=False)
    report={"decision":frozen["decision"],"passed_development_checks":frozen["passed_development_checks"],
            "frozen_selection_sha256":frozen_hash,"trial_scored_even_if_guard_failed":True,
            "test_period":[str(data["dates"][test[0]]),str(data["dates"][test[-1]])],
            "test_days":len(test),"daylight_samples":int(data["mask"][test].sum()),
            "forecast_input_days":len(good),"coverage_pct":100*len(good)/len(test),
            "fallback_dates":data["dates"][test[~nwp["available"][test]]].tolist(),
            "metrics":rows,"paired_seven_day_block_bootstrap":paired,"bootstrap_repeats":2000,"bootstrap_seed":42,
            "target_both_below_50":bool(trial["rmse"]<50 and trial["mae"]<50),
            "final_training":info,"xgboost_version":xgb.__version__,"elapsed_seconds":budget.elapsed,
            "artifact_hashes":{name:N.sha256(out/name) for name in ("xgboost_base_pred.npz","final_model.json","final_training.json")},
            "limitations":LIMITATIONS}
    N.write_json(out/"evaluation.json",report)
    write_report(out,frozen,report)
    print(pd.DataFrame(rows).query("season == 'all'")[["method","rmse","mae"]].to_string(index=False),flush=True)


def write_report(out,frozen,report):
    tuning=json.loads((out/"tuning_selection.json").read_text())
    selected=frozen["selection"]
    lines=["# Separate XGBoost–NWP experiment", "",
           "Midnight-issued Mumbai day-ahead GHI; native ten-minute daylight errors in W/m².", "",
           f"Decision: **{report['decision']}**. The existing dashboard default is unchanged.", "",
           f"Frozen trial: `{selected['candidate']}`, new-model weight **{selected['weight']:.2f}**. "
           "The remaining weight uses the saved uncorrected BiLSTM–HGB hybrid; no old model is retrained.", "",
           "Four residual-XGBoost configurations × four blend weights; 180 complete warmup days, "
           "90-day expanding OOF blocks, chronological inner 80/20 iteration selection and a one-day purge. "
           "First 214 validation dates rank trials; the final 110 check only the frozen choice. "
           "Both errors must improve over the saved NWP-HGB incumbent in both splits to pass.", "",
           "| Development comparison | RMSE | MAE |", "| --- | ---: | ---: |"]
    for name,score in (("Tuning incumbent",tuning["incumbent_tuning"]),("Tuning trial",selected),
                        ("Later guard incumbent",frozen["guard_incumbent"]),("Later guard trial",frozen["guard_trial"])):
        lines.append(f"| {name} | {score['rmse']:.5f} | {score['mae']:.5f} |")
    lines += ["", "The frozen XGBoost trial is scored once for an exploratory comparison even if development "
              "checks fail. A better test result cannot reverse a failed development decision.", "",
              f"## Full evaluation year: {report['test_period'][0]} to {report['test_period'][1]}", "",
              f"{report['test_days']} days / {report['daylight_samples']:,} daylight samples; "
              f"NWP coverage **{report['coverage_pct']:.2f}%**; fallback dates: `{report['fallback_dates']}`. "
              "Whole missing-input days use the saved corrected historical forecast and remain in scoring.", "",
              "| Method | RMSE | MAE | RMSE skill vs NWP-HGB |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {r['method']} | {r['rmse']:.5f} | {r['mae']:.5f} | {r['rmse_skill_vs_incumbent_pct']:.2f}% |"
              for r in report["metrics"] if r["season"]=="all"]
    lines += ["", "## Seasonal results", "", "| Method | Season | RMSE | MAE |", "| --- | --- | ---: | ---: |"]
    lines += [f"| {r['method']} | {r['season']} | {r['rmse']:.5f} | {r['mae']:.5f} |"
              for r in report["metrics"] if r["season"]!="all"]
    lines += ["", "## Paired seven-day block-bootstrap changes", "",
              "Trial minus comparator; negative favors XGBoost. 2,000 paired circular block resamples of pooled errors.", "",
              "| Comparator | Metric | Change | 95% interval |", "| --- | --- | ---: | --- |"]
    for name,pair in report["paired_seven_day_block_bootstrap"].items():
        for metric,value in pair.items():
            lines.append(f"| {name} | {metric} | {value['difference']:.5f} | [{value['ci95_lo']:.5f}, {value['ci95_hi']:.5f}] |")
    lines += ["", f"Final selected iterations: **{report['final_training']['selected_iterations']} / {MAX_ROUNDS}**; "
              f"XGBoost **{report['xgboost_version']}**. Both errors below 50: **{report['target_both_below_50']}**.", "",
              "## Limitations", ""]+[f"- {s}" for s in LIMITATIONS]
    lines += ["", "All new code is in `pipelines/xgboost_nwp/`; all experiment artifacts are here. "
              "The prior paper, HGB results and source cache are preserved. `final_model.json` saves only the "
              "new residual tree; complete future inference still needs the historical hybrid and feature pipeline.", ""]
    (out/"REPORT.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("development","evaluate","all"),default="development")
    parser.add_argument("--prep",type=Path,default=C.PREP_PATH)
    parser.add_argument("--cache",type=Path,default=N.DEFAULT_CACHE)
    parser.add_argument("--history",type=Path,default=C.RESULTS_DIR/"ghi_search")
    parser.add_argument("--incumbent",type=Path,default=C.RESULTS_DIR/"nwp_optimization")
    parser.add_argument("--out",type=Path,default=C.RESULTS_DIR/"xgboost_nwp")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--max-seconds",type=float,default=3600)
    args=parser.parse_args()
    if args.threads<1 or args.max_seconds<=0 or args.out.resolve() in (args.history.resolve(),args.incumbent.resolve()):
        parser.error("Use positive threads/runtime and a separate output folder")
    budget=N.Budget(args.max_seconds)
    args.out.mkdir(parents=True,exist_ok=True)
    nwp,_=N.load_forecasts(args.cache,args.prep)
    with np.load(args.prep,allow_pickle=False) as saved:
        data={k:saved[k] for k in saved.files}
    if not np.array_equal(nwp["dates"],data["dates"]):
        raise ValueError("Forecast dates differ")
    incumbent_freeze=json.loads((args.incumbent/"frozen_selection.json").read_text())
    E.check_hashes(args.incumbent,incumbent_freeze["artifact_hashes"])
    inc_report=json.loads((args.incumbent/"evaluation.json").read_text())
    if N.sha256(args.incumbent/"selected_base_pred.npz")!=inc_report["prediction_sha256"]:
        raise ValueError("Incumbent prediction hash differs")
    sources={}
    for label,folder,files in (("history",args.history,("selected_development.npz","correction_validation.npz","selected_base_pred.npz","selected_stage2.npz")),
                               ("incumbent",args.incumbent,("selected_development.npz","selected_base_pred.npz","frozen_selection.json","evaluation.json"))):
        for filename in files:sources[f"{label}/{filename}"]=N.sha256(folder/filename)
    code=[HERE/"run.py",HERE/"training.py",ROOT/"pv_pipeline/experiment_nwp.py",ROOT/"pv_pipeline/experiment_ghi_search.py",
          ROOT/"pv_pipeline/hgb_training.py",ROOT/"pv_pipeline/common.py",ROOT/"pv_pipeline/dwt_utils.py",ROOT/"pv_pipeline/01_base_dwt_bilstm.py"]
    manifest={"experiment_version":1,"backend":"xgboost","xgboost_version":xgb.__version__,
              "prep_sha256":N.sha256(args.prep),"weather_cache_sha256":N.sha256(args.cache/"forecasts.npz"),
              "weather_manifest_sha256":N.sha256(args.cache/"manifest.json"),"protocol":N.PROTOCOL,"sources":sources,
              "code_sha256":{str(p.relative_to(ROOT)):N.sha256(p) for p in code},
              "configurations":dict(configurations(args.threads)),"weights":list(WEIGHTS),"max_rounds":MAX_ROUNDS,
              "tuning_days":214,"guard_days":110,"threads":args.threads,
              "initial_complete_training_days":180,"fold_calendar_days":90,"purge_days":1}
    path=args.out/"manifest.json"
    if path.exists():
        if json.loads(path.read_text())!=manifest:raise ValueError("Sources/code/settings changed; use a new output folder")
    elif args.phase=="evaluate":raise ValueError("Freeze a development experiment before evaluation")
    else:N.write_json(path,manifest)
    try:
        with threadpool_limits(limits=args.threads):
            if args.phase in ("development","all"):develop(args,data,nwp,manifest,budget)
            if args.phase in ("evaluate","all"):evaluate(args,data,nwp,manifest,budget)
        N.write_json(args.out/"run_status.json",{"phase":args.phase,"complete":True,"elapsed_seconds":budget.elapsed})
    except N.BudgetExpired as exc:
        days=np.concatenate([p for _,p in E.outer_folds(data,nwp["available"])])
        pending=[]
        for name,params in configurations(args.threads):
            checkpoint=args.out/name/"oof.npz"
            if not checkpoint.exists() or not np.isfinite(E.load_checkpoint(checkpoint,data,contract(N.sha256(path),name,params))[days]).all():
                pending.append(name)
        N.write_json(args.out/"run_status.json",{"phase":args.phase,"complete":False,"elapsed_seconds":budget.elapsed,
                      "reason":str(exc),"unfinished_candidates":pending,"resume_supported":True})
        raise SystemExit(2)


if __name__=="__main__":
    main()
