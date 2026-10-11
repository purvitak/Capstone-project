"""Fixed-weight DWT-BiLSTM + context-tree ensemble; score development only.

Both source forecasts must come from the same prepared dataset and dates.
No weights are fitted here. Predictions are emitted only where BOTH models
have predictions, preserving the OOF and held-out-test boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import common as C


def blend_arrays(a, b, weight=0.5):
    if a.shape != b.shape:
        raise ValueError("Prediction shapes differ")
    if not 0 <= weight <= 1:
        raise ValueError("BiLSTM weight must be between 0 and 1")
    result = np.full(a.shape, np.nan, dtype=float)
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    result[valid] = (weight * np.clip(a[valid], *C.KT_CLIP)
                     + (1 - weight) * np.clip(b[valid], *C.KT_CLIP))
    return result


def load_source(path, dates, shape, prep_hash, backend):
    with np.load(path, allow_pickle=False) as source:
        if "config" not in source:
            raise ValueError(f"{path}: source lacks configuration/provenance")
        config = json.loads(str(source["config"]))
        if config.get("backend") != backend:
            raise ValueError(f"{path}: expected backend {backend}")
        if config.get("leaky") or config.get("smoke"):
            raise ValueError(f"{path}: leaky/smoke forecasts cannot be used")
        if config.get("prep_sha256") != prep_hash:
            raise ValueError(f"{path}: prepared-data fingerprint differs")
        if not np.array_equal(source["dates"], dates):
            raise ValueError(f"{path}: dates differ")
        predictions = source["kt_base"]
        if predictions.shape != shape:
            raise ValueError(f"{path}: prediction shape differs from prepared data")
        return predictions, config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep", default=str(C.PREP_PATH))
    ap.add_argument("--dwt", default=str(C.RESULTS_DIR / "scaled_100epochs/dwt_base_pred.npz"))
    ap.add_argument("--context", default=str(C.RESULTS_DIR / "context_hgb/dwt_base_pred.npz"))
    ap.add_argument("--weight", type=float, default=0.5,
                    help="Fixed BiLSTM weight; choose on development data only")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "hybrid"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(Path(args.prep).read_bytes()).hexdigest()
    with np.load(args.prep, allow_pickle=False) as d:
        dates, ghi, cs, mask = d["dates"], d["ghi"], d["cs_proxy"], d["mask"]
        a, a_config = load_source(args.dwt, dates, ghi.shape, fingerprint, "torch")
        b, b_config = load_source(args.context, dates, ghi.shape, fingerprint, "hgb")
        if a_config["resolution"] != b_config["resolution"]:
            raise ValueError("Source resolutions differ")
        if b_config.get("features") != "context":
            raise ValueError("Tree source must use --features context")
        blend = blend_arrays(a, b, args.weight)
        dev = d["dev_idx"]
        days = dev[np.isfinite(blend[dev]).all(axis=1)]
        if not len(days):
            raise ValueError("No shared development forecasts to compare")
        expected = len(dev) - 730
        methods = {"mean3": C.baseline_mean3_kt(d["kt"]), "dwt_bilstm": a,
                   "context_hgb": b, "hybrid": blend}
        rows = []
        for season, selected in (("all", np.ones(len(days), bool)),
                                 ("monsoon", d["is_monsoon"][days]),
                                 ("non_monsoon", ~d["is_monsoon"][days])):
            idx = days[selected]
            if not len(idx):
                continue
            for name, kt_hat in methods.items():
                metrics = C.point_metrics(ghi[idx], np.clip(kt_hat[idx], *C.KT_CLIP) * cs[idx], mask[idx])
                rows.append({"method": name, "season": season, "n_days": len(idx), **metrics})
                if season == "all":
                    print(f"[development] {name}: RMSE {metrics['rmse']:.2f} W/m2 on {len(idx)} days")
        config = {"model": "fixed_dwt_bilstm_context_hgb_blend", "weight": args.weight,
                  "prep_sha256": fingerprint,
                  "sources": {"dwt": str(Path(args.dwt).resolve()),
                              "context": str(Path(args.context).resolve())},
                  "source_configs": {"dwt": a_config, "context": b_config},
                  "source_sha256": {"dwt": hashlib.sha256(Path(args.dwt).read_bytes()).hexdigest(),
                                    "context": hashlib.sha256(Path(args.context).read_bytes()).hexdigest()}}
        np.savez_compressed(out / "hybrid_base_pred.npz", kt_base=blend, dates=dates,
                            config=json.dumps(config, sort_keys=True))
        report = {"config": config, "development_metrics": rows,
                  "complete_oof": len(days) == expected,
                  "n_oof_days": len(days), "expected_oof_days": expected,
                  "test_scored": False,
                  "test_forecast_days": int(np.isfinite(blend[d["test_idx"]]).all(axis=1).sum())}
        (out / "hybrid_development.json").write_text(json.dumps(report, indent=2))
        print(f"[done] {out / 'hybrid_base_pred.npz'}; test year not scored")
        if not report["complete_oof"]:
            print("[note] Partial development coverage; compare only matching days")


if __name__ == "__main__":
    main()
