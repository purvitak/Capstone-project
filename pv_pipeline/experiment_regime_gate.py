"""Development-only, issue-time regime gate for BiLSTM and context-HGB forecasts.

Both input forecasts must be complete expanding-window OOF predictions. The
gate is trained on the first 75% of common OOF days and scored on the last 25%.
No held-out test-day prediction is used for model selection or emitted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import blend_predictions as B
import common as C


def issue_time_features(prep, days):
    """Day-D descriptor from D-1 realised values only."""
    previous = np.asarray(days, dtype=int) - 1
    if (previous < 0).any():
        raise ValueError("All forecast days need an earlier day")
    mask = prep["mask"][previous]
    kt = np.where(mask, prep["kt"][previous], np.nan)
    temp = prep["air_temp"][previous]
    rh = prep["rh"][previous]
    with np.errstate(invalid="ignore"):
        return np.column_stack([
            np.nanmean(kt, axis=1),
            np.nanstd(kt, axis=1),
            np.nanmean(temp, axis=1),
            np.nanmean(rh, axis=1),
        ])


def soft_membership(scaled, centers, temperature):
    squared = ((scaled[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    logits = -squared / (2 * temperature)
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return probability / probability.sum(axis=1, keepdims=True)


def paired_block_bootstrap(ghi, cs, mask, reference, candidate,
                           block_days=7, repeats=1000, seed=42):
    """RMSE(candidate) - RMSE(reference), resampling adjacent days in blocks."""
    actual = ghi
    def daily_errors(pred):
        error = np.where(mask, (np.clip(pred, *C.KT_CLIP) * cs - actual) ** 2, 0.0)
        return error.sum(axis=1), mask.sum(axis=1)
    ref_sse, count = daily_errors(reference)
    cand_sse, _ = daily_errors(candidate)
    n = len(ghi)
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats)
    for j in range(repeats):
        starts = rng.integers(0, max(n - block_days + 1, 1),
                              size=int(np.ceil(n / block_days)))
        indices = np.concatenate([np.arange(start, min(start + block_days, n))
                                  for start in starts])[:n]
        total = count[indices].sum()
        differences[j] = (np.sqrt(cand_sse[indices].sum() / total)
                          - np.sqrt(ref_sse[indices].sum() / total))
    return [float(v) for v in np.quantile(differences, [0.025, 0.975])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep", default=str(C.PREP_PATH))
    ap.add_argument("--dwt", default=str(C.RESULTS_DIR / "dwt_10min_150_paper/dwt_base_pred.npz"))
    ap.add_argument("--context", default=str(C.RESULTS_DIR / "context_hgb/dwt_base_pred.npz"))
    ap.add_argument("--stage2", default=str(C.RESULTS_DIR / "dwt_10min_150_paper/dwt_stage2.npz"))
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "archive" / "novelty_regime_gate"))
    args = ap.parse_args()

    prep_hash = hashlib.sha256(Path(args.prep).read_bytes()).hexdigest()
    with np.load(args.prep, allow_pickle=False) as prep:
        dates, ghi, cs, mask = (prep[k] for k in ("dates", "ghi", "cs_proxy", "mask"))
        a, a_config = B.load_source(args.dwt, dates, ghi.shape, prep_hash, "torch")
        b, b_config = B.load_source(args.context, dates, ghi.shape, prep_hash, "hgb")
        if (a_config.get("resolution") != "10min" or b_config.get("resolution") != "10min"
                or b_config.get("features") != "context"):
            raise ValueError("Expected 10-minute BiLSTM and context-HGB forecasts")
        dev = prep["dev_idx"]
        days = dev[np.isfinite(a[dev]).all(axis=1) & np.isfinite(b[dev]).all(axis=1)]
        if len(days) != len(dev) - 730:
            raise ValueError("Both sources need complete, matching development OOF coverage")
        cut = int(0.75 * len(days))
        train, valid = days[:cut], days[cut:]
        raw_features = issue_time_features(prep, days)
        if not np.isfinite(raw_features).all():
            raise ValueError("Regime descriptors must be finite on all OOF days")
        scaler = StandardScaler().fit(raw_features[:cut])
        x_train = scaler.transform(raw_features[:cut])
        x_valid = scaler.transform(raw_features[cut:])
        cluster = KMeans(n_clusters=3, n_init=10, random_state=42).fit(x_train)
        train_labels = cluster.labels_
        sqdist = ((x_train - cluster.cluster_centers_[train_labels]) ** 2).sum(axis=1)
        temperature = max(float(np.median(sqdist)), 1e-6)

        def blend_for(idx, weights):
            return weights[:, None] * np.clip(a[idx], *C.KT_CLIP) + (1 - weights[:, None]) * np.clip(b[idx], *C.KT_CLIP)

        def rmse(idx, prediction):
            return C.point_metrics(ghi[idx], prediction * cs[idx], mask[idx])["rmse"]

        grid = np.linspace(0, 1, 11)
        global_scores = [rmse(train, blend_for(train, np.full(len(train), w))) for w in grid]
        global_weight = float(grid[int(np.argmin(global_scores))])
        cluster_weights = np.zeros(3)
        counts = []
        for k in range(3):
            group = train[train_labels == k]
            counts.append(int(len(group)))
            scores = [rmse(group, blend_for(group, np.full(len(group), w))) for w in grid]
            cluster_weights[k] = grid[int(np.argmin(scores))]

        probabilities = soft_membership(x_valid, cluster.cluster_centers_, temperature)
        soft_weight = probabilities @ cluster_weights
        candidates = {
            "dwt_bilstm": np.clip(a[valid], *C.KT_CLIP),
            "context_hgb": np.clip(b[valid], *C.KT_CLIP),
            "fixed_50_50": blend_for(valid, np.full(len(valid), 0.5)),
            "global_train_tuned": blend_for(valid, np.full(len(valid), global_weight)),
            "soft_regime_gate": blend_for(valid, soft_weight),
        }
        stage2_path = Path(args.stage2)
        if stage2_path.is_file():
            with np.load(stage2_path, allow_pickle=False) as stage2:
                if not np.array_equal(dates, stage2["dates"]):
                    raise ValueError("Stage 2 dates differ")
                if np.isfinite(stage2["kt_final"][valid][mask[valid]]).all():
                    candidates["dwt_stage2_reference"] = np.clip(stage2["kt_final"][valid], *C.KT_CLIP)

        result = {}
        for name, prediction in candidates.items():
            result[name] = {
                "rmse_wm2": rmse(valid, prediction),
                "monsoon_rmse_wm2": rmse(valid[prep["is_monsoon"][valid]],
                                         prediction[prep["is_monsoon"][valid]]),
                "non_monsoon_rmse_wm2": rmse(valid[~prep["is_monsoon"][valid]],
                                             prediction[~prep["is_monsoon"][valid]]),
            }
            print(f"[validation] {name:22s} RMSE {result[name]['rmse_wm2']:.2f} W/m2")

        gate_ci = {
            reference: paired_block_bootstrap(ghi[valid], cs[valid], mask[valid],
                                              candidates[reference], candidates["soft_regime_gate"])
            for reference in ("fixed_50_50", "global_train_tuned")
        }
        for reference, interval in gate_ci.items():
            print(f"[bootstrap] gate minus {reference}: 95% interval "
                  f"[{interval[0]:+.2f}, {interval[1]:+.2f}] W/m2")

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        gate_kt = np.full_like(a, np.nan)
        gate_kt[valid] = candidates["soft_regime_gate"]
        np.savez_compressed(out / "regime_gate_validation.npz", kt_gate=gate_kt,
                            dates=dates, validation_days=valid)
        report = {
            "source_dwt": str(Path(args.dwt).resolve()),
            "source_context": str(Path(args.context).resolve()),
            "training_days": len(train), "validation_days": len(valid),
            "validation_period": [str(dates[valid[0]]), str(dates[valid[-1]])],
            "descriptor_names": ["previous_day_kt_mean", "previous_day_kt_std",
                                 "previous_day_air_temp_mean", "previous_day_rh_mean"],
            "cluster_train_counts": counts,
            "dwt_weight_by_cluster": cluster_weights.tolist(),
            "global_train_tuned_dwt_weight": global_weight,
            "soft_membership_temperature": temperature,
            "validation_metrics": result,
            "gate_rmse_difference_95pct_block_bootstrap_ci_wm2": gate_ci,
            "bootstrap_block_days": 7,
            "test_scored": False,
            "note": "Gate is fit on early OOF days and scored on later OOF days; no held-out test prediction is emitted.",
        }
        (out / "regime_gate_development.json").write_text(json.dumps(report, indent=2))
        print(f"[done] {out / 'regime_gate_development.json'}")


if __name__ == "__main__":
    main()
