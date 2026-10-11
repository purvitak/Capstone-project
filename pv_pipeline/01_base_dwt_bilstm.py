"""DWT-BiLSTM day-ahead kt (Bektas & Altas 2024, leak-safe).

--resolution hourly : 24-step reproduction; predictions are expanded to (n_days, 144)
                     so Stage 2 / evaluate keep the same contract.
--resolution 10min  : 144-step Extension 1. Default DWT window is the previous day.
--lookback-days 3   : concatenate DWT profiles from D-1 through D-3 at each
                      ten-minute clock position; no target-day observations.

--leaky-weather-type : Bektas Model 1 ablation (forecast-day K from day-D GHI).

Default backend is PyTorch BiLSTM. If torch is missing, HistGradientBoosting is used
on the same leak-safe DWT features so the rest of the pipeline can still run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as C
from dwt_utils import (
    daily_clearness,
    dwt_matrix,
    expand_hourly_to_10min,
    hourly_mean,
    sky_type_from_k,
)

INITIAL_TRAIN_DAYS = 730
FOLD_DAYS = 90
PURGE_DAYS = 1


def _daily_types(ghi, cs, mask) -> np.ndarray:
    n = len(ghi)
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        out[i] = sky_type_from_k(daily_clearness(ghi[i], cs[i], mask[i]))
    return out


def _window_ghi(ghi, i: int, hourly: bool, lag: int = 1) -> np.ndarray:
    prev = ghi[i - lag]
    return hourly_mean(prev) if hourly else np.asarray(prev, dtype=float)


def build_sample(ghi, cs, mask, types, i: int, hourly: bool, leaky: bool,
                 levels: int, lookback_days: int = 1):
    """Features known at 00:00 of day i, except leaky sky type of day i."""
    if lookback_days < 1 or i < lookback_days:
        raise ValueError("Not enough previous days for requested lookback")
    bands = [dwt_matrix(_window_ghi(ghi, i, hourly, lag), levels=levels)
             for lag in range(1, lookback_days + 1)]
    t = bands[0].shape[0]
    wt_d1 = np.full((t, 1), float(types[i - 1]))
    wt_d = np.full((t, 1), float(types[i] if leaky else types[i - 1]))
    older_types = [np.full((t, 1), float(types[i - lag]))
                   for lag in range(2, lookback_days + 1)]
    feat = np.concatenate([*bands, wt_d1, wt_d, *older_types], axis=1)
    if hourly:
        with np.errstate(divide="ignore", invalid="ignore"):
            kt_row = np.where(cs[i] > C.DAYLIGHT_CS_MIN, ghi[i] / cs[i], 0.0)
        kt_row = np.clip(np.nan_to_num(kt_row, nan=0.0), *C.KT_CLIP)
        y = hourly_mean(kt_row)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            kt_row = np.where(cs[i] > C.DAYLIGHT_CS_MIN, ghi[i] / cs[i], 0.0)
        y = np.clip(np.nan_to_num(kt_row, nan=0.0), *C.KT_CLIP)
    return feat.astype(np.float32), y.astype(np.float32)


def pack_days(ghi, cs, mask, types, days, hourly, leaky, levels,
              lookback_days: int = 1):
    xs, ys, ok = [], [], []
    for i in days:
        i = int(i)
        if i < lookback_days:
            continue
        x, y = build_sample(ghi, cs, mask, types, i, hourly, leaky, levels,
                            lookback_days)
        xs.append(x)
        ys.append(y)
        ok.append(i)
    if not xs:
        return None, None, np.array([], dtype=int)
    return np.stack(xs), np.stack(ys), np.asarray(ok, dtype=int)


def ghi_rmse(kt_true, kt_pred, cs, ghi, mask) -> float:
    return C.point_metrics(ghi, np.clip(kt_pred, *C.KT_CLIP) * cs, mask)["rmse"]


def _try_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except Exception:
        return None, None


class BiLSTMNet:
    """Thin wrapper so the training loop does not import torch at module load."""

    def __init__(self, n_feat, n_out, hidden=64, layers=1, lr=1e-3):
        torch, nn = _try_torch()
        if torch is None:
            raise RuntimeError("torch not installed")
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers,
                                    batch_first=True, bidirectional=True)
                self.head = nn.Linear(hidden * 2, 1)

            def forward(self, x):
                h, _ = self.lstm(x)
                return self.head(h).squeeze(-1)

        self.model = _Net().to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def fit(self, X, y, epochs, batch_size, weights=None):
        # Scaling is fitted only on training inputs, never validation/test days.
        self.mean = X.mean(axis=(0, 1), keepdims=True)
        self.scale = X.std(axis=(0, 1), keepdims=True)
        self.scale = np.where(self.scale < 1e-6, 1.0, self.scale)
        X = ((X - self.mean) / self.scale).astype(np.float32)
        torch = self.torch
        self.model.train()
        n = len(X)
        for _ in range(epochs):
            for s in range(0, n, batch_size):
                idx = np.arange(s, min(s + batch_size, n))
                xb = torch.from_numpy(X[idx]).to(self.device)
                yb = torch.from_numpy(y[idx]).to(self.device)
                self.opt.zero_grad()
                pred = self.model(xb)
                if weights is None:
                    loss = self.loss_fn(pred, yb)
                else:
                    wb = torch.from_numpy(weights[idx]).to(self.device)
                    # GHI error = kt error * CS. Nighttime has zero weight.
                    loss = ((pred - yb).square() * wb).sum() / (wb > 0).sum().clamp_min(1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

    def predict(self, X, batch_size=64):
        torch = self.torch
        X = ((X - self.mean) / self.scale).astype(np.float32)
        self.model.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[s:s + batch_size]).to(self.device)
                outs.append(self.model(xb).cpu().numpy())
        return np.concatenate(outs, axis=0)

    def fit_validated(self, X, y, epochs, batch_size, validation, patience,
                      weights=None, validation_offset=None):
        """Choose epochs on a chronological inner validation slice, then refit.

        Refit uses all supplied training days and fresh scaling. Neither OOF
        forecast days nor test days participate in epoch selection.
        """
        cut, cs, ghi, mask, seed = validation
        Xfit, yfit = X[:cut - 1], y[:cut - 1]  # one-day inner purge
        best_score, best_epoch, stale = np.inf, 1, 0
        # fit() is called once per epoch; retain the first training-only scaler.
        for epoch in range(1, epochs + 1):
            self.fit(Xfit, yfit, epochs=1, batch_size=batch_size,
                     weights=None if weights is None else weights[:cut - 1])
            pred = self.predict(X[cut:])
            if validation_offset is not None:
                pred = pred + validation_offset
            score = C.point_metrics(ghi, np.clip(pred, *C.KT_CLIP) * cs, mask)["rmse"]
            if score < best_score:
                best_score, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
            if stale >= patience:
                break
        torch = self.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Reset model and optimizer for a fresh fit on all fold training days.
        for module in self.model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.opt.param_groups[0]["lr"])
        self.fit(X, y, epochs=best_epoch, batch_size=batch_size, weights=weights)
        return {"selected_epochs": best_epoch, "epochs_checked": epoch,
                "inner_validation_rmse_wm2": best_score,
                "inner_train_days": cut - 1, "inner_validation_days": len(X) - cut}


def _flat_hgb_x(X):
    n, t, f = X.shape
    step = np.repeat(np.arange(t, dtype=np.float32)[None, :], n, axis=0)[..., None] / t
    return np.concatenate([X, step], axis=-1).reshape(n * t, f + 1)


def fit_hgb(X, y):
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=40, random_state=0,
    )
    model.fit(_flat_hgb_x(X), y.ravel())
    return model


def predict_hgb(model, X):
    n, t, _ = X.shape
    return model.predict(_flat_hgb_x(X)).reshape(n, t)


def to_144(pred_block, hourly: bool) -> np.ndarray:
    if not hourly:
        return pred_block
    return np.stack([expand_hourly_to_10min(row) for row in pred_block])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=str(C.PREP_PATH))
    ap.add_argument("--out", default=str(C.RESULTS_DIR))
    ap.add_argument("--resolution", choices=("hourly", "10min"), default="10min")
    ap.add_argument("--leaky-weather-type", action="store_true")
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--lookback-days", type=int, default=1,
                    help="Past daily DWT profiles supplied as separate clock-aligned channels")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--features", choices=("dwt", "context"), default="dwt",
                    help="context adds existing historical kt/weather and known calendar/solar features")
    ap.add_argument("--loss", choices=("kt", "ghi"), default="kt",
                    help="ghi minimizes daylight irradiance error rather than all-day kt error")
    ap.add_argument("--residual-baseline", choices=("none", "mean3"), default="none",
                    help="Learn an adjustment to the 3-day mean kt profile")
    ap.add_argument("--fold-indices", default="",
                    help="Comma-separated zero-based folds to screen; requires --development-only")
    ap.add_argument("--development-only", action="store_true",
                    help="Run OOF folds only; do not fit or score the held-out test year")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--backend", choices=("auto", "torch", "hgb"), default="auto")
    args = ap.parse_args()
    if args.epochs < 1 or args.patience < 1 or not 0 < args.val_fraction < 0.5:
        ap.error("epochs/patience must be positive and val-fraction must be between 0 and 0.5")
    if args.lookback_days < 1 or args.lookback_days > 7:
        ap.error("--lookback-days must be between 1 and 7")
    if args.fold_indices and not args.development_only:
        ap.error("--fold-indices requires --development-only")
    if args.loss == "ghi" and args.resolution == "hourly":
        ap.error("--loss ghi currently supports 10min resolution only")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "dwt_hourly" if args.resolution == "hourly" else "dwt"
    if args.leaky_weather_type:
        tag += "_leaky"
    ckpt = out / f"{tag}_base_pred.npz"
    hourly = args.resolution == "hourly"
    levels = 3 if hourly else args.levels

    d = np.load(args.prep, allow_pickle=True)
    ghi, cs, mask = d["ghi"], d["cs_proxy"], d["mask"]
    kt = d["kt"]
    dates = d["dates"]
    dev_idx, test_idx = d["dev_idx"], d["test_idx"]
    purge_idx = int(d["purge_idx"])
    n_days = len(dates)
    types = _daily_types(ghi, cs, mask)
    mean3 = C.baseline_mean3_kt(kt)
    context = d["X"] if args.features == "context" else None

    torch, _ = _try_torch()
    if args.backend == "torch" and torch is None:
        raise SystemExit("torch requested but not installed")
    use_torch = (args.backend == "torch") or (args.backend == "auto" and torch is not None)
    backend = "torch" if use_torch else "hgb"
    if args.loss == "ghi" and not use_torch:
        ap.error("--loss ghi requires --backend torch")
    config = {"trainer_version": 2, "backend": backend, "resolution": args.resolution,
              "levels": levels, "leaky": args.leaky_weather_type,
              "epochs": args.epochs, "hidden": args.hidden, "batch_size": args.batch_size,
              "learning_rate": args.learning_rate, "seed": args.seed,
              "patience": args.patience, "val_fraction": args.val_fraction,
              "smoke": args.smoke, "prep": str(Path(args.prep).resolve()),
              "prep_sha256": hashlib.sha256(Path(args.prep).read_bytes()).hexdigest()}
    # Preserve compatibility with version-2 runs using the original options.
    if (args.features, args.loss, args.residual_baseline) != ("dwt", "kt", "none"):
        config.update(trainer_version=3, features=args.features, loss=args.loss,
                      residual_baseline=args.residual_baseline)
    if args.lookback_days != 1:
        config.update(trainer_version=4, lookback_days=args.lookback_days)
    config_json = json.dumps(config, sort_keys=True)
    print(f"[dwt] resolution={args.resolution} levels={levels} leaky={args.leaky_weather_type} "
          f"backend={backend}")

    if args.resume and ckpt.is_file():
        saved = np.load(ckpt, allow_pickle=True)
        if "config" not in saved or str(saved["config"]) != config_json:
            raise SystemExit("Checkpoint settings differ or checkpoint uses the old trainer. "
                             "Use a fresh --out directory.")
        if not np.array_equal(saved["dates"], dates):
            raise SystemExit("Checkpoint dates differ from prepared data; use a fresh --out directory.")
        kt_base = saved["kt_base"]
        print(f"[resume] {ckpt} finite days {int(np.isfinite(kt_base).any(1).sum())}")
    else:
        kt_base = np.full((n_days, C.STEPS_PER_DAY), np.nan)

    def train_predict(train_days, pred_days):
        info = {}
        Xtr, ytr, _ = pack_days(ghi, cs, mask, types, train_days, hourly,
                                args.leaky_weather_type, levels, args.lookback_days)
        Xpr, _, pidx = pack_days(ghi, cs, mask, types, pred_days, hourly,
                                 args.leaky_weather_type, levels, args.lookback_days)
        if Xtr is None or Xpr is None:
            return
        def at_resolution(a):
            if not hourly:
                return a
            return a.reshape((len(a), 24, 6) + a.shape[2:]).mean(axis=2)
        if context is not None:
            extra_tr = at_resolution(context[train_days]).astype(np.float32)
            extra_pr = at_resolution(context[pidx]).astype(np.float32)
            if not (np.isfinite(extra_tr).all() and np.isfinite(extra_pr).all()):
                raise ValueError("Context features must be finite on every selected day")
            Xtr = np.concatenate([Xtr, extra_tr], axis=-1)
            Xpr = np.concatenate([Xpr, extra_pr], axis=-1)
        offset_tr = at_resolution(mean3[train_days]).astype(np.float32)
        offset_pr = at_resolution(mean3[pidx]).astype(np.float32)
        weights = None
        if args.loss == "ghi":
            ytr = np.divide(ghi[train_days], cs[train_days],
                             out=np.zeros_like(ghi[train_days]),
                             where=cs[train_days] > C.DAYLIGHT_CS_MIN).astype(np.float32)
            weights = (mask[train_days] * (cs[train_days] / 1000.0) ** 2).astype(np.float32)
        if args.residual_baseline == "mean3":
            ytr = ytr - offset_tr
        if use_torch:
            fold_seed = args.seed + int(train_days[-1])
            torch.manual_seed(fold_seed)
            torch.cuda.manual_seed_all(fold_seed)
            np.random.seed(fold_seed)
            net = BiLSTMNet(Xtr.shape[-1], ytr.shape[-1], hidden=args.hidden,
                            lr=args.learning_rate)
            cut = int(len(Xtr) * (1 - args.val_fraction))
            vdays = train_days[cut:]
            vcs, vghi, vmask = cs[vdays], ghi[vdays], mask[vdays]
            if hourly:
                def agg(a):
                    return np.nanmean(np.where(vmask, a, np.nan).reshape(len(vdays), 24, 6), axis=2)
                vcs, vghi = agg(vcs), agg(vghi)
                vmask = vmask.reshape(len(vdays), 24, 6).any(axis=2)
            info = net.fit_validated(Xtr, ytr, epochs=2 if args.smoke else args.epochs,
                                    batch_size=args.batch_size,
                                    validation=(cut, vcs, vghi, vmask, fold_seed),
                                    patience=args.patience, weights=weights,
                                    validation_offset=offset_tr[cut:] if args.residual_baseline == "mean3" else None)
            print(f"[train] selected {info['selected_epochs']}/{info['epochs_checked']} epochs "
                  f"on inner validation (RMSE {info['inner_validation_rmse_wm2']:.2f}); "
                  f"refit on {len(Xtr)} days")
            pred = net.predict(Xpr)
        else:
            model = fit_hgb(Xtr, ytr)
            pred = predict_hgb(model, Xpr)
        if args.residual_baseline == "mean3":
            pred = pred + offset_pr
        pred144 = to_144(pred, hourly)
        for row, day in zip(pred144, pidx):
            kt_base[int(day)] = row
        return info

    starts = list(range(args.smoke and 365 or INITIAL_TRAIN_DAYS, len(dev_idx), FOLD_DAYS))
    if args.smoke:
        starts = starts[:1]
    if args.fold_indices:
        try:
            selected = [int(v) for v in args.fold_indices.split(",")]
            if any(i < 0 or i >= len(starts) for i in selected):
                raise ValueError()
            starts = [starts[i] for i in sorted(set(selected))]
        except ValueError:
            ap.error("--fold-indices must contain valid zero-based fold numbers")
    # Keep previously completed fold logs when resuming.
    previous = out / f"{tag}_development.json"
    folds = json.loads(previous.read_text()).get("folds", []) if args.resume and previous.exists() else []
    for a in starts:
        b = min(a + FOLD_DAYS, len(dev_idx))
        train_days = dev_idx[: max(a - PURGE_DAYS, 0)]
        pred_days = dev_idx[a:b]
        if np.isfinite(kt_base[pred_days]).all():
            print(f"[oof] skip {dates[pred_days[0]]} -> {dates[pred_days[-1]]}")
            continue
        print(f"[oof] train {len(train_days)}d  predict {dates[pred_days[0]]} -> {dates[pred_days[-1]]}")
        t0 = time.time()
        training_info = train_predict(train_days, pred_days)
        r = ghi_rmse(kt[pred_days], kt_base[pred_days], cs[pred_days], ghi[pred_days],
                     mask[pred_days])
        folds.append({"first": str(dates[pred_days[0]]), "last": str(dates[pred_days[-1]]),
                      "rmse_wm2": r, "seconds": time.time() - t0,
                      "training": training_info})
        print(f"[oof] RMSE {r:7.2f} W/m2  ({time.time()-t0:.0f}s)")
        np.savez_compressed(ckpt, kt_base=kt_base, dates=dates, config=config_json)

    if args.development_only:
        oof = dev_idx[np.isfinite(kt_base[dev_idx]).all(axis=1)]
        mean3 = C.baseline_mean3_kt(kt)
        r1 = ghi_rmse(None, kt_base[oof], cs[oof], ghi[oof], mask[oof])
        rb = ghi_rmse(None, mean3[oof], cs[oof], ghi[oof], mask[oof])
        summary = {"config": config, "folds": folds, "n_oof_days": len(oof),
                   "oof_rmse_wm2": r1, "oof_mean3_rmse_wm2": rb,
                   "oof_skill_vs_mean3": C.skill(r1, rb)}
        expected_oof = len(dev_idx) - INITIAL_TRAIN_DAYS
        summary["expected_oof_days"] = expected_oof
        summary["complete_oof"] = len(oof) == expected_oof and not args.smoke
        (out / f"{tag}_development.json").write_text(json.dumps(summary, indent=2))
        np.savez_compressed(ckpt, kt_base=kt_base, dates=dates, config=config_json)
        print(f"[development] RMSE {r1:.2f} vs 3-day {rb:.2f}; test year not scored")
        if not summary["complete_oof"]:
            print(f"[development] PARTIAL SCREEN: {len(oof)}/{expected_oof} OOF days; "
                  "compare models on matching days only")
        return

    need = np.concatenate([[purge_idx], test_idx])
    if not (args.resume and np.isfinite(kt_base[need]).all()):
        print(f"[final] fit on {len(dev_idx)} development days")
        t0 = time.time()
        train_days = dev_idx if not args.smoke else dev_idx[:400]
        train_predict(train_days, need)
        print(f"[final] done in {time.time()-t0:.0f}s")

    mean3 = C.baseline_mean3_kt(kt)
    r_model = ghi_rmse(kt[test_idx], kt_base[test_idx], cs[test_idx], ghi[test_idx],
                       mask[test_idx])
    r_base = ghi_rmse(kt[test_idx], mean3[test_idx], cs[test_idx], ghi[test_idx],
                      mask[test_idx])
    summary = {
        "config": config,
        "backend": backend, "resolution": args.resolution, "levels": levels,
        "leaky_weather_type": args.leaky_weather_type, "folds": folds,
        "test_rmse_wm2": r_model, "test_mean3_rmse_wm2": r_base,
        "test_skill_vs_mean3": C.skill(r_model, r_base),
        "n_finite": int(np.isfinite(kt_base).any(1).sum()),
    }
    print(f"[test] DWT-BiLSTM RMSE {r_model:.2f} vs 3-day {r_base:.2f}  "
          f"skill {summary['test_skill_vs_mean3']*100:+.2f}%")
    np.savez_compressed(ckpt, kt_base=kt_base, dates=dates, config=config_json)
    (out / f"{tag}_base.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] {ckpt}")
    print(f"[next] python 02_corrector.py --base {ckpt} --prefix {tag}_")


if __name__ == "__main__":
    main()
