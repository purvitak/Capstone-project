"""DWT-BiLSTM day-ahead kt (Bektas & Altas 2024, leak-safe).

--resolution hourly : 24-step reproduction; predictions are expanded to (n_days, 144)
                     so Stage 2 / evaluate keep the same contract.
--resolution 10min  : 144-step Extension 1. DWT window is the previous 144 steps only.

--leaky-weather-type : Bektas Model 1 ablation (forecast-day K from day-D GHI).

Default backend is PyTorch BiLSTM. If torch is missing, HistGradientBoosting is used
on the same leak-safe DWT features so the rest of the pipeline can still run.
"""

from __future__ import annotations

import argparse
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


def _window_ghi(ghi, i: int, hourly: bool) -> np.ndarray:
    prev = ghi[i - 1]
    return hourly_mean(prev) if hourly else np.asarray(prev, dtype=float)


def build_sample(ghi, cs, mask, types, i: int, hourly: bool, leaky: bool, levels: int):
    """Features known at 00:00 of day i, except leaky sky type of day i."""
    x_ghi = _window_ghi(ghi, i, hourly)
    bands = dwt_matrix(x_ghi, levels=levels)
    t = x_ghi.shape[0]
    wt_d1 = np.full((t, 1), float(types[i - 1]))
    wt_d = np.full((t, 1), float(types[i] if leaky else types[i - 1]))
    feat = np.concatenate([bands, wt_d1, wt_d], axis=1)
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


def pack_days(ghi, cs, mask, types, days, hourly, leaky, levels):
    xs, ys, ok = [], [], []
    for i in days:
        i = int(i)
        if i < 1:
            continue
        x, y = build_sample(ghi, cs, mask, types, i, hourly, leaky, levels)
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

    def fit(self, X, y, epochs, batch_size):
        torch = self.torch
        self.model.train()
        n = len(X)
        for _ in range(epochs):
            perm = np.random.permutation(n)
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                xb = torch.from_numpy(X[idx]).to(self.device)
                yb = torch.from_numpy(y[idx]).to(self.device)
                self.opt.zero_grad()
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                loss.backward()
                self.opt.step()

    def predict(self, X, batch_size=64):
        torch = self.torch
        self.model.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[s:s + batch_size]).to(self.device)
                outs.append(self.model(xb).cpu().numpy())
        return np.concatenate(outs, axis=0)


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
    ap.add_argument("--prep", default="results/prep.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--resolution", choices=("hourly", "10min"), default="10min")
    ap.add_argument("--leaky-weather-type", action="store_true")
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--backend", choices=("auto", "torch", "hgb"), default="auto")
    args = ap.parse_args()

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

    torch, _ = _try_torch()
    if args.backend == "torch" and torch is None:
        raise SystemExit("torch requested but not installed")
    use_torch = (args.backend == "torch") or (args.backend == "auto" and torch is not None)
    backend = "torch" if use_torch else "hgb"
    print(f"[dwt] resolution={args.resolution} levels={levels} leaky={args.leaky_weather_type} "
          f"backend={backend}")

    if args.resume and ckpt.is_file():
        kt_base = np.load(ckpt, allow_pickle=True)["kt_base"]
        print(f"[resume] {ckpt} finite days {int(np.isfinite(kt_base).any(1).sum())}")
    else:
        kt_base = np.full((n_days, C.STEPS_PER_DAY), np.nan)

    def train_predict(train_days, pred_days):
        Xtr, ytr, _ = pack_days(ghi, cs, mask, types, train_days, hourly,
                                args.leaky_weather_type, levels)
        Xpr, _, pidx = pack_days(ghi, cs, mask, types, pred_days, hourly,
                                 args.leaky_weather_type, levels)
        if Xtr is None or Xpr is None:
            return
        if use_torch:
            net = BiLSTMNet(Xtr.shape[-1], ytr.shape[-1], hidden=args.hidden)
            net.fit(Xtr, ytr, epochs=2 if args.smoke else args.epochs,
                    batch_size=args.batch_size)
            pred = net.predict(Xpr)
        else:
            model = fit_hgb(Xtr, ytr)
            pred = predict_hgb(model, Xpr)
        pred144 = to_144(pred, hourly)
        for row, day in zip(pred144, pidx):
            kt_base[int(day)] = row

    starts = list(range(args.smoke and 365 or INITIAL_TRAIN_DAYS, len(dev_idx), FOLD_DAYS))
    if args.smoke:
        starts = starts[:1]
    folds = []
    for a in starts:
        b = min(a + FOLD_DAYS, len(dev_idx))
        train_days = dev_idx[: max(a - PURGE_DAYS, 0)]
        pred_days = dev_idx[a:b]
        if np.isfinite(kt_base[pred_days]).any():
            print(f"[oof] skip {dates[pred_days[0]]} -> {dates[pred_days[-1]]}")
            continue
        print(f"[oof] train {len(train_days)}d  predict {dates[pred_days[0]]} -> {dates[pred_days[-1]]}")
        t0 = time.time()
        train_predict(train_days, pred_days)
        r = ghi_rmse(kt[pred_days], kt_base[pred_days], cs[pred_days], ghi[pred_days],
                     mask[pred_days])
        folds.append({"first": str(dates[pred_days[0]]), "last": str(dates[pred_days[-1]]),
                      "rmse_wm2": r, "seconds": time.time() - t0})
        print(f"[oof] RMSE {r:7.2f} W/m2  ({time.time()-t0:.0f}s)")
        np.savez_compressed(ckpt, kt_base=kt_base, dates=dates)

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
        "backend": backend, "resolution": args.resolution, "levels": levels,
        "leaky_weather_type": args.leaky_weather_type, "folds": folds,
        "test_rmse_wm2": r_model, "test_mean3_rmse_wm2": r_base,
        "test_skill_vs_mean3": C.skill(r_model, r_base),
        "n_finite": int(np.isfinite(kt_base).any(1).sum()),
    }
    print(f"[test] DWT-BiLSTM RMSE {r_model:.2f} vs 3-day {r_base:.2f}  "
          f"skill {summary['test_skill_vs_mean3']*100:+.2f}%")
    np.savez_compressed(ckpt, kt_base=kt_base, dates=dates)
    (out / f"{tag}_base.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] {ckpt}")
    print(f"[next] python 02_corrector.py --base {ckpt} --prefix {tag}_")


if __name__ == "__main__":
    main()
