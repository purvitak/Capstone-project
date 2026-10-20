"""Chronological daylight-only HGB fitting with a GHI prediction contract."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import common as C


def flat_features(x):
    n, steps, _ = x.shape
    clock = np.broadcast_to(np.arange(steps, dtype=np.float32) / steps,
                            (n, steps))[..., None]
    return np.concatenate((x, clock), axis=-1).reshape(n * steps, -1)


def ghi_to_kt(ghi, cs):
    """Zero outside the known daylight proxy; retain the existing kt bounds."""
    return np.clip(np.divide(ghi, cs, out=np.zeros_like(ghi, dtype=float),
                             where=cs > C.DAYLIGHT_CS_MIN), *C.KT_CLIP)


def new_model(*, loss="squared_error", max_iter=200, learning_rate=0.05,
              max_leaf_nodes=31, min_samples_leaf=40, seed=42, **extra):
    return HistGradientBoostingRegressor(
        loss=loss, max_iter=max_iter, learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
        early_stopping=False, random_state=seed, **extra)


def fit_direct_ghi(x, ghi, cs, mask, *, val_fraction=0.2, **params):
    """Select iterations on trailing training days, purge one day, then refit.

    No prediction-day arrays are accepted here. Trees do not require scaling.
    Selection scores the clipped forecast in physical GHI units, not raw index.
    """
    n, steps, _ = x.shape
    if ghi.shape != (n, steps) or cs.shape != ghi.shape or mask.shape != ghi.shape:
        raise ValueError("Training inputs and day-by-step targets must align")
    if not 0 < val_fraction < 0.5 or int(n * (1 - val_fraction)) < 2:
        raise ValueError("Insufficient chronological training/validation days")
    cut = int(n * (1 - val_fraction))
    flat = flat_features(x).reshape(n, steps, -1)
    fit_mask = mask[:cut - 1] & np.isfinite(ghi[:cut - 1])
    val_mask = mask[cut:] & np.isfinite(ghi[cut:])
    if not fit_mask.any() or not val_mask.any():
        raise ValueError("Training and validation need observed daylight targets")
    model = new_model(**params).fit(flat[:cut - 1][fit_mask], ghi[:cut - 1][fit_mask])
    cv_cs = cs[cut:][val_mask]
    truth = ghi[cut:][val_mask]
    best = (np.inf, np.inf, 0)
    for iteration, raw in enumerate(model.staged_predict(flat[cut:][val_mask]), 1):
        error = ghi_to_kt(raw, cv_cs) * cv_cs - truth
        score = (float(np.sqrt(np.mean(error ** 2))),
                 float(np.mean(np.abs(error))), iteration)
        if score[:2] < best[:2]:
            best = score
    selected = dict(params, max_iter=best[2])
    all_mask = mask & np.isfinite(ghi)
    final = new_model(**selected).fit(flat[all_mask], ghi[all_mask])
    info = {"selected_iterations": best[2], "iterations_checked": model.max_iter,
            "inner_validation_rmse_wm2": best[0], "inner_validation_mae_wm2": best[1],
            "inner_train_days": cut - 1, "inner_purge_days": 1,
            "inner_validation_days": n - cut, "refit_days": n,
            "training_daylight_rows": int(all_mask.sum()), "early_stopping": False}
    return final, info


def predict_direct_kt(model, x, cs):
    raw = model.predict(flat_features(x)).reshape(x.shape[:2])
    return ghi_to_kt(raw, cs)
