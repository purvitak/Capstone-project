"""Daylight GHI-residual XGBoost fitting, with chronological iteration selection."""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from hgb_training import flat_features, ghi_to_kt


class BudgetCallback(xgb.callback.TrainingCallback):
    def __init__(self, budget):
        self.budget = budget

    def after_iteration(self, model, epoch, evals_log):
        if self.budget is not None:
            self.budget.check()
        return False


def fit_residual(x, ghi, cs, mask, baseline, params, max_rounds=300, budget=None):
    """No test arrays accepted. Select clipped physical-GHI RMSE, then refit."""
    n, steps, _ = x.shape
    if ghi.shape != (n, steps) or any(a.shape != ghi.shape for a in (cs, mask, baseline)):
        raise ValueError("Residual training arrays must align")
    if mask.dtype.kind != "b" or not np.isfinite(baseline[mask]).all():
        raise ValueError("Expected a boolean mask and finite daylight NWP baseline")
    if params.get("objective") != "reg:squarederror" or max_rounds < 1:
        raise ValueError("Expected squared-error regression and positive rounds")
    cut = int(n * .8)
    if cut < 2 or cut >= n:
        raise ValueError("Insufficient chronological fitting/validation days")
    features = flat_features(x).reshape(n, steps, -1)
    mask = mask & np.isfinite(ghi) & np.isfinite(cs)
    fit_mask, val_mask = mask[:cut-1], mask[cut:]
    if not fit_mask.any() or not val_mask.any():
        raise ValueError("Fitting and validation require observed daylight")
    threads = params["nthread"]
    train = xgb.DMatrix(features[:cut-1][fit_mask],
                        label=(ghi-baseline)[:cut-1][fit_mask], nthread=threads)
    model = xgb.train(params, train, num_boost_round=max_rounds,
                      callbacks=[BudgetCallback(budget)])
    validation = xgb.DMatrix(features[cut:][val_mask], nthread=threads)
    truth, offset, solar = ghi[cut:][val_mask], baseline[cut:][val_mask], cs[cut:][val_mask]
    best = (np.inf, np.inf, 0)
    # The native DMatrix prediction cache extends each previous tree prefix.
    # Supplying iteration_range explicitly avoids native early-stopping ambiguity.
    for rounds in range(1, max_rounds+1):
        if budget is not None:
            budget.check()
        raw = model.predict(validation, iteration_range=(0, rounds))
        error = ghi_to_kt(raw + offset, solar) * solar - truth
        score = (float(np.sqrt(np.mean(error**2))), float(np.mean(abs(error))), rounds)
        if score[:2] < best[:2]:
            best = score
    full = xgb.DMatrix(features[mask], label=(ghi-baseline)[mask], nthread=threads)
    refitted = xgb.train(params, full, num_boost_round=best[2],
                         callbacks=[BudgetCallback(budget)])
    return refitted, {"target":"ghi_residual_to_nwp", "loss":"squared_error",
                       "selected_iterations":best[2], "iterations_checked":max_rounds,
                       "inner_train_days":cut-1, "inner_purge_days":1,
                       "inner_validation_days":n-cut, "refit_days":n,
                       "training_daylight_rows":int(mask.sum()),
                       "inner_validation_rmse_wm2":best[0], "inner_validation_mae_wm2":best[1],
                       "random_validation_split":False, "parameters":params}


def predict_kt(model, x, cs, baseline):
    if x.shape[:2] != cs.shape or baseline.shape != cs.shape or not np.isfinite(baseline).all():
        raise ValueError("Prediction arrays must align with finite NWP baseline")
    raw = model.inplace_predict(flat_features(x)).reshape(cs.shape)
    return ghi_to_kt(raw + baseline, cs)
