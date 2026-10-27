"""Bektaş & Altaş (2024) DWT-BiLSTM components, made issue-time safe.

Paper: "DWT-BILSTM-based models for day-ahead hourly global horizontal solar
irradiance forecasting", Neural Computing and Applications 36:13243-13253,
DOI 10.1007/s00521-024-09701-2.

Faithful parts: hourly sequences, discrete Meyer (dmey) six-level decomposition into
one approximation and six details (Eq. 8), weather type from daily GHI divided by
extraterrestrial irradiation with thresholds 0.25/0.45 (Eqs. 1-5), BiLSTM with ReLU
activation, MSE loss, Adam, batch 32, at most 500 epochs, patience 30.

Documented departures:
- The wavelet decomposition uses a trailing window that ends at the last hour of D−1.
  Decomposing the complete series would let later observations alter D−1 components.
- Eq. (4) prints 0.33; the standard eccentricity coefficient 0.033 is used.
- Early stopping restores the best validation weights.
"""
from __future__ import annotations

import copy

import numpy as np
import pywt

SOLAR_CONSTANT = 1367.0
K_SUNNY, K_CLOUDY = 0.45, 0.25
WAVELET, LEVELS = "dmey", 6
# 168 days = 4032 hours, the shortest whole-day window allowing six dmey levels
# without every coefficient being a boundary coefficient (pywt.dwt_max_level = 6).
WINDOW_DAYS = 168
N_COMPONENTS = LEVELS + 1


# ==========================================================================
# Weather type (paper Section 2.1)
# ==========================================================================

def extraterrestrial_daily_wh(doy, lat_deg: float) -> np.ndarray:
    """Daily extraterrestrial horizontal irradiation, Wh/m² (Eqs. 2-5)."""
    n = np.asarray(doy, dtype=float)
    phi = np.deg2rad(lat_deg)
    delta = np.deg2rad(23.45 * np.sin(np.deg2rad(360.0 * (284.0 + n) / 365.0)))
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))
    io = SOLAR_CONSTANT * (1.0 + 0.033 * np.cos(np.deg2rad(360.0 * n / 365.0)))
    return 24.0 * io / np.pi * (np.cos(phi) * np.cos(delta) * np.sin(ws)
                                + ws * np.sin(phi) * np.sin(delta))


def daily_clearness(hourly_ghi: np.ndarray, doy, lat_deg: float) -> np.ndarray:
    """K = daily GHI / extraterrestrial irradiation; NaN if any hour is missing."""
    daily_wh = np.asarray(hourly_ghi, dtype=float).sum(axis=1)  # 1-hour means → Wh/m²
    return daily_wh / extraterrestrial_daily_wh(doy, lat_deg)


def weather_types(k) -> np.ndarray:
    """0 = rainy (K < 0.25), 1 = cloudy, 2 = sunny (K > 0.45); NaN stays NaN."""
    k = np.asarray(k, dtype=float)
    out = np.where(k > K_SUNNY, 2.0, np.where(k > K_CLOUDY, 1.0, 0.0))
    out[~np.isfinite(k)] = np.nan
    return out


# ==========================================================================
# Discrete wavelet decomposition (paper Section 2.2)
# ==========================================================================

def mra(x: np.ndarray, wavelet: str = WAVELET, levels: int = LEVELS) -> np.ndarray:
    """(T, levels+1) additive components [A_L, D_L, ..., D_1].

    They sum exactly to the wavelet reconstruction of x. PyWavelets' dmey is a
    finite approximation of the Meyer wavelet, so that reconstruction differs from
    x by about 1% RMS on hourly GHI.
    """
    x = np.asarray(x, dtype=float)
    coeffs = pywt.wavedec(x, wavelet, mode="symmetric", level=levels)
    parts = []
    for j in range(len(coeffs)):
        keep = [c if i == j else np.zeros_like(c) for i, c in enumerate(coeffs)]
        parts.append(pywt.waverec(keep, wavelet, mode="symmetric")[: len(x)])
    return np.stack(parts, axis=-1)


def forward_fill_days(hourly: np.ndarray) -> np.ndarray:
    """Replace a missing day by the previous day: uses only earlier observations."""
    out = np.array(hourly, dtype=float)
    for i in range(1, len(out)):
        bad = ~np.isfinite(out[i])
        out[i, bad] = out[i - 1, bad]
    return out


def causal_dwt(hourly_filled: np.ndarray, day: int, window_days: int = WINDOW_DAYS) -> np.ndarray:
    """(24, 7) components of D−1, from a window ending at the last hour of D−1."""
    if day < window_days:
        raise ValueError("Not enough history for the wavelet window")
    series = hourly_filled[day - window_days: day].ravel()
    if not np.isfinite(series).all():
        raise ValueError("Wavelet window contains missing values")
    return mra(series)[-24:]


def dwt_by_day(hourly_filled: np.ndarray, days, window_days: int = WINDOW_DAYS) -> np.ndarray:
    out = np.full((len(hourly_filled), 24, N_COMPONENTS), np.nan)
    for day in days:
        out[day] = causal_dwt(hourly_filled, int(day), window_days)
    return out


# ==========================================================================
# Forecast-day inputs from NWP
# ==========================================================================

def quantile_map(source_fit, target_fit, values, n_quantiles: int = 101) -> np.ndarray:
    """Monotone empirical quantile mapping fitted only on the supplied pairs' marginals."""
    q = np.linspace(0.0, 1.0, n_quantiles)
    src = np.quantile(np.asarray(source_fit, float), q)
    dst = np.quantile(np.asarray(target_fit, float), q)
    # np.interp needs increasing x; ties from repeated values are harmless.
    src = np.maximum.accumulate(src)
    return np.interp(np.asarray(values, float), src, dst)


# ==========================================================================
# Inputs (paper Table 1)
# ==========================================================================

VARIANTS = ("oracle", "gfs_raw", "gfs_calibrated", "persistence")
VARIANT_NOTES = {
    "oracle": "forecast-day weather type (and Model 2 weather) from day-D observations, as published",
    "gfs_raw": "forecast-day weather type and weather from the archived GFS day-2 forecast",
    "gfs_calibrated": "GFS values after quantile mapping fitted on pre-test development days",
    "persistence": "forecast-day weather type and weather copied from D−1 (no forecast)",
}


def forecast_day_inputs(sources: dict, day: int, variant: str):
    """Weather type and hourly weather assumed for day D under each variant."""
    if variant == "oracle":
        return sources["type_obs"][day], sources["met_obs"][day]
    if variant == "persistence":
        return sources["type_obs"][day - 1], sources["met_obs"][day - 1]
    if variant in ("gfs_raw", "gfs_calibrated"):
        suffix = variant.split("_", 1)[1]
        return sources[f"type_gfs_{suffix}"][day], sources[f"met_gfs_{suffix}"][day]
    raise ValueError(f"Unknown variant {variant!r}")


def build_inputs(sources: dict, days, model: int, variant: str, norm: dict) -> np.ndarray:
    """(n, 24, 9) for Model 1, (n, 24, 15) for Model 2, normalized with training statistics.

    Model 1: seven D−1 DWT components, weather type D−1, weather type D.
    Model 2: adds temperature, humidity and wind speed of D−1 and of D.
    """
    if model not in (1, 2):
        raise ValueError("model must be 1 or 2")
    rows = []
    for day in days:
        day = int(day)
        type_d, met_d = forecast_day_inputs(sources, day, variant)
        parts = [sources["dwt"][day] / norm["ghi_scale"],
                 np.full((24, 1), sources["type_obs"][day - 1] / 2.0),
                 np.full((24, 1), type_d / 2.0)]
        if model == 2:
            lo, span = norm["met_min"], norm["met_span"]
            parts += [(sources["met_obs"][day - 1] - lo) / span, (met_d - lo) / span]
        rows.append(np.concatenate(parts, axis=1))
    x = np.stack(rows).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite inputs for model {model}, variant {variant}")
    return x


# ==========================================================================
# BiLSTM with ReLU cell activation (paper Sections 2.3-2.4)
# ==========================================================================

def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def make_network(n_features: int, hidden: int):
    """Keras-style Bidirectional(LSTM(hidden, activation='relu')) + per-hour Dense(1)."""
    torch, nn = _torch()

    class Direction(nn.Module):
        def __init__(self):
            super().__init__()
            self.kernel = nn.Linear(n_features, 4 * hidden)
            self.recurrent = nn.Linear(hidden, 4 * hidden, bias=False)
            # Keras defaults: Glorot kernel, orthogonal recurrent, unit forget bias.
            nn.init.xavier_uniform_(self.kernel.weight)
            for block in self.recurrent.weight.data.chunk(4, 0):
                nn.init.orthogonal_(block)
            nn.init.zeros_(self.kernel.bias)
            with torch.no_grad():
                self.kernel.bias[hidden:2 * hidden] = 1.0

        def forward(self, x, reverse: bool):
            batch, steps, _ = x.shape
            h = x.new_zeros(batch, hidden)
            c = x.new_zeros(batch, hidden)
            projected = self.kernel(x)
            order = range(steps - 1, -1, -1) if reverse else range(steps)
            out = [None] * steps
            for t in order:
                i, f, g, o = (projected[:, t] + self.recurrent(h)).chunk(4, dim=1)
                c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.relu(g)
                h = torch.sigmoid(o) * torch.relu(c)
                out[t] = h
            return torch.stack(out, dim=1)

    class PaperBiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_dir = Direction()
            self.backward_dir = Direction()
            self.head = nn.Linear(2 * hidden, 1)
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, x):
            h = torch.cat([self.forward_dir(x, False), self.backward_dir(x, True)], dim=-1)
            return self.head(h).squeeze(-1)

    return PaperBiLSTM()


def train_network(x_fit, y_fit, x_val, y_val, hidden: int, seed: int, *,
                  max_epochs: int = 500, patience: int = 30, batch_size: int = 32,
                  learning_rate: float = 1e-3):
    """Adam/MSE with shuffled mini-batches; stop after `patience` epochs without
    validation improvement and keep the best validation weights."""
    torch, nn = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_network(x_fit.shape[-1], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    xf, yf = torch.from_numpy(x_fit), torch.from_numpy(y_fit)
    xv, yv = torch.from_numpy(x_val), torch.from_numpy(y_val)
    best, best_state, best_epoch, stale, history = np.inf, None, 0, 0, []
    for epoch in range(1, max_epochs + 1):
        net.train()
        order = rng.permutation(len(xf))
        for start in range(0, len(order), batch_size):
            idx = torch.from_numpy(order[start:start + batch_size])
            opt.zero_grad()
            loss = loss_fn(net(xf[idx]), yf[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(loss_fn(net(xv), yv))
        if not np.isfinite(val):
            break
        history.append(val)
        if val < best:
            best, best_state, best_epoch, stale = val, copy.deepcopy(net.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training diverged before any finite validation loss")
    net.load_state_dict(best_state)
    return net, {"hidden": hidden, "best_epoch": best_epoch, "epochs_run": len(history),
                 "best_validation_mse": best, "validation_mse_history": history}


def predict(net, x: np.ndarray) -> np.ndarray:
    torch, _ = _torch()
    net.eval()
    with torch.no_grad():
        return net(torch.from_numpy(x)).numpy().astype(float)
