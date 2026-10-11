"""Local Streamlit dashboard for saved Mumbai day-ahead forecasts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.backend.forecast_data import day_report, discover_sources, load_forecast
from pv_pipeline.pv_model import PVParameters


st.set_page_config(page_title="SolarCast · Mumbai", page_icon="☀️", layout="wide")

st.markdown("""
<style>
  .stApp { background: #f4f7f5; color: #172d29; }
  [data-testid="stSidebar"] { background: #e7f1ed; }
  .block-container { max-width: 1450px; padding-top: 1.8rem; }
  .hero { background: linear-gradient(115deg, #0b413c, #127463);
          border-radius: 20px; padding: 28px 34px; color: white; margin-bottom: 22px; }
  .eyebrow { color: #abdcd2; font-size: .8rem; font-weight: 700;
             letter-spacing: .12em; text-transform: uppercase; }
  .hero h1 { color: white; margin: 8px 0 4px; font-size: 2.25rem; }
  .hero p { color: #e1f4ee; margin: 0; }
  [data-testid="stMetric"] { background: white; border: 1px solid #dde9e4;
                            border-radius: 14px; padding: 16px 18px; }
  [data-testid="stMetricLabel"] { color: #53736a; }
  [data-testid="stMetricValue"] { color: #0b5146; }
  h2, h3 { color: #164a41; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading forecast arrays…")
def cached_forecast(base: str, base_mtime: int, stage2: str, stage2_mtime: int):
    # File modification times invalidate cache when a training run finishes.
    return load_forecast(base_path=base, stage2_path=stage2 or None)


st.markdown("""
<div class="hero">
  <div class="eyebrow">Mumbai · Solar forecasting</div>
  <h1>Day-ahead GHI &amp; PV outlook</h1>
  <p>144 ten-minute forecasts per day · issue time 00:00 IST · historical test-year replay</p>
</div>
""", unsafe_allow_html=True)

sources = discover_sources()
if not sources:
    st.error("No saved forecast was found. Run the DWT-BiLSTM trainer first.")
    st.stop()

preferred = next((i for i, name in enumerate(sources)
                  if name == "dwt_10min_150_paper/dwt · corrected"), None)
if preferred is None:
    preferred = next((i for i, name in enumerate(sources)
                      if name.startswith("15epochs/") and name.endswith("· base")), 0)
with st.sidebar:
    st.header("Forecast controls")
    choice = st.selectbox("Prediction run", list(sources), index=preferred)
    base_path, stage2_path = sources[choice]
    st.caption("Only complete held-out test-day predictions can be displayed.")
    st.divider()
    st.subheader("PV system")
    capacity = st.number_input("Rated capacity (kW)", min_value=0.1, max_value=100000.0,
                               value=100.0, step=10.0)
    noct = st.number_input("NOCT (°C)", min_value=20.0, max_value=80.0,
                           value=45.0, step=1.0)
    beta_pct = st.number_input("Temperature loss (% / °C)", min_value=0.0,
                               max_value=2.0, value=0.45, step=0.01)
    st.caption("PV power is calculated when you change these controls; the GHI model is unchanged.")

try:
    store = cached_forecast(str(base_path), base_path.stat().st_mtime_ns,
                            str(stage2_path) if stage2_path else "",
                            stage2_path.stat().st_mtime_ns if stage2_path else 0)
except (OSError, KeyError, ValueError) as exc:
    st.error(f"Cannot load this prediction run: {exc}")
    st.stop()

available = store.available_test_idx
if available.size == 0:
    st.warning("This run has no complete test-year forecasts yet. Select the 15epochs run, or finish training this run.")
    st.stop()

with st.sidebar:
    selected = st.selectbox("Forecast date (IST)", list(available[::-1]),
                            format_func=lambda i: str(store.dates[i]))

params = PVParameters(capacity_kw=capacity, noct_c=noct,
                      temperature_coefficient=beta_pct / 100.0)
frame, metrics = day_report(store, int(selected), params)

st.caption(f"**{metrics['date']}** · {choice} · {len(available)} available test days · source: {store.source.relative_to(ROOT)}")
cards = st.columns(4)
cards[0].metric("Peak predicted GHI", f"{metrics['peak_predicted_ghi_wm2']:,.0f} W/m²")
cards[1].metric("Peak estimated PV", f"{metrics['peak_predicted_pv_kw']:,.1f} kW")
cards[2].metric("Predicted PV energy", f"{metrics['predicted_energy_kwh']:,.1f} kWh")
cards[3].metric("Daylight GHI RMSE", f"{metrics['daylight_rmse_wm2']:,.1f} W/m²")

st.subheader("Global horizontal irradiance")
st.caption("Actual = Solcast satellite estimate. Predicted = saved 00:00 IST day-ahead forecast. All 144 periods are shown.")
ghi_cols = ["Actual GHI (W/m²)", "Predicted GHI (W/m²)"]
if "Predicted GHI lower (W/m²)" in frame:
    ghi_cols += ["Predicted GHI lower (W/m²)", "Predicted GHI upper (W/m²)"]
st.line_chart(frame, x="Time (IST)", y=ghi_cols,
              color=["#193f38", "#e6a23c"] + (["#99bfb5", "#99bfb5"] if len(ghi_cols) > 2 else []),
              height=370)

left, right = st.columns([2, 1])
with left:
    st.subheader("Estimated PV power")
    st.caption("NOCT model using the selected system settings; forecast uses previous three days' temperature.")
    st.line_chart(frame, x="Time (IST)",
                  y=["Estimated PV from actual GHI (kW)", "Estimated PV from predicted GHI (kW)"],
                  color=["#193f38", "#e6a23c"], height=320)
with right:
    st.subheader("Daily report")
    st.metric("Estimated energy from actual GHI", f"{metrics['actual_estimated_energy_kwh']:,.1f} kWh")
    st.metric("Daylight MAE", f"{metrics['daylight_mae_wm2']:,.1f} W/m²")
    st.metric("Scored daylight periods", f"{metrics['scored_steps']} / 144")

with st.expander("Ten-minute data and calculation details", expanded=False):
    st.dataframe(frame, hide_index=True)
    st.markdown("""
    **PV calculation:** `T_cell = T_air + (NOCT − 20) × GHI / 800`;
    `P_kW = max(0, capacity_kW × GHI / 1000 × [1 − β × (T_cell − 25)])`.
    Daily energy is the sum of ten-minute mean power divided by 6.

    Predicted PV uses a same-time median of the preceding three days' air temperature.
    The actual-GHI comparison uses that day's observed air temperature, so it is a
    historical reference, not a value available when the forecast was issued.
    These curves estimate horizontal-irradiance PV output; they do not include tilt,
    shading, inverter clipping, or measured plant production.
    """)

st.download_button("Download this day's CSV", frame.to_csv(index=False).encode("utf-8"),
                   file_name=f"mumbai_forecast_{metrics['date']}.csv", mime="text/csv")
