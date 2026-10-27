"""Streamlit UI. Run via dashboard/run.py, or use the documented separate services."""
from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard import charts
from dashboard.calculations import csv_bytes, select_view, summary
from dashboard.data_service import DataServiceError, get_dates, get_forecast, mode

st.set_page_config(page_title="Mumbai Solar Forecast",page_icon="☀",layout="wide",initial_sidebar_state="expanded")
st.markdown(f"<style>{(Path(__file__).parent/'assets/style.css').read_text()}</style>",unsafe_allow_html=True)


def navigate(view):
    st.session_state.view = view


def reset_scope():
    st.session_state.view = "Day"
    st.session_state.hour_picker = 12
    st.session_state.interval_picker = 72
    st.session_state.inspector = 72


def change_hour(delta=0):
    hour = max(0,min(23,st.session_state.get("hour_picker",12)+delta))
    st.session_state.hour_picker = hour
    st.session_state.interval_picker = hour*6
    st.session_state.inspector = hour*6


def inspect_interval():
    slot = st.session_state.inspector
    st.session_state.interval_picker = slot
    st.session_state.hour_picker = slot//6
    st.session_state.view = "10-minute"


def select_hour_chart():
    event = st.session_state.get("hourly_chart",{})
    points = event.get("selection",{}).get("points",[])
    if points:
        hour = int(points[-1]["x"])
        st.session_state.hour_picker = hour
        change_hour()
        st.session_state.view = "Hour"


def clear_pv():
    st.session_state.capacity_kwp = None
    st.session_state.performance_ratio = None


def heading(title,note=""):
    st.markdown(f'<div class="section-title">{escape(title)}</div><div class="section-note">{escape(note)}</div>',unsafe_allow_html=True)


def value(number,places=1):
    return "—" if number is None else f"{number:,.{places}f}"


def cards(metrics,scope):
    entries=[("PEAK FORECAST GHI",value(metrics["peak_ghi_wm2"]),"W/m²","☀","",scope),
             ("PEAK ESTIMATED PV",value(metrics["peak_pv_kw"],2),"kW","↗","teal","GHI-based estimate" if metrics["peak_pv_kw"] is not None else "Set PV parameters to enable"),
             ("ESTIMATED ENERGY",value(metrics["energy_kwh"],2),"kWh","ϟ","teal",scope if metrics["energy_kwh"] is not None else "Set PV parameters to enable"),
             ("PEAK IRRADIANCE TIME",metrics["peak_time"] or "—","IST","◷","amber","Start of peak 10-minute period"),
             ("DAYTIME MEAN GHI",value(metrics["daytime_mean_ghi_wm2"]),"W/m²","◒","","Daylight samples in this view")]
    html='<div class="kpi-grid">'
    for label,num,unit,icon,color,note in entries:
        html+=f'<div class="solar-kpi"><div class="kpi-top"><span class="kpi-label">{label}</span><span class="kpi-icon {color}">{icon}</span></div><div class="kpi-value">{escape(num)}<span class="kpi-unit">{unit}</span></div><div class="kpi-note">{escape(note)}</div></div>'
    st.markdown(html+'</div>',unsafe_allow_html=True)


def main():
    for key,default in {"view":"Day","hour_picker":12,"interval_picker":72,"inspector":72}.items():
        st.session_state.setdefault(key,default)
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">☀</div><div><div class="brand-name">SOLAR</div><div class="brand-sub">MUMBAI WORKSPACE</div></div></div>',unsafe_allow_html=True)
        st.markdown('<div class="side-label">EXPLORE</div>',unsafe_allow_html=True)
        st.button("◫  Forecast overview",key="overview_nav",width="stretch",type="primary" if st.session_state.view=="Day" else "secondary",on_click=navigate,args=("Day",))
        st.button("◷  Interval explorer",key="explorer_nav",width="stretch",type="primary" if st.session_state.view!="Day" else "secondary",on_click=navigate,args=("Hour",))
        st.markdown('<div class="side-label">PV SYSTEM</div>',unsafe_allow_html=True)
        with st.expander("Configure estimated output",expanded=True):
            capacity=st.number_input("Installed DC capacity (kWp)",min_value=.01,value=None,step=.1,key="capacity_kwp",placeholder="Enter system capacity")
            ratio=st.number_input("Performance ratio (0–1)",min_value=.01,max_value=1.0,value=None,step=.01,key="performance_ratio",placeholder="Enter performance ratio")
            st.caption("Enter both values and press Enter after each. GHI is used as a plane-of-array proxy.")
            st.button("Clear PV settings",on_click=clear_pv,width="stretch")
    try:
        available=get_dates()
        if not available:
            raise DataServiceError("There are no saved forecasts to display.")
    except DataServiceError as exc:
        st.markdown('<div class="eyebrow">MUMBAI · INDIA</div><div class="page-title">Solar Energy Forecast Dashboard</div>',unsafe_allow_html=True)
        st.warning(str(exc))
        st.info("Forecast data is temporarily unavailable. Retry when the service is available.")
        st.button("Retry connection")
        st.stop()
    with st.sidebar:
        st.markdown(f'<div class="archive-panel"><div class="source-badge"><span class="source-dot"></span>Saved forecast archive</div><div class="archive-title">{len(available)} forecast days available</div><div class="archive-copy">{date.fromisoformat(available[0]):%d %b %Y} – {date.fromisoformat(available[-1]):%d %b %Y}<br>10-minute resolution · IST</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="side-foot">Weather forecasts: Open-Meteo · NOAA GFS, ECMWF IFS, DWD ICON<br>Historical reference: Solcast<br>All times shown in Indian Standard Time.</div>',unsafe_allow_html=True)
    title,controls=st.columns([1.85,1.15],gap="large",vertical_alignment="center")
    with title:
        st.markdown('<div class="eyebrow">SOLAR INTELLIGENCE · MUMBAI, INDIA</div><div class="page-title">Solar Energy Forecast Dashboard</div><div class="page-subtitle">Day-ahead outlook · 144 ten-minute periods · midnight IST issue</div>',unsafe_allow_html=True)
    with controls:
        picker,refresh,reset=st.columns([3,1,1],vertical_alignment="bottom")
        with picker:
            selected=st.date_input("Forecast date",value=date.fromisoformat(available[-1]),min_value=date.fromisoformat(available[0]),max_value=date.fromisoformat(available[-1]),key="forecast_date",on_change=reset_scope)
        with refresh:
            st.button("↻",help="Reload this forecast from the service",width="stretch",key="refresh")
        with reset:
            st.button("⟲",help="Reset exploration to day view",width="stretch",on_click=reset_scope,key="reset")
    if selected.isoformat() not in available:
        st.warning("No saved forecast is available for this date. Choose another archive date.")
        st.stop()
    parameters_ready=capacity is not None and ratio is not None
    try:
        payload,frame=get_forecast(selected,capacity if parameters_ready else None,ratio if parameters_ready else None)
    except DataServiceError as exc:
        st.warning(str(exc));st.button("Retry forecast");st.stop()
    badge="API connected" if mode()=="api" else "Connected data"
    st.markdown(f'<span class="source-badge"><span class="source-dot"></span>{badge} · Saved forecast</span>',unsafe_allow_html=True)
    if not parameters_ready:
        missing=" and ".join(label for label,item in [("installed DC capacity (kWp)",capacity),("performance ratio (0–1)",ratio)] if item is None)
        st.info(f"PV estimates are off. In the sidebar, open ‘Configure estimated output’ and enter {missing}. Press Enter after each value to calculate power and energy.")
    scope,levels=st.columns([1.1,1],vertical_alignment="center")
    with scope:
        st.markdown(f'<div class="scope-label" style="margin-top:20px">FORECAST EXPLORER</div><div class="scope-title">{selected:%A, %d %B %Y}</div>',unsafe_allow_html=True)
    with levels:
        st.segmented_control("Explore resolution",["Day","Hour","10-minute"],key="view",selection_mode="single",required=True,width="stretch",label_visibility="collapsed")
    view=st.session_state.view
    if view!="Day":
        back,hourcol,prev,next_,intervalcol=st.columns([1.3,2,.55,.55,2.5],vertical_alignment="bottom")
        with back:st.button("← Day view",on_click=navigate,args=("Day",),width="stretch")
        with hourcol:st.selectbox("Hour · IST",range(24),format_func=lambda h:f"{h:02d}:00–{h+1:02d}:00",key="hour_picker",on_change=change_hour)
        with prev:st.button("‹",key="prev_hour",help="Previous hour",disabled=st.session_state.hour_picker==0,on_click=change_hour,args=(-1,),width="stretch")
        with next_:st.button("›",key="next_hour",help="Next hour",disabled=st.session_state.hour_picker==23,on_click=change_hour,args=(1,),width="stretch")
        hour=st.session_state.hour_picker
        if st.session_state.interval_picker//6!=hour:st.session_state.interval_picker=hour*6
        with intervalcol:
            if view=="10-minute":
                st.selectbox("10-minute period",range(hour*6,(hour+1)*6),format_func=lambda s:f"{s//6:02d}:{s%6*10:02d}–{(s+1)//6:02d}:{(s+1)%6*10:02d}",key="interval_picker")
            else:st.caption("6 ten-minute intervals in this hour")
    hour=None if view=="Day" else st.session_state.hour_picker
    interval=st.session_state.interval_picker if view=="10-minute" else None
    current=select_view(frame,hour,interval)
    scope_note={"Day":"Complete 24-hour forecast","Hour":"Selected hour","10-minute":"Selected 10-minute interval"}[view]
    cards(summary(current),scope_note)
    ghi_col,pv_col=st.columns([1.5,1],gap="medium")
    # Keep all six points visible when inspecting one ten-minute interval.
    plot_frame=select_view(frame,hour) if view=="10-minute" else current
    with ghi_col,st.container(border=True,key="panel_ghi"):
        heading("Solar irradiance forecast","GHI · ten-minute period means · W/m²")
        st.plotly_chart(charts.profile(plot_frame,interval=interval),width="stretch",config=charts.CONFIG,key="ghi_profile")
    with pv_col,st.container(border=True,key="panel_pv"):
        heading("Estimated PV power","GHI-based illustrative output · kW")
        if parameters_ready:
            st.plotly_chart(charts.profile(plot_frame,"pv_power_kw",interval=interval),width="stretch",config=charts.CONFIG,key="pv_profile")
            st.caption(f"{capacity:g} kWp DC · performance ratio {ratio:.2f} · no measured plant-power validation")
        else:
            st.markdown('<div class="pv-empty"><div class="empty-icon">ϟ</div><div class="empty-title">Add your PV system</div><div class="empty-copy">Enter installed capacity and performance ratio in the sidebar to view estimated power and energy.</div></div>',unsafe_allow_html=True)
    hourlycol,detailcol=st.columns([1.7,1],gap="medium")
    with hourlycol,st.container(border=True,key="panel_hourly"):
        heading("Hourly solar generation","Choose a bar or use the hour selector to explore six intervals.")
        choices=["GHI"]+(["PV power","Energy"] if parameters_ready else [])
        if st.session_state.get("hourly_metric") not in choices:
            st.session_state.hourly_metric="GHI"
        metric=st.segmented_control("Hourly measure",choices,key="hourly_metric",required=True,label_visibility="collapsed")
        mapping={"GHI":"ghi_mean_wm2","PV power":"pv_mean_kw","Energy":"energy_kwh"}
        st.plotly_chart(charts.hourly(payload["hourly"],mapping.get(metric,"ghi_mean_wm2"),hour),width="stretch",config=charts.CONFIG,key="hourly_chart",on_select=select_hour_chart,selection_mode="points")
    with detailcol,st.container(border=True,key="panel_inspector"):
        heading("Interval inspector","Exact ten-minute period · local time in IST")
        # Stable manual selection remains usable without Plotly click events.
        if interval is not None:st.session_state.inspector=interval
        inspect_options=list(current.interval_index) if view!="Day" else list(range(144))
        if st.session_state.inspector not in inspect_options:st.session_state.inspector=inspect_options[0]
        slot=st.selectbox("Select interval",inspect_options,format_func=lambda s:f"{s//6:02d}:{s%6*10:02d} IST",key="inspector",on_change=inspect_interval,label_visibility="collapsed")
        row=frame.loc[frame.interval_index==slot].iloc[0]
        st.markdown(f'<div class="interval-header"><div class="interval-time">{row.timestamp:%H:%M} – {row.interval_end:%H:%M}<span class="kpi-unit">IST</span></div><div class="interval-date">{selected:%d %B %Y} · interval {slot+1} of 144</div></div>',unsafe_allow_html=True)
        for label,num,unit in [("Forecast GHI",value(float(row.ghi_forecast_wm2),2),"W/m²"),("Estimated PV",value(float(row.pv_power_kw),3) if parameters_ready else "—","kW")]:
            st.markdown(f'<div class="interval-tile"><span class="interval-key">{label}</span><span class="interval-value">{num} <span class="kpi-unit">{unit}</span></span></div>',unsafe_allow_html=True)
        st.caption("Daylight period" if row.daylight else "Outside the daylight evaluation window")
    if view!="Day":
        with st.container(border=True,key="panel_overview"):
            heading("Daily forecast overview","Your selected hour is highlighted within the complete day.")
            st.plotly_chart(charts.profile(frame,selected_hour=hour,interval=interval,mini=True),width="stretch",config=charts.CONFIG,key="day_overview")
    with st.expander(f"Detailed forecast · {len(current)} intervals",expanded=False):
        display=current.copy()
        display["Period start (IST)"]=display.timestamp.dt.strftime("%Y-%m-%d %H:%M")
        display["Period end (IST)"]=display.interval_end.dt.strftime("%H:%M")
        display=display.rename(columns={"ghi_forecast_wm2":"GHI forecast (W/m²)","pv_power_kw":"Estimated PV (kW)","hour":"Hour","interval_index":"Interval index"})
        columns=["Period start (IST)","Period end (IST)","GHI forecast (W/m²)"]+(["Estimated PV (kW)"] if parameters_ready else [])+["Hour","Interval index"]
        st.dataframe(display[columns],hide_index=True,width="stretch")
    download,description=st.columns([1,2],vertical_alignment="center")
    with download:st.download_button("↓  Download this view as CSV",data=csv_bytes(current),file_name=f"mumbai-solar-{selected}-{view.lower()}.csv",mime="text/csv",width="stretch")
    with description:st.caption(f"{len(current)} periods · timezone-aware timestamps · "+("GHI and estimated PV" if parameters_ready else "GHI forecasts"))
    st.markdown('<div class="footer"><span>Mumbai solar forecast workspace · IST (UTC+05:30)</span><span>Saved forecasts · illustrative PV estimates when configured · not a live feed</span></div>',unsafe_allow_html=True)


if __name__=="__main__":
    main()
