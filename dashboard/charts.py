"""Plotly figures with explicit units and local period-midpoint time axes."""
import pandas as pd
import plotly.graph_objects as go

BLUE = "#3877e8"
TEAL = "#12a58a"
AMBER = "#edac32"
CONFIG = {"displayModeBar": False, "scrollZoom": False, "displaylogo": False}


def base(height=310):
    fig = go.Figure()
    fig.update_layout(height=height, margin=dict(l=6, r=14, t=25, b=12),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, Arial, sans-serif", size=11, color="#75839a"),
                      hoverlabel=dict(bgcolor="#142239", font_color="white", bordercolor="#142239"),
                      hovermode="x unified", showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, tickfont_size=10),
                      yaxis=dict(gridcolor="#edf1f6", zeroline=False, rangemode="tozero", tickfont_size=10))
    return fig


def profile(frame, field="ghi_forecast_wm2", selected_hour=None, interval=None, mini=False):
    power = field == "pv_power_kw"
    color = TEAL if power else BLUE
    unit = "kW" if power else "W/m²"
    fig = base(120 if mini else 310)
    # Plotly receives naive LOCAL midpoints; tooltips explicitly identify IST.
    # This avoids a browser timezone silently shifting an already-local time axis.
    x = (frame.timestamp + pd.Timedelta(minutes=5)).dt.tz_localize(None)
    periods = frame.timestamp.dt.strftime("%H:%M") + "–" + frame.interval_end.dt.strftime("%H:%M")
    custom = [[str(label), int(slot)] for label, slot in zip(periods, frame.interval_index)]
    fig.add_trace(go.Scatter(x=x, y=frame[field], mode="lines", line=dict(color=color, width=2.5),
                            fill="tozeroy", fillcolor="rgba(18,165,138,.09)" if power else "rgba(56,119,232,.07)",
                            name="Estimated PV" if power else "Forecast GHI", customdata=custom,
                            hovertemplate="%{customdata[0]} IST<br>%{y:.2f} " + unit + "<extra></extra>"))
    if not mini and not frame.empty and frame[field].notna().all():
        peak = int(frame[field].to_numpy().argmax())
        fig.add_trace(go.Scatter(x=[x.iloc[peak]], y=[frame[field].iloc[peak]], mode="markers",
                                marker=dict(size=8,color=AMBER,line=dict(width=2,color="white")),
                                customdata=[custom[peak]], hovertemplate="Peak · %{customdata[0]} IST<br>%{y:.2f} " + unit + "<extra></extra>"))
    day = pd.Timestamp(frame.timestamp.iloc[0]).normalize().tz_localize(None)
    if len(frame) >= 100:
        fig.update_xaxes(range=[day, day + pd.Timedelta(days=1)], dtick=10_800_000, tickformat="%H:%M")
    else:
        start = pd.Timestamp(frame.timestamp.iloc[0]).tz_localize(None)
        end = pd.Timestamp(frame.interval_end.iloc[-1]).tz_localize(None)
        fig.update_xaxes(range=[start, end], dtick=600_000, tickformat="%H:%M")
    fig.update_yaxes(title_text=None if mini else unit, title_font_size=10)
    if selected_hour is not None:
        start = day + pd.Timedelta(hours=selected_hour)
        fig.add_vrect(x0=start,x1=start+pd.Timedelta(hours=1),fillcolor=AMBER,opacity=.10,line_width=0)
    if interval is not None:
        row = frame.loc[frame.interval_index == interval]
        if not row.empty:
            point = row.iloc[0]
            fig.add_vline(x=(point.timestamp+pd.Timedelta(minutes=5)).tz_localize(None).timestamp()*1000,
                          line_width=1,line_dash="dot",line_color=AMBER)
    if mini:
        fig.update_yaxes(showticklabels=False,showgrid=False)
        fig.update_layout(margin=dict(l=0,r=4,t=2,b=5))
    return fig


def hourly(rows, metric="ghi_mean_wm2", selected_hour=None):
    frame = pd.DataFrame(rows)
    color = BLUE if metric == "ghi_mean_wm2" else TEAL
    unit = {"ghi_mean_wm2":"W/m²", "pv_mean_kw":"kW", "energy_kwh":"kWh"}[metric]
    fig = base(245)
    fig.add_trace(go.Bar(x=frame.hour, y=frame[metric],
                        marker=dict(color=[AMBER if h==selected_hour else color for h in frame.hour],
                                    cornerradius=3),
                        customdata=[[f"{int(h):02d}:00–{int(h)+1:02d}:00",int(h)] for h in frame.hour],
                        hovertemplate="%{customdata[0]} IST<br>%{y:.2f} "+unit+"<extra></extra>"))
    fig.update_xaxes(tickvals=list(range(0,24,3)),ticktext=[f"{h:02d}:00" for h in range(0,24,3)],
                     range=[-.7,23.7],title_text="Local time · IST",title_font_size=10)
    fig.update_yaxes(title_text=unit,title_font_size=10)
    fig.update_layout(clickmode="event+select",dragmode=False,bargap=.32)
    return fig
