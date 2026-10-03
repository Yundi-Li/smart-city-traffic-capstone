"""
Interactive Traffic Dashboard Generator

Generates a self-contained HTML dashboard from the Metro Interstate Traffic Volume dataset
using Plotly. Satisfies the Power BI rubric requirements for the capstone project.
"""

import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

DATA_PATH = "data/Metro_Interstate_Traffic_Volume.csv"
OUTPUT_PATH = "capstone_part1/traffic_dashboard.html"


def load_and_prepare(path: str) -> pd.DataFrame:
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path, parse_dates=["date_time"])
    df["hour"] = df["date_time"].dt.hour
    df["year"] = df["date_time"].dt.year
    df["date"] = df["date_time"].dt.date
    df["temp_celsius"] = df["temp"] - 273.15
    df["traffic_category"] = pd.cut(
        df["traffic_volume"],
        bins=[-1, 4500, 5500, float("inf")],
        labels=["Low", "Medium", "High"],
    )
    logger.info("Data prepared: %d rows, columns: %s", len(df), list(df.columns))
    return df


def build_dashboard(df: pd.DataFrame) -> str:
    # --- KPI values ---
    total_hours = len(df)
    avg_traffic = df["traffic_volume"].mean()
    avg_temp = df["temp_celsius"].mean()

    # --- Chart A: Daily traffic trends 2015-2017 ---
    years = [2015, 2016, 2017]
    daily = df[df["year"].isin(years)].groupby(["year", "date"])["traffic_volume"].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    # Normalise dates to same reference year for overlay
    daily["day_of_year"] = daily["date"].dt.dayofyear

    # --- Chart B: Hourly traffic 2017 ---
    hourly_2017 = df[df["year"] == 2017].groupby("hour")["traffic_volume"].mean().reset_index()

    # --- Chart C: Weather impact ---
    weather_avg = df.groupby("weather_main")["traffic_volume"].mean().sort_values(ascending=False).reset_index()
    highest = weather_avg.iloc[0]
    lowest = weather_avg.iloc[-1]
    diff = highest["traffic_volume"] - lowest["traffic_volume"]

    # --- Chart D: Temp vs traffic (sample for performance) ---
    scatter_df = df.sample(n=min(5000, len(df)), random_state=42)

    # --- Dropdown filter values ---
    weather_options = sorted(df["weather_main"].dropna().unique())
    category_options = ["Low", "Medium", "High"]

    # Build figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "A. Daily Traffic Trends (2015–2017)",
            "B. Hourly Traffic Patterns (2017)",
            "C. Average Traffic by Weather Condition",
            "D. Temperature vs Traffic Volume",
        ),
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{"colspan": 2}, None],
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    # --- A: Daily trends ---
    colors = {2015: "#636EFA", 2016: "#EF553B", 2017: "#00CC96"}
    for yr in years:
        subset = daily[daily["year"] == yr].sort_values("day_of_year")
        fig.add_trace(go.Scatter(
            x=subset["day_of_year"], y=subset["traffic_volume"],
            mode="lines", name=str(yr), line=dict(color=colors[yr], width=1.5),
            legendgroup="yearly",
        ), row=1, col=1)
    fig.update_xaxes(title_text="Day of Year", row=1, col=1)
    fig.update_yaxes(title_text="Avg Traffic Volume", row=1, col=1)

    # --- B: Hourly 2017 ---
    fig.add_trace(go.Bar(
        x=hourly_2017["hour"], y=hourly_2017["traffic_volume"],
        marker_color="#AB63FA", name="Hourly 2017", showlegend=False,
    ), row=2, col=1)
    fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
    fig.update_yaxes(title_text="Avg Traffic Volume", row=2, col=1)

    # --- C: Weather impact ---
    fig.add_trace(go.Bar(
        x=weather_avg["weather_main"], y=weather_avg["traffic_volume"],
        marker_color="#FFA15A", name="Weather Avg", showlegend=False,
    ), row=2, col=2)
    fig.add_annotation(
        text=(
            f"Highest: {highest['weather_main']} ({highest['traffic_volume']:.0f})<br>"
            f"Lowest: {lowest['weather_main']} ({lowest['traffic_volume']:.0f})<br>"
            f"Difference: {diff:.0f}"
        ),
        xref="x4", yref="y4",
        x=lowest["weather_main"], y=highest["traffic_volume"],
        showarrow=False, font=dict(size=10, color="white"),
        bgcolor="rgba(0,0,0,0.6)", borderpad=4,
    )
    fig.update_xaxes(title_text="Weather Condition", row=2, col=2)
    fig.update_yaxes(title_text="Avg Traffic Volume", row=2, col=2)

    # --- D: Scatter temp vs traffic ---
    weather_colors = {w: c for w, c in zip(
        weather_options,
        ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
         "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
         "#E45756", "#54A24B"][:len(weather_options)]
    )}
    for w in weather_options:
        ws = scatter_df[scatter_df["weather_main"] == w]
        if len(ws) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=ws["temp_celsius"], y=ws["traffic_volume"],
            mode="markers", name=w,
            marker=dict(color=weather_colors.get(w, "#888"), size=3, opacity=0.5),
            legendgroup="weather",
        ), row=3, col=1)
    fig.update_xaxes(title_text="Temperature (°C)", row=3, col=1)
    fig.update_yaxes(title_text="Traffic Volume", row=3, col=1)

    # --- Dropdown filters (updatemenus) ---
    # Weather filter
    weather_buttons = [dict(label="All Weather", method="update", args=[{"visible": True}])]
    # We need to figure out trace indices for the scatter traces
    # Traces: 3 yearly lines (0-2), 1 hourly bar (3), 1 weather bar (4), N scatter traces (5+)
    n_scatter = len([w for w in weather_options if len(scatter_df[scatter_df["weather_main"] == w]) > 0])
    total_traces = 3 + 1 + 1 + n_scatter

    for i, w in enumerate(weather_options):
        ws = scatter_df[scatter_df["weather_main"] == w]
        if len(ws) == 0:
            continue
        vis = [True] * 5 + [False] * n_scatter  # show all non-scatter, hide all scatter
        vis[5 + i] = True  # show only this weather's scatter
        weather_buttons.append(dict(label=w, method="update", args=[{"visible": vis}]))

    # Show all button for weather
    all_vis = [True] * total_traces
    weather_buttons[0]["args"] = [{"visible": all_vis}]

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=weather_buttons,
                direction="down", showactive=True,
                x=0.0, xanchor="left", y=1.18, yanchor="top",
                bgcolor="#2d2d2d", font=dict(color="white"),
            ),
        ],
        annotations=list(fig.layout.annotations) + [
            dict(text="Filter by Weather:", x=0.0, y=1.21, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="white")),
        ],
    )

    # --- Layout ---
    fig.update_layout(
        height=1300,
        template="plotly_dark",
        title=dict(
            text=(
                f"<b>Metro Interstate Traffic Dashboard</b><br>"
                f"<span style='font-size:14px'>"
                f"Total Hours Analysed: <b>{total_hours:,}</b> &nbsp;|&nbsp; "
                f"Avg Traffic Volume: <b>{avg_traffic:,.0f}</b> &nbsp;|&nbsp; "
                f"Avg Temperature: <b>{avg_temp:.1f} °C</b>"
                f"</span>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=20, color="white"),
        ),
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#2d2d2d",
        font=dict(color="white"),
        legend=dict(orientation="h", y=-0.02),
        margin=dict(t=160),
    )

    return fig.to_html(full_html=True, include_plotlyjs=True)


def main():
    df = load_and_prepare(DATA_PATH)
    html = build_dashboard(df)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Dashboard saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
