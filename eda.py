"""
Descriptive Analysis & Visualization Module
=============================================
Produces 6 Matplotlib visualizations + a descriptive-statistics summary
for the CA-2 portfolio presentation (Descriptive Analysis & Visualization
criterion, 5 marks: comprehensive analysis + 5+ meaningful Matplotlib
visualizations with interpretation).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VIS_DIR = Path(__file__).resolve().parent.parent / "visuals"
VIS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.titleweight": "bold",
    "font.size": 11,
})

ACCENT = "#1f6feb"
ACCENT2 = "#e8622c"
ACCENT3 = "#2ca58d"


def load():
    stations = pd.read_csv(DATA_DIR / "station_master_clean.csv")
    hist = pd.read_csv(DATA_DIR / "hourly_demand_history.csv", parse_dates=["date"])
    return stations, hist


def fig1_capacity_distribution(stations):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(stations["capacity"], bins=20, color=ACCENT, edgecolor="white")
    ax.set_title("Distribution of Real Station Dock Capacity (Citi Bike NYC GBFS)")
    ax.set_xlabel("Station capacity (docks)")
    ax.set_ylabel("Number of stations")
    fig.tight_layout()
    fig.savefig(VIS_DIR / "01_capacity_distribution.png", dpi=150)
    plt.close(fig)


def fig2_occupancy_by_tier(stations):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ["Low-traffic (<=20 docks)", "Mid-traffic (21-40 docks)", "High-traffic (40+ docks)"]
    data = [stations.loc[stations["tier"] == t, "occupancy_rate"] for t in order]
    bp = ax.boxplot(data, tick_labels=["Low", "Mid", "High"], patch_artist=True)
    for patch, c in zip(bp["boxes"], [ACCENT3, ACCENT, ACCENT2]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_title("Live Bike Occupancy Rate by Station Traffic Tier")
    ax.set_xlabel("Station tier")
    ax.set_ylabel("Occupancy rate (bikes / capacity)")
    fig.tight_layout()
    fig.savefig(VIS_DIR / "02_occupancy_by_tier.png", dpi=150)
    plt.close(fig)


def fig3_hourly_demand_pattern(hist):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    g = hist.groupby(["hour", "is_weekend"])["net_demand"].mean().reset_index()
    for wk, label, color in [(0, "Weekday", ACCENT), (1, "Weekend", ACCENT2)]:
        sub = g[g["is_weekend"] == wk]
        ax.plot(sub["hour"], sub["net_demand"], marker="o", label=label, color=color, linewidth=2)
    ax.axhline(0, color="#999", linewidth=1)
    ax.set_title("Average Net Station Demand by Hour of Day")
    ax.set_xlabel("Hour of day (0-23)")
    ax.set_ylabel("Avg net demand (checkouts - returns)")
    ax.legend()
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(VIS_DIR / "03_hourly_demand_pattern.png", dpi=150)
    plt.close(fig)


def fig4_weather_effect(hist):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    daily = hist.groupby(["date", "rain"])["net_demand"].apply(lambda s: s.abs().sum()).reset_index()
    rain_vol = daily.loc[daily["rain"] == 1, "net_demand"]
    dry_vol = daily.loc[daily["rain"] == 0, "net_demand"]
    ax.boxplot([dry_vol, rain_vol], tick_labels=["Dry day", "Rainy day"], patch_artist=True,
               boxprops=dict(facecolor=ACCENT, alpha=0.7))
    ax.set_title("Daily System-Wide Ride Volume: Dry vs. Rainy Days")
    ax.set_ylabel("Total daily absolute demand (proxy for ride volume)")
    fig.tight_layout()
    fig.savefig(VIS_DIR / "04_weather_effect.png", dpi=150)
    plt.close(fig)


def fig5_top_stations(hist):
    fig, ax = plt.subplots(figsize=(8, 5))
    top = (hist.assign(abs_demand=hist["net_demand"].abs())
                .groupby("name")["abs_demand"].mean()
                .sort_values(ascending=False).head(12))
    ax.barh(top.index[::-1], top.values[::-1], color=ACCENT3)
    ax.set_title("Top 12 Real Stations by Average Hourly Demand Intensity")
    ax.set_xlabel("Avg |net demand| per hour")
    fig.tight_layout()
    fig.savefig(VIS_DIR / "05_top_stations.png", dpi=150)
    plt.close(fig)


def fig6_station_map(stations):
    fig, ax = plt.subplots(figsize=(6.5, 7))
    sizes = (stations["capacity"] / stations["capacity"].max()) * 300 + 20
    sc = ax.scatter(stations["lon"], stations["lat"], s=sizes,
                     c=stations["occupancy_rate"], cmap="RdYlGn", edgecolor="#333", linewidth=0.4, alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Live occupancy rate")
    ax.set_title("Real Citi Bike NYC Station Network\n(bubble size = dock capacity)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(VIS_DIR / "06_station_map.png", dpi=150)
    plt.close(fig)


def fig7_correlation_heatmap(hist):
    fig, ax = plt.subplots(figsize=(6, 5))
    corr_df = hist[["hour", "is_weekend", "temp_c", "rain", "capacity", "net_demand"]].copy()
    corr_df["abs_demand"] = hist["net_demand"].abs()
    corr = corr_df.drop(columns=["net_demand"]).corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    ax.set_title("Feature Correlation with Demand Intensity")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(VIS_DIR / "07_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def descriptive_summary(stations, hist) -> str:
    lines = []
    lines.append(f"Active real stations analyzed: {len(stations)}")
    lines.append(f"Total station-hour records (60-day history): {len(hist):,}")
    lines.append(f"Avg dock capacity: {stations['capacity'].mean():.1f} (min {stations['capacity'].min()}, max {stations['capacity'].max()})")
    lines.append(f"Avg live occupancy rate: {stations['occupancy_rate'].mean():.1%}")
    lines.append(f"Stations currently <20% full (restock risk): {(stations['occupancy_rate'] < 0.2).sum()}")
    lines.append(f"Stations currently >80% full (overflow risk): {(stations['occupancy_rate'] > 0.8).sum()}")
    wk = hist[hist["is_weekend"] == 0]["net_demand"].abs().mean()
    we = hist[hist["is_weekend"] == 1]["net_demand"].abs().mean()
    lines.append(f"Avg hourly demand intensity — weekday: {wk:.2f} vs weekend: {we:.2f} ({(wk/we - 1):.0%} higher on weekdays)")
    return "\n".join(lines)


if __name__ == "__main__":
    stations, hist = load()
    fig1_capacity_distribution(stations)
    fig2_occupancy_by_tier(stations)
    fig3_hourly_demand_pattern(hist)
    fig4_weather_effect(hist)
    fig5_top_stations(hist)
    fig6_station_map(stations)
    fig7_correlation_heatmap(hist)
    summary = descriptive_summary(stations, hist)
    print(summary)
    (VIS_DIR.parent / "reports" / "descriptive_summary.txt").write_text(summary)
    print(f"\nSaved 7 visualizations -> {VIS_DIR}")
