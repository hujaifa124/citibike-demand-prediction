"""
Data Collection & Integration Module
=====================================
Citi Bike NYC GBFS (General Bikeshare Feed Specification) - Official Live System Data

Source: https://gbfs.citibikenyc.com/gbfs/en/station_information.json
        https://gbfs.citibikenyc.com/gbfs/en/station_status.json

This module loads the REAL, live-pulled station inventory (station_information)
and the REAL, live-pulled real-time system state (station_status) captured on
2026-10-08 from Citi Bike NYC's public GBFS endpoints — the same feed that
powers the official Citi Bike app.

GBFS by design only exposes the CURRENT real-time state of the system (how many
bikes/docks are free right now at each station) — it does not expose historical
trip-level demand. To build a genuine PREDICTIVE model (station-level, hour-by-hour
demand forecasting) a time-series of demand is required. We therefore layer a
historical hourly demand series on top of the real station network (real station
IDs, names, coordinates and dock capacities), generated with a parameterized
demand model whose parameters (rush-hour multipliers, weekday/weekend effect,
weather sensitivity, station-tier base rates) are calibrated against publicly
documented Citi Bike ridership patterns (AM/PM commute peaks, Manhattan-core vs.
outer-borough demand gradients, weather sensitivity reported in NYC DOT's annual
Citi Bike reports). This hybrid approach (real network + modeled time axis) is
the standard workaround used in coursework/industry prototypes whenever only a
live snapshot feed (not a trip-history data lake) is available, and is called
out explicitly and transparently here and in the README/report.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_station_information() -> pd.DataFrame:
    """Load the real GBFS station_information snapshot."""
    df = pd.read_csv(DATA_DIR / "station_information_raw.csv")
    df = df.dropna(subset=["lat", "lon"]).copy()
    df["capacity"] = df["capacity"].fillna(0).astype(int)
    # Drop duplicate/zero-capacity ghost stations (closed/decommissioned docks,
    # a known quirk of the GBFS feed where retired stations remain listed briefly)
    df = df[df["capacity"] > 0].reset_index(drop=True)
    return df


def load_station_status() -> pd.DataFrame:
    """Load the real GBFS station_status live snapshot."""
    df = pd.read_csv(DATA_DIR / "station_status_raw.csv")
    df = df[df["is_renting"] == 1].reset_index(drop=True)
    return df


def build_station_master() -> pd.DataFrame:
    """Merge information + live status into one cleaned station master table."""
    info = load_station_information()
    status = load_station_status()
    master = info.merge(status, on="station_id", how="inner")
    master["occupancy_rate"] = master["num_bikes_available"] / master["capacity"].replace(0, np.nan)
    master["occupancy_rate"] = master["occupancy_rate"].clip(0, 1).fillna(0)

    # Tier stations by capacity -> proxy for how central / high-traffic a dock is.
    # (Manhattan core stations in the real GBFS feed consistently show far larger
    # capacities than outer-borough residential stations — capacity itself is a
    # reasonable, data-driven proxy for station importance.)
    master["tier"] = pd.cut(
        master["capacity"],
        bins=[0, 20, 40, 1000],
        labels=["Low-traffic (<=20 docks)", "Mid-traffic (21-40 docks)", "High-traffic (40+ docks)"],
    )
    return master


if __name__ == "__main__":
    m = build_station_master()
    print(f"Loaded {len(m)} real, currently-active Citi Bike NYC stations.")
    print(m[["station_id", "name", "capacity", "num_bikes_available", "occupancy_rate", "tier"]].head(10))
    out = DATA_DIR / "station_master_clean.csv"
    m.to_csv(out, index=False)
    print(f"Saved cleaned station master -> {out}")
