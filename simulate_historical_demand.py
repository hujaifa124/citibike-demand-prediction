"""
Historical Demand Layer
========================
GBFS exposes only a live snapshot, so to train a genuine hour-by-hour demand
PREDICTOR we construct a 60-day hourly net-demand history on top of the REAL
station network pulled in data_collection.py (real station_id, name, lat/lon,
capacity, borough-tier). Net demand = bikes checked out minus bikes returned,
per station per hour.

Parameters below (AM/PM peak multipliers, weekday/weekend ratio, rain
sensitivity, temperature response) are calibrated to match the broad,
publicly-reported ridership patterns for Citi Bike NYC: two sharp commute
peaks on weekdays (~8AM, ~5-6PM), a flatter single midday peak on weekends,
materially lower ridership in rain/cold, and higher baseline demand at
high-capacity (Manhattan-core) stations than low-capacity outer-borough ones.

This is explicitly a MODELED historical layer (documented as such throughout
the project) built on top of 100% real station master data, used because raw
historical trip-level CSVs are not reachable from this environment's sandboxed
network and GBFS itself carries no history.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RNG = np.random.default_rng(42)

N_DAYS = 60
START_DATE = pd.Timestamp("2026-08-10")


def hourly_profile(hour: int, is_weekend: bool) -> float:
    """Relative demand intensity for a given hour of day."""
    if not is_weekend:
        # Weekday: AM commute peak ~8am, PM commute peak ~5-6pm
        am = np.exp(-((hour - 8) ** 2) / (2 * 1.3 ** 2))
        pm = np.exp(-((hour - 17.5) ** 2) / (2 * 1.6 ** 2))
        base = 0.08
        return base + 1.0 * am + 1.15 * pm
    else:
        # Weekend: one broad midday/afternoon leisure peak
        mid = np.exp(-((hour - 14) ** 2) / (2 * 3.2 ** 2))
        base = 0.12
        return base + 0.9 * mid


def simulate() -> pd.DataFrame:
    stations = pd.read_csv(DATA_DIR / "station_master_clean.csv")

    # base hourly demand rate scales with station capacity (proxy for how
    # central / high-traffic the real station is) and tier
    tier_multiplier = {
        "Low-traffic (<=20 docks)": 0.6,
        "Mid-traffic (21-40 docks)": 1.0,
        "High-traffic (40+ docks)": 1.9,
    }
    stations["base_rate"] = stations["capacity"].clip(upper=80) / 20.0 * stations["tier"].map(tier_multiplier)

    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    rows = []

    # one simulated weather series shared across stations (station-agnostic days)
    daily_temp = 24 - 10 * np.sin(np.linspace(0, np.pi, N_DAYS)) + RNG.normal(0, 2, N_DAYS)  # deg C, summer->autumn cooldown
    daily_rain = RNG.random(N_DAYS) < 0.22  # ~22% chance of rain that day

    for d_idx, date in enumerate(dates):
        is_weekend = date.dayofweek >= 5
        temp = daily_temp[d_idx]
        rain = daily_rain[d_idx]
        weather_factor = (1.0 if temp > 15 else 0.75 if temp > 5 else 0.55)
        weather_factor *= 0.55 if rain else 1.0

        for hour in range(24):
            hp = hourly_profile(hour, is_weekend)
            for _, st in stations.iterrows():
                lam = max(st["base_rate"] * hp * weather_factor * 4.0, 0.05)
                checkouts = RNG.poisson(lam=lam)
                returns = RNG.poisson(lam=lam * 0.92)
                net_demand = checkouts - returns
                rows.append(
                    (
                        st["station_id"],
                        st["name"],
                        st["tier"],
                        st["capacity"],
                        st["lat"],
                        st["lon"],
                        date,
                        hour,
                        int(is_weekend),
                        round(temp, 1),
                        int(rain),
                        checkouts,
                        returns,
                        net_demand,
                    )
                )

    cols = [
        "station_id", "name", "tier", "capacity", "lat", "lon",
        "date", "hour", "is_weekend", "temp_c", "rain", "checkouts", "returns", "net_demand",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    return df


if __name__ == "__main__":
    df = simulate()
    out = DATA_DIR / "hourly_demand_history.csv"
    df.to_csv(out, index=False)
    print(f"Simulated {len(df):,} station-hour records across {df['station_id'].nunique()} real stations "
          f"and {df['date'].nunique()} days -> {out}")
