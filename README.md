# Predictive Analytics for Bike-Sharing Demand
### Optimizing Station-Level Bike Redistribution Using Python

**Lovely Professional University — Mittal School of Business**
Course: MGNM801 (Business Analytics Using Python) | Academic Task 2 — Presentation Portfolio
**Student:** MD Hujaifa Ahmed | **Reg. No.:** 12420188 | **Roll No.:** RQ1P2410A01

---

## Problem Statement

Citi Bike NYC operates thousands of docking stations citywide. Bike availability is highly
uneven across the day — popular stations run **empty** during rush hours while others **fill
up** with no free docks, hurting the rider experience and operational efficiency. This project
collects real, live station-network data from Citi Bike's official GBFS feed, analyzes current
system imbalance, and trains a predictive model that forecasts near-term station demand —
enabling a proactive, data-driven bike redistribution plan.

## Data Source

**Citi Bike NYC — Official GBFS (General Bikeshare Feed Specification) Live System Data**
- `station_information.json` — real station IDs, names, coordinates, dock capacities
- `station_status.json` — real-time bike/dock availability, live-pulled 2026-10-08

> GBFS exposes only the *current* system snapshot, not historical trip-level demand. To train a
> genuine predictive model, a 60-day hourly historical demand series is layered on top of the
> **real station network** (real IDs, names, coordinates, capacities), using a parameterized
> model calibrated to Citi Bike NYC's publicly documented ridership patterns (AM/PM commute
> peaks, weekday/weekend differences, weather sensitivity). This is done transparently and is
> documented in `src/simulate_historical_demand.py`.

## Project Structure

```
citibike_project/
├── data/
│   ├── station_information_raw.csv      # real GBFS station attributes
│   ├── station_status_raw.csv           # real GBFS live snapshot
│   ├── station_master_clean.csv         # cleaned/merged/feature-engineered
│   └── hourly_demand_history.csv        # 60-day hourly demand history
├── src/
│   ├── data_collection.py               # load + clean real GBFS data
│   ├── simulate_historical_demand.py    # historical demand layer
│   ├── eda.py                           # descriptive analysis + 7 visualizations
│   └── predictive_model.py              # Random Forest demand model + redistribution logic
├── notebooks/
│   └── CA2_MGNM801_Business_Analytics_Portfolio.ipynb   # full executed portfolio notebook
├── visuals/                             # 9 generated PNG charts
├── reports/
│   ├── descriptive_summary.txt
│   ├── model_report.txt
│   └── redistribution_recommendations.csv
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python src/data_collection.py            # builds station_master_clean.csv
python src/simulate_historical_demand.py # builds hourly_demand_history.csv
python src/eda.py                        # generates 7 visualizations + summary
python src/predictive_model.py           # trains model + redistribution recommendations
```

Or open `notebooks/CA2_MGNM801_Business_Analytics_Portfolio.ipynb` to see the full workflow
with narrative, charts and outputs already run end-to-end.

## Key Results

- **75 real, currently-active Citi Bike NYC stations** analyzed (avg. 31.5 docks/station).
- **13 stations** currently below 20% occupancy (restock risk) and **19 stations** above 80%
  (overflow risk) at the live snapshot moment — ~43% of the network already imbalanced.
- Demand is strongly **bimodal on weekdays** (AM/PM commute) and **unimodal on weekends**
  (midday leisure).
- **Random Forest Regressor** predicting hourly checkout demand: **R² = 0.877**, MAE = 1.11,
  RMSE = 1.74 on held-out test data.
- Top predictive features: hour-of-day (cyclical), station capacity, and traffic tier.
- **24 real stations flagged** for proactive redistribution at the weekday 5 PM commute peak
  (see `reports/redistribution_recommendations.csv`).

## Business Recommendations

1. Dispatch redistribution trucks ~30-45 minutes ahead of the 8 AM / 5 PM weekday peaks to the
   flagged restock-priority stations.
2. Prioritize high-capacity, high-traffic-tier stations first — they drive the largest share of
   demand variance.
3. Scale back dispatch on weekends and rainy days, where system-wide ridership is materially
   lower.
4. Re-run the pipeline daily against the live GBFS feed so recommendations stay current.

## Tools Used

Python 3, pandas, NumPy, Matplotlib, scikit-learn, Jupyter/nbconvert.

## Declaration

This project is my individual work for Academic Task 2 (MGNM801), built on real Citi Bike NYC
GBFS data collected via Python, with the historical demand layer transparently documented as a
calibrated simulation layered on the real station network.
