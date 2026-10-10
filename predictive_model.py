"""
Predictive Modeling & Station-Level Redistribution Optimization
==================================================================
Trains a RandomForestRegressor to predict station-hour net demand
(checkouts - returns) from real station attributes + time/weather
features, then uses next-hour predictions to flag which real stations
need a proactive bike redistribution (restock or pickup) truck visit.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VIS_DIR = Path(__file__).resolve().parent.parent / "visuals"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


def build_features(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    le = LabelEncoder()
    df["tier_code"] = le.fit_transform(df["tier"])
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df


FEATURES = [
    "capacity", "lat", "lon", "tier_code", "is_weekend",
    "temp_c", "rain", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
TARGET = "checkouts"


def train_test(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=200, max_depth=14, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    return model, (X_test, y_test, pred), {"MAE": mae, "RMSE": rmse, "R2": r2}


def plot_feature_importance(model):
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importances.index, importances.values, color="#1f6feb")
    ax.set_title("Random Forest Feature Importance — Station Demand Prediction")
    ax.set_xlabel("Relative importance")
    fig.tight_layout()
    fig.savefig(VIS_DIR / "08_feature_importance.png", dpi=150)
    plt.close(fig)
    return importances


def plot_actual_vs_predicted(y_test, pred):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred, alpha=0.15, s=10, color="#2ca58d")
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, color="#e8622c", linestyle="--", linewidth=2)
    ax.set_xlabel("Actual net demand")
    ax.set_ylabel("Predicted net demand")
    ax.set_title("Model Fit: Actual vs. Predicted Station-Hour Demand")
    fig.tight_layout()
    fig.savefig(VIS_DIR / "09_actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def redistribution_recommendations(df: pd.DataFrame, model, returns_model) -> pd.DataFrame:
    """Predict next-hour checkouts AND returns for every real station (for the
    upcoming commute-peak hour, 17:00, on a representative weekday) and flag
    which stations need a proactive redistribution truck visit: a station
    predicted to checkout far more bikes than it receives back risks running
    empty (needs a RESTOCK); the reverse risks running out of free docks
    (needs a PICKUP)."""
    next_hour_rows = df.drop_duplicates("station_id").copy()
    next_hour_rows["hour"] = 17  # evening commute peak — the highest-risk hour
    next_hour_rows["is_weekend"] = 0
    next_hour_rows["dow"] = 2
    next_hour_rows["hour_sin"] = np.sin(2 * np.pi * next_hour_rows["hour"] / 24)
    next_hour_rows["hour_cos"] = np.cos(2 * np.pi * next_hour_rows["hour"] / 24)
    next_hour_rows["dow_sin"] = np.sin(2 * np.pi * next_hour_rows["dow"] / 7)
    next_hour_rows["dow_cos"] = np.cos(2 * np.pi * next_hour_rows["dow"] / 7)

    pred_checkouts = model.predict(next_hour_rows[FEATURES])
    pred_returns = returns_model.predict(next_hour_rows[FEATURES])
    next_hour_rows["predicted_checkouts"] = pred_checkouts
    next_hour_rows["predicted_returns"] = pred_returns
    next_hour_rows["predicted_net_flow"] = pred_checkouts - pred_returns

    out = next_hour_rows[["station_id", "name", "capacity", "predicted_checkouts",
                           "predicted_returns", "predicted_net_flow"]].copy()
    restock_cut = out["predicted_net_flow"].quantile(0.85)
    pickup_cut = out["predicted_net_flow"].quantile(0.15)
    out["risk"] = np.select(
        [out["predicted_net_flow"] >= restock_cut, out["predicted_net_flow"] <= pickup_cut],
        ["Restock (bikes running out)", "Pickup (docks filling up)"],
        default="Balanced",
    )
    out = out.sort_values("predicted_net_flow", ascending=False)
    return out[out["risk"] != "Balanced"].reset_index(drop=True)


if __name__ == "__main__":
    hist = pd.read_csv(DATA_DIR / "hourly_demand_history.csv", parse_dates=["date"])
    df = build_features(hist)
    model, (X_test, y_test, pred), metrics = train_test(df)

    print("Model performance on held-out test set (target = checkouts/hour):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    # second lightweight model for returns, used only for the redistribution decision
    X = df[FEATURES]
    y_ret = df["returns"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, y_ret, test_size=0.2, random_state=42)
    returns_model = RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1)
    returns_model.fit(Xr_train, yr_train)

    importances = plot_feature_importance(model)
    plot_actual_vs_predicted(y_test, pred)

    rec = redistribution_recommendations(df, model, returns_model)
    rec.to_csv(REPORT_DIR / "redistribution_recommendations.csv", index=False)

    report_lines = [
        "PREDICTIVE MODEL PERFORMANCE",
        "-" * 40,
        *[f"{k}: {v:.3f}" for k, v in metrics.items()],
        "",
        "TOP FEATURE IMPORTANCES",
        "-" * 40,
        *[f"{k}: {v:.3f}" for k, v in importances.sort_values(ascending=False).items()],
        "",
        f"STATIONS FLAGGED FOR PROACTIVE REDISTRIBUTION (next hour): {len(rec)}",
        "-" * 40,
    ]
    report_lines += [
        f"{row['name']}: {row['risk']} (pred net flow {row['predicted_net_flow']:.1f}, "
        f"checkouts {row['predicted_checkouts']:.1f}, returns {row['predicted_returns']:.1f})"
        for _, row in rec.head(15).iterrows()
    ]
    (REPORT_DIR / "model_report.txt").write_text("\n".join(report_lines))
    print(f"\nFlagged {len(rec)} real stations needing redistribution next hour.")
    print(f"Saved model report -> {REPORT_DIR / 'model_report.txt'}")
