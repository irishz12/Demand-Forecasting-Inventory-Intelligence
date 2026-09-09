import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb

DATA_PATH = Path("data/processed/model_data.parquet")
MODEL_PATH = Path("models/demand_forecaster_xgboost.json")

FEATURE_COLS = [
    "sell_price",
    "day_of_week",
    "month_num",
    "year_num",
    "week_of_year",
    "event_flag",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
]


def load_model():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


def recursive_forecast(history, future_rows, model):
    """
    Forecast future demand using only historical sales available
    before the forecast period.

    Future calendar/price features are taken from future_rows,
    while all demand-derived lag/rolling features come from
    historical actuals + prior model predictions.
    """

    sales_history = list(history["sales"].astype(float))

    forecasts = []

    for _, future_row in future_rows.iterrows():

        if len(sales_history) < 28:
            raise ValueError(
                "At least 28 historical observations are required."
            )

        window_7 = np.array(sales_history[-7:])
        window_14 = np.array(sales_history[-14:])
        window_28 = np.array(sales_history[-28:])

        row = pd.DataFrame([{
            "sell_price": float(future_row["sell_price"]),
            "day_of_week": int(future_row["day_of_week"]),
            "month_num": int(future_row["month_num"]),
            "year_num": int(future_row["year_num"]),
            "week_of_year": int(future_row["week_of_year"]),
            "event_flag": int(future_row["event_flag"]),
            "snap_CA": int(future_row["snap_CA"]),
            "snap_TX": int(future_row["snap_TX"]),
            "snap_WI": int(future_row["snap_WI"]),

            "lag_1": sales_history[-1],
            "lag_7": sales_history[-7],
            "lag_14": sales_history[-14],
            "lag_28": sales_history[-28:][-1],

            "rolling_mean_7": window_7.mean(),
            "rolling_mean_14": window_14.mean(),
            "rolling_mean_28": window_28.mean(),

            "rolling_std_7": window_7.std(),
            "rolling_std_14": window_14.std(),
            "rolling_std_28": window_28.std(),
        }])[FEATURE_COLS]

        prediction = float(model.predict(row)[0])
        prediction = max(0.0, prediction)

        forecasts.append(prediction)

        # Recursive prediction becomes available history.
        sales_history.append(prediction)

    return np.array(forecasts)


def simulate_series(series, model):
    series = series.sort_values("date").reset_index(drop=True)

    # Final 28 days are held out as the business simulation period.
    simulation_start = series["date"].max() - pd.Timedelta(days=27)

    history = series[series["date"] < simulation_start].copy()
    simulation = series[series["date"] >= simulation_start].copy()

    if len(history) < 35 or len(simulation) < 14:
        return None

    # Evaluate decisions every 7 days so each decision gets
    # a complete 7-day future demand window.
    decision_dates = simulation["date"].iloc[:-6].iloc[::7]

    baseline_stockouts = 0
    forecast_stockouts = 0

    baseline_excess = 0.0
    forecast_excess = 0.0

    baseline_cost = 0.0
    forecast_cost = 0.0

    total_days = 0

    for decision_date in decision_dates:

        future_start = decision_date + pd.Timedelta(days=1)
        future_end = decision_date + pd.Timedelta(days=7)

        future = series[
            (series["date"] >= future_start)
            & (series["date"] <= future_end)
        ].copy()

        if len(future) != 7:
            continue

        # --------------------------------------------------
        # FORECAST STRATEGY
        # --------------------------------------------------

        forecast_values = recursive_forecast(
            history=history,
            future_rows=future,
            model=model,
        )

        forecast_demand = forecast_values.sum()

        forecast_inventory = forecast_demand * 1.20

        # --------------------------------------------------
        # BASELINE STRATEGY
        # --------------------------------------------------

        baseline_daily_demand = history["sales"].tail(7).mean()
        baseline_demand = baseline_daily_demand * 7

        baseline_inventory = baseline_demand * 1.20

        # --------------------------------------------------
        # ACTUAL FUTURE DEMAND
        # --------------------------------------------------

        actual_demand = future["sales"].sum()

        # Stockout quantity
        baseline_shortage = max(
            0.0,
            actual_demand - baseline_inventory,
        )

        forecast_shortage = max(
            0.0,
            actual_demand - forecast_inventory,
        )

        # Excess inventory
        baseline_leftover = max(
            0.0,
            baseline_inventory - actual_demand,
        )

        forecast_leftover = max(
            0.0,
            forecast_inventory - actual_demand,
        )

        if baseline_shortage > 0:
            baseline_stockouts += 1

        if forecast_shortage > 0:
            forecast_stockouts += 1

        baseline_excess += baseline_leftover
        forecast_excess += forecast_leftover

        # Simple business cost assumptions.
        holding_cost = 0.10
        stockout_cost = 2.00

        baseline_cost += (
            baseline_leftover * holding_cost
            + baseline_shortage * stockout_cost
        )

        forecast_cost += (
            forecast_leftover * holding_cost
            + forecast_shortage * stockout_cost
        )

        total_days += 7

        # IMPORTANT:
        # Only actual demand up to the decision period becomes
        # available for the next decision.
        history = pd.concat(
            [
                history,
                series[
                    (series["date"] >= decision_date)
                    & (series["date"] <= future_end)
                ],
            ],
            ignore_index=True,
        )

    if total_days == 0:
        return None

    decisions = len(decision_dates)

    return {
        "days_simulated": total_days,
        "decisions": decisions,

        "baseline_stockout_rate":
            baseline_stockouts / decisions,

        "forecast_stockout_rate":
            forecast_stockouts / decisions,

        "baseline_service_level":
            1 - (baseline_stockouts / decisions),

        "forecast_service_level":
            1 - (forecast_stockouts / decisions),

        "baseline_avg_excess_inventory":
            baseline_excess / decisions,

        "forecast_avg_excess_inventory":
            forecast_excess / decisions,

        "baseline_inventory_cost":
            baseline_cost / decisions,

        "forecast_inventory_cost":
            forecast_cost / decisions,
    }


if __name__ == "__main__":

    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    print(f"Total rows: {len(df):,}")

    model = load_model()

    results = []

    for (store_id, item_id), group in df.groupby(
        ["store_id", "item_id"]
    ):

        result = simulate_series(
            group,
            model,
        )

        if result is not None:
            result["store_id"] = store_id
            result["item_id"] = item_id
            results.append(result)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise RuntimeError(
            "No simulation results generated."
        )

    metrics = {
        "series_evaluated":
            len(results_df),

        "baseline_stockout_rate":
            results_df["baseline_stockout_rate"].mean(),

        "forecast_stockout_rate":
            results_df["forecast_stockout_rate"].mean(),

        "baseline_service_level":
            results_df["baseline_service_level"].mean(),

        "forecast_service_level":
            results_df["forecast_service_level"].mean(),

        "baseline_avg_excess_inventory":
            results_df["baseline_avg_excess_inventory"].mean(),

        "forecast_avg_excess_inventory":
            results_df["forecast_avg_excess_inventory"].mean(),

        "baseline_inventory_cost":
            results_df["baseline_inventory_cost"].mean(),

        "forecast_inventory_cost":
            results_df["forecast_inventory_cost"].mean(),
    }

    output_path = Path(
        "reports/inventory_simulation_decision_time.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame([metrics]).to_csv(
        output_path,
        index=False,
    )

    print("\n=== DECISION-TIME INVENTORY SIMULATION ===")

    for key, value in metrics.items():

        if "rate" in key or "level" in key:
            print(f"{key}: {value:.2%}")

        elif "cost" in key or "excess" in key:
            print(f"{key}: {value:.2f}")

        else:
            print(f"{key}: {value}")

    print(f"\nSaved: {output_path}")
    print("\nDecision-time simulation completed successfully.")
