import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import mlflow

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


def forecast_series(store_id, item_id, forecast_days=7):
    if forecast_days not in [7, 14, 30]:
        raise ValueError("forecast_days must be 7, 14, or 30")

    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    series = df[
        (df["store_id"] == store_id)
        & (df["item_id"] == item_id)
    ].copy()

    if series.empty:
        raise ValueError(
            f"No data found for store={store_id}, item={item_id}"
        )

    series = series.sort_values("date").reset_index(drop=True)

    model = load_model()

    # Historical sales used to build recursive lag features.
    sales_history = list(series["sales"].astype(float).values)

    latest = series.iloc[-1]

    last_date = pd.Timestamp(latest["date"])

    # Use latest known values for features that cannot be
    # reliably projected without additional external data.
    sell_price = float(latest["sell_price"])

    event_flag = int(latest["event_flag"])
    snap_CA = int(latest["snap_CA"])
    snap_TX = int(latest["snap_TX"])
    snap_WI = int(latest["snap_WI"])

    forecasts = []

    for step in range(1, forecast_days + 1):

        future_date = last_date + pd.Timedelta(days=step)

        # Calendar features
        day_of_week = future_date.dayofweek
        month_num = future_date.month
        year_num = future_date.year
        week_of_year = future_date.isocalendar().week

        # Recursive lag features
        lag_1 = sales_history[-1]
        lag_7 = sales_history[-7]
        lag_14 = sales_history[-14]
        lag_28 = sales_history[-28]

        # Rolling features based only on information available
        # before the forecast date.
        window_7 = np.array(sales_history[-7:])
        window_14 = np.array(sales_history[-14:])
        window_28 = np.array(sales_history[-28:])

        rolling_mean_7 = window_7.mean()
        rolling_mean_14 = window_14.mean()
        rolling_mean_28 = window_28.mean()

        rolling_std_7 = window_7.std()
        rolling_std_14 = window_14.std()
        rolling_std_28 = window_28.std()

        row = pd.DataFrame([{
            "sell_price": sell_price,
            "day_of_week": day_of_week,
            "month_num": month_num,
            "year_num": year_num,
            "week_of_year": int(week_of_year),
            "event_flag": event_flag,
            "snap_CA": snap_CA,
            "snap_TX": snap_TX,
            "snap_WI": snap_WI,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "lag_28": lag_28,
            "rolling_mean_7": rolling_mean_7,
            "rolling_mean_14": rolling_mean_14,
            "rolling_mean_28": rolling_mean_28,
            "rolling_std_7": rolling_std_7,
            "rolling_std_14": rolling_std_14,
            "rolling_std_28": rolling_std_28,
        }])[FEATURE_COLS]

        prediction = float(model.predict(row)[0])
        prediction = max(0.0, prediction)

        forecasts.append({
            "date": future_date,
            "store_id": store_id,
            "item_id": item_id,
            "forecast": prediction,
        })

        # Critical recursive step:
        # today's prediction becomes future history.
        sales_history.append(prediction)

    result = pd.DataFrame(forecasts)

    return result


if __name__ == "__main__":

    print("Loading sample series...")

    df = pd.read_parquet(DATA_PATH)

    sample = (
        df[["store_id", "item_id"]]
        .drop_duplicates()
        .iloc[0]
    )

    store_id = sample["store_id"]
    item_id = sample["item_id"]

    print(f"Sample store: {store_id}")
    print(f"Sample item: {item_id}")

    forecast = forecast_series(
        store_id=store_id,
        item_id=item_id,
        forecast_days=7,
    )

    print("\n=== 7-DAY FORECAST ===")
    print(forecast.to_string(index=False))

    print("\nTotal forecast demand:")
    print(f"{forecast['forecast'].sum():.2f}")

    print("\nForecast completed successfully.")
