from functools import lru_cache
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import mlflow

DATA_PATH = Path("data/processed/model_data.parquet")
MODEL_PATH = Path("models/demand_forecaster_xgboost.json")
CALENDAR_PATH = Path("data/raw/calendar.csv")

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


@lru_cache(maxsize=1)
def load_data(data_path=DATA_PATH):
    print("Loading data...")
    return pd.read_parquet(data_path)


@lru_cache(maxsize=1)
def load_model(model_path=MODEL_PATH):
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


@lru_cache(maxsize=4)
def load_calendar_lookup(calendar_path=CALENDAR_PATH):
    """
    Load M5 calendar and build a lightweight date lookup mapping:
    date_str (YYYY-MM-DD) -> {event_flag: int, snap_CA: int, snap_TX: int, snap_WI: int}

    Returns an empty dict if the calendar file is unavailable.
    """
    path = Path(calendar_path)
    if not path.exists():
        return {}

    try:
        cal = pd.read_csv(
            path,
            usecols=[
                "date",
                "event_name_1",
                "event_name_2",
                "snap_CA",
                "snap_TX",
                "snap_WI",
            ],
        )
        event_flag = (
            cal["event_name_1"].notna() | cal["event_name_2"].notna()
        ).astype(int)

        cal_lookup = {}
        for d, ef, ca, tx, wi in zip(
            cal["date"],
            event_flag,
            cal["snap_CA"].fillna(0).astype(int),
            cal["snap_TX"].fillna(0).astype(int),
            cal["snap_WI"].fillna(0).astype(int),
        ):
            cal_lookup[str(d)] = {
                "event_flag": int(ef),
                "snap_CA": int(ca),
                "snap_TX": int(tx),
                "snap_WI": int(wi),
            }
        return cal_lookup
    except Exception:
        return {}


def forecast_series(store_id, item_id, forecast_days=7):
    if forecast_days not in [7, 14, 30]:
        raise ValueError("forecast_days must be 7, 14, or 30")

    df = load_data()

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

    # Load calendar lookup for future exogenous features (event_flag, SNAP)
    calendar_lookup = load_calendar_lookup(CALENDAR_PATH)

    # Use latest known selling price across the forecast horizon.
    sell_price = float(latest["sell_price"])

    # Fallback exogenous values if calendar data is unavailable for a date
    fallback_event_flag = int(latest["event_flag"])
    fallback_snap_CA = int(latest["snap_CA"])
    fallback_snap_TX = int(latest["snap_TX"])
    fallback_snap_WI = int(latest["snap_WI"])

    forecasts = []

    for step in range(1, forecast_days + 1):

        future_date = last_date + pd.Timedelta(days=step)
        future_date_str = future_date.strftime("%Y-%m-%d")

        # Calendar features
        day_of_week = future_date.dayofweek
        month_num = future_date.month
        year_num = future_date.year
        week_of_year = future_date.isocalendar().week

        # Exogenous event and SNAP features from actual future calendar
        if future_date_str in calendar_lookup:
            cal_features = calendar_lookup[future_date_str]
            event_flag = int(cal_features["event_flag"])
            snap_CA = int(cal_features["snap_CA"])
            snap_TX = int(cal_features["snap_TX"])
            snap_WI = int(cal_features["snap_WI"])
        else:
            event_flag = fallback_event_flag
            snap_CA = fallback_snap_CA
            snap_TX = fallback_snap_TX
            snap_WI = fallback_snap_WI

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

        rolling_std_7 = window_7.std(ddof=1)
        rolling_std_14 = window_14.std(ddof=1)
        rolling_std_28 = window_28.std(ddof=1)

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
