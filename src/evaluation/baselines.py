import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT = Path("data/processed/model_data.parquet")

print("Loading modeling data...")
df = pd.read_parquet(INPUT)

df = df.sort_values(["store_id", "item_id", "date"])

# Last 28 days = validation period
max_date = df["date"].max()
validation_start = max_date - pd.Timedelta(days=27)

train = df[df["date"] < validation_start].copy()
valid = df[df["date"] >= validation_start].copy()

print(f"Train: {train.shape}")
print(f"Validation: {valid.shape}")
print(f"Validation period: {validation_start.date()} -> {max_date.date()}")

group_cols = ["store_id", "item_id"]

# -------------------------
# Naive: yesterday's demand
# -------------------------
valid["naive_pred"] = valid["lag_1"]

# -------------------------
# Seasonal Naive: same day last week
# -------------------------
valid["seasonal_naive_pred"] = valid["lag_7"]

# -------------------------
# Moving Average: last 7 days
# -------------------------
valid["moving_average_pred"] = valid["rolling_mean_7"]

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    denominator = np.abs(y_true).sum()
    wape = (
        np.abs(y_true - y_pred).sum() / denominator
        if denominator != 0 else np.nan
    )

    return mae, rmse, wape

results = []

for name, column in [
    ("Naive", "naive_pred"),
    ("Seasonal Naive", "seasonal_naive_pred"),
    ("Moving Average", "moving_average_pred"),
]:
    mae, rmse, wape = evaluate(
        valid["sales"],
        valid[column]
    )

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
    })

results_df = pd.DataFrame(results)

print("\n=== BASELINE RESULTS ===")
print(results_df.to_string(index=False))

results_df.to_csv(
    "reports/baseline_results.csv",
    index=False
)

print("\nSaved: reports/baseline_results.csv")
