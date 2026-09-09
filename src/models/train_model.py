import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT = Path("data/processed/model_data.parquet")
MODEL_PATH = Path("models/demand_forecaster.joblib")

print("Loading modeling data...")
df = pd.read_parquet(INPUT)

df = df.sort_values("date").reset_index(drop=True)

# Last 28 days = validation
max_date = df["date"].max()
validation_start = max_date - pd.Timedelta(days=27)

train = df[df["date"] < validation_start].copy()
valid = df[df["date"] >= validation_start].copy()

feature_cols = [
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

X_train = train[feature_cols]
y_train = train["sales"]

X_valid = valid[feature_cols]
y_valid = valid["sales"]

print(f"Train rows: {len(train):,}")
print(f"Validation rows: {len(valid):,}")
print(f"Features: {len(feature_cols)}")
print(f"Validation period: {validation_start.date()} -> {max_date.date()}")

model = HistGradientBoostingRegressor(
    learning_rate=0.08,
    max_iter=300,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

mlflow.set_experiment("demand-forecasting")

with mlflow.start_run():

    print("Training model...")

    model.fit(X_train, y_train)

    print("Generating predictions...")

    predictions = model.predict(X_valid)

    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_valid, predictions)
    rmse = np.sqrt(mean_squared_error(y_valid, predictions))

    denominator = np.abs(y_valid).sum()
    wape = np.abs(y_valid - predictions).sum() / denominator

    print("\n=== ML MODEL RESULTS ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"WAPE: {wape:.4%}")

    mlflow.log_params({
        "model": "HistGradientBoostingRegressor",
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "features": len(feature_cols),
    })

    mlflow.log_metrics({
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
    })

    mlflow.sklearn.log_model(
        model,
        "demand_forecaster"
    )

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved: {MODEL_PATH}")
    print("MLflow run logged.")
