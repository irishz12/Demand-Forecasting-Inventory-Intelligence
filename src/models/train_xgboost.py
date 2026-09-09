import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
import mlflow.xgboost

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.evaluation.promotion import promote_candidate

INPUT = Path("data/processed/model_data.parquet")
MODEL_PATH = Path("models/demand_forecaster_xgboost.json")

print("Loading modeling data...")
df = pd.read_parquet(INPUT)

df = df.sort_values("date").reset_index(drop=True)

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

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="mae",
    n_jobs=4,
    random_state=42,
)

mlflow.set_experiment("demand-forecasting-xgboost")

with mlflow.start_run() as run:

    print("Training XGBoost...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    print("Generating predictions...")

    predictions = model.predict(X_valid)
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_valid, predictions)
    rmse = np.sqrt(mean_squared_error(y_valid, predictions))

    denominator = np.abs(y_valid).sum()
    wape = np.abs(y_valid - predictions).sum() / denominator

    print("\n=== XGBOOST RESULTS ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"WAPE: {wape:.4%}")

    mlflow.log_params({
        "model": "XGBRegressor",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "features": len(feature_cols),
    })

    mlflow.log_metrics({
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
    })

    mlflow.xgboost.log_model(
        model,
        "demand_forecaster"
    )

    candidates_dir = Path("models/candidates")
    candidates_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidates_dir / f"demand_forecaster_xgboost_{run.info.run_id}.json"

    model.save_model(candidate_path)

    print(f"\nCandidate model saved: {candidate_path}")
    print("MLflow run logged.")

    print("\nEvaluating candidate with Quality Gate & Promotion Orchestrator...")
    result = promote_candidate(
        candidate_run_id=run.info.run_id,
        candidate_path=candidate_path,
        target_runtime_path=MODEL_PATH,
        model_name="DemandForecasterXGBoost",
    )

    if result.status == "PROMOTED":
        print("\n=== PROMOTION STATUS: PROMOTED ===")
        print(f"Model successfully promoted to registered version: {result.registered_version}")
        print(f"Runtime artifact updated: {MODEL_PATH}")
        for reason in result.reasons:
            print(f"  - {reason}")
    elif result.status == "REJECTED":
        print("\n=== PROMOTION STATUS: REJECTED ===")
        print("Candidate did not satisfy Quality Gate criteria. Runtime champion remains untouched.")
        for reason in result.reasons:
            print(f"  - {reason}")
    else:  # FAILED
        print("\n=== PROMOTION STATUS: FAILED ===")
        print(f"Operational/infrastructure failure during promotion: {result.error}")
        for reason in result.reasons:
            print(f"  - {reason}")
        sys.exit(1)
