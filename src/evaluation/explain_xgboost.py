import pandas as pd
import numpy as np
from pathlib import Path
import shap
import xgboost as xgb

DATA_PATH = Path("data/processed/model_data.parquet")
MODEL_PATH = Path("models/demand_forecaster_xgboost.json")
OUTPUT_PATH = Path("reports/xgboost_shap_feature_importance.csv")

print("Loading data...")
df = pd.read_parquet(DATA_PATH)

max_date = df["date"].max()
validation_start = max_date - pd.Timedelta(days=27)

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

X_valid = valid[feature_cols]

# Sample for efficient SHAP calculation
X_sample = X_valid.sample(
    n=min(1000, len(X_valid)),
    random_state=42
)

print(f"Validation rows: {len(X_valid):,}")
print(f"SHAP sample: {len(X_sample):,}")

print("Loading XGBoost model...")
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

importance = np.abs(shap_values).mean(axis=0)

result = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": importance,
})

result = result.sort_values(
    "mean_abs_shap",
    ascending=False
).reset_index(drop=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_PATH, index=False)

print("\n=== XGBOOST SHAP FEATURE IMPORTANCE ===")
print(result.to_string(index=False))

print(f"\nSaved: {OUTPUT_PATH}")
print("XGBoost SHAP analysis completed successfully.")
