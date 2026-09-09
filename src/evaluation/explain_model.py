import pandas as pd
import shap
import joblib
from pathlib import Path

DATA = Path("data/processed/model_data.parquet")
MODEL = Path("models/demand_forecaster.joblib")
OUTPUT = Path("reports/shap_feature_importance.csv")

print("Loading model and validation data...")

df = pd.read_parquet(DATA)

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

model = joblib.load(MODEL)

# Sample validation rows for efficient SHAP calculation
X = valid[feature_cols].sample(
    n=min(1000, len(valid)),
    random_state=42
)

print(f"SHAP sample size: {len(X):,}")
print("Calculating SHAP values...")

explainer = shap.Explainer(model, X)
shap_values = explainer(X, check_additivity=False)

importance = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": abs(shap_values.values).mean(axis=0)
})

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False
).reset_index(drop=True)

importance.to_csv(
    OUTPUT,
    index=False
)

print("\n=== TOP FEATURES ===")
print(importance.head(10).to_string(index=False))

print(f"\nSaved: {OUTPUT}")
print("SHAP analysis complete.")
