import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/m5_sales_long.parquet")
OUTPUT = Path("data/processed/model_data.parquet")

print("Loading processed data...")

df = pd.read_parquet(INPUT)

# Keep a manageable subset:
# top 50 store-item combinations by total historical sales
top_series = (
    df.groupby(["store_id", "item_id"])["sales"]
    .sum()
    .nlargest(50)
    .reset_index()[["store_id", "item_id"]]
)

df = df.merge(
    top_series,
    on=["store_id", "item_id"],
    how="inner"
)

print(f"Selected rows: {len(df):,}")
print(f"Selected series: {df[['store_id', 'item_id']].drop_duplicates().shape[0]}")

df = df.sort_values(
    ["store_id", "item_id", "date"]
).reset_index(drop=True)

# Calendar features
df["day_of_week"] = df["date"].dt.dayofweek
df["month_num"] = df["date"].dt.month
df["year_num"] = df["date"].dt.year
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

# Price
df["sell_price"] = (
    df.groupby(["store_id", "item_id"])["sell_price"]
    .ffill()
    .fillna(0)
)

group = df.groupby(["store_id", "item_id"])["sales"]

print("Creating lag features...")

for lag in [1, 7, 14, 28]:
    df[f"lag_{lag}"] = group.shift(lag)

print("Creating rolling features...")

for window in [7, 14, 28]:
    shifted = group.shift(1)

    df[f"rolling_mean_{window}"] = (
        shifted.groupby(
            [df["store_id"], df["item_id"]]
        ).transform(
            lambda x: x.rolling(window).mean()
        )
    )

    df[f"rolling_std_{window}"] = (
        shifted.groupby(
            [df["store_id"], df["item_id"]]
        ).transform(
            lambda x: x.rolling(window).std()
        )
    )

# Event flag
df["event_flag"] = (
    df["event_name_1"].notna() |
    df["event_name_2"].notna()
).astype("int8")

# SNAP
for col in ["snap_CA", "snap_TX", "snap_WI"]:
    df[col] = df[col].fillna(0).astype("int8")

feature_cols = [
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

# Remove rows without enough history
df = df.dropna(subset=feature_cols).reset_index(drop=True)

model_cols = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "date",
    "sales",
    "sell_price",
    "day_of_week",
    "month_num",
    "year_num",
    "week_of_year",
    "event_flag",
    "snap_CA",
    "snap_TX",
    "snap_WI",
] + feature_cols

df = df[model_cols]

df.to_parquet(
    OUTPUT,
    index=False,
    compression="snappy"
)

print(f"Saved: {OUTPUT}")
print(f"Final shape: {df.shape}")
print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
print("Feature engineering complete.")
