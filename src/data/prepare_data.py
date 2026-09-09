import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

print("Loading M5 data...")

calendar = pd.read_csv(RAW / "calendar.csv")
sales = pd.read_csv(RAW / "sales_train_validation.csv")
prices = pd.read_csv(RAW / "sell_prices.csv")

print(f"Calendar: {calendar.shape}")
print(f"Sales: {sales.shape}")
print(f"Prices: {prices.shape}")

# Convert sales from wide → long format
id_cols = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]

day_cols = [c for c in sales.columns if c.startswith("d_")]

sales_long = sales.melt(
    id_vars=id_cols,
    value_vars=day_cols,
    var_name="d",
    value_name="sales",
)

print(f"Long sales: {sales_long.shape}")

# Add calendar information
sales_long = sales_long.merge(
    calendar,
    on="d",
    how="left",
)

# Add weekly selling price
sales_long = sales_long.merge(
    prices,
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left",
)

# Convert date
sales_long["date"] = pd.to_datetime(sales_long["date"])

# Sort for downstream time-series processing
sales_long = sales_long.sort_values(
    ["store_id", "item_id", "date"]
).reset_index(drop=True)

# Save processed data
output = PROCESSED / "m5_sales_long.parquet"
sales_long.to_parquet(output, index=False)

print(f"Saved: {output}")
print(f"Final shape: {sales_long.shape}")
print(f"Date range: {sales_long['date'].min()} → {sales_long['date'].max()}")
print(f"Missing prices: {sales_long['sell_price'].isna().sum():,}")
print("Preparation complete.")
