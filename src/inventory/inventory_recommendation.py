import pandas as pd
from pathlib import Path

from src.inference.forecast import forecast_series

DATA_PATH = Path("data/processed/model_data.parquet")


def calculate_safety_stock(forecast_demand, safety_factor=0.20):
    """
    Simple MVP assumption:
    Safety Stock = Forecast Demand × Safety Factor
    """
    return forecast_demand * safety_factor


def calculate_recommended_order(
    forecast_demand,
    available_inventory,
    safety_stock,
):
    """
    Recommended Order =
    Forecast Demand + Safety Stock - Available Inventory

    Never recommend a negative order quantity.
    """
    recommended_order = (
        forecast_demand
        + safety_stock
        - available_inventory
    )

    return max(0.0, recommended_order)


def generate_inventory_recommendation(
    store_id,
    item_id,
    available_inventory,
    forecast_days=7,
):
    if forecast_days not in [7, 14, 30]:
        raise ValueError("forecast_days must be 7, 14, or 30")

    print("Generating recursive forecast...")

    forecast_df = forecast_series(
        store_id=store_id,
        item_id=item_id,
        forecast_days=forecast_days,
    )

    forecast_demand = forecast_df["forecast"].sum()

    safety_stock = calculate_safety_stock(
        forecast_demand
    )

    recommended_order = calculate_recommended_order(
        forecast_demand=forecast_demand,
        available_inventory=available_inventory,
        safety_stock=safety_stock,
    )

    result = {
        "store_id": store_id,
        "item_id": item_id,
        "forecast_days": forecast_days,
        "forecast_demand": round(forecast_demand, 2),
        "available_inventory": available_inventory,
        "safety_stock": round(safety_stock, 2),
        "recommended_order": round(recommended_order, 2),
    }

    return result, forecast_df


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

    result, forecast_df = generate_inventory_recommendation(
        store_id=store_id,
        item_id=item_id,
        available_inventory=50,
        forecast_days=7,
    )

    print("\n=== FORECAST ===")
    print(forecast_df.to_string(index=False))

    print("\n=== INVENTORY INTELLIGENCE ===")

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nInventory intelligence v2 completed successfully.")
