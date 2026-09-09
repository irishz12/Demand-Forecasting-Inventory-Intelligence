import unittest
from unittest.mock import patch
import pandas as pd

from src.inventory.inventory_recommendation import (
    calculate_safety_stock,
    calculate_recommended_order,
    generate_inventory_recommendation,
)


class TestSafetyStockCalculation(unittest.TestCase):
    """Test safety stock determination under various demand profiles."""

    def test_default_safety_factor(self):
        # Default safety factor is 20% (0.20)
        demand = 100.0
        safety_stock = calculate_safety_stock(demand)
        self.assertAlmostEqual(safety_stock, 20.0, places=4)

    def test_custom_safety_factor(self):
        demand = 200.0
        safety_stock = calculate_safety_stock(demand, safety_factor=0.15)
        self.assertAlmostEqual(safety_stock, 30.0, places=4)

    def test_zero_forecast_demand(self):
        safety_stock = calculate_safety_stock(0.0)
        self.assertAlmostEqual(safety_stock, 0.0, places=4)

    def test_float_demand_precision(self):
        demand = 37.5
        safety_stock = calculate_safety_stock(demand, safety_factor=0.20)
        self.assertAlmostEqual(safety_stock, 7.5, places=4)


class TestRecommendedOrderCalculation(unittest.TestCase):
    """Test inventory replenishment order calculations with deterministic inputs."""

    def test_known_inputs_requiring_order(self):
        # Demand: 100, Available: 30, Safety Stock: 20
        # Expected: 100 + 20 - 30 = 90
        order = calculate_recommended_order(
            forecast_demand=100.0,
            available_inventory=30.0,
            safety_stock=20.0,
        )
        self.assertAlmostEqual(order, 90.0, places=4)

    def test_never_negative_when_inventory_is_excess(self):
        # Demand: 50, Available: 100, Safety Stock: 10
        # Formula raw: 50 + 10 - 100 = -40 -> Clamped to 0.0
        order = calculate_recommended_order(
            forecast_demand=50.0,
            available_inventory=100.0,
            safety_stock=10.0,
        )
        self.assertEqual(order, 0.0)

    def test_inventory_exactly_sufficient(self):
        # Demand: 80, Available: 96, Safety Stock: 16
        # Expected: 80 + 16 - 96 = 0.0
        order = calculate_recommended_order(
            forecast_demand=80.0,
            available_inventory=96.0,
            safety_stock=16.0,
        )
        self.assertEqual(order, 0.0)

    def test_zero_available_inventory(self):
        # Demand: 100, Available: 0, Safety Stock: 20
        # Expected: 100 + 20 - 0 = 120.0
        order = calculate_recommended_order(
            forecast_demand=100.0,
            available_inventory=0.0,
            safety_stock=20.0,
        )
        self.assertAlmostEqual(order, 120.0, places=4)

    def test_fractional_units(self):
        # Demand: 15.5, Available: 5.25, Safety Stock: 3.1
        # Expected: 15.5 + 3.1 - 5.25 = 13.35
        order = calculate_recommended_order(
            forecast_demand=15.5,
            available_inventory=5.25,
            safety_stock=3.1,
        )
        self.assertAlmostEqual(order, 13.35, places=2)


class TestGenerateInventoryRecommendation(unittest.TestCase):
    """Test recommendation orchestration and input validation."""

    def test_invalid_forecast_days_rejected(self):
        for invalid_days in [0, 1, 5, 10, 21, 45, -7]:
            with self.subTest(forecast_days=invalid_days):
                with self.assertRaises(ValueError) as ctx:
                    generate_inventory_recommendation(
                        store_id="CA_1",
                        item_id="FOODS_1_001",
                        available_inventory=50.0,
                        forecast_days=invalid_days,
                    )
                self.assertIn("forecast_days must be 7, 14, or 30", str(ctx.exception))

    @patch("src.inventory.inventory_recommendation.forecast_series")
    def test_deterministic_recommendation_pipeline(self, mock_forecast):
        # Mock 7-day forecast with 10 units per day (total 70.0)
        mock_dates = pd.date_range("2016-04-25", periods=7, freq="D")
        mock_df = pd.DataFrame({
            "date": mock_dates,
            "store_id": ["CA_1"] * 7,
            "item_id": ["FOODS_1_001"] * 7,
            "forecast": [10.0] * 7,
        })
        mock_forecast.return_value = mock_df

        result, forecast_df = generate_inventory_recommendation(
            store_id="CA_1",
            item_id="FOODS_1_001",
            available_inventory=50.0,
            forecast_days=7,
        )

        mock_forecast.assert_called_once_with(
            store_id="CA_1",
            item_id="FOODS_1_001",
            forecast_days=7,
        )

        # Total demand = 70.0, Safety stock = 14.0 (20%), Available = 50.0
        # Recommended order = 70.0 + 14.0 - 50.0 = 34.0
        self.assertEqual(result["store_id"], "CA_1")
        self.assertEqual(result["item_id"], "FOODS_1_001")
        self.assertEqual(result["forecast_days"], 7)
        self.assertEqual(result["forecast_demand"], 70.0)
        self.assertEqual(result["available_inventory"], 50.0)
        self.assertEqual(result["safety_stock"], 14.0)
        self.assertEqual(result["recommended_order"], 34.0)
        self.assertEqual(len(forecast_df), 7)


if __name__ == "__main__":
    unittest.main()
