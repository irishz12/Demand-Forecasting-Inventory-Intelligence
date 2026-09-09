import unittest
from unittest.mock import patch
import pandas as pd
from pydantic import ValidationError
from fastapi import HTTPException

from src.api import ForecastRequest, app, health, forecast

try:
    from fastapi.testclient import TestClient
    HAS_TESTCLIENT = True
except Exception:
    HAS_TESTCLIENT = False


class TestForecastRequestValidation(unittest.TestCase):
    """Test Pydantic schema validation for forecast requests."""

    def test_valid_forecast_horizons_accepted(self):
        for horizon in [7, 14, 30]:
            with self.subTest(forecast_days=horizon):
                req = ForecastRequest(
                    store_id="CA_1",
                    item_id="FOODS_1_001",
                    available_inventory=25.0,
                    forecast_days=horizon,
                )
                self.assertEqual(req.forecast_days, horizon)
                self.assertEqual(req.store_id, "CA_1")
                self.assertEqual(req.item_id, "FOODS_1_001")
                self.assertEqual(req.available_inventory, 25.0)

    def test_default_forecast_days_is_7(self):
        req = ForecastRequest(
            store_id="TX_2",
            item_id="HOBBIES_1_002",
            available_inventory=10.0,
        )
        self.assertEqual(req.forecast_days, 7)

    def test_invalid_forecast_horizons_rejected(self):
        for invalid_horizon in [1, 5, 10, 15, 28, 60, -7, 0]:
            with self.subTest(forecast_days=invalid_horizon):
                with self.assertRaises(ValidationError):
                    ForecastRequest(
                        store_id="CA_1",
                        item_id="FOODS_1_001",
                        available_inventory=25.0,
                        forecast_days=invalid_horizon,
                    )

    def test_negative_available_inventory_rejected(self):
        for invalid_inv in [-1.0, -0.01, -50.0]:
            with self.subTest(available_inventory=invalid_inv):
                with self.assertRaises(ValidationError):
                    ForecastRequest(
                        store_id="CA_1",
                        item_id="FOODS_1_001",
                        available_inventory=invalid_inv,
                        forecast_days=7,
                    )

    def test_zero_available_inventory_accepted(self):
        req = ForecastRequest(
            store_id="CA_1",
            item_id="FOODS_1_001",
            available_inventory=0.0,
            forecast_days=7,
        )
        self.assertEqual(req.available_inventory, 0.0)

    def test_missing_required_fields_rejected(self):
        # Missing store_id
        with self.assertRaises(ValidationError):
            ForecastRequest(
                item_id="FOODS_1_001",
                available_inventory=10.0,
            )
        # Missing item_id
        with self.assertRaises(ValidationError):
            ForecastRequest(
                store_id="CA_1",
                available_inventory=10.0,
            )
        # Missing available_inventory
        with self.assertRaises(ValidationError):
            ForecastRequest(
                store_id="CA_1",
                item_id="FOODS_1_001",
            )


class TestDirectEndpointFunctions(unittest.TestCase):
    """Test endpoint handler functions directly."""

    def test_health_endpoint_response(self):
        res = health()
        self.assertEqual(res, {
            "status": "healthy",
            "service": "demand-forecasting-api",
        })

    @patch("src.api.generate_inventory_recommendation")
    def test_forecast_endpoint_success(self, mock_rec):
        mock_result = {
            "store_id": "CA_1",
            "item_id": "FOODS_1_001",
            "forecast_days": 7,
            "forecast_demand": 70.0,
            "available_inventory": 20.0,
            "safety_stock": 14.0,
            "recommended_order": 64.0,
        }
        mock_df = pd.DataFrame({
            "date": pd.date_range("2016-05-01", periods=7, freq="D"),
            "forecast": [10.0] * 7,
        })
        mock_rec.return_value = (mock_result, mock_df)

        req = ForecastRequest(
            store_id="CA_1",
            item_id="FOODS_1_001",
            available_inventory=20.0,
            forecast_days=7,
        )
        response = forecast(req)

        self.assertEqual(response["store_id"], "CA_1")
        self.assertEqual(response["item_id"], "FOODS_1_001")
        self.assertEqual(response["forecast_days"], 7)
        self.assertEqual(len(response["daily_forecast"]), 7)
        self.assertEqual(response["daily_forecast"][0], {"date": "2016-05-01", "forecast": 10.0})
        self.assertEqual(response["inventory"]["recommended_order"], 64.0)

    @patch("src.api.generate_inventory_recommendation")
    def test_forecast_endpoint_value_error_raises_http_400(self, mock_rec):
        mock_rec.side_effect = ValueError("No data found for store=CA_1, item=UNKNOWN")

        req = ForecastRequest(
            store_id="CA_1",
            item_id="UNKNOWN",
            available_inventory=10.0,
            forecast_days=7,
        )
        with self.assertRaises(HTTPException) as ctx:
            forecast(req)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("No data found", ctx.exception.detail)


@unittest.skipUnless(HAS_TESTCLIENT, "fastapi.testclient (httpx) not installed")
class TestFastAPIClient(unittest.TestCase):
    """Test full HTTP routing, serialization, and error status codes via TestClient."""

    def setUp(self):
        self.client = TestClient(app)

    def test_http_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "healthy",
            "service": "demand-forecasting-api",
        })

    def test_http_invalid_horizon_returns_422(self):
        payload = {
            "store_id": "CA_1",
            "item_id": "FOODS_1_001",
            "available_inventory": 15.0,
            "forecast_days": 10,
        }
        response = self.client.post("/forecast", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_http_negative_inventory_returns_422(self):
        payload = {
            "store_id": "CA_1",
            "item_id": "FOODS_1_001",
            "available_inventory": -5.0,
            "forecast_days": 7,
        }
        response = self.client.post("/forecast", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_http_missing_field_returns_422(self):
        payload = {
            "store_id": "CA_1",
            "forecast_days": 7,
        }
        response = self.client.post("/forecast", json=payload)
        self.assertEqual(response.status_code, 422)

    @patch("src.api.generate_inventory_recommendation")
    def test_http_successful_forecast(self, mock_rec):
        mock_result = {
            "store_id": "CA_1",
            "item_id": "FOODS_1_001",
            "forecast_days": 7,
            "forecast_demand": 70.0,
            "available_inventory": 20.0,
            "safety_stock": 14.0,
            "recommended_order": 64.0,
        }
        mock_df = pd.DataFrame({
            "date": pd.date_range("2016-05-01", periods=7, freq="D"),
            "forecast": [10.0] * 7,
        })
        mock_rec.return_value = (mock_result, mock_df)

        payload = {
            "store_id": "CA_1",
            "item_id": "FOODS_1_001",
            "available_inventory": 20.0,
            "forecast_days": 7,
        }
        response = self.client.post("/forecast", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["store_id"], "CA_1")
        self.assertEqual(data["inventory"]["recommended_order"], 64.0)
        self.assertEqual(len(data["daily_forecast"]), 7)

    @patch("src.api.generate_inventory_recommendation")
    def test_http_value_error_returns_400(self, mock_rec):
        mock_rec.side_effect = ValueError("No data found for store=CA_1, item=UNKNOWN")

        payload = {
            "store_id": "CA_1",
            "item_id": "UNKNOWN",
            "available_inventory": 10.0,
            "forecast_days": 7,
        }
        response = self.client.post("/forecast", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("No data found", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
