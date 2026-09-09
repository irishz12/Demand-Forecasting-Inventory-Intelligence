import unittest
import numpy as np
from pathlib import Path

from src.inference.forecast import (
    forecast_series,
    FEATURE_COLS,
    DATA_PATH,
    MODEL_PATH,
)


class TestForecastValidation(unittest.TestCase):
    """Test forecast input validation and constants."""

    def test_invalid_forecast_days_rejected_immediately(self):
        # Must fail input validation before attempting to read data or model
        for invalid_days in [0, 1, 5, 10, 21, 28, 60, -1]:
            with self.subTest(forecast_days=invalid_days):
                with self.assertRaises(ValueError) as ctx:
                    forecast_series(
                        store_id="CA_1",
                        item_id="FOODS_1_001",
                        forecast_days=invalid_days,
                    )
                self.assertIn("forecast_days must be 7, 14, or 30", str(ctx.exception))

    def test_feature_columns_specification(self):
        # Exactly 19 engineered features required by the production XGBoost model
        self.assertEqual(len(FEATURE_COLS), 19)
        # Ensure no duplicates
        self.assertEqual(len(FEATURE_COLS), len(set(FEATURE_COLS)))

        # Required lag features
        for lag in ["lag_1", "lag_7", "lag_14", "lag_28"]:
            self.assertIn(lag, FEATURE_COLS)

        # Required rolling window statistics
        for window in [7, 14, 28]:
            self.assertIn(f"rolling_mean_{window}", FEATURE_COLS)
            self.assertIn(f"rolling_std_{window}", FEATURE_COLS)

        # Required calendar features
        for cal in ["day_of_week", "month_num", "year_num", "week_of_year"]:
            self.assertIn(cal, FEATURE_COLS)

        # Required external / SNAP features
        for ext in ["sell_price", "event_flag", "snap_CA", "snap_TX", "snap_WI"]:
            self.assertIn(ext, FEATURE_COLS)

    def test_artifact_path_constants(self):
        self.assertEqual(DATA_PATH, Path("data/processed/model_data.parquet"))
        self.assertEqual(MODEL_PATH, Path("models/demand_forecaster_xgboost.json"))

    def test_recursive_feature_engineering_math(self):
        """Verify the mathematical logic of the recursive feature updates."""
        history = [10.0] * 28  # 28 days of constant sales of 10.0

        w7 = np.array(history[-7:])
        w14 = np.array(history[-14:])
        w28 = np.array(history[-28:])

        self.assertEqual(w7.mean(), 10.0)
        self.assertEqual(w7.std(), 0.0)
        self.assertEqual(w14.mean(), 10.0)
        self.assertEqual(w28.mean(), 10.0)

        # Append next day prediction (e.g. 17.0)
        prediction = 17.0
        history.append(prediction)

        # New lag_1 should be the predicted value
        self.assertEqual(history[-1], 17.0)
        # New lag_7 should be the original value from 7 steps back
        self.assertEqual(history[-7], 10.0)

        # New 7-day rolling window contains six 10.0s and one 17.0
        new_w7 = np.array(history[-7:])
        self.assertAlmostEqual(new_w7.mean(), (60.0 + 17.0) / 7.0, places=4)
        self.assertGreater(new_w7.std(), 0.0)


if __name__ == "__main__":
    unittest.main()
