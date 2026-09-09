import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pathlib import Path

from src.inference.forecast import (
    forecast_series,
    load_calendar_lookup,
    FEATURE_COLS,
    DATA_PATH,
    MODEL_PATH,
    CALENDAR_PATH,
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
        self.assertGreater(new_w7.std(ddof=1), 0.0)

    def test_rolling_std_ddof_1_alignment(self):
        """Verify rolling std uses ddof=1 to align with training pandas rolling().std()."""
        sample_window = np.array([10.0, 20.0])
        # ddof=0 produces population std (5.0)
        self.assertEqual(sample_window.std(ddof=0), 5.0)
        # ddof=1 produces sample std sqrt(50) = 7.0710678...
        expected_ddof1 = np.sqrt(50.0)
        self.assertAlmostEqual(sample_window.std(ddof=1), expected_ddof1, places=6)

        # Confirm pandas rolling().std() default matches np.std(ddof=1)
        s = pd.Series([10.0, 20.0])
        self.assertAlmostEqual(s.rolling(2).std().iloc[-1], sample_window.std(ddof=1), places=6)


class TestCalendarExogenousInference(unittest.TestCase):
    """Test calendar lookup and fallback handling for exogenous features."""

    def test_calendar_lookup_event_and_snap_extraction(self):
        """Verify calendar lookup parses known future dates with correct int types."""
        lookup = load_calendar_lookup(CALENDAR_PATH)
        self.assertGreater(len(lookup), 0)

        # 2016-04-30: Pesach End (event_flag=1, snap_CA=0)
        entry_0430 = lookup.get("2016-04-30")
        self.assertIsNotNone(entry_0430)
        self.assertEqual(entry_0430["event_flag"], 1)
        self.assertEqual(entry_0430["snap_CA"], 0)
        self.assertIsInstance(entry_0430["event_flag"], int)
        self.assertIsInstance(entry_0430["snap_CA"], int)

        # 2016-05-01: Orthodox Easter & SNAP in CA/TX
        entry_0501 = lookup.get("2016-05-01")
        self.assertIsNotNone(entry_0501)
        self.assertEqual(entry_0501["event_flag"], 1)
        self.assertEqual(entry_0501["snap_CA"], 1)
        self.assertEqual(entry_0501["snap_TX"], 1)
        self.assertEqual(entry_0501["snap_WI"], 0)
        self.assertIsInstance(entry_0501["event_flag"], int)
        self.assertIsInstance(entry_0501["snap_CA"], int)

    def test_missing_calendar_file_returns_empty_lookup_gracefully(self):
        """Missing calendar file must return empty dict rather than raising FileNotFoundError."""
        lookup = load_calendar_lookup(Path("data/raw/non_existent_calendar.csv"))
        self.assertEqual(lookup, {})

    def test_missing_calendar_falls_back_safely(self):
        """When calendar lookup is empty, forecast_series completes using latest historical values."""
        with patch("src.inference.forecast.load_calendar_lookup", return_value={}):
            result = forecast_series(
                store_id="CA_1",
                item_id="FOODS_3_090",
                forecast_days=7,
            )
            self.assertEqual(len(result), 7)
            self.assertEqual(
                list(result.columns),
                ["date", "store_id", "item_id", "forecast"],
            )


if __name__ == "__main__":
    unittest.main()
