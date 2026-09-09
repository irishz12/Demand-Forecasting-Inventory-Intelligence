import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from src.evaluation.inventory_simulation import simulate_series


class TestInventorySimulationHistoryRollover(unittest.TestCase):
    """Regression test: verify history rollover does not duplicate decision dates."""

    def test_history_rollover_does_not_duplicate_decision_date(self):
        # 70 days: 42 days initial history + 28 days simulation window (4 weekly decision cycles)
        dates = pd.date_range("2016-01-01", periods=70, freq="D")
        series = pd.DataFrame({
            "date": dates,
            "sales": [10.0] * 70,
            "sell_price": [2.0] * 70,
            "day_of_week": dates.dayofweek,
            "month_num": dates.month,
            "year_num": dates.year,
            "week_of_year": dates.isocalendar().week.astype(int),
            "event_flag": [0] * 70,
            "snap_CA": [0] * 70,
            "snap_TX": [0] * 70,
            "snap_WI": [0] * 70,
            "lag_1": [10.0] * 70,
            "lag_7": [10.0] * 70,
            "lag_14": [10.0] * 70,
            "lag_28": [10.0] * 70,
            "rolling_mean_7": [10.0] * 70,
            "rolling_mean_14": [10.0] * 70,
            "rolling_mean_28": [10.0] * 70,
            "rolling_std_7": [0.0] * 70,
            "rolling_std_14": [0.0] * 70,
            "rolling_std_28": [0.0] * 70,
        })

        mock_model = MagicMock()
        history_snapshots = []

        def record_forecast(history, future_rows, model):
            history_snapshots.append(history.copy())
            return np.full(len(future_rows), 10.0)

        with patch(
            "src.evaluation.inventory_simulation.recursive_forecast",
            side_effect=record_forecast,
        ):
            result = simulate_series(series, mock_model)

        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(history_snapshots), 2)

        # Verify that no snapshot contains duplicated dates across weekly rollover cycles
        for cycle_idx, snap in enumerate(history_snapshots):
            duplicated_dates = snap[snap["date"].duplicated(keep=False)]
            dup_list = duplicated_dates["date"].tolist()
            self.assertEqual(
                len(duplicated_dates),
                0,
                f"Cycle {cycle_idx + 1} has duplicated decision dates in history: {dup_list}",
            )
            self.assertEqual(len(snap), snap["date"].nunique())

        # Verify history grows by exactly 7 rows per weekly decision cycle
        for i in range(1, len(history_snapshots)):
            self.assertEqual(
                len(history_snapshots[i]) - len(history_snapshots[i - 1]),
                7,
                f"History did not grow by exactly 7 rows between cycles {i} and {i + 1}",
            )


if __name__ == "__main__":
    unittest.main()
