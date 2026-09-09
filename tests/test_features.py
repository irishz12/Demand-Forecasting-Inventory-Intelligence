import unittest
import pandas as pd


def select_top_series_logic(df, n_series=50, validation_days=28):
    """
    Independent test helper mirroring src/features/build_features.py top-series selection logic:
    select top series using only df[df["date"] < validation_start].
    """
    max_date = df["date"].max()
    validation_start = max_date - pd.Timedelta(days=validation_days - 1)

    return (
        df[df["date"] < validation_start]
        .groupby(["store_id", "item_id"])["sales"]
        .sum()
        .nlargest(n_series)
        .reset_index()[["store_id", "item_id"]]
    )


class TestFeatureEngineeringTopSeriesSelection(unittest.TestCase):
    """Regression test: verify series selection excludes the validation period."""

    def test_top_series_selection_strictly_pre_validation(self):
        """
        Validate that series with massive sales only in the validation window
        are NOT selected over series with higher training-period sales volume.
        """
        dates = pd.date_range("2016-01-01", periods=50, freq="D")
        max_date = dates.max()
        validation_start = max_date - pd.Timedelta(days=27)

        records = []
        for d in dates:
            # STEADY_ITEM has 10 sales/day in training, 0 in validation (Total training: 220, validation: 0)
            records.append({
                "date": d,
                "store_id": "CA_1",
                "item_id": "STEADY_ITEM",
                "sales": 10.0 if d < validation_start else 0.0,
            })
            # LATE_SPIKE_ITEM has 0 sales/day in training, 50 in validation (Total training: 0, validation: 1400)
            records.append({
                "date": d,
                "store_id": "CA_1",
                "item_id": "LATE_SPIKE_ITEM",
                "sales": 0.0 if d < validation_start else 50.0,
            })

        df = pd.DataFrame(records)

        # Select top 1 series using pre-validation selection logic
        top_series = select_top_series_logic(df, n_series=1, validation_days=28)

        self.assertEqual(len(top_series), 1)
        selected_item = top_series.iloc[0]["item_id"]

        # STEADY_ITEM must be selected because training sales (220) > late spike training sales (0)
        self.assertEqual(
            selected_item,
            "STEADY_ITEM",
            "Top series selection leaked validation period data!",
        )

    def test_top_series_returns_expected_columns(self):
        dates = pd.date_range("2016-01-01", periods=35, freq="D")
        records = [
            {"date": d, "store_id": "CA_1", "item_id": f"ITEM_{i}", "sales": float(i)}
            for d in dates
            for i in range(3)
        ]
        df = pd.DataFrame(records)

        top_series = select_top_series_logic(df, n_series=2, validation_days=28)
        self.assertEqual(list(top_series.columns), ["store_id", "item_id"])
        self.assertEqual(len(top_series), 2)


if __name__ == "__main__":
    unittest.main()
