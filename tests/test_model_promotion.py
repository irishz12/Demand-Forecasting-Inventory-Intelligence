"""
Unit tests for isolated Model Promotion Orchestrator.

Verifies:
- Champion resolution (fail-closed, no hardcoded fallbacks)
- Candidate metric retrieval and validation
- Quality Gate integration (promotions, rejections, guardrails)
- Candidate registration in MLflow Model Registry
- Atomic filesystem replacement (POSIX os.replace, temporary staging)
- Model Registry alias updates
- Resilient failure handling without unsafe rollbacks
- Serialization and auditing
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.evaluation.promotion import (
    AliasUpdateError,
    AtomicReplacementError,
    CandidateMetricError,
    ChampionInfo,
    ChampionResolutionError,
    PromotionError,
    PromotionResult,
    RegistrationError,
    atomic_replace_file,
    evaluate_candidate,
    get_candidate_metrics,
    get_champion_metrics,
    promote_candidate,
    register_candidate_version,
    update_champion_alias,
)


class TestPromotionBase(unittest.TestCase):
    """Base test fixture providing isolated filesystem directories."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.candidate_path = self.base_path / "candidate.json"
        self.runtime_path = self.base_path / "models" / "runtime.json"
        self.runtime_path.parent.mkdir(parents=True, exist_ok=True)

        self.candidate_path.write_text('{"model": "candidate_model"}', encoding="utf-8")
        self.runtime_path.write_text('{"model": "champion_model"}', encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_mock_client(
        self,
        champion_version="1",
        champion_run_id="champ_run_100",
        champion_metrics=None,
        candidate_metrics=None,
        registered_version="2",
    ):
        if champion_metrics is None:
            champion_metrics = {"WAPE": 0.225481, "MAE": 8.6744, "RMSE": 12.6023}
        if candidate_metrics is None:
            candidate_metrics = {"WAPE": 0.210000, "MAE": 8.1000, "RMSE": 11.5000}

        client = MagicMock()

        mock_champ_ver = MagicMock()
        mock_champ_ver.version = champion_version
        mock_champ_ver.run_id = champion_run_id
        mock_champ_ver.aliases = ["champion"]
        client.get_model_version_by_alias.return_value = mock_champ_ver

        champ_run = MagicMock()
        champ_run.data.metrics = dict(champion_metrics)

        cand_run = MagicMock()
        cand_run.data.metrics = dict(candidate_metrics)

        def get_run_side_effect(run_id):
            if run_id == champion_run_id:
                return champ_run
            return cand_run

        client.get_run.side_effect = get_run_side_effect

        mock_new_ver = MagicMock()
        mock_new_ver.version = registered_version
        client.create_model_version.return_value = mock_new_ver

        return client


class TestPromotionOrchestrator(TestPromotionBase):
    """Tests for the top-level promote_candidate orchestrator."""

    def test_candidate_beats_champion_promoted(self):
        """Candidate with strictly lower WAPE and MAE is PROMOTED."""
        client = self.create_mock_client(
            champion_metrics={"WAPE": 0.225481, "MAE": 8.6744, "RMSE": 12.6023},
            candidate_metrics={"WAPE": 0.210000, "MAE": 8.1000, "RMSE": 11.5000},
            registered_version="2",
        )

        result = promote_candidate(
            candidate_run_id="cand_run_200",
            candidate_path=self.candidate_path,
            target_runtime_path=self.runtime_path,
            client=client,
        )

        self.assertEqual(result.status, "PROMOTED")
        self.assertEqual(result.registered_version, "2")
        self.assertEqual(self.runtime_path.read_text(encoding="utf-8"), '{"model": "candidate_model"}')
        client.create_model_version.assert_called_once()
        client.set_registered_model_alias.assert_called_once_with(
            "DemandForecasterXGBoost", "champion", "2"
        )

    def test_candidate_equals_champion_wape_promoted(self):
        """Candidate with identical WAPE is permitted by the <= champion rule."""
        client = self.create_mock_client(
            champion_metrics={"WAPE": 0.225481, "MAE": 8.6744, "RMSE": 12.6023},
            candidate_metrics={"WAPE": 0.225481, "MAE": 8.5000, "RMSE": 12.0000},
            registered_version="3",
        )

        result = promote_candidate(
            candidate_run_id="cand_run_201",
            candidate_path=self.candidate_path,
            target_runtime_path=self.runtime_path,
            client=client,
        )

        self.assertEqual(result.status, "PROMOTED")
        self.assertEqual(result.registered_version, "3")
        self.assertEqual(self.runtime_path.read_text(encoding="utf-8"), '{"model": "candidate_model"}')

    def test_candidate_worse_than_champion_rejected(self):
        """Candidate with higher WAPE than champion is REJECTED; runtime file untouched."""
        client = self.create_mock_client(
            champion_metrics={"WAPE": 0.225481, "MAE": 8.6744, "RMSE": 12.6023},
            candidate_metrics={"WAPE": 0.230000, "MAE": 8.5000, "RMSE": 12.0000},
        )

        result = promote_candidate(
            candidate_run_id="cand_run_202",
            candidate_path=self.candidate_path,
            target_runtime_path=self.runtime_path,
            client=client,
        )

        self.assertEqual(result.status, "REJECTED")
        self.assertIsNone(result.registered_version)
        self.assertEqual(self.runtime_path.read_text(encoding="utf-8"), '{"model": "champion_model"}')
        client.create_model_version.assert_not_called()
        client.set_registered_model_alias.assert_not_called()

    def test_candidate_fails_baseline_rejected(self):
        """Candidate failing baseline WAPE (>= 0.29245) is REJECTED."""
        client = self.create_mock_client(
            champion_metrics={"WAPE": 0.300000, "MAE": 8.6744, "RMSE": 12.6023},
            candidate_metrics={"WAPE": 0.295000, "MAE": 8.5000, "RMSE": 12.0000},
        )

        result = promote_candidate(
            candidate_run_id="cand_run_203",
            candidate_path=self.candidate_path,
            target_runtime_path=self.runtime_path,
            client=client,
        )

        self.assertEqual(result.status, "REJECTED")
        self.assertEqual(self.runtime_path.read_text(encoding="utf-8"), '{"model": "champion_model"}')
        client.create_model_version.assert_not_called()

    def test_candidate_exceeds_wape_guardrail_rejected(self):
        """Candidate with WAPE > 0.2500 is REJECTED by guardrail."""
        client = self.create_mock_client(
            champion_metrics={"WAPE": 0.280000, "MAE": 8.6744, "RMSE": 12.6023},
            candidate_metrics={"WAPE": 0.255000, "MAE": 8.5000, "RMSE": 12.0000},
        )

        result = promote_candidate(
            candidate_run_id="cand_run_204",
            candidate_path=self.candidate_path,
            target_runtime_path=self.runtime_path,
            client=client,
        )

        self.assertEqual(result.status, "REJECTED")
        self.assertEqual(self.runtime_path.read_text(encoding="utf-8"), '{"model": "champion_model"}')
        client.create_model_version.assert_not_called()

    def test_candidate_exceeds_mae_guardrail_rejected(self):
        """Candidate with MAE > 10.00 is REJECTED by guardrail."""
        client = self.create_mock_client(
            champion_metrics={"WAPE": 0.225481, "MAE": 8.6744, "RMSE": 12.6023},
            candidate_metrics={"WAPE": 0.210000, "MAE": 10.5000, "RMSE": 13.0000},
        )

        result = promote_candidate(
            candidate_run_id="cand_run_205",
            candidate_path=self.candidate_path,
            target_runtime_path=self.runtime_path,
            client=client,
        )

        self.assertEqual(result.status, "REJECTED")
        self.assertEqual(self.runtime_path.read_text(encoding="utf-8"), '{"model": "champion_model"}')
        client.create_model_version.assert_not_called()

    def test_missing_champion_alias_fails_closed(self):
        """When champion alias is absent, promotion FAILS CLOSED and does not promote."""
        client = self.create_mock_client()
        client.get_model_version_by_alias.side_effect = Exception("Alias 'champion' not found")

        result = promote_candidate(
            candidate_run_id="cand_run_206",
            candidate_path=self.candidate_path,
            target_runtime_path=self.runtime_path,
            client=client,
        )

        self.assertEqual(result.status, "FAILED")
        self.assertIn("Failed to resolve champion model version", result.error)
        self.assertEqual(self.runtime_path.read_text(encoding="utf-8"), '{"model": "champion_model"}')
        client.create_model_version.assert_not_called()

    def test_mlflow_client_failure_during_champion_lookup_fails_closed(self):
        """Connection error during champion lookup yields FAILED status and preserves runtime."""
        client = self.create_mock_client()
        client.get_model_version_by_alias.side_effect = RuntimeError("Database locked")

        result = promote_candidate(
            candidate_run_id="cand_run_207",
            candidate_path=self.candidate_path,
            target_runtime_path=self.runtime_path,
            client=client,
        )

        self.assertEqual(result.status, "FAILED")
        self.assertIn("Database locked", result.error)
        self.assertEqual(self.runtime_path.read_text(encoding="utf-8"), '{"model": "champion_model"}')
        client.create_model_version.assert_not_called()

    def test_missing_candidate_metrics_fails_closed(self):
        """When candidate run is missing WAPE or MAE, promotion FAILS CLOSED."""
        client = self.create_mock_client(
            candidate_metrics={"MAE": 8.1000, "RMSE": 11.5000}  # Missing WAPE
        )

        result = promote_candidate(
            candidate_run_id="cand_run_208",
            candidate_path=self.candidate_path,
            target_runtime_path=self.runtime_path,
            client=client,
        )

        self.assertEqual(result.status, "FAILED")
        self.assertIn("Missing required metric 'WAPE'", result.error)
        self.assertEqual(self.runtime_path.read_text(encoding="utf-8"), '{"model": "champion_model"}')
        client.create_model_version.assert_not_called()

    def test_registration_failure_fails_closed(self):
        """When MLflow model version creation fails, promotion FAILS CLOSED and runtime is unchanged."""
        client = self.create_mock_client()
        client.create_model_version.side_effect = RuntimeError("Registry unavailable")

        result = promote_candidate(
            candidate_run_id="cand_run_209",
            candidate_path=self.candidate_path,
            target_runtime_path=self.runtime_path,
            client=client,
        )

        self.assertEqual(result.status, "FAILED")
        self.assertIn("Failed to register model version", result.error)
        self.assertEqual(self.runtime_path.read_text(encoding="utf-8"), '{"model": "champion_model"}')
        client.set_registered_model_alias.assert_not_called()

    def test_atomic_replacement_failure_preserves_runtime(self):
        """Filesystem error during atomic replacement FAILS promotion and leaves runtime file intact."""
        client = self.create_mock_client()

        with patch("src.evaluation.promotion.atomic_replace_file", side_effect=AtomicReplacementError("Disk full")):
            result = promote_candidate(
                candidate_run_id="cand_run_210",
                candidate_path=self.candidate_path,
                target_runtime_path=self.runtime_path,
                client=client,
            )

        self.assertEqual(result.status, "FAILED")
        self.assertIn("Disk full", result.error)
        self.assertEqual(self.runtime_path.read_text(encoding="utf-8"), '{"model": "champion_model"}')
        client.set_registered_model_alias.assert_not_called()

    def test_alias_update_failure_reports_inconsistency_without_unsafe_rollback(self):
        """If alias update fails after file swap, reports FAILED without overwriting runtime model."""
        client = self.create_mock_client()
        client.set_registered_model_alias.side_effect = RuntimeError("Network timeout setting alias")

        result = promote_candidate(
            candidate_run_id="cand_run_211",
            candidate_path=self.candidate_path,
            target_runtime_path=self.runtime_path,
            client=client,
        )

        self.assertEqual(result.status, "FAILED")
        self.assertIn("CRITICAL PARTIAL FAILURE", result.error)
        # Runtime model has the new model, and was NOT destroyed/reverted unsafely
        self.assertEqual(self.runtime_path.read_text(encoding="utf-8"), '{"model": "candidate_model"}')


class TestAtomicReplacement(TestPromotionBase):
    """Tests for atomic_replace_file function."""

    def test_successful_atomic_replacement(self):
        """Target is safely replaced with candidate content."""
        new_candidate = self.base_path / "new_candidate.json"
        new_candidate.write_text('{"new_key": 42}', encoding="utf-8")

        atomic_replace_file(new_candidate, self.runtime_path)

        self.assertEqual(self.runtime_path.read_text(encoding="utf-8"), '{"new_key": 42}')
        # Confirm no temporary files left in the directory
        temp_files = list(self.runtime_path.parent.glob(".tmp_runtime_*"))
        self.assertEqual(len(temp_files), 0)

    def test_missing_candidate_raises_error(self):
        """Missing candidate file raises AtomicReplacementError; target unchanged."""
        missing_candidate = self.base_path / "non_existent.json"

        with self.assertRaises(AtomicReplacementError):
            atomic_replace_file(missing_candidate, self.runtime_path)

        self.assertEqual(self.runtime_path.read_text(encoding="utf-8"), '{"model": "champion_model"}')


class TestChampionAndCandidateResolution(unittest.TestCase):
    """Tests for get_champion_metrics and get_candidate_metrics."""

    def test_get_champion_metrics_success(self):
        client = MagicMock()
        mock_mv = MagicMock()
        mock_mv.version = "1"
        mock_mv.run_id = "run_abc"
        mock_mv.aliases = ["champion"]
        client.get_model_version_by_alias.return_value = mock_mv

        mock_run = MagicMock()
        mock_run.data.metrics = {"WAPE": 0.225481, "MAE": 8.6744, "RMSE": 12.6023}
        client.get_run.return_value = mock_run

        info = get_champion_metrics(client, "DemandForecasterXGBoost", "champion")
        self.assertIsInstance(info, ChampionInfo)
        self.assertEqual(info.version, "1")
        self.assertEqual(info.run_id, "run_abc")
        self.assertAlmostEqual(info.metrics["WAPE"], 0.225481)

    def test_get_champion_metrics_raises_on_nan(self):
        client = MagicMock()
        mock_mv = MagicMock()
        mock_mv.version = "1"
        mock_mv.run_id = "run_abc"
        client.get_model_version_by_alias.return_value = mock_mv

        mock_run = MagicMock()
        mock_run.data.metrics = {"WAPE": float("nan"), "MAE": 8.6744, "RMSE": 12.6023}
        client.get_run.return_value = mock_run

        with self.assertRaises(ChampionResolutionError) as ctx:
            get_champion_metrics(client, "DemandForecasterXGBoost", "champion")
        self.assertIn("cannot be NaN", str(ctx.exception))

    def test_get_candidate_metrics_raises_on_empty_run_id(self):
        client = MagicMock()
        with self.assertRaises(CandidateMetricError):
            get_candidate_metrics(client, "")

    def test_get_candidate_metrics_raises_on_negative_metric(self):
        client = MagicMock()
        mock_run = MagicMock()
        mock_run.data.metrics = {"WAPE": -0.05, "MAE": 8.0, "RMSE": 10.0}
        client.get_run.return_value = mock_run

        with self.assertRaises(CandidateMetricError) as ctx:
            get_candidate_metrics(client, "run_test")
        self.assertIn("must be non-negative", str(ctx.exception))


class TestPromotionResultSerialization(unittest.TestCase):
    """Tests for PromotionResult serialization and dictionary access."""

    def test_to_dict_serialization(self):
        res = PromotionResult(
            status="PROMOTED",
            candidate_run_id="run_1",
            candidate_metrics={"WAPE": 0.21, "MAE": 8.0, "RMSE": 11.0},
            champion_version="1",
            champion_run_id="run_0",
            champion_metrics={"WAPE": 0.22, "MAE": 8.5, "RMSE": 12.0},
            registered_version="2",
            target_runtime_path="models/demand_forecaster_xgboost.json",
            reasons=["Promoted"],
            error=None,
        )

        d = res.to_dict()
        self.assertEqual(d["status"], "PROMOTED")
        self.assertEqual(d["registered_version"], "2")
        self.assertEqual(res["status"], "PROMOTED")
        self.assertIn("status", res)
        self.assertEqual(res.get("missing_key", 999), 999)


if __name__ == "__main__":
    unittest.main()
