import unittest
import math
from src.evaluation.quality_gate import (
    evaluate_quality_gate,
    QualityGateResult,
    DEFAULT_BASELINE_WAPE,
    DEFAULT_WAPE_GUARDRAIL,
    DEFAULT_MAE_GUARDRAIL,
)


class TestQualityGatePromote(unittest.TestCase):
    """Test scenarios where candidate model passes all applicable rules and is promoted."""

    def test_candidate_strictly_beats_all(self):
        candidate = {"WAPE": 0.2200, "MAE": 8.50}
        champion = {"WAPE": 0.2300, "MAE": 8.90}
        result = evaluate_quality_gate(candidate, champion)

        self.assertEqual(result.status, "PROMOTE")
        self.assertTrue(result.passed)
        self.assertTrue(result.checks["baseline_superiority"]["passed"])
        self.assertTrue(result.checks["champion_comparison"]["passed"])
        self.assertTrue(result.checks["wape_guardrail"]["passed"])
        self.assertTrue(result.checks["mae_guardrail"]["passed"])
        self.assertEqual(len(result.reasons), 4)

    def test_candidate_no_champion_supplied(self):
        candidate = {"WAPE": 0.2250, "MAE": 8.60}
        result = evaluate_quality_gate(candidate, champion_metrics=None)

        self.assertEqual(result.status, "PROMOTE")
        self.assertTrue(result.passed)
        self.assertFalse(result.checks["champion_comparison"]["applicable"])
        self.assertTrue(result.checks["champion_comparison"]["passed"])
        self.assertIsNone(result.champion_metrics)

    def test_candidate_equals_champion_wape(self):
        # Equality against champion must be accepted
        candidate = {"WAPE": 0.22548, "MAE": 8.674}
        champion = {"WAPE": 0.22548, "MAE": 8.674}
        result = evaluate_quality_gate(candidate, champion)

        self.assertEqual(result.status, "PROMOTE")
        self.assertTrue(result.passed)
        self.assertTrue(result.checks["champion_comparison"]["passed"])

    def test_candidate_exactly_equals_wape_guardrail(self):
        # WAPE guardrail allows exact equality (<= 0.2500)
        candidate = {"WAPE": 0.2500, "MAE": 8.00}
        champion = {"WAPE": 0.2600, "MAE": 9.00}
        result = evaluate_quality_gate(candidate, champion)

        self.assertEqual(result.status, "PROMOTE")
        self.assertTrue(result.passed)
        self.assertTrue(result.checks["wape_guardrail"]["passed"])

    def test_candidate_exactly_equals_mae_guardrail(self):
        # MAE guardrail allows exact equality (<= 10.00)
        candidate = {"WAPE": 0.2200, "MAE": 10.00}
        champion = {"WAPE": 0.2300, "MAE": 10.50}
        result = evaluate_quality_gate(candidate, champion)

        self.assertEqual(result.status, "PROMOTE")
        self.assertTrue(result.passed)
        self.assertTrue(result.checks["mae_guardrail"]["passed"])

    def test_candidate_exactly_equals_both_guardrails(self):
        # Boundary test: WAPE == 0.2500 and MAE == 10.00
        candidate = {"WAPE": 0.2500, "MAE": 10.00}
        result = evaluate_quality_gate(candidate, champion_metrics=None)

        self.assertEqual(result.status, "PROMOTE")
        self.assertTrue(result.passed)
        self.assertTrue(result.checks["wape_guardrail"]["passed"])
        self.assertTrue(result.checks["mae_guardrail"]["passed"])

    def test_lowercase_metric_keys_supported(self):
        candidate = {"wape": 0.2150, "mae": 8.20}
        champion = {"wape": 0.2250, "mae": 8.60}
        result = evaluate_quality_gate(candidate, champion)

        self.assertEqual(result.status, "PROMOTE")
        self.assertTrue(result.passed)

    def test_integer_metric_values_accepted(self):
        candidate = {"WAPE": 0.20, "MAE": 8}
        result = evaluate_quality_gate(candidate)
        self.assertEqual(result.status, "PROMOTE")
        self.assertTrue(result.passed)


class TestQualityGateRejectBoundaries(unittest.TestCase):
    """Test boundary conditions and threshold rejections."""

    def test_candidate_fails_baseline_comparison_greater(self):
        # candidate WAPE (0.30) exceeds baseline (0.29245)
        candidate = {"WAPE": 0.3000, "MAE": 8.00}
        result = evaluate_quality_gate(candidate)

        self.assertEqual(result.status, "REJECT")
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["baseline_superiority"]["passed"])
        self.assertTrue(any("baseline" in r.lower() for r in result.reasons))

    def test_candidate_fails_baseline_comparison_exact_equality(self):
        # Baseline check is STRICT: cand_wape == baseline_wape MUST FAIL
        candidate = {"WAPE": DEFAULT_BASELINE_WAPE, "MAE": 8.00}
        result = evaluate_quality_gate(candidate, baseline_wape=DEFAULT_BASELINE_WAPE)

        self.assertEqual(result.status, "REJECT")
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["baseline_superiority"]["passed"])

    def test_candidate_fails_champion_comparison_greater(self):
        # candidate WAPE (0.23) is worse than champion WAPE (0.22)
        candidate = {"WAPE": 0.2300, "MAE": 8.00}
        champion = {"WAPE": 0.2200, "MAE": 8.00}
        result = evaluate_quality_gate(candidate, champion)

        self.assertEqual(result.status, "REJECT")
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["champion_comparison"]["passed"])
        self.assertTrue(result.checks["champion_comparison"]["applicable"])

    def test_candidate_exceeds_wape_guardrail(self):
        # candidate WAPE 0.2501 > 0.2500
        candidate = {"WAPE": 0.2501, "MAE": 8.00}
        result = evaluate_quality_gate(candidate)

        self.assertEqual(result.status, "REJECT")
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["wape_guardrail"]["passed"])

    def test_candidate_exceeds_mae_guardrail(self):
        # candidate MAE 10.01 > 10.00
        candidate = {"WAPE": 0.2200, "MAE": 10.01}
        result = evaluate_quality_gate(candidate)

        self.assertEqual(result.status, "REJECT")
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["mae_guardrail"]["passed"])

    def test_candidate_fails_multiple_rules_simultaneously(self):
        # Fails baseline (0.35 > 0.2925), champion (0.35 > 0.22), WAPE guardrail (0.35 > 0.25), and MAE (12.0 > 10.0)
        candidate = {"WAPE": 0.3500, "MAE": 12.00}
        champion = {"WAPE": 0.2200, "MAE": 8.50}
        result = evaluate_quality_gate(candidate, champion)

        self.assertEqual(result.status, "REJECT")
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["baseline_superiority"]["passed"])
        self.assertFalse(result.checks["champion_comparison"]["passed"])
        self.assertFalse(result.checks["wape_guardrail"]["passed"])
        self.assertFalse(result.checks["mae_guardrail"]["passed"])
        self.assertEqual(len(result.reasons), 4)


class TestQualityGateInvalidInputValidation(unittest.TestCase):
    """Test robust rejection of non-numeric, missing, NaN, infinite, and negative inputs."""

    def test_missing_wape_metric(self):
        candidate = {"MAE": 8.00}
        result = evaluate_quality_gate(candidate)
        self.assertEqual(result.status, "REJECT")
        self.assertFalse(result.passed)
        self.assertTrue(any("WAPE" in r for r in result.reasons))

    def test_missing_mae_metric(self):
        candidate = {"WAPE": 0.2200}
        result = evaluate_quality_gate(candidate)
        self.assertEqual(result.status, "REJECT")
        self.assertFalse(result.passed)
        self.assertTrue(any("MAE" in r for r in result.reasons))

    def test_empty_candidate_metrics(self):
        result = evaluate_quality_gate({})
        self.assertEqual(result.status, "REJECT")
        self.assertFalse(result.passed)

    def test_non_mapping_candidate_metrics(self):
        for bad_input in [None, [0.22, 8.0], "metrics", 123]:
            with self.subTest(candidate=bad_input):
                result = evaluate_quality_gate(bad_input)
                self.assertEqual(result.status, "REJECT")
                self.assertFalse(result.passed)

    def test_negative_wape_rejected(self):
        candidate = {"WAPE": -0.05, "MAE": 8.00}
        result = evaluate_quality_gate(candidate)
        self.assertEqual(result.status, "REJECT")
        self.assertFalse(result.passed)
        self.assertTrue(any("non-negative" in r for r in result.reasons))

    def test_negative_mae_rejected(self):
        candidate = {"WAPE": 0.2200, "MAE": -1.50}
        result = evaluate_quality_gate(candidate)
        self.assertEqual(result.status, "REJECT")
        self.assertFalse(result.passed)
        self.assertTrue(any("non-negative" in r for r in result.reasons))

    def test_nan_candidate_metrics_rejected(self):
        candidate_nan_wape = {"WAPE": float("nan"), "MAE": 8.00}
        result = evaluate_quality_gate(candidate_nan_wape)
        self.assertEqual(result.status, "REJECT")
        self.assertFalse(result.passed)
        self.assertTrue(any("NaN" in r for r in result.reasons))

        candidate_nan_mae = {"WAPE": 0.2200, "MAE": float("nan")}
        result2 = evaluate_quality_gate(candidate_nan_mae)
        self.assertEqual(result2.status, "REJECT")
        self.assertFalse(result2.passed)
        self.assertTrue(any("NaN" in r for r in result2.reasons))

    def test_infinite_candidate_metrics_rejected(self):
        candidate_inf = {"WAPE": float("inf"), "MAE": 8.00}
        result = evaluate_quality_gate(candidate_inf)
        self.assertEqual(result.status, "REJECT")
        self.assertFalse(result.passed)
        self.assertTrue(any("infinite" in r for r in result.reasons))

        candidate_neginf = {"WAPE": 0.2200, "MAE": float("-inf")}
        result2 = evaluate_quality_gate(candidate_neginf)
        self.assertEqual(result2.status, "REJECT")
        self.assertFalse(result2.passed)

    def test_invalid_string_type_rejected(self):
        candidate = {"WAPE": "0.2200", "MAE": 8.00}
        result = evaluate_quality_gate(candidate)
        self.assertEqual(result.status, "REJECT")
        self.assertFalse(result.passed)

    def test_boolean_type_rejected(self):
        # Booleans inherit from int in Python; must be explicitly rejected
        candidate = {"WAPE": True, "MAE": 8.00}
        result = evaluate_quality_gate(candidate)
        self.assertEqual(result.status, "REJECT")
        self.assertFalse(result.passed)

    def test_invalid_champion_metrics_rejected(self):
        candidate = {"WAPE": 0.2200, "MAE": 8.00}
        # Non-mapping champion
        result = evaluate_quality_gate(candidate, champion_metrics="invalid")
        self.assertEqual(result.status, "REJECT")

        # NaN champion
        result2 = evaluate_quality_gate(candidate, champion_metrics={"WAPE": float("nan")})
        self.assertEqual(result2.status, "REJECT")

        # Negative champion
        result3 = evaluate_quality_gate(candidate, champion_metrics={"WAPE": -0.10})
        self.assertEqual(result3.status, "REJECT")


class TestQualityGateResultStructure(unittest.TestCase):
    """Test serialization, dictionary subscripting, and real project data integration."""

    def test_dictionary_subscripting_and_membership(self):
        candidate = {"WAPE": 0.2100, "MAE": 8.00}
        result = evaluate_quality_gate(candidate)

        self.assertEqual(result["status"], "PROMOTE")
        self.assertTrue(result["passed"])
        self.assertIn("checks", result)
        self.assertIn("thresholds", result)
        self.assertEqual(result.get("status"), "PROMOTE")

    def test_to_dict_serialization(self):
        candidate = {"WAPE": 0.2100, "MAE": 8.00}
        champion = {"WAPE": 0.2255, "MAE": 8.67}
        result = evaluate_quality_gate(candidate, champion)

        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["status"], "PROMOTE")
        self.assertTrue(d["passed"])
        self.assertEqual(d["thresholds"]["baseline_wape"], DEFAULT_BASELINE_WAPE)
        self.assertEqual(d["thresholds"]["wape_guardrail"], DEFAULT_WAPE_GUARDRAIL)
        self.assertEqual(d["thresholds"]["mae_guardrail"], DEFAULT_MAE_GUARDRAIL)
        self.assertIn("baseline_superiority", d["checks"])
        self.assertIn("champion_comparison", d["checks"])
        self.assertIn("wape_guardrail", d["checks"])
        self.assertIn("mae_guardrail", d["checks"])

    def test_real_authoritative_project_metrics(self):
        # Authoritative verified project champion metrics:
        champion = {"WAPE": 0.225481, "MAE": 8.6744, "RMSE": 12.6023}

        # Case A: Challenger model improves WAPE to 21.5%
        improved_candidate = {"WAPE": 0.215000, "MAE": 8.2000, "RMSE": 12.1000}
        res_improve = evaluate_quality_gate(improved_candidate, champion)
        self.assertEqual(res_improve.status, "PROMOTE")

        # Case B: Challenger model matches current champion
        matching_candidate = {"WAPE": 0.225481, "MAE": 8.6744}
        res_match = evaluate_quality_gate(matching_candidate, champion)
        self.assertEqual(res_match.status, "PROMOTE")

        # Case C: Baseline Moving Average (WAPE 29.25%) tested as candidate
        baseline_candidate = {"WAPE": 0.292454, "MAE": 11.2509}
        res_baseline = evaluate_quality_gate(baseline_candidate, champion)
        self.assertEqual(res_baseline.status, "REJECT")
        self.assertFalse(res_baseline.checks["baseline_superiority"]["passed"])
        self.assertFalse(res_baseline.checks["champion_comparison"]["passed"])


if __name__ == "__main__":
    unittest.main()
