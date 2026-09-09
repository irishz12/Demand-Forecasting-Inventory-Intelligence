"""
Model Quality Gate for Demand Forecasting & Inventory Intelligence.

Deterministic Champion/Challenger evaluation gate comparing candidate model
metrics against baseline benchmarks, existing champion performance, and
absolute quality guardrails.
"""

from dataclasses import dataclass, asdict
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

DEFAULT_BASELINE_WAPE: float = 0.29245
DEFAULT_WAPE_GUARDRAIL: float = 0.2500
DEFAULT_MAE_GUARDRAIL: float = 10.00


@dataclass
class QualityGateResult:
    """Structured result from quality gate evaluation."""

    status: str  # "PROMOTE" or "REJECT"
    passed: bool
    checks: Dict[str, Dict[str, Any]]
    reasons: List[str]
    candidate_metrics: Dict[str, Any]
    champion_metrics: Optional[Dict[str, Any]]
    thresholds: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to a standard dictionary."""
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _validate_and_extract_metric(
    metrics: Any,
    candidate_keys: List[str],
    metric_name: str,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract and validate a numeric metric from a metrics mapping.

    Returns:
        (value, error_message): If valid, value is float and error_message is None.
                                If invalid, value is None and error_message describes the issue.
    """
    if not isinstance(metrics, Mapping):
        return None, f"Metrics object must be a mapping, got {type(metrics).__name__}"

    found_key = None
    for k in candidate_keys:
        if k in metrics:
            found_key = k
            break

    if found_key is None:
        return None, f"Missing required metric '{metric_name}'"

    raw_val = metrics[found_key]

    if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
        return None, (
            f"Metric '{metric_name}' must be numeric, "
            f"got {type(raw_val).__name__} ({raw_val!r})"
        )

    val = float(raw_val)

    if math.isnan(val):
        return None, f"Metric '{metric_name}' cannot be NaN"

    if math.isinf(val):
        return None, f"Metric '{metric_name}' cannot be infinite"

    if val < 0.0:
        return None, f"Metric '{metric_name}' must be non-negative, got {val}"

    return val, None


def evaluate_quality_gate(
    candidate_metrics: Any,
    champion_metrics: Optional[Any] = None,
    baseline_wape: float = DEFAULT_BASELINE_WAPE,
    wape_guardrail: float = DEFAULT_WAPE_GUARDRAIL,
    mae_guardrail: float = DEFAULT_MAE_GUARDRAIL,
) -> QualityGateResult:
    """
    Evaluate candidate model metrics against baseline, champion, and guardrails.

    Rules:
      1. Baseline Superiority: Candidate WAPE < baseline_wape (strict inequality)
      2. Champion Comparison: If champion provided, Candidate WAPE <= champion WAPE (allows equality)
         If champion not provided, marked as not applicable (passed=True, applicable=False)
      3. Guardrails: Candidate WAPE <= wape_guardrail (<= 0.2500)
                     Candidate MAE <= mae_guardrail (<= 10.00)

    Returns:
        QualityGateResult with status 'PROMOTE' if all applicable checks pass,
        otherwise 'REJECT'.
    """
    thresholds = {
        "baseline_wape": float(baseline_wape),
        "wape_guardrail": float(wape_guardrail),
        "mae_guardrail": float(mae_guardrail),
    }

    # Validate candidate metrics mapping
    if not isinstance(candidate_metrics, Mapping):
        reason = (
            f"Candidate metrics must be a valid mapping, got {type(candidate_metrics).__name__}"
        )
        return QualityGateResult(
            status="REJECT",
            passed=False,
            checks={
                "baseline_superiority": {
                    "passed": False,
                    "applicable": True,
                    "message": "Candidate metrics invalid",
                    "candidate_value": None,
                    "threshold_value": baseline_wape,
                },
                "champion_comparison": {
                    "passed": False,
                    "applicable": champion_metrics is not None,
                    "message": "Candidate metrics invalid",
                    "candidate_value": None,
                    "threshold_value": None,
                },
                "wape_guardrail": {
                    "passed": False,
                    "applicable": True,
                    "message": "Candidate metrics invalid",
                    "candidate_value": None,
                    "threshold_value": wape_guardrail,
                },
                "mae_guardrail": {
                    "passed": False,
                    "applicable": True,
                    "message": "Candidate metrics invalid",
                    "candidate_value": None,
                    "threshold_value": mae_guardrail,
                },
            },
            reasons=[reason],
            candidate_metrics={},
            champion_metrics=dict(champion_metrics) if isinstance(champion_metrics, Mapping) else None,
            thresholds=thresholds,
        )

    cand_wape, err_wape = _validate_and_extract_metric(
        candidate_metrics, ["WAPE", "wape"], "WAPE"
    )
    cand_mae, err_mae = _validate_and_extract_metric(
        candidate_metrics, ["MAE", "mae"], "MAE"
    )

    validation_errors = []
    if err_wape:
        validation_errors.append(f"Candidate: {err_wape}")
    if err_mae:
        validation_errors.append(f"Candidate: {err_mae}")

    # Validate champion metrics if supplied
    champ_wape = None
    if champion_metrics is not None:
        if not isinstance(champion_metrics, Mapping):
            validation_errors.append(
                f"Champion metrics must be a valid mapping, got {type(champion_metrics).__name__}"
            )
        else:
            champ_wape, err_champ_wape = _validate_and_extract_metric(
                champion_metrics, ["WAPE", "wape"], "WAPE"
            )
            if err_champ_wape:
                validation_errors.append(f"Champion: {err_champ_wape}")

    checks: Dict[str, Dict[str, Any]] = {}
    passed_reasons: List[str] = []
    failed_reasons: List[str] = []

    if validation_errors:
        failed_reasons.extend(validation_errors)
        checks["baseline_superiority"] = {
            "passed": False,
            "applicable": True,
            "message": "Check failed due to invalid metric input",
            "candidate_value": cand_wape,
            "threshold_value": baseline_wape,
        }
        checks["champion_comparison"] = {
            "passed": False,
            "applicable": champion_metrics is not None,
            "message": "Check failed due to invalid metric input",
            "candidate_value": cand_wape,
            "threshold_value": champ_wape,
        }
        checks["wape_guardrail"] = {
            "passed": False,
            "applicable": True,
            "message": "Check failed due to invalid metric input",
            "candidate_value": cand_wape,
            "threshold_value": wape_guardrail,
        }
        checks["mae_guardrail"] = {
            "passed": False,
            "applicable": True,
            "message": "Check failed due to invalid metric input",
            "candidate_value": cand_mae,
            "threshold_value": mae_guardrail,
        }

        return QualityGateResult(
            status="REJECT",
            passed=False,
            checks=checks,
            reasons=failed_reasons,
            candidate_metrics=dict(candidate_metrics),
            champion_metrics=dict(champion_metrics) if isinstance(champion_metrics, Mapping) else None,
            thresholds=thresholds,
        )

    # 1. Baseline superiority (strict: cand_wape < baseline_wape)
    if cand_wape < baseline_wape:
        checks["baseline_superiority"] = {
            "passed": True,
            "applicable": True,
            "message": (
                f"Candidate WAPE ({cand_wape:.4f}) strictly outperforms "
                f"Moving Average baseline ({baseline_wape:.4f})"
            ),
            "candidate_value": cand_wape,
            "threshold_value": baseline_wape,
        }
        passed_reasons.append(checks["baseline_superiority"]["message"])
    else:
        checks["baseline_superiority"] = {
            "passed": False,
            "applicable": True,
            "message": (
                f"Candidate WAPE ({cand_wape:.4f}) does not strictly beat "
                f"baseline WAPE ({baseline_wape:.4f})"
            ),
            "candidate_value": cand_wape,
            "threshold_value": baseline_wape,
        }
        failed_reasons.append(checks["baseline_superiority"]["message"])

    # 2. Champion comparison (cand_wape <= champ_wape)
    if champion_metrics is None:
        checks["champion_comparison"] = {
            "passed": True,
            "applicable": False,
            "message": "No champion model provided; champion comparison skipped",
            "candidate_value": cand_wape,
            "threshold_value": None,
        }
        passed_reasons.append(checks["champion_comparison"]["message"])
    else:
        if cand_wape <= champ_wape:
            checks["champion_comparison"] = {
                "passed": True,
                "applicable": True,
                "message": (
                    f"Candidate WAPE ({cand_wape:.4f}) matches or improves "
                    f"champion WAPE ({champ_wape:.4f})"
                ),
                "candidate_value": cand_wape,
                "threshold_value": champ_wape,
            }
            passed_reasons.append(checks["champion_comparison"]["message"])
        else:
            checks["champion_comparison"] = {
                "passed": False,
                "applicable": True,
                "message": (
                    f"Candidate WAPE ({cand_wape:.4f}) is worse than "
                    f"champion WAPE ({champ_wape:.4f})"
                ),
                "candidate_value": cand_wape,
                "threshold_value": champ_wape,
            }
            failed_reasons.append(checks["champion_comparison"]["message"])

    # 3. WAPE guardrail (cand_wape <= wape_guardrail)
    if cand_wape <= wape_guardrail:
        checks["wape_guardrail"] = {
            "passed": True,
            "applicable": True,
            "message": (
                f"Candidate WAPE ({cand_wape:.4f}) satisfies guardrail "
                f"(<= {wape_guardrail:.4f})"
            ),
            "candidate_value": cand_wape,
            "threshold_value": wape_guardrail,
        }
        passed_reasons.append(checks["wape_guardrail"]["message"])
    else:
        checks["wape_guardrail"] = {
            "passed": False,
            "applicable": True,
            "message": (
                f"Candidate WAPE ({cand_wape:.4f}) exceeds guardrail threshold "
                f"(> {wape_guardrail:.4f})"
            ),
            "candidate_value": cand_wape,
            "threshold_value": wape_guardrail,
        }
        failed_reasons.append(checks["wape_guardrail"]["message"])

    # 4. MAE guardrail (cand_mae <= mae_guardrail)
    if cand_mae <= mae_guardrail:
        checks["mae_guardrail"] = {
            "passed": True,
            "applicable": True,
            "message": (
                f"Candidate MAE ({cand_mae:.4f}) satisfies guardrail "
                f"(<= {mae_guardrail:.4f})"
            ),
            "candidate_value": cand_mae,
            "threshold_value": mae_guardrail,
        }
        passed_reasons.append(checks["mae_guardrail"]["message"])
    else:
        checks["mae_guardrail"] = {
            "passed": False,
            "applicable": True,
            "message": (
                f"Candidate MAE ({cand_mae:.4f}) exceeds guardrail threshold "
                f"(> {mae_guardrail:.4f})"
            ),
            "candidate_value": cand_mae,
            "threshold_value": mae_guardrail,
        }
        failed_reasons.append(checks["mae_guardrail"]["message"])

    all_passed = all(chk["passed"] for chk in checks.values())
    status = "PROMOTE" if all_passed else "REJECT"
    reasons = passed_reasons if all_passed else failed_reasons

    return QualityGateResult(
        status=status,
        passed=all_passed,
        checks=checks,
        reasons=reasons,
        candidate_metrics=dict(candidate_metrics),
        champion_metrics=dict(champion_metrics) if isinstance(champion_metrics, Mapping) else None,
        thresholds=thresholds,
    )
