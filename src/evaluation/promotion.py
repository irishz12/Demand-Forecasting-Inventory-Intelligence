"""
Model Promotion Orchestrator for Demand Forecasting & Inventory Intelligence.

Safely evaluates and promotes candidate models using the pure Quality Gate
and MLflow Model Registry champion aliases.

Architecture:
    Candidate Metrics
           ↓
    MLflow Champion Lookup (via alias "champion")
           ↓
    evaluate_quality_gate()
           ↓
    REJECT ──────────────→ Runtime champion untouched
           │
           └─ PROMOTE
                 ↓
          Register MLflow Version
                 ↓
          Atomic Runtime Replacement (POSIX os.replace)
                 ↓
          Update "champion" Alias

Concurrency Note:
    Champion resolution is performed immediately prior to candidate evaluation.
    This provides an optimistic compare-and-swap style check without requiring
    distributed locking or persistent server infrastructure for this MVP.
"""

from dataclasses import asdict, dataclass, field
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from src.evaluation.quality_gate import (
    DEFAULT_BASELINE_WAPE,
    DEFAULT_MAE_GUARDRAIL,
    DEFAULT_WAPE_GUARDRAIL,
    QualityGateResult,
    evaluate_quality_gate,
)

# Standard required metrics for model evaluation
REQUIRED_METRICS = ("MAE", "RMSE", "WAPE")


class PromotionError(Exception):
    """Base exception for model promotion failures."""
    pass


class ChampionResolutionError(PromotionError):
    """Raised when the current champion cannot be safely resolved from MLflow."""
    pass


class CandidateMetricError(PromotionError):
    """Raised when candidate metrics cannot be retrieved or are invalid."""
    pass


class RegistrationError(PromotionError):
    """Raised when registering a candidate in MLflow Model Registry fails."""
    pass


class AtomicReplacementError(PromotionError):
    """Raised when atomic filesystem replacement of the runtime artifact fails."""
    pass


class AliasUpdateError(PromotionError):
    """Raised when updating the champion alias in MLflow Model Registry fails."""
    pass


@dataclass(frozen=True)
class ChampionInfo:
    """Metadata and verified metrics for the active champion model."""

    version: str
    run_id: str
    metrics: Dict[str, float]
    aliases: List[str]


@dataclass
class PromotionResult:
    """Auditable result of a candidate model promotion evaluation."""

    status: str  # "PROMOTED", "REJECTED", or "FAILED"
    candidate_run_id: str
    candidate_metrics: Optional[Dict[str, float]] = None
    champion_version: Optional[str] = None
    champion_run_id: Optional[str] = None
    champion_metrics: Optional[Dict[str, float]] = None
    quality_gate_result: Optional[Dict[str, Any]] = None
    registered_version: Optional[str] = None
    target_runtime_path: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to a standard dictionary."""
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _extract_numeric_metric(
    metrics: Mapping[str, Any],
    metric_name: str,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract and validate a numeric metric from a metrics mapping.

    Checks:
    - Metric key presence (case-insensitive fallback: 'MAE' or 'mae')
    - Type is numeric (int or float, not bool)
    - Value is finite (not NaN or Inf)
    - Value is non-negative
    """
    found_key = None
    for candidate_key in (metric_name, metric_name.lower(), metric_name.upper()):
        if candidate_key in metrics:
            found_key = candidate_key
            break

    if found_key is None:
        return None, f"Missing required metric '{metric_name}'"

    raw_val = metrics[found_key]

    if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
        return None, f"Metric '{metric_name}' must be numeric, got {type(raw_val).__name__} ({raw_val!r})"

    val = float(raw_val)

    if math.isnan(val):
        return None, f"Metric '{metric_name}' cannot be NaN"

    if math.isinf(val):
        return None, f"Metric '{metric_name}' cannot be infinite"

    if val < 0.0:
        return None, f"Metric '{metric_name}' must be non-negative, got {val}"

    return val, None


def get_champion_metrics(
    client: Any,
    model_name: str = "DemandForecasterXGBoost",
    alias: str = "champion",
) -> ChampionInfo:
    """
    Resolve the current champion model version and metrics from MLflow.

    FAILS CLOSED:
    If the MLflow client fails, the registered model or alias does not exist,
    the run cannot be retrieved, or metrics are missing/invalid, this function
    raises ChampionResolutionError.

    DO NOT return hard-coded fallback champion metrics.
    """
    try:
        model_version = client.get_model_version_by_alias(model_name, alias)
    except Exception as exc:
        raise ChampionResolutionError(
            f"Failed to resolve champion model version for '{model_name}' alias '{alias}': {exc}"
        ) from exc

    if model_version is None:
        raise ChampionResolutionError(
            f"No champion model version found for '{model_name}' alias '{alias}'"
        )

    version_str = str(getattr(model_version, "version", "")).strip()
    run_id = str(getattr(model_version, "run_id", "")).strip()
    aliases = list(getattr(model_version, "aliases", []))

    if not version_str:
        raise ChampionResolutionError(
            f"Resolved champion version for '{model_name}@{alias}' has empty version identifier"
        )

    if not run_id:
        raise ChampionResolutionError(
            f"Resolved champion version {version_str} for '{model_name}@{alias}' has no associated run_id"
        )

    try:
        run = client.get_run(run_id)
    except Exception as exc:
        raise ChampionResolutionError(
            f"Failed to retrieve MLflow run '{run_id}' for champion version {version_str}: {exc}"
        ) from exc

    if run is None or not hasattr(run, "data") or not hasattr(run.data, "metrics"):
        raise ChampionResolutionError(
            f"MLflow run '{run_id}' for champion version {version_str} contains no data/metrics"
        )

    raw_metrics = run.data.metrics or {}
    validated_metrics: Dict[str, float] = {}

    for metric_name in REQUIRED_METRICS:
        val, err = _extract_numeric_metric(raw_metrics, metric_name)
        if err:
            raise ChampionResolutionError(
                f"Invalid champion metric for version {version_str} (run '{run_id}'): {err}"
            )
        assert val is not None
        validated_metrics[metric_name] = val

    return ChampionInfo(
        version=version_str,
        run_id=run_id,
        metrics=validated_metrics,
        aliases=aliases,
    )


def get_candidate_metrics(
    client: Any,
    candidate_run_id: str,
) -> Dict[str, float]:
    """
    Retrieve and validate candidate metrics from an MLflow run.

    Retrieves: MAE, RMSE, WAPE.
    Fails safely with CandidateMetricError if any metric is missing or invalid.
    """
    if not candidate_run_id or not str(candidate_run_id).strip():
        raise CandidateMetricError("Candidate run_id cannot be empty")

    try:
        run = client.get_run(candidate_run_id)
    except Exception as exc:
        raise CandidateMetricError(
            f"Failed to retrieve candidate MLflow run '{candidate_run_id}': {exc}"
        ) from exc

    if run is None or not hasattr(run, "data") or not hasattr(run.data, "metrics"):
        raise CandidateMetricError(
            f"Candidate MLflow run '{candidate_run_id}' contains no data/metrics"
        )

    raw_metrics = run.data.metrics or {}
    validated_metrics: Dict[str, float] = {}

    for metric_name in REQUIRED_METRICS:
        val, err = _extract_numeric_metric(raw_metrics, metric_name)
        if err:
            raise CandidateMetricError(
                f"Invalid candidate metric for run '{candidate_run_id}': {err}"
            )
        assert val is not None
        validated_metrics[metric_name] = val

    return validated_metrics


def evaluate_candidate(
    candidate_metrics: Mapping[str, Any],
    champion_metrics: Mapping[str, Any],
    baseline_wape: float = DEFAULT_BASELINE_WAPE,
    wape_guardrail: float = DEFAULT_WAPE_GUARDRAIL,
    mae_guardrail: float = DEFAULT_MAE_GUARDRAIL,
) -> QualityGateResult:
    """
    Delegate candidate evaluation to the pure Quality Gate.

    Does NOT duplicate Quality Gate logic.
    """
    return evaluate_quality_gate(
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        baseline_wape=baseline_wape,
        wape_guardrail=wape_guardrail,
        mae_guardrail=mae_guardrail,
    )


def register_candidate_version(
    client: Any,
    candidate_run_id: str,
    model_name: str = "DemandForecasterXGBoost",
    artifact_path: str = "demand_forecaster",
) -> str:
    """
    Register a candidate run as a new version in the MLflow Model Registry.

    Returns:
        The newly registered model version number as a string (e.g. "2").
    """
    try:
        source_uri = f"runs:/{candidate_run_id}/{artifact_path}"
        mv = client.create_model_version(
            name=model_name,
            source=source_uri,
            run_id=candidate_run_id,
        )
        return str(mv.version)
    except Exception as exc:
        raise RegistrationError(
            f"Failed to register model version for candidate run '{candidate_run_id}': {exc}"
        ) from exc


def atomic_replace_file(
    candidate_path: Union[str, Path],
    target_runtime_path: Union[str, Path],
) -> None:
    """
    Atomically replace target_runtime_path with candidate_path.

    Guarantees:
    - Target is never opened with 'w' or truncated directly.
    - Staged to a temporary file in the SAME directory/filesystem as target.
    - Flushed and fsynced before atomic rename via os.replace.
    - Temporary file cleaned up on any failure.
    - Existing target remains intact if replacement does not execute.
    """
    candidate = Path(candidate_path).resolve()
    target = Path(target_runtime_path).resolve()

    if not candidate.is_file():
        raise AtomicReplacementError(
            f"Candidate file does not exist or is not a regular file: {candidate}"
        )

    target_dir = target.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    temp_path: Optional[Path] = None
    try:
        # Create temp file in the same directory to ensure same filesystem for os.replace
        with tempfile.NamedTemporaryFile(
            dir=target_dir,
            prefix=".tmp_runtime_",
            suffix=".json",
            delete=False,
        ) as tmp_file:
            temp_path = Path(tmp_file.name)
            with open(candidate, "rb") as src_file:
                shutil.copyfileobj(src_file, tmp_file)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        # Atomic replacement under POSIX semantics
        os.replace(temp_path, target)
        temp_path = None
    except Exception as exc:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise AtomicReplacementError(
            f"Failed to atomically replace '{target}' with '{candidate}': {exc}"
        ) from exc


def update_champion_alias(
    client: Any,
    model_name: str,
    alias: str,
    version: str,
) -> None:
    """
    Update the champion alias in the MLflow Model Registry to point to version.
    """
    try:
        client.set_registered_model_alias(model_name, alias, str(version))
    except Exception as exc:
        raise AliasUpdateError(
            f"Failed to set alias '{alias}' on model '{model_name}' to version '{version}': {exc}"
        ) from exc


def promote_candidate(
    candidate_run_id: str,
    candidate_path: Union[str, Path],
    target_runtime_path: Union[str, Path] = "models/demand_forecaster_xgboost.json",
    model_name: str = "DemandForecasterXGBoost",
    alias: str = "champion",
    artifact_path: str = "demand_forecaster",
    tracking_uri: str = "sqlite:///mlflow.db",
    client: Optional[Any] = None,
    baseline_wape: float = DEFAULT_BASELINE_WAPE,
    wape_guardrail: float = DEFAULT_WAPE_GUARDRAIL,
    mae_guardrail: float = DEFAULT_MAE_GUARDRAIL,
    raise_on_failure: bool = False,
) -> PromotionResult:
    """
    Orchestrate safe candidate evaluation, registration, atomic replacement,
    and alias promotion.

    Workflow:
      1. Initialize MLflowClient if not injected.
      2. Retrieve candidate metrics from candidate_run_id.
      3. Retrieve current champion metrics via alias (FAILS CLOSED).
      4. Evaluate candidate against Quality Gate.
      5. If REJECT:
           - Return PromotionResult(status="REJECTED")
           - Runtime model remains untouched.
           - Candidate remains in MLflow run artifacts.
      6. If PROMOTE:
           - Register candidate as new version in MLflow Model Registry.
           - Atomically replace target runtime file via temp file + os.replace.
           - Update alias "champion" to the newly registered version.
           - Return PromotionResult(status="PROMOTED")

    Failure Handling:
      - All infrastructure failures yield status="FAILED" and preserve runtime artifact.
      - If alias update fails after atomic replacement, reports inconsistent state
        without unsafe automatic rollback.
    """
    # 1. Initialize client if not injected
    if client is None:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)
        except Exception as exc:
            err_msg = f"Failed to initialize MLflowClient with tracking URI '{tracking_uri}': {exc}"
            if raise_on_failure:
                raise PromotionError(err_msg) from exc
            return PromotionResult(
                status="FAILED",
                candidate_run_id=candidate_run_id,
                target_runtime_path=str(target_runtime_path),
                reasons=[err_msg],
                error=err_msg,
            )

    # 2. Retrieve candidate metrics
    try:
        candidate_metrics = get_candidate_metrics(client, candidate_run_id)
    except CandidateMetricError as exc:
        err_msg = str(exc)
        if raise_on_failure:
            raise
        return PromotionResult(
            status="FAILED",
            candidate_run_id=candidate_run_id,
            target_runtime_path=str(target_runtime_path),
            reasons=[err_msg],
            error=err_msg,
        )

    # 3. Retrieve champion metrics (Fail-Closed)
    try:
        champion_info = get_champion_metrics(
            client=client,
            model_name=model_name,
            alias=alias,
        )
    except ChampionResolutionError as exc:
        err_msg = str(exc)
        if raise_on_failure:
            raise
        return PromotionResult(
            status="FAILED",
            candidate_run_id=candidate_run_id,
            candidate_metrics=candidate_metrics,
            target_runtime_path=str(target_runtime_path),
            reasons=[err_msg],
            error=err_msg,
        )

    # 4. Evaluate candidate against Quality Gate
    gate_result = evaluate_candidate(
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_info.metrics,
        baseline_wape=baseline_wape,
        wape_guardrail=wape_guardrail,
        mae_guardrail=mae_guardrail,
    )

    # 5. Handle Quality Gate Rejection
    if gate_result.status != "PROMOTE":
        return PromotionResult(
            status="REJECTED",
            candidate_run_id=candidate_run_id,
            candidate_metrics=candidate_metrics,
            champion_version=champion_info.version,
            champion_run_id=champion_info.run_id,
            champion_metrics=champion_info.metrics,
            quality_gate_result=gate_result.to_dict(),
            registered_version=None,
            target_runtime_path=str(target_runtime_path),
            reasons=gate_result.reasons,
            error=None,
        )

    # 6. Candidate Passed Gate -> Proceed to Registration & Atomic Promotion
    # 6a. Register candidate version in Model Registry
    try:
        registered_version = register_candidate_version(
            client=client,
            candidate_run_id=candidate_run_id,
            model_name=model_name,
            artifact_path=artifact_path,
        )
    except RegistrationError as exc:
        err_msg = str(exc)
        if raise_on_failure:
            raise
        return PromotionResult(
            status="FAILED",
            candidate_run_id=candidate_run_id,
            candidate_metrics=candidate_metrics,
            champion_version=champion_info.version,
            champion_run_id=champion_info.run_id,
            champion_metrics=champion_info.metrics,
            quality_gate_result=gate_result.to_dict(),
            registered_version=None,
            target_runtime_path=str(target_runtime_path),
            reasons=[err_msg],
            error=err_msg,
        )

    # 6b. Atomic replacement of runtime model artifact
    try:
        atomic_replace_file(
            candidate_path=candidate_path,
            target_runtime_path=target_runtime_path,
        )
    except AtomicReplacementError as exc:
        err_msg = str(exc)
        if raise_on_failure:
            raise
        return PromotionResult(
            status="FAILED",
            candidate_run_id=candidate_run_id,
            candidate_metrics=candidate_metrics,
            champion_version=champion_info.version,
            champion_run_id=champion_info.run_id,
            champion_metrics=champion_info.metrics,
            quality_gate_result=gate_result.to_dict(),
            registered_version=registered_version,
            target_runtime_path=str(target_runtime_path),
            reasons=[err_msg],
            error=err_msg,
        )

    # 6c. Update champion alias in Model Registry
    try:
        update_champion_alias(
            client=client,
            model_name=model_name,
            alias=alias,
            version=registered_version,
        )
    except AliasUpdateError as exc:
        err_msg = (
            f"CRITICAL PARTIAL FAILURE: Runtime model '{target_runtime_path}' was replaced with "
            f"candidate version {registered_version}, but updating MLflow alias '{alias}' failed: {exc}. "
            f"Runtime model is now version {registered_version} while MLflow alias may still point to "
            f"version {champion_info.version}. Do NOT overwrite runtime file automatically; update alias manually."
        )
        if raise_on_failure:
            raise AliasUpdateError(err_msg) from exc
        return PromotionResult(
            status="FAILED",
            candidate_run_id=candidate_run_id,
            candidate_metrics=candidate_metrics,
            champion_version=champion_info.version,
            champion_run_id=champion_info.run_id,
            champion_metrics=champion_info.metrics,
            quality_gate_result=gate_result.to_dict(),
            registered_version=registered_version,
            target_runtime_path=str(target_runtime_path),
            reasons=[err_msg],
            error=err_msg,
        )

    # Promotion succeeded completely
    return PromotionResult(
        status="PROMOTED",
        candidate_run_id=candidate_run_id,
        candidate_metrics=candidate_metrics,
        champion_version=champion_info.version,
        champion_run_id=champion_info.run_id,
        champion_metrics=champion_info.metrics,
        quality_gate_result=gate_result.to_dict(),
        registered_version=registered_version,
        target_runtime_path=str(target_runtime_path),
        reasons=[
            f"Candidate passed Quality Gate and promoted to {model_name} version {registered_version} (alias: '{alias}')"
        ],
        error=None,
    )
