"""Deterministic Memory CI checks for staged TruthGit writes.

The LLM can propose claims and write actions, but this module owns the
repository-local validation pipeline. It is intentionally generic: checks
inspect claim content, source trust, branch context, active belief state, and
audit/provenance history without using evaluation IDs or task fixtures.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import crud, models
from app.conflict_engine import belief_score, classify_claim_against_current
from app.memory_ci_policy import MemoryCIPolicyConfig, PredicatePolicyClass, default_memory_ci_policy
from app.normalization import NormalizedClaim, canonical_text, normalize_extracted_claim, windows_overlap
from app.schemas import ExtractedClaim

PASS_STATUSES = {"checked"}
REVIEWABLE_STATUSES = {"pending", "proposed", "checked", "review_required"}
TERMINAL_STATUSES = {"applied", "rejected"}


@dataclass(frozen=True)
class CheckContext:
    """Inputs shared by all Memory CI checks."""

    db: Session
    staged: models.StagedCommit
    branch: models.Branch
    claims: list[NormalizedClaim]
    source_ref: str | None
    source_excerpt: str
    source_trust: float
    policy_config: MemoryCIPolicyConfig

    def policy_for(self, claim: NormalizedClaim) -> PredicatePolicyClass:
        """Return the configured predicate policy for one claim."""

        return self.policy_config.policy_for_predicate(claim.predicate)


@dataclass(frozen=True)
class MemoryCheckOutcome:
    """One check result before persistence."""

    check_name: str
    severity: str
    passed: bool
    reason_code: str
    message: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class MemoryCIDecision:
    """Computed policy decision from a check run."""

    overall_status: str
    decision: str
    score: float
    reason_summary: str | None


CheckFunction = Callable[[CheckContext], MemoryCheckOutcome]


@dataclass(frozen=True)
class MemoryCheckSpec:
    """Registry metadata for a deterministic Memory CI check."""

    name: str
    function: CheckFunction
    description: str


@dataclass(frozen=True)
class MemoryCheckRegistry:
    """Executable registry for Memory CI checks."""

    checks: tuple[MemoryCheckSpec, ...]

    def run(self, context: CheckContext) -> list[MemoryCheckOutcome]:
        """Run checks enabled by the policy config."""

        enabled = set(context.policy_config.enabled_checks)
        return [spec.function(context) for spec in self.checks if spec.name in enabled]


def run_memory_ci(
    db: Session,
    staged_commit_id: str,
    policy_config: MemoryCIPolicyConfig | None = None,
) -> models.MemoryCheckRun:
    """Run deterministic Memory CI and update the staged commit lifecycle."""

    config = policy_config or default_memory_ci_policy()
    staged = db.get(models.StagedCommit, staged_commit_id)
    if staged is None:
        raise ValueError(f"Unknown staged commit: {staged_commit_id}")
    branch = crud.get_branch(db, staged.branch_id)
    if branch is None:
        raise ValueError(f"Staged commit branch does not exist: {staged.branch_id}")
    if staged.status in TERMINAL_STATUSES:
        raise ValueError(f"Staged commit {staged.id} is terminal: {staged.status}")

    claims = _claims_from_staged(staged)
    context = CheckContext(
        db=db,
        staged=staged,
        branch=branch,
        claims=claims,
        source_ref=staged.source_ref,
        source_excerpt=staged.source_excerpt,
        source_trust=max(0.0, min(1.0, float(staged.source_trust_score))),
        policy_config=config,
    )
    crud.add_audit_event(
        db,
        event_type="memory_ci.started",
        entity_type="staged_commit",
        entity_id=0,
        entity_key=staged.id,
        payload={"suite_version": config.suite_version, "staged_commit_id": staged.id},
    )
    outcomes = DEFAULT_CHECK_REGISTRY.run(context)
    if not claims:
        outcomes.append(
            MemoryCheckOutcome(
                check_name="non_empty_claims_check",
                severity="fail",
                passed=False,
                reason_code="no_claims",
                message="No atomic claims were available for the proposed memory write.",
                payload={},
            )
        )
    decision = decide_memory_ci(staged=staged, outcomes=outcomes)
    now = models.utc_now()
    run = models.MemoryCheckRun(
        staged_commit_id=staged.id,
        suite_version=config.suite_version,
        overall_status=decision.overall_status,
        decision=decision.decision,
        score=decision.score,
        metadata_json={
            "claim_count": len(claims),
            "branch_id": branch.id,
            "branch_name": branch.name,
            "source_trust_score": context.source_trust,
            "preexisting_review_required": bool(staged.review_required),
            "predicate_policy_classes": {
                claim.predicate: context.policy_for(claim).name
                for claim in claims
            },
            "enabled_checks": list(config.enabled_checks),
        },
        completed_at=now,
    )
    db.add(run)
    db.flush()
    for outcome in outcomes:
        db.add(
            models.MemoryCheckResult(
                run_id=run.id,
                check_name=outcome.check_name,
                severity=outcome.severity,
                passed=outcome.passed,
                reason_code=outcome.reason_code,
                message=outcome.message,
                payload_json=outcome.payload,
            )
        )
    staged.latest_check_run_id = run.id
    staged.checked_at = now
    _apply_decision_to_staged(db, staged=staged, decision=decision, outcomes=outcomes)
    crud.add_audit_event(
        db,
        event_type="memory_ci.completed",
        entity_type="staged_commit",
        entity_id=0,
        entity_key=staged.id,
        payload={
            "run_id": run.id,
            "overall_status": decision.overall_status,
            "decision": decision.decision,
            "reason_summary": decision.reason_summary,
        },
    )
    db.flush()
    return run


def decide_memory_ci(
    *,
    staged: models.StagedCommit,
    outcomes: list[MemoryCheckOutcome],
) -> MemoryCIDecision:
    """Compute a deterministic lifecycle decision from check outcomes."""

    failing = [outcome for outcome in outcomes if not outcome.passed and outcome.severity == "fail"]
    warning = [outcome for outcome in outcomes if not outcome.passed and outcome.severity == "warn"]
    passed_count = sum(1 for outcome in outcomes if outcome.passed)
    score = round(passed_count / max(len(outcomes), 1), 4)
    if any(outcome.reason_code == "no_claims" for outcome in failing):
        return MemoryCIDecision(
            overall_status="fail",
            decision="reject",
            score=score,
            reason_summary="No atomic claims were available for the proposed memory write.",
        )
    if failing:
        return MemoryCIDecision(
            overall_status="fail",
            decision="quarantine",
            score=score,
            reason_summary="; ".join(_dedupe([outcome.message for outcome in failing])[:4]),
        )
    if warning or staged.review_required:
        reasons = [outcome.message for outcome in warning]
        if staged.review_required:
            reasons.extend(list(staged.risk_reasons or []))
        return MemoryCIDecision(
            overall_status="warn",
            decision="require_review",
            score=score,
            reason_summary="; ".join(_dedupe(reasons)[:4]) or "Policy requires review.",
        )
    return MemoryCIDecision(
        overall_status="pass",
        decision="auto_apply",
        score=score,
        reason_summary=None,
    )


def list_check_results(db: Session, run_id: int) -> list[models.MemoryCheckResult]:
    """Return persisted per-check results for one run."""

    return list(
        db.scalars(
            select(models.MemoryCheckResult)
            .where(models.MemoryCheckResult.run_id == run_id)
            .order_by(models.MemoryCheckResult.id)
        )
    )


def latest_check_report(db: Session, staged_commit_id: str) -> tuple[models.MemoryCheckRun | None, list[models.MemoryCheckResult]]:
    """Return the latest check run and its results for a staged commit."""

    staged = db.get(models.StagedCommit, staged_commit_id)
    if staged is None:
        raise ValueError(f"Unknown staged commit: {staged_commit_id}")
    if staged.latest_check_run_id is None:
        return None, []
    run = db.get(models.MemoryCheckRun, staged.latest_check_run_id)
    if run is None:
        return None, []
    return run, list_check_results(db, run.id)


def release_quarantine(
    db: Session,
    *,
    staged_commit_id: str,
    reviewer: str,
    notes: str,
) -> models.StagedCommit:
    """Release a quarantined item back to manual review without applying it."""

    staged = db.get(models.StagedCommit, staged_commit_id)
    if staged is None:
        raise ValueError(f"Unknown staged commit: {staged_commit_id}")
    if staged.status != "quarantined":
        raise ValueError(f"Staged commit {staged_commit_id} is {staged.status}, not quarantined")
    if not notes.strip():
        raise ValueError("Quarantine release requires reviewer notes")
    staged.status = "review_required"
    staged.review_required = True
    staged.quarantine_release_status = "released"
    staged.quarantine_reviewer = reviewer
    staged.quarantine_notes = notes
    staged.reviewed_at = models.utc_now()
    crud.add_audit_event(
        db,
        event_type="staged_commit.quarantine_released",
        entity_type="staged_commit",
        entity_id=0,
        entity_key=staged.id,
        payload={"staged_commit_id": staged.id, "reviewer": reviewer, "notes": notes},
    )
    db.flush()
    return staged


def check_report_payload(
    db: Session,
    staged: models.StagedCommit,
) -> dict[str, Any]:
    """Serialize the latest check report for API and visualization payloads."""

    run, results = latest_check_report(db, staged.id)
    if run is None:
        return {"run": None, "results": []}
    return {
        "run": _run_payload(run),
        "results": [_result_payload(result) for result in results],
    }


def _apply_decision_to_staged(
    db: Session,
    *,
    staged: models.StagedCommit,
    decision: MemoryCIDecision,
    outcomes: list[MemoryCheckOutcome],
) -> None:
    reason_codes = [outcome.reason_code for outcome in outcomes if not outcome.passed]
    warning_messages = [outcome.message for outcome in outcomes if not outcome.passed]
    staged.risk_reasons = _dedupe([*list(staged.risk_reasons or []), *reason_codes])
    staged.warnings_json = _dedupe([*list(staged.warnings_json or []), *warning_messages])
    if decision.decision == "reject":
        staged.status = "rejected"
        staged.review_required = False
        staged.reviewed_at = models.utc_now()
        crud.add_audit_event(
            db,
            event_type="staged_commit.rejected",
            entity_type="staged_commit",
            entity_id=0,
            entity_key=staged.id,
            payload={"staged_commit_id": staged.id, "reason": decision.reason_summary, "source": "memory_ci"},
        )
        return
    if decision.decision == "quarantine":
        staged.status = "quarantined"
        staged.review_required = True
        staged.quarantined_at = models.utc_now()
        staged.quarantine_reason_summary = decision.reason_summary
        staged.quarantine_release_status = "blocked"
        crud.add_audit_event(
            db,
            event_type="staged_commit.quarantined",
            entity_type="staged_commit",
            entity_id=0,
            entity_key=staged.id,
            payload={"staged_commit_id": staged.id, "reason": decision.reason_summary},
        )
        return
    if decision.decision == "require_review":
        staged.status = "review_required"
        staged.review_required = True
        crud.add_audit_event(
            db,
            event_type="staged_commit.review_required",
            entity_type="staged_commit",
            entity_id=0,
            entity_key=staged.id,
            payload={"staged_commit_id": staged.id, "reason": decision.reason_summary},
        )
        return
    staged.status = "checked"
    staged.review_required = False


def low_trust_source_check(context: CheckContext) -> MemoryCheckOutcome:
    """Flag weak provenance independently of the model's write action."""

    policies = _policies_for_claims(context)
    fail_threshold = max((policy.low_trust_fail_threshold for policy in policies), default=0.35)
    warn_threshold = max((policy.low_trust_warn_threshold for policy in policies), default=0.65)
    protected = any(policy.requires_review_on_main for policy in policies)
    if context.source_trust < fail_threshold:
        return _fail(
            "low_trust_source_check",
            "low_trust_source",
            "Source trust is below the quarantine threshold for a durable memory write.",
            {
                "source_trust_score": context.source_trust,
                "fail_threshold": fail_threshold,
                "protected_predicate": protected,
            },
        )
    if context.source_trust < warn_threshold:
        return _warn(
            "low_trust_source_check",
            "borderline_trust_source",
            "Source trust is low enough to require review before becoming active truth.",
            {
                "source_trust_score": context.source_trust,
                "warn_threshold": warn_threshold,
                "protected_predicate": protected,
            },
        )
    return _pass("low_trust_source_check", {"source_trust_score": context.source_trust})


def protected_predicate_review_check(context: CheckContext) -> MemoryCheckOutcome:
    """Require review for predicates with high user-impact or operational impact."""

    if context.branch.name != "main":
        return _pass("protected_predicate_review_check", {"branch_name": context.branch.name})
    protected_claims = [
        {"subject": claim.subject, "predicate": claim.predicate, "policy_class": context.policy_for(claim).name}
        for claim in context.claims
        if context.policy_for(claim).requires_review_on_main
    ]
    if protected_claims:
        return _warn(
            "protected_predicate_review_check",
            "protected_predicate_requires_review",
            "Protected predicates require review before automatic durable truth updates.",
            {"claims": protected_claims},
        )
    return _pass("protected_predicate_review_check", {})


def contradiction_spike_check(context: CheckContext) -> MemoryCheckOutcome:
    """Detect unsupported contradictions against the active branch belief state."""

    conflicts: list[dict[str, Any]] = []
    for claim in context.claims:
        belief = crud.get_belief_by_subject_predicate(
            context.db,
            subject=claim.subject,
            predicate=claim.predicate,
        )
        if belief is None:
            continue
        current = crud.get_current_versions(context.db, belief_id=belief.id, branch_id=context.branch.id)
        decision = classify_claim_against_current(
            context.db,
            claim=claim,
            current_versions=current,
            source_trust=context.source_trust,
            branch_is_hypothetical=context.branch.name != "main",
        )
        if decision.action != "unresolved_conflict":
            continue
        strongest = max((crud.belief_version_support_score(context.db, version) for version in current), default=0.0)
        proposed = belief_score(claim.confidence, context.source_trust)
        conflicts.append(
            {
                "subject": claim.subject,
                "predicate": claim.predicate,
                "object_value": claim.object_value,
                "proposed_score": proposed,
                "strongest_active_score": strongest,
                "contradiction_group": decision.contradiction_group,
            }
        )
    if conflicts:
        return _fail(
            "contradiction_spike_check",
            "contradicts_stronger_active_memory",
            "Proposed claim conflicts with stronger active memory without enough support.",
            {"conflicts": conflicts},
        )
    return _pass("contradiction_spike_check", {})


def temporal_overlap_check(context: CheckContext) -> MemoryCheckOutcome:
    """Detect overlapping windows that cannot be safely treated as coexistence."""

    overlaps: list[dict[str, Any]] = []
    for claim in context.claims:
        belief = crud.get_belief_by_subject_predicate(
            context.db,
            subject=claim.subject,
            predicate=claim.predicate,
        )
        if belief is None:
            continue
        for version in crud.get_current_versions(context.db, belief_id=belief.id, branch_id=context.branch.id):
            if version.normalized_object_value == claim.normalized_object_value:
                continue
            if not windows_overlap(version.valid_from, version.valid_to, claim.valid_from, claim.valid_to):
                continue
            if _newer_temporal_update(claim, version):
                continue
            overlaps.append(
                {
                    "belief_version_id": version.id,
                    "subject": claim.subject,
                    "predicate": claim.predicate,
                    "existing_object": version.object_value,
                    "proposed_object": claim.object_value,
                    "existing_valid_from": _iso(version.valid_from),
                    "existing_valid_to": _iso(version.valid_to),
                    "proposed_valid_from": _iso(claim.valid_from),
                    "proposed_valid_to": _iso(claim.valid_to),
                }
            )
    if overlaps:
        return _fail(
            "temporal_overlap_check",
            "unsafe_temporal_overlap",
            "Proposed active windows overlap an existing value without a clear supersession time.",
            {"overlaps": overlaps},
        )
    return _pass("temporal_overlap_check", {})


def branch_leakage_risk_check(context: CheckContext) -> MemoryCheckOutcome:
    """Block branch-only hypothetical information from entering main truth."""

    if context.branch.name != "main":
        return _pass("branch_leakage_risk_check", {"branch_name": context.branch.name})
    if _has_branch_only_cue(context.source_excerpt, context.claims):
        return _fail(
            "branch_leakage_risk_check",
            "branch_only_claim_on_main",
            "Future, temporary, or hypothetical memory should be written on a branch instead of main.",
            {"branch_name": context.branch.name},
        )
    return _pass("branch_leakage_risk_check", {"branch_name": context.branch.name})


def rollback_regression_check(context: CheckContext) -> MemoryCheckOutcome:
    """Detect attempts to reintroduce recently rolled-back content or provenance."""

    regressions: list[dict[str, Any]] = []
    rolled_back_versions = list(
        context.db.scalars(
            select(models.BeliefVersion)
            .where(models.BeliefVersion.status == "retracted")
            .order_by(models.BeliefVersion.id.desc())
            .limit(100)
        )
    )
    for claim in context.claims:
        for version in rolled_back_versions:
            belief = crud.get_belief(context.db, version.belief_id)
            if belief is None:
                continue
            if canonical_text(belief.subject) != canonical_text(claim.subject):
                continue
            if belief.predicate != claim.predicate:
                continue
            if version.normalized_object_value != claim.normalized_object_value:
                continue
            if "rolled_back_by_commit_id" in (version.metadata_json or {}) or "retracted_by_support_rollback_commit_id" in (version.metadata_json or {}):
                regressions.append(
                    {
                        "rolled_back_version_id": version.id,
                        "subject": claim.subject,
                        "predicate": claim.predicate,
                        "object_value": claim.object_value,
                    }
                )
    if _matches_rolled_back_source(context):
        regressions.append({"source_ref": context.source_ref, "source_excerpt": _shorten(context.source_excerpt)})
    if regressions:
        return _fail(
            "rollback_regression_check",
            "rollback_regression",
            "Proposed write reintroduces content or provenance that was previously rolled back.",
            {"regressions": regressions},
        )
    return _pass("rollback_regression_check", {})


def suspicious_same_object_corroboration_check(context: CheckContext) -> MemoryCheckOutcome:
    """Detect suspicious same-object support that may be poisoning or source hijacking."""

    suspicious: list[dict[str, Any]] = []
    for claim in context.claims:
        belief = crud.get_belief_by_subject_predicate(
            context.db,
            subject=claim.subject,
            predicate=claim.predicate,
        )
        if belief is None:
            continue
        same_object = [
            version
            for version in crud.get_current_versions(context.db, belief_id=belief.id, branch_id=context.branch.id)
            if version.normalized_object_value == claim.normalized_object_value
        ]
        for version in same_object:
            support_score = crud.belief_version_support_score(context.db, version)
            policy = context.policy_for(claim)
            if (
                context.source_trust >= policy.suspicious_corroboration_trust_jump
                and support_score < policy.suspicious_corroboration_max_existing_support
            ):
                suspicious.append(
                    {
                        "belief_version_id": version.id,
                        "proposed_source_trust": context.source_trust,
                        "current_support_score": support_score,
                        "policy_class": policy.name,
                    }
                )
    if suspicious:
        return _warn(
            "suspicious_same_object_corroboration_check",
            "suspicious_corroboration_spike",
            "Same-object corroboration has an unusually high trust jump and should be reviewed.",
            {"suspicious": suspicious},
        )
    return _pass("suspicious_same_object_corroboration_check", {})


def merge_conflict_policy_check(context: CheckContext) -> MemoryCheckOutcome:
    """Guard against staged writes that look like unsafe merge auto-resolution."""

    text = f"{context.source_excerpt} {context.staged.proposed_commit_message}".lower()
    if "merge" not in text:
        return _pass("merge_conflict_policy_check", {})
    conflicting_claims = []
    for claim in context.claims:
        belief = crud.get_belief_by_subject_predicate(
            context.db,
            subject=claim.subject,
            predicate=claim.predicate,
        )
        if belief is None:
            continue
        current = crud.get_current_versions(context.db, belief_id=belief.id, branch_id=context.branch.id)
        if any(
            version.normalized_object_value != claim.normalized_object_value
            and windows_overlap(version.valid_from, version.valid_to, claim.valid_from, claim.valid_to)
            for version in current
        ):
            conflicting_claims.append({"subject": claim.subject, "predicate": claim.predicate})
    if conflicting_claims:
        return _fail(
            "merge_conflict_policy_check",
            "unsafe_merge_auto_resolution",
            "Merge-like write would auto-resolve a conflict that should stay reviewable.",
            {"claims": conflicting_claims},
        )
    return _pass("merge_conflict_policy_check", {})


def duplicate_source_anomaly_check(context: CheckContext) -> MemoryCheckOutcome:
    """Detect repeated provenance attempting to inflate support."""

    duplicate_sources: list[dict[str, Any]] = []
    excerpt_key = canonical_text(context.source_excerpt)
    ref_key = canonical_text(context.source_ref or "")
    existing_sources = list(
        context.db.scalars(
            select(models.Source).order_by(models.Source.id.desc()).limit(200)
        )
    )
    for source in existing_sources:
        if excerpt_key and canonical_text(source.excerpt) == excerpt_key:
            duplicate_sources.append({"source_id": source.id, "match": "excerpt"})
        elif ref_key and source.source_ref and canonical_text(source.source_ref) == ref_key and excerpt_key:
            if canonical_text(source.excerpt) == excerpt_key:
                duplicate_sources.append({"source_id": source.id, "match": "source_ref_and_excerpt"})
    if duplicate_sources:
        return _warn(
            "duplicate_source_anomaly_check",
            "duplicate_source_anomaly",
            "Proposed write repeats an existing source/excerpt and may inflate support.",
            {"matches": duplicate_sources[:5]},
        )
    return _pass("duplicate_source_anomaly_check", {})


def support_gap_check(context: CheckContext) -> MemoryCheckOutcome:
    """Apply a minimal support-count policy for sensitive durable claims."""

    gaps: list[dict[str, Any]] = []
    if context.branch.name != "main":
        return _pass("support_gap_check", {"branch_name": context.branch.name})
    for claim in context.claims:
        policy = context.policy_for(claim)
        if policy.min_support_count_for_auto_apply <= 1 and not policy.requires_review_on_main:
            continue
        belief = crud.get_belief_by_subject_predicate(
            context.db,
            subject=claim.subject,
            predicate=claim.predicate,
        )
        existing_support_count = 0
        if belief is not None:
            existing_support_count = sum(
                len(crud.active_support_links(context.db, version.id))
                for version in crud.get_current_versions(context.db, belief_id=belief.id, branch_id=context.branch.id)
                if version.normalized_object_value == claim.normalized_object_value
            )
        effective_support_count = existing_support_count + 1
        if (
            context.source_trust < policy.min_auto_apply_trust
            and effective_support_count < policy.min_support_count_for_auto_apply
        ):
            gaps.append(
                {
                    "subject": claim.subject,
                    "predicate": claim.predicate,
                    "object_value": claim.object_value,
                    "source_trust_score": context.source_trust,
                    "active_same_object_support_count": existing_support_count,
                    "required_support_count": policy.min_support_count_for_auto_apply,
                    "min_auto_apply_trust": policy.min_auto_apply_trust,
                    "policy_class": policy.name,
                }
            )
    if gaps:
        return _warn(
            "support_gap_check",
            "support_gap_for_protected_claim",
            "Protected claim lacks enough corroborating support for automatic application.",
            {"gaps": gaps},
        )
    return _pass("support_gap_check", {})


DEFAULT_CHECK_REGISTRY = MemoryCheckRegistry(
    checks=(
        MemoryCheckSpec(
            name="low_trust_source_check",
            function=low_trust_source_check,
            description="Route weak provenance to review or quarantine.",
        ),
        MemoryCheckSpec(
            name="protected_predicate_review_check",
            function=protected_predicate_review_check,
            description="Apply per-predicate review policy classes.",
        ),
        MemoryCheckSpec(
            name="contradiction_spike_check",
            function=contradiction_spike_check,
            description="Detect unsupported contradictions against active truth.",
        ),
        MemoryCheckSpec(
            name="temporal_overlap_check",
            function=temporal_overlap_check,
            description="Detect unsafe overlapping temporal windows.",
        ),
        MemoryCheckSpec(
            name="branch_leakage_risk_check",
            function=branch_leakage_risk_check,
            description="Keep branch-only scenarios out of main truth.",
        ),
        MemoryCheckSpec(
            name="rollback_regression_check",
            function=rollback_regression_check,
            description="Block reintroduction of rolled-back content or sources.",
        ),
        MemoryCheckSpec(
            name="suspicious_same_object_corroboration_check",
            function=suspicious_same_object_corroboration_check,
            description="Review suspicious support spikes for the same object.",
        ),
        MemoryCheckSpec(
            name="merge_conflict_policy_check",
            function=merge_conflict_policy_check,
            description="Prevent unsafe merge-like auto-resolution.",
        ),
        MemoryCheckSpec(
            name="duplicate_source_anomaly_check",
            function=duplicate_source_anomaly_check,
            description="Detect duplicate provenance used to inflate support.",
        ),
        MemoryCheckSpec(
            name="support_gap_check",
            function=support_gap_check,
            description="Require enough evidence for configured predicate classes.",
        ),
    )
)

# Backwards-compatible public name for callers that imported the registry.
CHECK_REGISTRY = DEFAULT_CHECK_REGISTRY


def _claims_from_staged(staged: models.StagedCommit) -> list[NormalizedClaim]:
    claims: list[NormalizedClaim] = []
    for raw_claim in staged.claims_json or []:
        claim = ExtractedClaim.model_validate(raw_claim)
        claims.append(normalize_extracted_claim(claim))
    return claims


def _policies_for_claims(context: CheckContext) -> list[PredicatePolicyClass]:
    return [context.policy_for(claim) for claim in context.claims]


def _newer_temporal_update(claim: NormalizedClaim, version: models.BeliefVersion) -> bool:
    if claim.valid_from is None:
        return False
    if version.valid_from is None:
        return True
    return claim.valid_from > version.valid_from


def _has_branch_only_cue(source_excerpt: str, claims: list[NormalizedClaim]) -> bool:
    text = source_excerpt.lower()
    confirmed_current = re.search(r"\b(confirmed|already|currently|right now|now)\b", text)
    speculative = re.search(
        r"\b(would|could|might|hypothetical|counterfactual|what if|temporary|planning|plan|scenario|draft)\b",
        text,
    )
    bounded = re.search(
        r"\b(during|itinerary|trip|conference|conference week|travel|visit|vacation|relocation|remote work)\b",
        text,
    )
    if (speculative or bounded) and not confirmed_current:
        return True
    today = date.today()
    for claim in claims:
        if claim.valid_from is not None and claim.valid_from > today and not confirmed_current:
            return True
    return False


def _matches_rolled_back_source(context: CheckContext) -> bool:
    ref_key = canonical_text(context.source_ref or "")
    excerpt_key = canonical_text(context.source_excerpt)
    rolled_back_links = list(
        context.db.scalars(
            select(models.BeliefVersionSourceLink)
            .where(models.BeliefVersionSourceLink.status == "rolled_back")
            .order_by(models.BeliefVersionSourceLink.id.desc())
            .limit(100)
        )
    )
    for link in rolled_back_links:
        source = context.db.get(models.Source, link.source_id)
        if source is None:
            continue
        if ref_key and source.source_ref and canonical_text(source.source_ref) == ref_key:
            return True
        if excerpt_key and canonical_text(source.excerpt) == excerpt_key:
            return True
    return False


def _pass(check_name: str, payload: dict[str, Any]) -> MemoryCheckOutcome:
    return MemoryCheckOutcome(
        check_name=check_name,
        severity="info",
        passed=True,
        reason_code="passed",
        message="Check passed.",
        payload=payload,
    )


def _warn(check_name: str, reason_code: str, message: str, payload: dict[str, Any]) -> MemoryCheckOutcome:
    return MemoryCheckOutcome(
        check_name=check_name,
        severity="warn",
        passed=False,
        reason_code=reason_code,
        message=message,
        payload=payload,
    )


def _fail(check_name: str, reason_code: str, message: str, payload: dict[str, Any]) -> MemoryCheckOutcome:
    return MemoryCheckOutcome(
        check_name=check_name,
        severity="fail",
        passed=False,
        reason_code=reason_code,
        message=message,
        payload=payload,
    )


def _run_payload(run: models.MemoryCheckRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "staged_commit_id": run.staged_commit_id,
        "suite_version": run.suite_version,
        "overall_status": run.overall_status,
        "decision": run.decision,
        "score": run.score,
        "metadata_json": run.metadata_json,
        "created_at": _iso(run.created_at),
        "completed_at": _iso(run.completed_at),
    }


def _result_payload(result: models.MemoryCheckResult) -> dict[str, Any]:
    return {
        "id": result.id,
        "run_id": result.run_id,
        "check_name": result.check_name,
        "severity": result.severity,
        "passed": result.passed,
        "reason_code": result.reason_code,
        "message": result.message,
        "payload_json": result.payload_json,
        "created_at": _iso(result.created_at),
    }


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _shorten(value: str, limit: int = 140) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    return compact[:limit] + ("..." if len(compact) > limit else "")


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
