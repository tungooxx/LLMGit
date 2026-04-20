"""Deterministic belief conflict classification."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app import crud, models
from app.normalization import NormalizedClaim, windows_overlap


@dataclass(frozen=True)
class ConflictDecision:
    """Result of deterministic conflict analysis."""

    action: str
    supersede_version_ids: list[int]
    contradiction_group: str | None = None
    warnings: tuple[str, ...] = ()


def belief_score(confidence: float, trust_score: float) -> float:
    """Compute a deterministic source-weighted score."""

    return round(confidence * trust_score, 4)


def _version_score(db: Session, version: models.BeliefVersion) -> float:
    return belief_score(version.confidence, crud.source_trust(db, version.source_id))


def _newer_temporal_claim(
    claim: NormalizedClaim,
    existing: models.BeliefVersion,
) -> bool:
    if claim.valid_from is None:
        return False
    if existing.valid_from is None:
        return True
    return claim.valid_from >= existing.valid_from


def classify_claim_against_current(
    db: Session,
    *,
    claim: NormalizedClaim,
    current_versions: list[models.BeliefVersion],
    source_trust: float,
    branch_is_hypothetical: bool,
) -> ConflictDecision:
    """Classify how a normalized claim should interact with current truth."""

    if not current_versions:
        return ConflictDecision(action="add", supersede_version_ids=())

    same_object = [
        version
        for version in current_versions
        if version.normalized_object_value == claim.normalized_object_value
        and windows_overlap(version.valid_from, version.valid_to, claim.valid_from, claim.valid_to)
    ]
    if same_object:
        return ConflictDecision(
            action="duplicate",
            supersede_version_ids=(),
            warnings=("Existing active belief already has the same object value.",),
        )

    overlapping = [
        version
        for version in current_versions
        if windows_overlap(version.valid_from, version.valid_to, claim.valid_from, claim.valid_to)
    ]
    if not overlapping:
        return ConflictDecision(action="coexist", supersede_version_ids=())

    if branch_is_hypothetical:
        return ConflictDecision(
            action="branch_override",
            supersede_version_ids=[version.id for version in overlapping],
            warnings=("Hypothetical branch version overrides inherited truth only on this branch.",),
        )

    new_score = belief_score(claim.confidence, source_trust)
    supersede_ids: list[int] = []
    unresolved: list[models.BeliefVersion] = []

    for version in overlapping:
        old_score = _version_score(db, version)
        if _newer_temporal_claim(claim, version) or new_score >= old_score:
            supersede_ids.append(version.id)
        else:
            unresolved.append(version)

    if unresolved:
        group = f"conflict:{claim.subject.lower()}:{claim.predicate}"
        return ConflictDecision(
            action="unresolved_conflict",
            supersede_version_ids=supersede_ids,
            contradiction_group=group,
            warnings=(
                "Potential conflict: overlapping active belief has a stronger trust/confidence score.",
            ),
        )

    return ConflictDecision(action="supersede", supersede_version_ids=supersede_ids)
