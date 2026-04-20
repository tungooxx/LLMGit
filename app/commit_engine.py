"""Version-control operations for TruthGit belief memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import crud, models
from app.conflict_engine import classify_claim_against_current
from app.normalization import NormalizedClaim


@dataclass
class CommitResult:
    """Result of a commit-style operation."""

    commit: models.Commit
    introduced_versions: list[models.BeliefVersion] = field(default_factory=list)
    restored_versions: list[models.BeliefVersion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def ensure_main_branch(db: Session) -> models.Branch:
    """Ensure the default main branch exists."""

    existing = crud.get_branch_by_name(db, "main")
    if existing is not None:
        return existing
    branch = models.Branch(name="main", description="Default TruthGit branch", status="active")
    db.add(branch)
    db.flush()
    crud.add_audit_event(
        db,
        event_type="branch.created",
        entity_type="branch",
        entity_id=branch.id,
        payload={"name": branch.name},
    )
    return branch


def create_branch(
    db: Session,
    *,
    name: str,
    description: str | None = None,
    parent_branch_id: int | None = None,
) -> models.Branch:
    """Create a branch for hypothetical or alternate belief histories."""

    if crud.get_branch_by_name(db, name) is not None:
        raise ValueError(f"Branch already exists: {name}")
    if parent_branch_id is None and name != "main":
        parent_branch_id = ensure_main_branch(db).id
    if parent_branch_id is not None and crud.get_branch(db, parent_branch_id) is None:
        raise ValueError(f"Parent branch does not exist: {parent_branch_id}")

    branch = models.Branch(
        name=name,
        description=description,
        parent_branch_id=parent_branch_id,
        status="active",
    )
    db.add(branch)
    db.flush()
    crud.add_audit_event(
        db,
        event_type="branch.created",
        entity_type="branch",
        entity_id=branch.id,
        payload={"name": name, "parent_branch_id": parent_branch_id},
    )
    return branch


def create_commit(
    db: Session,
    *,
    branch_id: int,
    operation_type: str,
    message: str,
    created_by: str,
    model_name: str | None = None,
) -> models.Commit:
    """Create a commit with the latest branch commit as parent."""

    branch = crud.get_branch(db, branch_id)
    if branch is None:
        raise ValueError(f"Branch does not exist: {branch_id}")
    parent = crud.get_latest_commit(db, branch_id)
    commit = models.Commit(
        branch_id=branch_id,
        parent_commit_id=parent.id if parent else None,
        operation_type=operation_type,
        message=message,
        created_by=created_by,
        model_name=model_name,
    )
    db.add(commit)
    db.flush()
    crud.add_audit_event(
        db,
        event_type="commit.created",
        entity_type="commit",
        entity_id=commit.id,
        payload={
            "branch_id": branch_id,
            "operation_type": operation_type,
            "message": message,
            "parent_commit_id": commit.parent_commit_id,
        },
    )
    return commit


def _branch_version_status(branch: models.Branch) -> str:
    return "active" if branch.name == "main" and branch.parent_branch_id is None else "hypothetical"


def _merge_metadata(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if extra:
        merged.update(extra)
    return merged


def _apply_claim_to_commit(
    db: Session,
    *,
    commit: models.Commit,
    branch: models.Branch,
    source: models.Source,
    claim: NormalizedClaim,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[models.BeliefVersion | None, list[str]]:
    """Apply one normalized claim under an already-created commit."""

    belief = crud.get_or_create_belief(db, subject=claim.subject, predicate=claim.predicate)
    current_versions = crud.get_current_versions(db, belief_id=belief.id, branch_id=branch.id)
    branch_is_hypothetical = _branch_version_status(branch) == "hypothetical"
    decision = classify_claim_against_current(
        db,
        claim=claim,
        current_versions=current_versions,
        source_trust=source.trust_score,
        branch_is_hypothetical=branch_is_hypothetical,
    )
    warnings = list(decision.warnings)
    if source.trust_score < 0.4:
        warnings.append("Low-trust source was used; review before relying on this belief.")
    if decision.action == "duplicate":
        return None, warnings

    supersedes_id = decision.supersede_version_ids[0] if decision.supersede_version_ids else None
    for version in current_versions:
        if version.id in decision.supersede_version_ids and version.branch_id == branch.id:
            version.status = "superseded"
            version.metadata_json = _merge_metadata(
                version.metadata_json,
                {"superseded_by_commit_id": commit.id},
            )

    if decision.contradiction_group:
        for version in current_versions:
            if version.branch_id == branch.id:
                version.contradiction_group = decision.contradiction_group

    version = models.BeliefVersion(
        belief_id=belief.id,
        commit_id=commit.id,
        branch_id=branch.id,
        object_value=claim.object_value,
        normalized_object_value=claim.normalized_object_value,
        confidence=claim.confidence,
        valid_from=claim.valid_from,
        valid_to=claim.valid_to,
        status=_branch_version_status(branch),
        source_id=source.id,
        supersedes_version_id=supersedes_id,
        contradiction_group=decision.contradiction_group,
        metadata_json=_merge_metadata(
            {
                "decision": decision.action,
                "is_negation": claim.is_negation,
                "source_quote": claim.source_quote,
                "notes": claim.notes,
            },
            extra_metadata,
        ),
    )
    db.add(version)
    db.flush()
    crud.add_audit_event(
        db,
        event_type="belief_version.created",
        entity_type="belief_version",
        entity_id=version.id,
        payload={
            "belief_id": belief.id,
            "commit_id": commit.id,
            "branch_id": branch.id,
            "decision": decision.action,
            "supersedes_version_id": supersedes_id,
        },
    )
    return version, warnings


def apply_claims(
    db: Session,
    *,
    claims: list[NormalizedClaim],
    branch_id: int,
    source: models.Source,
    message: str,
    created_by: str = "user",
    model_name: str | None = None,
    operation_type: str = "update",
) -> CommitResult:
    """Create a commit and apply validated normalized claims."""

    branch = crud.get_branch(db, branch_id)
    if branch is None:
        raise ValueError(f"Branch does not exist: {branch_id}")
    commit = create_commit(
        db,
        branch_id=branch.id,
        operation_type=operation_type,
        message=message,
        created_by=created_by,
        model_name=model_name,
    )
    result = CommitResult(commit=commit)
    for claim in claims:
        version, warnings = _apply_claim_to_commit(
            db,
            commit=commit,
            branch=branch,
            source=source,
            claim=claim,
        )
        result.warnings.extend(warnings)
        if version is not None:
            result.introduced_versions.append(version)
    db.flush()
    return result


def retract_version(
    db: Session,
    *,
    version_id: int,
    branch_id: int,
    message: str,
    created_by: str = "user",
) -> CommitResult:
    """Retract one belief version without deleting history."""

    version = db.get(models.BeliefVersion, version_id)
    if version is None:
        raise ValueError(f"Belief version does not exist: {version_id}")
    commit = create_commit(
        db,
        branch_id=branch_id,
        operation_type="retract",
        message=message,
        created_by=created_by,
    )
    version.status = "retracted"
    version.metadata_json = _merge_metadata(version.metadata_json, {"retracted_by_commit_id": commit.id})
    crud.add_audit_event(
        db,
        event_type="belief_version.retracted",
        entity_type="belief_version",
        entity_id=version.id,
        payload={"commit_id": commit.id},
    )
    return CommitResult(commit=commit)


def rollback_commit(
    db: Session,
    *,
    commit_id: int,
    message: str | None = None,
    created_by: str = "user",
) -> CommitResult:
    """Rollback a commit by retracting introduced versions and restoring superseded predecessors."""

    target = db.get(models.Commit, commit_id)
    if target is None:
        raise ValueError(f"Commit does not exist: {commit_id}")
    rollback = create_commit(
        db,
        branch_id=target.branch_id,
        operation_type="rollback",
        message=message or f"Rollback commit {commit_id}",
        created_by=created_by,
    )
    result = CommitResult(commit=rollback)
    introduced = list(
        db.scalars(
            select(models.BeliefVersion)
            .where(models.BeliefVersion.commit_id == target.id)
            .order_by(models.BeliefVersion.id)
        )
    )
    for version in introduced:
        if version.status != "retracted":
            version.status = "retracted"
            version.metadata_json = _merge_metadata(
                version.metadata_json,
                {"rolled_back_by_commit_id": rollback.id},
            )
            result.introduced_versions.append(version)
        if version.supersedes_version_id is None:
            continue
        previous = db.get(models.BeliefVersion, version.supersedes_version_id)
        if previous is None or previous.status != "superseded":
            continue
        db.flush()
        still_overridden = db.scalar(
            select(models.BeliefVersion.id).where(
                models.BeliefVersion.id != version.id,
                models.BeliefVersion.supersedes_version_id == previous.id,
                models.BeliefVersion.branch_id == previous.branch_id,
                models.BeliefVersion.status.in_(crud.CURRENT_STATUSES),
            )
        )
        if still_overridden is None:
            previous_branch = crud.get_branch(db, previous.branch_id)
            if previous_branch is not None:
                previous.status = _branch_version_status(previous_branch)
                previous.metadata_json = _merge_metadata(
                    previous.metadata_json,
                    {"restored_by_rollback_commit_id": rollback.id},
                )
                result.restored_versions.append(previous)
    crud.add_audit_event(
        db,
        event_type="commit.rolled_back",
        entity_type="commit",
        entity_id=target.id,
        payload={
            "rollback_commit_id": rollback.id,
            "retracted_version_ids": [version.id for version in result.introduced_versions],
            "restored_version_ids": [version.id for version in result.restored_versions],
        },
    )
    db.flush()
    return result


def list_branch_diffs(db: Session, *, branch_id: int) -> list[models.BeliefVersion]:
    """Return current versions introduced directly on a branch."""

    return list(
        db.scalars(
            select(models.BeliefVersion)
            .where(
                models.BeliefVersion.branch_id == branch_id,
                models.BeliefVersion.status.in_(crud.CURRENT_STATUSES),
            )
            .order_by(models.BeliefVersion.id)
        )
    )


def merge_branch(
    db: Session,
    *,
    source_branch_id: int,
    target_branch_id: int,
    message: str,
    created_by: str = "user",
) -> CommitResult:
    """Merge branch-local current belief versions into a target branch."""

    source_branch = crud.get_branch(db, source_branch_id)
    target_branch = crud.get_branch(db, target_branch_id)
    if source_branch is None:
        raise ValueError(f"Source branch does not exist: {source_branch_id}")
    if target_branch is None:
        raise ValueError(f"Target branch does not exist: {target_branch_id}")

    commit = create_commit(
        db,
        branch_id=target_branch.id,
        operation_type="merge",
        message=message,
        created_by=created_by,
    )
    result = CommitResult(commit=commit)
    for source_version in list_branch_diffs(db, branch_id=source_branch.id):
        source = db.get(models.Source, source_version.source_id)
        belief = db.get(models.Belief, source_version.belief_id)
        if source is None or belief is None:
            continue
        claim = NormalizedClaim(
            subject=belief.subject,
            predicate=belief.predicate,
            object_value=source_version.object_value,
            normalized_object_value=source_version.normalized_object_value,
            confidence=source_version.confidence,
            valid_from=source_version.valid_from,
            valid_to=source_version.valid_to,
            is_negation=bool(source_version.metadata_json.get("is_negation", False)),
            source_quote=source_version.metadata_json.get("source_quote"),
            notes=source_version.metadata_json.get("notes"),
        )
        version, warnings = _apply_claim_to_commit(
            db,
            commit=commit,
            branch=target_branch,
            source=source,
            claim=claim,
            extra_metadata={
                "merged_from_branch_id": source_branch.id,
                "merged_from_version_id": source_version.id,
            },
        )
        result.warnings.extend(warnings)
        if version is not None:
            result.introduced_versions.append(version)

    source_branch.status = "merged"
    crud.add_audit_event(
        db,
        event_type="branch.merged",
        entity_type="branch",
        entity_id=source_branch.id,
        payload={
            "target_branch_id": target_branch.id,
            "merge_commit_id": commit.id,
            "introduced_version_ids": [version.id for version in result.introduced_versions],
        },
    )
    db.flush()
    return result
