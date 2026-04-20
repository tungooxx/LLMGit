"""Repository-style database helpers for TruthGit."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session

from app import models
from app.normalization import canonical_key

CURRENT_STATUSES = {"active", "hypothetical"}


def add_audit_event(
    db: Session,
    *,
    event_type: str,
    entity_type: str,
    entity_id: int,
    payload: dict[str, Any] | None = None,
) -> models.AuditEvent:
    """Append an audit event."""

    event = models.AuditEvent(
        event_type=event_type,
        entity_type=entity_type,
        entity_id=entity_id,
        payload_json=payload or {},
    )
    db.add(event)
    db.flush()
    return event


def create_source(
    db: Session,
    *,
    source_type: str,
    source_ref: str | None,
    excerpt: str,
    trust_score: float,
) -> models.Source:
    """Create a provenance source."""

    source = models.Source(
        source_type=source_type,
        source_ref=source_ref,
        excerpt=excerpt,
        trust_score=max(0.0, min(1.0, trust_score)),
    )
    db.add(source)
    db.flush()
    add_audit_event(db, event_type="source.created", entity_type="source", entity_id=source.id)
    return source


def get_branch(db: Session, branch_id: int) -> models.Branch | None:
    """Fetch a branch by id."""

    return db.get(models.Branch, branch_id)


def get_branch_by_name(db: Session, name: str) -> models.Branch | None:
    """Fetch a branch by name."""

    return db.scalar(select(models.Branch).where(models.Branch.name == name))


def list_branches(db: Session) -> list[models.Branch]:
    """List branches by id."""

    return list(db.scalars(select(models.Branch).order_by(models.Branch.id)))


def get_belief(db: Session, belief_id: int) -> models.Belief | None:
    """Fetch a belief by id."""

    return db.get(models.Belief, belief_id)


def get_or_create_belief(db: Session, *, subject: str, predicate: str) -> models.Belief:
    """Fetch or create a stable belief identity."""

    key = canonical_key(subject, predicate)
    existing = db.scalar(select(models.Belief).where(models.Belief.canonical_key == key))
    if existing is not None:
        return existing
    belief = models.Belief(subject=subject.strip(), predicate=predicate.strip(), canonical_key=key)
    db.add(belief)
    db.flush()
    add_audit_event(
        db,
        event_type="belief.created",
        entity_type="belief",
        entity_id=belief.id,
        payload={"canonical_key": key},
    )
    return belief


def get_belief_by_subject_predicate(
    db: Session,
    *,
    subject: str,
    predicate: str,
) -> models.Belief | None:
    """Fetch a belief by normalized subject and predicate."""

    key = canonical_key(subject, predicate)
    return db.scalar(select(models.Belief).where(models.Belief.canonical_key == key))


def list_belief_versions(db: Session, belief_id: int) -> list[models.BeliefVersion]:
    """Return all versions for a belief in chronological order."""

    return list(
        db.scalars(
            select(models.BeliefVersion)
            .where(models.BeliefVersion.belief_id == belief_id)
            .order_by(models.BeliefVersion.created_at, models.BeliefVersion.id)
        )
    )


def get_local_current_versions(
    db: Session,
    *,
    belief_id: int,
    branch_id: int,
) -> list[models.BeliefVersion]:
    """Return current versions stored directly on a branch."""

    return list(
        db.scalars(
            select(models.BeliefVersion)
            .where(
                models.BeliefVersion.belief_id == belief_id,
                models.BeliefVersion.branch_id == branch_id,
                models.BeliefVersion.status.in_(CURRENT_STATUSES),
            )
            .order_by(models.BeliefVersion.created_at.desc(), models.BeliefVersion.id.desc())
        )
    )


def get_current_versions(
    db: Session,
    *,
    belief_id: int,
    branch_id: int,
    include_inherited: bool = True,
) -> list[models.BeliefVersion]:
    """Return branch-current versions, falling back through parents if requested."""

    local = get_local_current_versions(db, belief_id=belief_id, branch_id=branch_id)
    if local or not include_inherited:
        return local
    branch = get_branch(db, branch_id)
    if branch is None or branch.parent_branch_id is None:
        return []
    return get_current_versions(
        db,
        belief_id=belief_id,
        branch_id=branch.parent_branch_id,
        include_inherited=True,
    )


def get_latest_commit(db: Session, branch_id: int) -> models.Commit | None:
    """Return latest commit for a branch."""

    return db.scalar(
        select(models.Commit)
        .where(models.Commit.branch_id == branch_id)
        .order_by(models.Commit.created_at.desc(), models.Commit.id.desc())
    )


def search_beliefs(
    db: Session,
    *,
    query: str,
    branch_id: int | None = None,
    include_inactive: bool = False,
    limit: int = 25,
) -> list[models.BeliefVersion]:
    """Search belief versions by simple SQL predicates for MVP retrieval."""

    like = f"%{query.lower()}%"
    stmt: Select[tuple[models.BeliefVersion]] = (
        select(models.BeliefVersion)
        .join(models.Belief, models.Belief.id == models.BeliefVersion.belief_id)
        .where(
            or_(
                models.Belief.subject.ilike(like),
                models.Belief.predicate.ilike(like),
                models.BeliefVersion.object_value.ilike(like),
                models.BeliefVersion.normalized_object_value.ilike(like),
            )
        )
        .order_by(models.BeliefVersion.created_at.desc(), models.BeliefVersion.id.desc())
        .limit(limit)
    )
    if branch_id is not None:
        stmt = stmt.where(models.BeliefVersion.branch_id == branch_id)
    if not include_inactive:
        stmt = stmt.where(models.BeliefVersion.status.in_(CURRENT_STATUSES))
    return list(db.scalars(stmt))


def list_commits(
    db: Session,
    *,
    commit_id: int | None = None,
    belief_id: int | None = None,
    branch_id: int | None = None,
) -> list[models.Commit]:
    """List commit history filtered by commit, belief, or branch."""

    if commit_id is not None:
        commit = db.get(models.Commit, commit_id)
        return [commit] if commit else []

    stmt = select(models.Commit).order_by(models.Commit.created_at.desc(), models.Commit.id.desc())
    if branch_id is not None:
        stmt = stmt.where(models.Commit.branch_id == branch_id)
    if belief_id is not None:
        stmt = stmt.join(models.BeliefVersion, models.BeliefVersion.commit_id == models.Commit.id).where(
            models.BeliefVersion.belief_id == belief_id
        )
    return list(db.scalars(stmt).unique())


def list_audit_events(db: Session, limit: int = 100) -> list[models.AuditEvent]:
    """Return recent audit events."""

    return list(
        db.scalars(select(models.AuditEvent).order_by(models.AuditEvent.id.desc()).limit(limit))
    )


def source_trust(db: Session, source_id: int) -> float:
    """Fetch a source trust score."""

    source = db.get(models.Source, source_id)
    return source.trust_score if source else 0.5


def ids(items: Iterable[Any]) -> list[int]:
    """Return ids from SQLAlchemy model instances."""

    return [int(item.id) for item in items]
