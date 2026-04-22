"""Repository-style database helpers for TruthGit."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session

from app import models
from app.normalization import canonical_key

CURRENT_STATUSES = {"active", "hypothetical"}
ACTIVE_SUPPORT_STATUSES = {"active"}


def add_audit_event(
    db: Session,
    *,
    event_type: str,
    entity_type: str,
    entity_id: int,
    entity_key: str | None = None,
    payload: dict[str, Any] | None = None,
) -> models.AuditEvent:
    """Append an audit event."""

    event = models.AuditEvent(
        event_type=event_type,
        entity_type=entity_type,
        entity_id=entity_id,
        entity_key=entity_key,
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


def list_audit_events(
    db: Session,
    limit: int = 100,
    *,
    entity_type: str | None = None,
    entity_id: int | None = None,
    entity_key: str | None = None,
) -> list[models.AuditEvent]:
    """Return recent audit events."""

    stmt = select(models.AuditEvent).order_by(models.AuditEvent.id.desc()).limit(limit)
    if entity_type is not None:
        stmt = stmt.where(models.AuditEvent.entity_type == entity_type)
    if entity_id is not None:
        stmt = stmt.where(models.AuditEvent.entity_id == entity_id)
    if entity_key is not None:
        stmt = stmt.where(models.AuditEvent.entity_key == entity_key)
    return list(db.scalars(stmt))


def source_trust(db: Session, source_id: int) -> float:
    """Fetch a source trust score."""

    source = db.get(models.Source, source_id)
    return source.trust_score if source else 0.5


def add_belief_source_link(
    db: Session,
    *,
    belief_version_id: int,
    source_id: int,
    relation_type: str,
    commit_id: int | None = None,
    status: str = "active",
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> models.BeliefVersionSourceLink:
    """Attach a support or opposition source to a belief version."""

    existing = db.scalar(
        select(models.BeliefVersionSourceLink).where(
            models.BeliefVersionSourceLink.belief_version_id == belief_version_id,
            models.BeliefVersionSourceLink.source_id == source_id,
            models.BeliefVersionSourceLink.relation_type == relation_type,
            models.BeliefVersionSourceLink.status == status,
        )
    )
    if existing is not None:
        return existing

    link = models.BeliefVersionSourceLink(
        belief_version_id=belief_version_id,
        source_id=source_id,
        commit_id=commit_id,
        relation_type=relation_type,
        status=status,
        reason=reason,
        metadata_json=metadata or {},
    )
    db.add(link)
    db.flush()
    add_audit_event(
        db,
        event_type=f"belief_source.{relation_type}_added",
        entity_type="belief_version",
        entity_id=belief_version_id,
        payload={
            "belief_version_id": belief_version_id,
            "source_id": source_id,
            "commit_id": commit_id,
            "relation_type": relation_type,
            "status": status,
            "reason": reason,
        },
    )
    return link


def list_belief_source_links(
    db: Session,
    *,
    belief_version_id: int,
    relation_type: str | None = None,
    statuses: set[str] | None = None,
) -> list[models.BeliefVersionSourceLink]:
    """List support/opposition source links for a belief version."""

    stmt = select(models.BeliefVersionSourceLink).where(
        models.BeliefVersionSourceLink.belief_version_id == belief_version_id
    )
    if relation_type is not None:
        stmt = stmt.where(models.BeliefVersionSourceLink.relation_type == relation_type)
    if statuses is not None:
        stmt = stmt.where(models.BeliefVersionSourceLink.status.in_(statuses))
    return list(db.scalars(stmt.order_by(models.BeliefVersionSourceLink.id)))


def active_support_links(db: Session, belief_version_id: int) -> list[models.BeliefVersionSourceLink]:
    """Return active support links for one belief version."""

    return list_belief_source_links(
        db,
        belief_version_id=belief_version_id,
        relation_type="support",
        statuses=ACTIVE_SUPPORT_STATUSES,
    )


def active_opposition_links(db: Session, belief_version_id: int) -> list[models.BeliefVersionSourceLink]:
    """Return active opposition links for one belief version."""

    return list_belief_source_links(
        db,
        belief_version_id=belief_version_id,
        relation_type="opposition",
        statuses=ACTIVE_SUPPORT_STATUSES,
    )


def belief_version_support_score(db: Session, version: models.BeliefVersion) -> float:
    """Compute current source-weighted support minus opposition for a belief version."""

    supports = active_support_links(db, version.id)
    oppositions = active_opposition_links(db, version.id)
    any_support_links = list_belief_source_links(db, belief_version_id=version.id, relation_type="support")
    if not any_support_links:
        support_score = source_trust(db, version.source_id)
    else:
        support_score = sum(source_trust(db, link.source_id) for link in supports)
    opposition_score = sum(source_trust(db, link.source_id) for link in oppositions)
    return round(max(0.0, support_score - opposition_score) * version.confidence, 4)


def source_link_payload(db: Session, link: models.BeliefVersionSourceLink) -> dict[str, Any]:
    """Serialize a source link with source details for API/tool responses."""

    source = db.get(models.Source, link.source_id)
    return {
        "id": link.id,
        "source_id": link.source_id,
        "source_type": source.source_type if source else None,
        "source_ref": source.source_ref if source else None,
        "excerpt": source.excerpt if source else None,
        "trust_score": source.trust_score if source else None,
        "relation_type": link.relation_type,
        "status": link.status,
        "commit_id": link.commit_id,
        "removed_by_commit_id": link.removed_by_commit_id,
        "reason": link.reason,
        "metadata_json": link.metadata_json,
        "created_at": link.created_at,
    }


def support_graph_payload(db: Session, version: models.BeliefVersion) -> dict[str, Any]:
    """Return support/opposition sources and current support score for a version."""

    support_links = list_belief_source_links(db, belief_version_id=version.id, relation_type="support")
    opposition_links = list_belief_source_links(db, belief_version_id=version.id, relation_type="opposition")
    active_supports = [link for link in support_links if link.status in ACTIVE_SUPPORT_STATUSES]
    active_oppositions = [link for link in opposition_links if link.status in ACTIVE_SUPPORT_STATUSES]
    support_trust_total = round(sum(source_trust(db, link.source_id) for link in active_supports), 4)
    opposition_trust_total = round(sum(source_trust(db, link.source_id) for link in active_oppositions), 4)
    return {
        "support_score": belief_version_support_score(db, version),
        "governing_source_id": version.source_id,
        "active_support_trust_total": support_trust_total,
        "active_opposition_trust_total": opposition_trust_total,
        "net_evidence_trust": round(max(0.0, support_trust_total - opposition_trust_total), 4),
        "active_support_count": len(active_supports),
        "active_opposition_count": len(active_oppositions),
        "support_sources": [source_link_payload(db, link) for link in support_links],
        "opposition_sources": [source_link_payload(db, link) for link in opposition_links],
    }


def ids(items: Iterable[Any]) -> list[int]:
    """Return ids from SQLAlchemy model instances."""

    return [int(item.id) for item in items]
