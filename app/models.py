"""SQLAlchemy data model for version-controlled belief memory."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Index, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class Source(Base):
    """Provenance record for a belief version."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_type: Mapped[str] = mapped_column(String(50), index=True)
    source_ref: Mapped[str | None] = mapped_column(String(500), nullable=True)
    excerpt: Mapped[str] = mapped_column(Text)
    trust_score: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Branch(Base):
    """Version-control branch for belief versions."""

    __tablename__ = "branches"
    __table_args__ = (UniqueConstraint("name", name="uq_branches_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_branch_id: Mapped[int | None] = mapped_column(ForeignKey("branches.id"), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(30), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Commit(Base):
    """Audit-friendly commit describing a memory operation."""

    __tablename__ = "commits"

    id: Mapped[int] = mapped_column(primary_key=True)
    branch_id: Mapped[int] = mapped_column(ForeignKey("branches.id"), index=True)
    parent_commit_id: Mapped[int | None] = mapped_column(ForeignKey("commits.id"), nullable=True, index=True)
    operation_type: Mapped[str] = mapped_column(String(30))
    message: Mapped[str] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(80))
    model_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Belief(Base):
    """Stable identity for a belief topic such as subject+predicate."""

    __tablename__ = "beliefs"
    __table_args__ = (
        UniqueConstraint("canonical_key", name="uq_beliefs_canonical_key"),
        Index("ix_beliefs_subject_predicate", "subject", "predicate"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    subject: Mapped[str] = mapped_column(String(300))
    predicate: Mapped[str] = mapped_column(String(120))
    canonical_key: Mapped[str] = mapped_column(String(500), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class BeliefVersion(Base):
    """Append-only belief content plus lifecycle status."""

    __tablename__ = "belief_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    belief_id: Mapped[int] = mapped_column(ForeignKey("beliefs.id"), index=True)
    commit_id: Mapped[int] = mapped_column(ForeignKey("commits.id"), index=True)
    branch_id: Mapped[int] = mapped_column(ForeignKey("branches.id"), index=True)
    object_value: Mapped[str] = mapped_column(Text)
    normalized_object_value: Mapped[str] = mapped_column(String(500))
    confidence: Mapped[float] = mapped_column(Float)
    valid_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    valid_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), index=True)
    supersedes_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("belief_versions.id"), nullable=True
    )
    contradiction_group: Mapped[str | None] = mapped_column(String(120), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class StagedCommit(Base):
    """Durable review queue entry for proposed belief-memory writes."""

    __tablename__ = "staged_commits"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    branch_id: Mapped[int] = mapped_column(ForeignKey("branches.id"), index=True)
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    claims_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    source_type: Mapped[str] = mapped_column(String(50))
    source_ref: Mapped[str | None] = mapped_column(String(500), nullable=True)
    source_excerpt: Mapped[str] = mapped_column(Text)
    source_trust_score: Mapped[float] = mapped_column(Float, default=0.5)
    proposed_commit_message: Mapped[str] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(80), default="agent")
    model_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    review_required: Mapped[bool] = mapped_column(Boolean, default=False)
    risk_reasons: Mapped[list[str]] = mapped_column(JSON, default=list)
    warnings_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    reviewer: Mapped[str | None] = mapped_column(String(80), nullable=True)
    review_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    applied_commit_id: Mapped[int | None] = mapped_column(ForeignKey("commits.id"), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditEvent(Base):
    """Append-only event log for TruthGit operations."""

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_entity", "entity_type", "entity_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    entity_type: Mapped[str] = mapped_column(String(80))
    entity_id: Mapped[int] = mapped_column()
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
