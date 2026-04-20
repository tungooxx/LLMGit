"""Initial TruthGit schema.

Revision ID: 0001_initial_truthgit
Revises:
Create Date: 2026-04-20
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial_truthgit"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("source_ref", sa.String(length=500), nullable=True),
        sa.Column("excerpt", sa.Text(), nullable=False),
        sa.Column("trust_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_sources_source_type", "sources", ["source_type"])

    op.create_table(
        "branches",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("parent_branch_id", sa.Integer(), sa.ForeignKey("branches.id"), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("name", name="uq_branches_name"),
    )
    op.create_index("ix_branches_parent_branch_id", "branches", ["parent_branch_id"])

    op.create_table(
        "commits",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("branch_id", sa.Integer(), sa.ForeignKey("branches.id"), nullable=False),
        sa.Column("parent_commit_id", sa.Integer(), sa.ForeignKey("commits.id"), nullable=True),
        sa.Column("operation_type", sa.String(length=30), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("created_by", sa.String(length=80), nullable=False),
        sa.Column("model_name", sa.String(length=120), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_commits_branch_id", "commits", ["branch_id"])
    op.create_index("ix_commits_parent_commit_id", "commits", ["parent_commit_id"])

    op.create_table(
        "beliefs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("subject", sa.String(length=300), nullable=False),
        sa.Column("predicate", sa.String(length=120), nullable=False),
        sa.Column("canonical_key", sa.String(length=500), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("canonical_key", name="uq_beliefs_canonical_key"),
    )
    op.create_index("ix_beliefs_canonical_key", "beliefs", ["canonical_key"])
    op.create_index("ix_beliefs_subject_predicate", "beliefs", ["subject", "predicate"])

    op.create_table(
        "belief_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("belief_id", sa.Integer(), sa.ForeignKey("beliefs.id"), nullable=False),
        sa.Column("commit_id", sa.Integer(), sa.ForeignKey("commits.id"), nullable=False),
        sa.Column("branch_id", sa.Integer(), sa.ForeignKey("branches.id"), nullable=False),
        sa.Column("object_value", sa.Text(), nullable=False),
        sa.Column("normalized_object_value", sa.String(length=500), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("valid_from", sa.Date(), nullable=True),
        sa.Column("valid_to", sa.Date(), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("source_id", sa.Integer(), sa.ForeignKey("sources.id"), nullable=False),
        sa.Column("supersedes_version_id", sa.Integer(), sa.ForeignKey("belief_versions.id"), nullable=True),
        sa.Column("contradiction_group", sa.String(length=120), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_belief_versions_belief_id", "belief_versions", ["belief_id"])
    op.create_index("ix_belief_versions_branch_id", "belief_versions", ["branch_id"])
    op.create_index("ix_belief_versions_commit_id", "belief_versions", ["commit_id"])
    op.create_index("ix_belief_versions_source_id", "belief_versions", ["source_id"])
    op.create_index("ix_belief_versions_status", "belief_versions", ["status"])

    op.create_table(
        "audit_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_type", sa.String(length=80), nullable=False),
        sa.Column("entity_type", sa.String(length=80), nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_audit_events_entity", "audit_events", ["entity_type", "entity_id"])
    op.create_index("ix_audit_events_event_type", "audit_events", ["event_type"])


def downgrade() -> None:
    op.drop_index("ix_audit_events_event_type", table_name="audit_events")
    op.drop_index("ix_audit_events_entity", table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_index("ix_belief_versions_status", table_name="belief_versions")
    op.drop_index("ix_belief_versions_source_id", table_name="belief_versions")
    op.drop_index("ix_belief_versions_commit_id", table_name="belief_versions")
    op.drop_index("ix_belief_versions_branch_id", table_name="belief_versions")
    op.drop_index("ix_belief_versions_belief_id", table_name="belief_versions")
    op.drop_table("belief_versions")
    op.drop_index("ix_beliefs_subject_predicate", table_name="beliefs")
    op.drop_index("ix_beliefs_canonical_key", table_name="beliefs")
    op.drop_table("beliefs")
    op.drop_index("ix_commits_parent_commit_id", table_name="commits")
    op.drop_index("ix_commits_branch_id", table_name="commits")
    op.drop_table("commits")
    op.drop_index("ix_branches_parent_branch_id", table_name="branches")
    op.drop_table("branches")
    op.drop_index("ix_sources_source_type", table_name="sources")
    op.drop_table("sources")
