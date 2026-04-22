"""Add belief support and opposition source links.

Revision ID: 0004_belief_support_sets
Revises: 0003_audit_entity_key
Create Date: 2026-04-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_belief_support_sets"
down_revision: str | None = "0003_audit_entity_key"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "belief_version_source_links",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("belief_version_id", sa.Integer(), sa.ForeignKey("belief_versions.id"), nullable=False),
        sa.Column("source_id", sa.Integer(), sa.ForeignKey("sources.id"), nullable=False),
        sa.Column("commit_id", sa.Integer(), sa.ForeignKey("commits.id"), nullable=True),
        sa.Column("relation_type", sa.String(length=30), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="active"),
        sa.Column("removed_by_commit_id", sa.Integer(), sa.ForeignKey("commits.id"), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index(
        "ix_belief_source_links_belief_version_id",
        "belief_version_source_links",
        ["belief_version_id"],
    )
    op.create_index("ix_belief_source_links_source_id", "belief_version_source_links", ["source_id"])
    op.create_index("ix_belief_source_links_commit_id", "belief_version_source_links", ["commit_id"])
    op.create_index(
        "ix_belief_source_links_removed_by_commit_id",
        "belief_version_source_links",
        ["removed_by_commit_id"],
    )
    op.create_index("ix_belief_source_links_relation_type", "belief_version_source_links", ["relation_type"])
    op.create_index("ix_belief_source_links_status", "belief_version_source_links", ["status"])
    op.create_index(
        "ix_belief_source_links_version_relation",
        "belief_version_source_links",
        ["belief_version_id", "relation_type", "status"],
    )
    op.create_index(
        "ix_belief_source_links_source_relation",
        "belief_version_source_links",
        ["source_id", "relation_type", "status"],
    )

    op.execute(
        """
        INSERT INTO belief_version_source_links (
            belief_version_id,
            source_id,
            commit_id,
            relation_type,
            status,
            reason,
            metadata_json,
            created_at
        )
        SELECT
            id,
            source_id,
            commit_id,
            'support',
            CASE WHEN status = 'retracted' THEN 'removed' ELSE 'active' END,
            'backfilled from belief_versions.source_id',
            '{}',
            created_at
        FROM belief_versions
        """
    )


def downgrade() -> None:
    op.drop_index("ix_belief_source_links_source_relation", table_name="belief_version_source_links")
    op.drop_index("ix_belief_source_links_version_relation", table_name="belief_version_source_links")
    op.drop_index("ix_belief_source_links_status", table_name="belief_version_source_links")
    op.drop_index("ix_belief_source_links_relation_type", table_name="belief_version_source_links")
    op.drop_index("ix_belief_source_links_removed_by_commit_id", table_name="belief_version_source_links")
    op.drop_index("ix_belief_source_links_commit_id", table_name="belief_version_source_links")
    op.drop_index("ix_belief_source_links_source_id", table_name="belief_version_source_links")
    op.drop_index("ix_belief_source_links_belief_version_id", table_name="belief_version_source_links")
    op.drop_table("belief_version_source_links")
