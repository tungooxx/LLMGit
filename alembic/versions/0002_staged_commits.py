"""Add durable staged commits.

Revision ID: 0002_staged_commits
Revises: 0001_initial_truthgit
Create Date: 2026-04-20
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_staged_commits"
down_revision: str | None = "0001_initial_truthgit"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "staged_commits",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("branch_id", sa.Integer(), sa.ForeignKey("branches.id"), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="pending"),
        sa.Column("claims_json", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("source_ref", sa.String(length=500), nullable=True),
        sa.Column("source_excerpt", sa.Text(), nullable=False),
        sa.Column("source_trust_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("proposed_commit_message", sa.Text(), nullable=False),
        sa.Column("created_by", sa.String(length=80), nullable=False, server_default="agent"),
        sa.Column("model_name", sa.String(length=120), nullable=True),
        sa.Column("review_required", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("risk_reasons", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("warnings_json", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("reviewer", sa.String(length=80), nullable=True),
        sa.Column("review_notes", sa.Text(), nullable=True),
        sa.Column("applied_commit_id", sa.Integer(), sa.ForeignKey("commits.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_staged_commits_applied_commit_id", "staged_commits", ["applied_commit_id"])
    op.create_index("ix_staged_commits_branch_id", "staged_commits", ["branch_id"])
    op.create_index("ix_staged_commits_status", "staged_commits", ["status"])


def downgrade() -> None:
    op.drop_index("ix_staged_commits_status", table_name="staged_commits")
    op.drop_index("ix_staged_commits_branch_id", table_name="staged_commits")
    op.drop_index("ix_staged_commits_applied_commit_id", table_name="staged_commits")
    op.drop_table("staged_commits")
