"""Add durable Memory CI and quarantine workflow.

Revision ID: 0005_memory_ci
Revises: 0004_belief_support_sets
Create Date: 2026-04-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_memory_ci"
down_revision: str | None = "0004_belief_support_sets"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("staged_commits", sa.Column("latest_check_run_id", sa.Integer(), nullable=True))
    op.add_column("staged_commits", sa.Column("checked_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("staged_commits", sa.Column("quarantined_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("staged_commits", sa.Column("quarantine_reason_summary", sa.Text(), nullable=True))
    op.add_column("staged_commits", sa.Column("quarantine_release_status", sa.String(length=30), nullable=True))
    op.add_column("staged_commits", sa.Column("quarantine_reviewer", sa.String(length=80), nullable=True))
    op.add_column("staged_commits", sa.Column("quarantine_notes", sa.Text(), nullable=True))
    op.create_index("ix_staged_commits_latest_check_run_id", "staged_commits", ["latest_check_run_id"])

    op.create_table(
        "memory_check_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("staged_commit_id", sa.String(length=36), sa.ForeignKey("staged_commits.id"), nullable=False),
        sa.Column("suite_version", sa.String(length=80), nullable=False),
        sa.Column("overall_status", sa.String(length=30), nullable=False),
        sa.Column("decision", sa.String(length=30), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_memory_check_runs_staged_commit_id", "memory_check_runs", ["staged_commit_id"])
    op.create_index("ix_memory_check_runs_overall_status", "memory_check_runs", ["overall_status"])
    op.create_index("ix_memory_check_runs_decision", "memory_check_runs", ["decision"])
    op.create_index(
        "ix_memory_check_runs_staged_status",
        "memory_check_runs",
        ["staged_commit_id", "overall_status"],
    )

    op.create_table(
        "memory_check_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("memory_check_runs.id"), nullable=False),
        sa.Column("check_name", sa.String(length=120), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("reason_code", sa.String(length=120), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_memory_check_results_run_id", "memory_check_results", ["run_id"])
    op.create_index("ix_memory_check_results_check_name", "memory_check_results", ["check_name"])
    op.create_index("ix_memory_check_results_severity", "memory_check_results", ["severity"])
    op.create_index("ix_memory_check_results_passed", "memory_check_results", ["passed"])
    op.create_index(
        "ix_memory_check_results_run_check",
        "memory_check_results",
        ["run_id", "check_name"],
    )
    op.create_index(
        "ix_memory_check_results_severity_passed",
        "memory_check_results",
        ["severity", "passed"],
    )


def downgrade() -> None:
    op.drop_index("ix_memory_check_results_severity_passed", table_name="memory_check_results")
    op.drop_index("ix_memory_check_results_run_check", table_name="memory_check_results")
    op.drop_index("ix_memory_check_results_passed", table_name="memory_check_results")
    op.drop_index("ix_memory_check_results_severity", table_name="memory_check_results")
    op.drop_index("ix_memory_check_results_check_name", table_name="memory_check_results")
    op.drop_index("ix_memory_check_results_run_id", table_name="memory_check_results")
    op.drop_table("memory_check_results")

    op.drop_index("ix_memory_check_runs_staged_status", table_name="memory_check_runs")
    op.drop_index("ix_memory_check_runs_decision", table_name="memory_check_runs")
    op.drop_index("ix_memory_check_runs_overall_status", table_name="memory_check_runs")
    op.drop_index("ix_memory_check_runs_staged_commit_id", table_name="memory_check_runs")
    op.drop_table("memory_check_runs")

    op.drop_index("ix_staged_commits_latest_check_run_id", table_name="staged_commits")
    op.drop_column("staged_commits", "quarantine_notes")
    op.drop_column("staged_commits", "quarantine_reviewer")
    op.drop_column("staged_commits", "quarantine_release_status")
    op.drop_column("staged_commits", "quarantine_reason_summary")
    op.drop_column("staged_commits", "quarantined_at")
    op.drop_column("staged_commits", "checked_at")
    op.drop_column("staged_commits", "latest_check_run_id")
