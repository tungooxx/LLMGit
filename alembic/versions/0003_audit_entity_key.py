"""Add string audit entity key.

Revision ID: 0003_audit_entity_key
Revises: 0002_staged_commits
Create Date: 2026-04-20
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_audit_entity_key"
down_revision: str | None = "0002_staged_commits"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("audit_events", sa.Column("entity_key", sa.String(length=120), nullable=True))
    op.create_index("ix_audit_events_entity_key", "audit_events", ["entity_type", "entity_key"])


def downgrade() -> None:
    op.drop_index("ix_audit_events_entity_key", table_name="audit_events")
    op.drop_column("audit_events", "entity_key")
