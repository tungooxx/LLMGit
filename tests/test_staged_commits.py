from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.commit_engine import ensure_main_branch
from app.db import Base
from app.schemas import ExtractedClaim, SourceCreate
from app.tools import approve_staged_commit, stage_belief_changes


def test_staged_commit_survives_new_session(tmp_path) -> None:
    db_path = tmp_path / "truthgit-staging.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)
    Base.metadata.create_all(bind=engine)

    first_session = SessionLocal()
    try:
        branch = ensure_main_branch(first_session)
        staged = stage_belief_changes(
            first_session,
            claims=[
                ExtractedClaim(
                    subject="Alice",
                    predicate="lives_in",
                    object="Seoul",
                    confidence=0.8,
                )
            ],
            branch_id=branch.id,
            source=SourceCreate(
                source_type="manual",
                source_ref="restart-test",
                excerpt="Alice lives in Seoul.",
                trust_score=0.9,
            ),
            proposed_commit_message="Restart-safe staged write",
        )
        staged_id = staged.id
        first_session.commit()
    finally:
        first_session.close()

    second_session = SessionLocal()
    try:
        result = approve_staged_commit(
            second_session,
            staged_commit_id=staged_id,
            reviewer="tester",
            notes="approved after session restart",
        )
        second_session.commit()
        assert result.commit.id is not None
        assert result.introduced_versions[0].object_value == "Seoul"
        assert second_session.get(type(staged), staged_id).status == "applied"
    finally:
        second_session.close()
