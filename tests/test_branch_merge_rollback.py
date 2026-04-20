from __future__ import annotations

from datetime import date

from app import crud
from app.commit_engine import apply_claims, create_branch, merge_branch, rollback_commit
from app.normalization import NormalizedClaim


def _source(db_session, excerpt: str, trust: float = 0.8):
    return crud.create_source(
        db_session,
        source_type="manual",
        source_ref="test",
        excerpt=excerpt,
        trust_score=trust,
    )


def _claim(predicate: str, obj: str, valid_from=None) -> NormalizedClaim:
    return NormalizedClaim(
        subject="Alice",
        predicate=predicate,
        object_value=obj,
        normalized_object_value=obj.lower(),
        confidence=0.85,
        valid_from=valid_from,
        valid_to=None,
    )


def test_branch_isolation_and_merge(db_session) -> None:
    main = crud.get_branch_by_name(db_session, "main")
    apply_claims(
        db_session,
        claims=[_claim("lives_in", "Seoul")],
        branch_id=main.id,
        source=_source(db_session, "Alice lives in Seoul."),
        message="main seed",
    )
    branch = create_branch(db_session, name="trip-plan", description="Conference week")
    branch_result = apply_claims(
        db_session,
        claims=[_claim("stays_in", "Tokyo")],
        branch_id=branch.id,
        source=_source(db_session, "During the conference week, Alice will stay in Tokyo."),
        message="trip plan",
    )

    assert branch_result.introduced_versions[0].status == "hypothetical"
    main_lives = crud.get_belief_by_subject_predicate(db_session, subject="Alice", predicate="lives_in")
    assert crud.get_current_versions(db_session, belief_id=main_lives.id, branch_id=main.id)[0].object_value == "Seoul"

    merge = merge_branch(
        db_session,
        source_branch_id=branch.id,
        target_branch_id=main.id,
        message="Merge trip-plan",
    )
    assert merge.commit.operation_type == "merge"
    assert merge.introduced_versions[0].status == "active"
    assert crud.get_branch(db_session, branch.id).status == "merged"


def test_rollback_restores_superseded_version(db_session) -> None:
    main = crud.get_branch_by_name(db_session, "main")
    first = apply_claims(
        db_session,
        claims=[_claim("lives_in", "Seoul")],
        branch_id=main.id,
        source=_source(db_session, "Alice lives in Seoul."),
        message="seed",
    )
    second = apply_claims(
        db_session,
        claims=[_claim("lives_in", "Busan", date(2026, 3, 1))],
        branch_id=main.id,
        source=_source(db_session, "Alice moved to Busan in March 2026."),
        message="move",
    )

    rolled_back = rollback_commit(db_session, commit_id=second.commit.id)

    assert second.introduced_versions[0].status == "retracted"
    assert first.introduced_versions[0].status == "active"
    assert rolled_back.restored_versions[0].id == first.introduced_versions[0].id
