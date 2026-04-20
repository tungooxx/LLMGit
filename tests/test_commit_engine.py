from __future__ import annotations

from datetime import date

from app import crud
from app.commit_engine import apply_claims
from app.normalization import NormalizedClaim


def _claim(obj: str, valid_from=None) -> NormalizedClaim:
    return NormalizedClaim(
        subject="Alice",
        predicate="lives_in",
        object_value=obj,
        normalized_object_value=obj.lower(),
        confidence=0.85,
        valid_from=valid_from,
        valid_to=None,
    )


def test_supersession_preserves_lineage(db_session) -> None:
    main = crud.get_branch_by_name(db_session, "main")
    source = crud.create_source(
        db_session,
        source_type="manual",
        source_ref="test",
        excerpt="Alice lives in Seoul.",
        trust_score=0.8,
    )
    first = apply_claims(
        db_session,
        claims=[_claim("Seoul")],
        branch_id=main.id,
        source=source,
        message="Alice lives in Seoul",
    )
    second = apply_claims(
        db_session,
        claims=[_claim("Busan", date(2026, 3, 1))],
        branch_id=main.id,
        source=crud.create_source(
            db_session,
            source_type="manual",
            source_ref="test",
            excerpt="Alice moved to Busan in March 2026.",
            trust_score=0.8,
        ),
        message="Alice moved to Busan",
    )

    seoul = first.introduced_versions[0]
    busan = second.introduced_versions[0]
    assert seoul.status == "superseded"
    assert busan.status == "active"
    assert busan.supersedes_version_id == seoul.id

    belief = crud.get_belief_by_subject_predicate(db_session, subject="Alice", predicate="lives_in")
    current = crud.get_current_versions(db_session, belief_id=belief.id, branch_id=main.id)
    assert [version.object_value for version in current] == ["Busan"]
