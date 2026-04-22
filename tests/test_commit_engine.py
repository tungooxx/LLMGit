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
    busan_graph = crud.support_graph_payload(db_session, busan)
    assert busan_graph["active_opposition_count"] == 0
    assert busan_graph["opposition_sources"][0]["status"] == "superseded"

    belief = crud.get_belief_by_subject_predicate(db_session, subject="Alice", predicate="lives_in")
    current = crud.get_current_versions(db_session, belief_id=belief.id, branch_id=main.id)
    assert [version.object_value for version in current] == ["Busan"]


def test_stronger_same_object_source_updates_current_justification(db_session) -> None:
    main = crud.get_branch_by_name(db_session, "main")
    early_source = crud.create_source(
        db_session,
        source_type="manual",
        source_ref="community-note",
        excerpt="Alice lives in Seoul.",
        trust_score=0.45,
    )
    official_source = crud.create_source(
        db_session,
        source_type="manual",
        source_ref="official-registry",
        excerpt="Official registry confirms Alice lives in Seoul.",
        trust_score=0.95,
    )

    first = apply_claims(
        db_session,
        claims=[_claim("Seoul")],
        branch_id=main.id,
        source=early_source,
        message="early note",
    )
    second = apply_claims(
        db_session,
        claims=[_claim("Seoul")],
        branch_id=main.id,
        source=official_source,
        message="official corroboration",
    )

    assert first.introduced_versions[0].status == "active"
    assert second.introduced_versions == []

    belief = crud.get_belief_by_subject_predicate(db_session, subject="Alice", predicate="lives_in")
    current = crud.get_current_versions(db_session, belief_id=belief.id, branch_id=main.id)
    assert len(current) == 1
    assert current[0].source_id == early_source.id
    support_graph = crud.support_graph_payload(db_session, current[0])
    assert support_graph["governing_source_id"] == early_source.id
    assert support_graph["active_support_count"] == 2
    assert support_graph["active_support_trust_total"] == 1.4
    assert support_graph["net_evidence_trust"] == 1.4
    assert {source["source_ref"] for source in support_graph["support_sources"]} == {
        "community-note",
        "official-registry",
    }


def test_rollback_of_corroborating_source_keeps_original_belief(db_session) -> None:
    main = crud.get_branch_by_name(db_session, "main")
    first_source = crud.create_source(
        db_session,
        source_type="manual",
        source_ref="first",
        excerpt="Alice lives in Seoul.",
        trust_score=0.8,
    )
    second_source = crud.create_source(
        db_session,
        source_type="manual",
        source_ref="second",
        excerpt="Directory also says Alice lives in Seoul.",
        trust_score=0.7,
    )
    first = apply_claims(
        db_session,
        claims=[_claim("Seoul")],
        branch_id=main.id,
        source=first_source,
        message="first support",
    )
    second = apply_claims(
        db_session,
        claims=[_claim("Seoul")],
        branch_id=main.id,
        source=second_source,
        message="second support",
    )

    from app.commit_engine import rollback_commit

    rollback_commit(db_session, commit_id=second.commit.id)

    belief = crud.get_belief_by_subject_predicate(db_session, subject="Alice", predicate="lives_in")
    current = crud.get_current_versions(db_session, belief_id=belief.id, branch_id=main.id)
    assert [version.id for version in current] == [first.introduced_versions[0].id]
    support_graph = crud.support_graph_payload(db_session, current[0])
    assert support_graph["active_support_count"] == 1
    assert support_graph["support_sources"][0]["source_ref"] == "first"
    assert any(source["status"] == "rolled_back" for source in support_graph["support_sources"])
