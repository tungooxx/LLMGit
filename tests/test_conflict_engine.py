from __future__ import annotations

from app import crud
from app.commit_engine import apply_claims
from app.conflict_engine import classify_claim_against_current
from app.normalization import NormalizedClaim


def test_low_trust_conflict_does_not_supersede_high_trust_current(db_session) -> None:
    main = crud.get_branch_by_name(db_session, "main")
    source = crud.create_source(
        db_session,
        source_type="manual",
        source_ref="test",
        excerpt="Alice lives in Seoul.",
        trust_score=0.9,
    )
    apply_claims(
        db_session,
        claims=[
            NormalizedClaim(
                subject="Alice",
                predicate="lives_in",
                object_value="Seoul",
                normalized_object_value="seoul",
                confidence=0.9,
                valid_from=None,
                valid_to=None,
            )
        ],
        branch_id=main.id,
        source=source,
        message="seed",
    )
    belief = crud.get_belief_by_subject_predicate(db_session, subject="Alice", predicate="lives_in")
    current = crud.get_current_versions(db_session, belief_id=belief.id, branch_id=main.id)

    decision = classify_claim_against_current(
        db_session,
        claim=NormalizedClaim(
            subject="Alice",
            predicate="lives_in",
            object_value="Mars",
            normalized_object_value="mars",
            confidence=0.6,
            valid_from=None,
            valid_to=None,
        ),
        current_versions=current,
        source_trust=0.2,
        branch_is_hypothetical=False,
    )

    assert decision.action == "unresolved_conflict"
    assert decision.supersede_version_ids == []
    assert decision.contradiction_group is not None
