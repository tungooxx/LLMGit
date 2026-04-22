from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import select

from app import models
from app.commit_engine import ensure_main_branch, rollback_commit
from app.memory_ci import latest_check_report, release_quarantine
from app.memory_ci_policy import default_memory_ci_policy
from app.schemas import ExtractedClaim, SourceCreate
from app.tools import apply_staged_commit, approve_staged_commit, stage_belief_changes
from experiments.governance_benchmark import export_governance_results, run_governance_benchmark


def test_pass_can_auto_apply_when_policy_allows(db_session) -> None:
    branch = ensure_main_branch(db_session)
    staged = _stage(
        db_session,
        branch_id=branch.id,
        subject="Alice",
        predicate="favorite_color",
        obj="green",
        source_ref="profile-color-pass",
        excerpt="Alice's public profile says her favorite color is green.",
        trust=0.86,
        confidence=0.82,
    )

    assert staged.status == "checked"
    assert staged.review_required is False
    result = apply_staged_commit(
        db_session,
        staged_commit_id=staged.id,
        commit_message="Auto apply low-risk CI pass",
    )

    assert staged.status == "applied"
    assert result.commit.id == staged.applied_commit_id
    assert result.introduced_versions[0].object_value == "green"


def test_warn_routes_to_review_required_not_quarantine(db_session) -> None:
    branch = ensure_main_branch(db_session)
    staged = _stage(
        db_session,
        branch_id=branch.id,
        subject="Alice",
        predicate="works_at",
        obj="Lab A",
        source_ref="personnel-review",
        excerpt="Alice works at Lab A.",
        trust=0.88,
        confidence=0.86,
    )

    run, results = latest_check_report(db_session, staged.id)
    assert run is not None
    assert run.overall_status == "warn"
    assert run.decision == "require_review"
    assert staged.status == "review_required"
    assert staged.quarantined_at is None
    assert any(result.reason_code == "protected_predicate_requires_review" for result in results)


def test_fail_enters_quarantine(db_session) -> None:
    branch = ensure_main_branch(db_session)
    _seed_active(
        db_session,
        branch_id=branch.id,
        subject="Alice",
        predicate="lives_in",
        obj="Busan",
        source_ref="verified-registry",
        excerpt="Verified registry says Alice lives in Busan.",
        trust=0.94,
        confidence=0.9,
    )
    staged = _stage(
        db_session,
        branch_id=branch.id,
        subject="Alice",
        predicate="lives_in",
        obj="Atlantis",
        source_ref="anonymous-forum",
        excerpt="Anonymous forum joke says Alice lives in Atlantis.",
        trust=0.2,
        confidence=0.7,
    )

    assert staged.status == "quarantined"
    assert staged.quarantined_at is not None
    assert staged.quarantine_reason_summary


def test_quarantined_item_cannot_become_active_accidentally(db_session) -> None:
    branch = ensure_main_branch(db_session)
    staged = _stage(
        db_session,
        branch_id=branch.id,
        subject="Alice",
        predicate="stays_in",
        obj="Tokyo",
        source_ref="main-branch-leak",
        excerpt="During the conference week, Alice will stay in Tokyo.",
        trust=0.8,
        confidence=0.78,
    )

    assert staged.status == "quarantined"
    with pytest.raises(ValueError, match="quarantined"):
        apply_staged_commit(
            db_session,
            staged_commit_id=staged.id,
            commit_message="Should not apply",
        )
    belief = models.Belief
    assert db_session.scalar(select(belief).where(belief.subject == "Alice", belief.predicate == "stays_in")) is None


def test_release_from_quarantine_emits_audit(db_session) -> None:
    branch = ensure_main_branch(db_session)
    staged = _stage(
        db_session,
        branch_id=branch.id,
        subject="Alice",
        predicate="stays_in",
        obj="Tokyo",
        source_ref="release-branch-leak",
        excerpt="During the conference week, Alice will stay in Tokyo.",
        trust=0.8,
        confidence=0.78,
    )

    release_quarantine(
        db_session,
        staged_commit_id=staged.id,
        reviewer="tester",
        notes="Reviewed and will keep this as manual review.",
    )

    assert staged.status == "review_required"
    event_types = [
        event.event_type
        for event in db_session.scalars(
            select(models.AuditEvent)
            .where(models.AuditEvent.entity_key == staged.id)
            .order_by(models.AuditEvent.id)
        )
    ]
    assert "staged_commit.quarantined" in event_types
    assert "staged_commit.quarantine_released" in event_types


def test_rollback_regression_check_quarantines_reintroduced_content(db_session) -> None:
    branch = ensure_main_branch(db_session)
    seeded = _seed_active(
        db_session,
        branch_id=branch.id,
        subject="Alice",
        predicate="favorite_snack",
        obj="mango",
        source_ref="bad-snack-import",
        excerpt="Bad import says Alice's favorite snack is mango.",
        trust=0.9,
        confidence=0.86,
    )
    rollback_commit(db_session, commit_id=seeded.applied_commit_id or 0, message="Rollback bad snack import")

    staged = _stage(
        db_session,
        branch_id=branch.id,
        subject="Alice",
        predicate="favorite_snack",
        obj="mango",
        source_ref="bad-snack-import",
        excerpt="Bad import says Alice's favorite snack is mango.",
        trust=0.91,
        confidence=0.86,
    )

    assert staged.status == "quarantined"
    assert "rollback_regression" in staged.risk_reasons


def test_branch_leakage_risk_check_quarantines_main_write(db_session) -> None:
    branch = ensure_main_branch(db_session)
    staged = _stage(
        db_session,
        branch_id=branch.id,
        subject="Alice",
        predicate="stays_in",
        obj="Tokyo",
        source_ref="branch-leakage",
        excerpt="During the conference week, Alice will stay in Tokyo.",
        trust=0.8,
        confidence=0.78,
    )

    assert staged.status == "quarantined"
    assert "branch_only_claim_on_main" in staged.risk_reasons


def test_temporal_overlap_check_quarantines_unsafe_overlap(db_session) -> None:
    branch = ensure_main_branch(db_session)
    _seed_active(
        db_session,
        branch_id=branch.id,
        subject="Alice",
        predicate="office_location",
        obj="Lab A",
        source_ref="office-calendar-a",
        excerpt="Alice's office location is Lab A for 2026.",
        trust=0.88,
        confidence=0.82,
        valid_from=date(2026, 1, 1),
        valid_to=date(2026, 12, 31),
    )

    staged = _stage(
        db_session,
        branch_id=branch.id,
        subject="Alice",
        predicate="office_location",
        obj="Lab B",
        source_ref="office-calendar-b",
        excerpt="Alice's office location is Lab B for 2026.",
        trust=0.88,
        confidence=0.82,
        valid_from=date(2026, 1, 1),
        valid_to=date(2026, 12, 31),
    )

    assert staged.status == "quarantined"
    assert "unsafe_temporal_overlap" in staged.risk_reasons


def test_duplicate_source_anomaly_routes_to_review(db_session) -> None:
    branch = ensure_main_branch(db_session)
    _seed_active(
        db_session,
        branch_id=branch.id,
        subject="Alice",
        predicate="favorite_city",
        obj="Busan",
        source_ref="favorite-city-profile",
        excerpt="Profile page says Alice's favorite city is Busan.",
        trust=0.82,
        confidence=0.8,
    )

    staged = _stage(
        db_session,
        branch_id=branch.id,
        subject="Alice",
        predicate="favorite_city",
        obj="Busan",
        source_ref="favorite-city-profile",
        excerpt="Profile page says Alice's favorite city is Busan.",
        trust=0.82,
        confidence=0.8,
    )

    assert staged.status == "review_required"
    assert "duplicate_source_anomaly" in staged.risk_reasons


def test_per_predicate_policy_classes_are_configured() -> None:
    policy = default_memory_ci_policy()

    assert policy.policy_for_predicate("lives_in").name == "identity_state"
    assert policy.policy_for_predicate("salary_amount").name == "financial"
    assert policy.policy_for_predicate("project_launch_date").name == "operational_deadline"
    assert policy.policy_for_predicate("favorite_color").name == "low_risk"


def test_governance_benchmark_exports_results(tmp_path: Path) -> None:
    results = run_governance_benchmark()
    export_governance_results(results, tmp_path)

    metrics = {row["metric"]: row["score"] for row in results["metric_summary"]}
    assert "quarantine_precision" in metrics
    assert "unsafe_commit_block_rate" in metrics
    assert (tmp_path / "governance_benchmark_results.json").exists()
    assert (tmp_path / "governance_metric_summary.csv").exists()


def test_memory_ci_and_quarantine_endpoints(client) -> None:
    staged_response = client.post("/chat", json={"message": "Alice lives in Seoul.", "auto_commit": False})
    staged_id = staged_response.json()["staged_commit_id"]

    checks = client.get(f"/staged/{staged_id}/checks")
    assert checks.status_code == 200
    assert checks.json()["run"]["overall_status"] == "warn"

    rerun = client.post(f"/staged/{staged_id}/run-checks")
    assert rerun.status_code == 200
    assert rerun.json()["run"]["decision"] == "require_review"

    quarantine_response = client.post("/chat", json={"message": "Alice lives in Atlantis.", "auto_commit": False})
    quarantined_id = quarantine_response.json()["staged_commit_id"]
    queued = client.get("/quarantine")
    assert queued.status_code == 200
    assert any(row["id"] == quarantined_id for row in queued.json())

    detail = client.get(f"/quarantine/{quarantined_id}")
    assert detail.status_code == 200
    assert detail.json()["run"]["decision"] == "quarantine"

    released = client.post(
        f"/quarantine/{quarantined_id}/release",
        json={"reviewer": "tester", "notes": "Keep this under manual review."},
    )
    assert released.status_code == 200
    assert released.json()["status"] == "review_required"


def test_memory_ci_has_no_obvious_benchmark_conditionals() -> None:
    source = (
        Path("app/memory_ci.py").read_text(encoding="utf-8")
        + Path("app/memory_ci_policy.py").read_text(encoding="utf-8")
    ).lower()
    forbidden = ["case_id", "question_id", "synthetic", "longmemeval", "benchmark"]
    assert not any(pattern in source for pattern in forbidden)


def _seed_active(
    db_session,
    *,
    branch_id: int,
    subject: str,
    predicate: str,
    obj: str,
    source_ref: str,
    excerpt: str,
    trust: float,
    confidence: float,
    valid_from: date | None = None,
    valid_to: date | None = None,
) -> models.StagedCommit:
    staged = _stage(
        db_session,
        branch_id=branch_id,
        subject=subject,
        predicate=predicate,
        obj=obj,
        source_ref=source_ref,
        excerpt=excerpt,
        trust=trust,
        confidence=confidence,
        valid_from=valid_from,
        valid_to=valid_to,
    )
    approve_staged_commit(
        db_session,
        staged_commit_id=staged.id,
        reviewer="tester",
        notes="Seed approved for Memory CI test.",
    )
    return staged


def _stage(
    db_session,
    *,
    branch_id: int,
    subject: str,
    predicate: str,
    obj: str,
    source_ref: str,
    excerpt: str,
    trust: float,
    confidence: float,
    valid_from: date | None = None,
    valid_to: date | None = None,
) -> models.StagedCommit:
    return stage_belief_changes(
        db_session,
        claims=[
            ExtractedClaim(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=confidence,
                valid_from=valid_from,
                valid_to=valid_to,
                source_quote=excerpt,
            )
        ],
        branch_id=branch_id,
        source=SourceCreate(
            source_type="document",
            source_ref=source_ref,
            excerpt=excerpt,
            trust_score=trust,
        ),
        proposed_commit_message=f"Memory CI test write: {source_ref}",
        created_by="test",
        model_name="test",
    )
