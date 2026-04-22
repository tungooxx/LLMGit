"""Generate a qualitative Memory CI/CD quarantine case study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import crud, models
from app.commit_engine import ensure_main_branch
from app.db import Base
from app.memory_ci import check_report_payload
from app.schemas import ExtractedClaim, SourceCreate
from app.tools import approve_staged_commit, reject_staged_commit, stage_belief_changes


def run_case_study() -> dict[str, Any]:
    """Run a small qualitative scenario through real TruthGit services."""

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)
    db = SessionLocal()
    try:
        branch = ensure_main_branch(db)
        db.flush()

        good = stage_belief_changes(
            db,
            claims=[
                ExtractedClaim(
                    subject="Alice",
                    predicate="favorite_color",
                    object="green",
                    confidence=0.86,
                    source_quote="Alice's public profile says her favorite color is green.",
                )
            ],
            branch_id=branch.id,
            source=SourceCreate(
                source_type="document",
                source_ref="public-profile-color",
                excerpt="Alice's public profile says her favorite color is green.",
                trust_score=0.86,
            ),
            proposed_commit_message="Case study: benign profile update",
            created_by="case-study",
        )
        good_result = approve_staged_commit(
            db,
            staged_commit_id=good.id,
            reviewer="memory-ci",
            notes="Auto-applied after CI pass.",
        )

        seed = stage_belief_changes(
            db,
            claims=[
                ExtractedClaim(
                    subject="Alice",
                    predicate="lives_in",
                    object="Busan",
                    confidence=0.9,
                    source_quote="Verified city registry says Alice lives in Busan.",
                )
            ],
            branch_id=branch.id,
            source=SourceCreate(
                source_type="api",
                source_ref="verified-city-registry",
                excerpt="Verified city registry says Alice lives in Busan.",
                trust_score=0.95,
            ),
            proposed_commit_message="Case study: verified residence seed",
            created_by="case-study",
        )
        seed_result = approve_staged_commit(
            db,
            staged_commit_id=seed.id,
            reviewer="reviewer",
            notes="Verified registry source.",
        )

        suspicious = stage_belief_changes(
            db,
            claims=[
                ExtractedClaim(
                    subject="Alice",
                    predicate="lives_in",
                    object="Atlantis",
                    confidence=0.72,
                    source_quote="Anonymous forum joke says Alice lives in Atlantis.",
                )
            ],
            branch_id=branch.id,
            source=SourceCreate(
                source_type="document",
                source_ref="anonymous-forum-joke",
                excerpt="Anonymous forum joke says Alice lives in Atlantis.",
                trust_score=0.2,
            ),
            proposed_commit_message="Case study: suspicious residence update",
            created_by="case-study",
        )
        reject_staged_commit(
            db,
            staged_commit_id=suspicious.id,
            reviewer="professor-demo-reviewer",
            notes="Rejected: low-trust source contradicts verified registry.",
        )
        db.flush()

        versions = list(db.scalars(select(models.BeliefVersion).order_by(models.BeliefVersion.id)))
        audit = list(db.scalars(select(models.AuditEvent).order_by(models.AuditEvent.id)))
        staged = [good, seed, suspicious]
        payload = {
            "title": "TruthGit Memory CI/CD quarantine case study",
            "steps": [
                {
                    "step": "good_update_passes",
                    "staged_commit_id": good.id,
                    "status": good.status,
                    "applied_commit_id": good_result.commit.id,
                    "checks": check_report_payload(db, good),
                },
                {
                    "step": "trusted_seed_reviewed",
                    "staged_commit_id": seed.id,
                    "status": seed.status,
                    "applied_commit_id": seed_result.commit.id,
                    "checks": check_report_payload(db, seed),
                },
                {
                    "step": "suspicious_update_quarantined_and_rejected",
                    "staged_commit_id": suspicious.id,
                    "status": suspicious.status,
                    "quarantine_reason_summary": suspicious.quarantine_reason_summary,
                    "checks": check_report_payload(db, suspicious),
                },
            ],
            "belief_versions": [_version_payload(db, version) for version in versions],
            "staged_commits": [_staged_payload(staged_commit) for staged_commit in staged],
            "audit_events": [
                {
                    "id": event.id,
                    "event_type": event.event_type,
                    "entity_type": event.entity_type,
                    "entity_key": event.entity_key,
                    "payload_json": event.payload_json,
                    "created_at": event.created_at.isoformat() if event.created_at else None,
                }
                for event in audit
            ],
        }
        return payload
    finally:
        db.close()
        engine.dispose()


def export_case_study(output_path: Path) -> None:
    """Write the qualitative case study JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(run_case_study(), indent=2, default=str), encoding="utf-8")


def _version_payload(db: Any, version: models.BeliefVersion) -> dict[str, Any]:
    belief = crud.get_belief(db, version.belief_id)
    payload = {
        "id": version.id,
        "subject": belief.subject if belief else None,
        "predicate": belief.predicate if belief else None,
        "object_value": version.object_value,
        "status": version.status,
        "source_id": version.source_id,
        "commit_id": version.commit_id,
    }
    payload.update(crud.support_graph_payload(db, version))
    return payload


def _staged_payload(staged: models.StagedCommit) -> dict[str, Any]:
    return {
        "id": staged.id,
        "status": staged.status,
        "review_required": staged.review_required,
        "risk_reasons": staged.risk_reasons,
        "warnings": staged.warnings_json,
        "quarantine_reason_summary": staged.quarantine_reason_summary,
        "quarantine_release_status": staged.quarantine_release_status,
        "reviewer": staged.reviewer,
        "review_notes": staged.review_notes,
        "applied_commit_id": staged.applied_commit_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/results/memory_ci_case_study.json")
    args = parser.parse_args()
    export_case_study(Path(args.output))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
