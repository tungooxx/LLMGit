"""Seed and demonstrate TruthGit research scenarios."""

from __future__ import annotations

from datetime import date

from rich.console import Console
from sqlalchemy.orm import Session

from app import crud
from app.commit_engine import apply_claims, create_branch, ensure_main_branch, merge_branch, rollback_commit
from app.db import Base, SessionLocal, engine
from app.models import Source
from app.normalization import NormalizedClaim

console = Console()


def _source(db: Session, excerpt: str, trust: float = 0.8) -> Source:
    return crud.create_source(
        db,
        source_type="manual",
        source_ref="demo",
        excerpt=excerpt,
        trust_score=trust,
    )


def _claim(subject: str, predicate: str, obj: str, *, valid_from: date | None = None) -> NormalizedClaim:
    return NormalizedClaim(
        subject=subject,
        predicate=predicate,
        object_value=obj,
        normalized_object_value=obj.lower(),
        confidence=0.85,
        valid_from=valid_from,
        valid_to=None,
    )


def run_demo(reset: bool = True) -> None:
    """Run all MVP scenarios against the configured SQLite database."""

    if reset:
        Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        main = ensure_main_branch(db)
        db.commit()

        console.rule("1. Supersession")
        seoul = apply_claims(
            db,
            claims=[_claim("Alice", "lives_in", "Seoul")],
            branch_id=main.id,
            source=_source(db, "Alice lives in Seoul."),
            message="Alice lives in Seoul",
            created_by="system",
        )
        busan = apply_claims(
            db,
            claims=[_claim("Alice", "lives_in", "Busan", valid_from=date(2026, 3, 1))],
            branch_id=main.id,
            source=_source(db, "Alice moved to Busan in March 2026."),
            message="Alice moved to Busan",
            created_by="system",
        )
        db.commit()
        console.print({"seoul_commit": seoul.commit.id, "busan_commit": busan.commit.id})

        console.rule("2. Branching")
        trip = create_branch(db, name="trip-plan", description="Hypothetical conference travel")
        trip_result = apply_claims(
            db,
            claims=[_claim("Alice", "stays_in", "Tokyo")],
            branch_id=trip.id,
            source=_source(db, "During the conference week, Alice will stay in Tokyo."),
            message="Conference week stay",
            created_by="system",
        )
        db.commit()
        console.print({"trip_branch": trip.id, "hypothetical_versions": [v.id for v in trip_result.introduced_versions]})

        console.rule("3. Merge")
        merge = merge_branch(db, source_branch_id=trip.id, target_branch_id=main.id, message="Merge trip plan")
        db.commit()
        console.print({"merge_commit": merge.commit.id, "merged_versions": [v.id for v in merge.introduced_versions]})

        console.rule("4. Rollback")
        bad = apply_claims(
            db,
            claims=[_claim("Alice", "lives_in", "Atlantis")],
            branch_id=main.id,
            source=_source(db, "Unverified rumor: Alice lives in Atlantis.", trust=0.2),
            message="Bad low-trust memory",
            created_by="system",
        )
        db.commit()
        rolled_back = rollback_commit(db, commit_id=bad.commit.id, message="Rollback bad low-trust memory")
        db.commit()
        console.print({"bad_commit": bad.commit.id, "rollback_commit": rolled_back.commit.id})

        console.rule("5. Conflict explanation")
        belief = crud.get_belief_by_subject_predicate(db, subject="Alice", predicate="lives_in")
        if belief:
            timeline = crud.list_belief_versions(db, belief.id)
            for version in timeline:
                console.print(
                    {
                        "version": version.id,
                        "object": version.object_value,
                        "status": version.status,
                        "supersedes": version.supersedes_version_id,
                        "commit": version.commit_id,
                    }
                )
    finally:
        db.close()


if __name__ == "__main__":
    run_demo()
