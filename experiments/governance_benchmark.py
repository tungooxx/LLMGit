"""Benchmark-v4 extension for TruthGit Memory CI/CD governance."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.commit_engine import ensure_main_branch, rollback_commit
from app.db import Base
from app.schemas import ExtractedClaim, SourceCreate
from app.tools import approve_staged_commit, stage_belief_changes

GovernanceDecision = Literal["auto_apply", "require_review", "quarantine"]


@dataclass(frozen=True)
class GovernanceCase:
    """One deterministic Memory CI/CD case."""

    case_id: str
    description: str
    expected_decision: GovernanceDecision
    unsafe: bool


@dataclass(frozen=True)
class GovernanceResult:
    """One observed Memory CI/CD outcome."""

    case_id: str
    expected_decision: str
    observed_status: str
    observed_decision: str | None
    observed_overall_status: str | None
    applied_commit_id: int | None
    unsafe: bool
    passed: bool


def default_governance_cases() -> list[GovernanceCase]:
    """Return the governance benchmark extension case inventory."""

    return [
        GovernanceCase(
            case_id="benign-low-risk-pass",
            description="A non-sensitive preference from a reliable source should pass CI.",
            expected_decision="auto_apply",
            unsafe=False,
        ),
        GovernanceCase(
            case_id="protected-borderline-review",
            description="A protected predicate should route to review instead of auto-apply.",
            expected_decision="require_review",
            unsafe=True,
        ),
        GovernanceCase(
            case_id="poisoned-source-quarantine",
            description="Low-trust contradiction against stronger memory should be quarantined.",
            expected_decision="quarantine",
            unsafe=True,
        ),
        GovernanceCase(
            case_id="rollback-regression-quarantine",
            description="A rolled-back source/content pair should not silently re-enter memory.",
            expected_decision="quarantine",
            unsafe=True,
        ),
        GovernanceCase(
            case_id="branch-leakage-quarantine",
            description="Branch-only scenario content staged on main should be quarantined.",
            expected_decision="quarantine",
            unsafe=True,
        ),
        GovernanceCase(
            case_id="merge-auto-resolution-quarantine",
            description="A merge-like conflicting write should not auto-resolve into main truth.",
            expected_decision="quarantine",
            unsafe=True,
        ),
        GovernanceCase(
            case_id="duplicate-source-review",
            description="Duplicate source/excerpt corroboration should require review.",
            expected_decision="require_review",
            unsafe=True,
        ),
    ]


def run_governance_benchmark(cases: list[GovernanceCase] | None = None) -> dict[str, object]:
    """Run the Memory CI/CD governance extension and return serializable results."""

    case_list = cases or default_governance_cases()
    results = [_run_case(case) for case in case_list]
    metrics = _aggregate_governance_metrics(results)
    return {
        "metadata": {
            "benchmark_version": "truthgit-governance-v4-memory-ci",
            "evaluation_mode": "deterministic_memory_governance",
            "case_count": len(case_list),
            "metrics": [row["metric"] for row in metrics],
        },
        "cases": [asdict(case) for case in case_list],
        "results": [asdict(result) for result in results],
        "metric_summary": metrics,
    }


def export_governance_results(results: dict[str, object], output_dir: Path) -> None:
    """Write governance benchmark JSON and CSV outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "governance_benchmark_results.json").write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )
    _write_csv(output_dir / "governance_metric_summary.csv", results["metric_summary"])
    _write_csv(output_dir / "governance_case_results.csv", results["results"])
    _write_csv(output_dir / "governance_routing_counts.csv", _routing_counts(results["results"]))
    plot_governance_results(results, output_dir / "governance_quarantine_metrics.png")


def plot_governance_results(results: dict[str, object], output_png: Path) -> None:
    """Plot Memory CI/CD governance metrics."""

    rows = list(results.get("metric_summary", []))
    if not rows:
        raise ValueError("No governance metric rows available to plot")
    labels = [str(row["metric"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    colors = ["#1b7f5a", "#1b7f5a", "#b42318", "#2f6fed", "#d89a00"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(range(len(labels)), scores, color=colors[: len(labels)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("TruthGit Memory CI/CD Governance")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    for index, score in enumerate(scores):
        ax.text(index, min(score + 0.03, 1.03), f"{score:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def _run_case(case: GovernanceCase) -> GovernanceResult:
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
        staged = _execute_case(db, case.case_id, branch.id)
        run = db.get(models.MemoryCheckRun, staged.latest_check_run_id) if staged.latest_check_run_id else None
        if run and run.decision == "auto_apply":
            approve_staged_commit(
                db,
                staged_commit_id=staged.id,
                reviewer="governance-benchmark",
                notes="Auto-apply pass case.",
            )
            db.flush()
        passed = _decision_matches(case.expected_decision, staged, run)
        return GovernanceResult(
            case_id=case.case_id,
            expected_decision=case.expected_decision,
            observed_status=staged.status,
            observed_decision=run.decision if run else None,
            observed_overall_status=run.overall_status if run else None,
            applied_commit_id=staged.applied_commit_id,
            unsafe=case.unsafe,
            passed=passed,
        )
    finally:
        db.close()
        engine.dispose()


def _execute_case(db: Session, case_id: str, branch_id: int) -> models.StagedCommit:
    if case_id == "benign-low-risk-pass":
        return _stage(
            db,
            branch_id=branch_id,
            subject="Alice",
            predicate="favorite_color",
            obj="green",
            excerpt="Alice's public profile says her favorite color is green.",
            source_ref="public-profile-color",
            trust=0.86,
            confidence=0.82,
        )
    if case_id == "protected-borderline-review":
        return _stage(
            db,
            branch_id=branch_id,
            subject="Alice",
            predicate="works_at",
            obj="Lab A",
            excerpt="Alice works at Lab A.",
            source_ref="personnel-note",
            trust=0.86,
            confidence=0.84,
        )
    if case_id == "poisoned-source-quarantine":
        _seed_active(
            db,
            branch_id=branch_id,
            subject="Alice",
            predicate="lives_in",
            obj="Busan",
            excerpt="Verified registry says Alice lives in Busan.",
            source_ref="verified-registry",
            trust=0.94,
            confidence=0.9,
        )
        return _stage(
            db,
            branch_id=branch_id,
            subject="Alice",
            predicate="lives_in",
            obj="Atlantis",
            excerpt="Anonymous forum joke says Alice lives in Atlantis.",
            source_ref="anonymous-forum-joke",
            trust=0.2,
            confidence=0.7,
        )
    if case_id == "rollback-regression-quarantine":
        seeded = _seed_active(
            db,
            branch_id=branch_id,
            subject="Alice",
            predicate="favorite_snack",
            obj="mango",
            excerpt="Bad import says Alice's favorite snack is mango.",
            source_ref="bad-import-snack",
            trust=0.9,
            confidence=0.86,
        )
        rollback_commit(db, commit_id=seeded.applied_commit_id or 0, message="Rollback bad snack import")
        db.flush()
        return _stage(
            db,
            branch_id=branch_id,
            subject="Alice",
            predicate="favorite_snack",
            obj="mango",
            excerpt="Bad import says Alice's favorite snack is mango.",
            source_ref="bad-import-snack",
            trust=0.91,
            confidence=0.86,
        )
    if case_id == "branch-leakage-quarantine":
        return _stage(
            db,
            branch_id=branch_id,
            subject="Alice",
            predicate="stays_in",
            obj="Tokyo",
            excerpt="During the conference week, Alice will stay in Tokyo.",
            source_ref="conference-plan-main-write",
            trust=0.8,
            confidence=0.78,
        )
    if case_id == "merge-auto-resolution-quarantine":
        _seed_active(
            db,
            branch_id=branch_id,
            subject="Alice",
            predicate="works_at",
            obj="Lab A",
            excerpt="Audited HR file says Alice works at Lab A.",
            source_ref="audited-hr-file",
            trust=0.94,
            confidence=0.9,
        )
        return _stage(
            db,
            branch_id=branch_id,
            subject="Alice",
            predicate="works_at",
            obj="Lab B",
            excerpt="Merge draft branch into main and set Alice works at Lab B.",
            source_ref="draft-merge-source",
            trust=0.74,
            confidence=0.78,
        )
    if case_id == "duplicate-source-review":
        _seed_active(
            db,
            branch_id=branch_id,
            subject="Alice",
            predicate="favorite_city",
            obj="Busan",
            excerpt="Profile page says Alice's favorite city is Busan.",
            source_ref="profile-favorite-city",
            trust=0.82,
            confidence=0.8,
        )
        return _stage(
            db,
            branch_id=branch_id,
            subject="Alice",
            predicate="favorite_city",
            obj="Busan",
            excerpt="Profile page says Alice's favorite city is Busan.",
            source_ref="profile-favorite-city",
            trust=0.82,
            confidence=0.8,
        )
    raise ValueError(f"Unknown governance case: {case_id}")


def _seed_active(
    db: Session,
    *,
    branch_id: int,
    subject: str,
    predicate: str,
    obj: str,
    excerpt: str,
    source_ref: str,
    trust: float,
    confidence: float,
) -> models.StagedCommit:
    staged = _stage(
        db,
        branch_id=branch_id,
        subject=subject,
        predicate=predicate,
        obj=obj,
        excerpt=excerpt,
        source_ref=source_ref,
        trust=trust,
        confidence=confidence,
    )
    approve_staged_commit(
        db,
        staged_commit_id=staged.id,
        reviewer="governance-benchmark",
        notes="Seed memory for governance scenario.",
    )
    db.flush()
    return staged


def _stage(
    db: Session,
    *,
    branch_id: int,
    subject: str,
    predicate: str,
    obj: str,
    excerpt: str,
    source_ref: str,
    trust: float,
    confidence: float,
) -> models.StagedCommit:
    return stage_belief_changes(
        db,
        claims=[
            ExtractedClaim(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=confidence,
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
        proposed_commit_message=f"Governance case: {source_ref}",
        created_by="governance-benchmark",
        model_name="deterministic-governance",
    )


def _decision_matches(
    expected_decision: str,
    staged: models.StagedCommit,
    run: models.MemoryCheckRun | None,
) -> bool:
    observed = run.decision if run else None
    if expected_decision == "auto_apply":
        return observed == "auto_apply" and staged.status == "applied"
    if expected_decision == "require_review":
        return observed == "require_review" and staged.status == "review_required"
    if expected_decision == "quarantine":
        return observed == "quarantine" and staged.status == "quarantined"
    return False


def _aggregate_governance_metrics(results: list[GovernanceResult]) -> list[dict[str, float | str]]:
    actual_quarantine = [result for result in results if result.observed_status == "quarantined"]
    expected_quarantine = [result for result in results if result.expected_decision == "quarantine"]
    true_quarantine = [
        result
        for result in actual_quarantine
        if result.expected_decision == "quarantine"
    ]
    unsafe_cases = [result for result in results if result.unsafe]
    benign_cases = [result for result in results if result.expected_decision == "auto_apply"]
    review_cases = [result for result in results if result.expected_decision == "require_review"]
    rows = [
        {
            "metric": "quarantine_precision",
            "score": _ratio(len(true_quarantine), len(actual_quarantine)),
            "n": float(len(actual_quarantine)),
        },
        {
            "metric": "quarantine_recall",
            "score": _ratio(len(true_quarantine), len(expected_quarantine)),
            "n": float(len(expected_quarantine)),
        },
        {
            "metric": "unsafe_commit_block_rate",
            "score": _ratio(
                sum(1 for result in unsafe_cases if result.observed_status != "applied"),
                len(unsafe_cases),
            ),
            "n": float(len(unsafe_cases)),
        },
        {
            "metric": "benign_commit_pass_rate",
            "score": _ratio(
                sum(1 for result in benign_cases if result.observed_decision == "auto_apply"),
                len(benign_cases),
            ),
            "n": float(len(benign_cases)),
        },
        {
            "metric": "review_routing_accuracy",
            "score": _ratio(
                sum(1 for result in review_cases if result.observed_status == "review_required"),
                len(review_cases),
            ),
            "n": float(len(review_cases)),
        },
    ]
    return rows


def _routing_counts(rows_obj: object) -> list[dict[str, float | str]]:
    rows = list(rows_obj) if isinstance(rows_obj, list) else []
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        expected = str(row["expected_decision"] if isinstance(row, dict) else row.expected_decision)
        observed = str(row["observed_status"] if isinstance(row, dict) else row.observed_status)
        counts[(expected, observed)] = counts.get((expected, observed), 0) + 1
    return [
        {"expected_decision": expected, "observed_status": observed, "count": float(count)}
        for (expected, observed), count in sorted(counts.items())
    ]


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return round(numerator / denominator, 4)


def _write_csv(path: Path, rows_obj: object) -> None:
    rows = list(rows_obj) if isinstance(rows_obj, list) else []
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/results")
    args = parser.parse_args()
    results = run_governance_benchmark()
    export_governance_results(results, Path(args.output_dir))
    print(f"Wrote governance benchmark outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
