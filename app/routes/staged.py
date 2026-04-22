"""Review endpoints for durable staged memory writes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models
from app.db import get_db
from app.memory_ci import latest_check_report, release_quarantine, run_memory_ci
from app.schemas import (
    CommitResultRead,
    MemoryCheckReport,
    QuarantineApproveRequest,
    QuarantineReleaseRequest,
    StagedCommitRead,
    StagedRejectRequest,
    StagedReviewRequest,
)
from app.tools import approve_staged_commit, reject_staged_commit

router = APIRouter(prefix="/staged", tags=["staged"])
quarantine_router = APIRouter(prefix="/quarantine", tags=["quarantine"])


@router.get("", response_model=list[StagedCommitRead])
def list_staged_commits(
    status: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[models.StagedCommit]:
    """List staged memory writes."""

    stmt = select(models.StagedCommit).order_by(models.StagedCommit.created_at.desc())
    if status is not None:
        stmt = stmt.where(models.StagedCommit.status == status)
    return list(db.scalars(stmt))


@router.get("/{staged_commit_id}", response_model=StagedCommitRead)
def get_staged_commit(
    staged_commit_id: str,
    db: Session = Depends(get_db),
) -> models.StagedCommit:
    """Return one staged memory write."""

    staged = db.get(models.StagedCommit, staged_commit_id)
    if staged is None:
        raise HTTPException(status_code=404, detail="Staged commit not found")
    return staged


@router.post("/{staged_commit_id}/run-checks", response_model=MemoryCheckReport)
def run_staged_checks(
    staged_commit_id: str,
    db: Session = Depends(get_db),
) -> object:
    """Run deterministic Memory CI for a staged write."""

    staged = db.get(models.StagedCommit, staged_commit_id)
    if staged is None:
        raise HTTPException(status_code=404, detail="Staged commit not found")
    try:
        run = run_memory_ci(db, staged_commit_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    results = list(
        db.scalars(
            select(models.MemoryCheckResult)
            .where(models.MemoryCheckResult.run_id == run.id)
            .order_by(models.MemoryCheckResult.id)
        )
    )
    db.commit()
    return {"staged_commit": staged, "run": run, "results": results}


@router.get("/{staged_commit_id}/checks", response_model=MemoryCheckReport)
def get_staged_checks(
    staged_commit_id: str,
    db: Session = Depends(get_db),
) -> object:
    """Return the latest Memory CI check report for one staged write."""

    staged = db.get(models.StagedCommit, staged_commit_id)
    if staged is None:
        raise HTTPException(status_code=404, detail="Staged commit not found")
    try:
        run, results = latest_check_report(db, staged_commit_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"staged_commit": staged, "run": run, "results": results}


@router.post("/{staged_commit_id}/approve", response_model=CommitResultRead)
def approve_staged(
    staged_commit_id: str,
    request: StagedReviewRequest,
    db: Session = Depends(get_db),
) -> object:
    """Approve a staged write and apply it through the commit engine."""

    try:
        result = approve_staged_commit(
            db,
            staged_commit_id=staged_commit_id,
            reviewer=request.reviewer,
            notes=request.notes,
            commit_message=request.commit_message,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    db.commit()
    return result


@router.post("/{staged_commit_id}/reject", response_model=StagedCommitRead)
def reject_staged(
    staged_commit_id: str,
    request: StagedRejectRequest,
    db: Session = Depends(get_db),
) -> models.StagedCommit:
    """Reject a staged write without applying belief-memory changes."""

    try:
        staged = reject_staged_commit(
            db,
            staged_commit_id=staged_commit_id,
            reviewer=request.reviewer,
            notes=request.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    db.commit()
    return staged


@quarantine_router.get("", response_model=list[StagedCommitRead])
def list_quarantine(db: Session = Depends(get_db)) -> list[models.StagedCommit]:
    """List quarantined staged memory writes."""

    return list(
        db.scalars(
            select(models.StagedCommit)
            .where(models.StagedCommit.status == "quarantined")
            .order_by(models.StagedCommit.quarantined_at.desc(), models.StagedCommit.created_at.desc())
        )
    )


@quarantine_router.get("/{staged_commit_id}", response_model=MemoryCheckReport)
def get_quarantined(
    staged_commit_id: str,
    db: Session = Depends(get_db),
) -> object:
    """Return a quarantined staged write and its latest CI report."""

    staged = db.get(models.StagedCommit, staged_commit_id)
    if staged is None:
        raise HTTPException(status_code=404, detail="Quarantined staged commit not found")
    if staged.status != "quarantined":
        raise HTTPException(status_code=400, detail=f"Staged commit is {staged.status}, not quarantined")
    run, results = latest_check_report(db, staged_commit_id)
    return {"staged_commit": staged, "run": run, "results": results}


@quarantine_router.post("/{staged_commit_id}/release", response_model=StagedCommitRead)
def release_quarantined(
    staged_commit_id: str,
    request: QuarantineReleaseRequest,
    db: Session = Depends(get_db),
) -> models.StagedCommit:
    """Release quarantine back into manual review without applying memory."""

    try:
        staged = release_quarantine(
            db,
            staged_commit_id=staged_commit_id,
            reviewer=request.reviewer,
            notes=request.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    db.commit()
    return staged


@quarantine_router.post("/{staged_commit_id}/reject", response_model=StagedCommitRead)
def reject_quarantined(
    staged_commit_id: str,
    request: StagedRejectRequest,
    db: Session = Depends(get_db),
) -> models.StagedCommit:
    """Reject a quarantined staged write."""

    try:
        staged = reject_staged_commit(
            db,
            staged_commit_id=staged_commit_id,
            reviewer=request.reviewer,
            notes=request.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    db.commit()
    return staged


@quarantine_router.post("/{staged_commit_id}/approve-and-apply", response_model=CommitResultRead)
def approve_and_apply_quarantined(
    staged_commit_id: str,
    request: QuarantineApproveRequest,
    db: Session = Depends(get_db),
) -> object:
    """Explicitly override quarantine and apply after reviewer notes."""

    try:
        result = approve_staged_commit(
            db,
            staged_commit_id=staged_commit_id,
            reviewer=request.reviewer,
            notes=request.notes,
            commit_message=request.commit_message,
            override_quarantine=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    db.commit()
    return result
