"""Review endpoints for durable staged memory writes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models
from app.db import get_db
from app.schemas import CommitResultRead, StagedCommitRead, StagedRejectRequest, StagedReviewRequest
from app.tools import approve_staged_commit, reject_staged_commit

router = APIRouter(prefix="/staged", tags=["staged"])


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
