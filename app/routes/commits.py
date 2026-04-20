"""Commit and audit endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud, models
from app.commit_engine import rollback_commit
from app.db import get_db
from app.schemas import AuditEventRead, CommitRead, CommitResultRead, RollbackRequest

router = APIRouter(tags=["commits"])


@router.get("/commits/{commit_id}", response_model=CommitRead)
def get_commit(commit_id: int, db: Session = Depends(get_db)) -> object:
    """Return one commit."""

    commit = db.get(models.Commit, commit_id)
    if commit is None:
        raise HTTPException(status_code=404, detail="Commit not found")
    return commit


@router.post("/commits/{commit_id}/rollback", response_model=CommitResultRead)
def post_rollback_commit(
    commit_id: int,
    request: RollbackRequest,
    db: Session = Depends(get_db),
) -> object:
    """Rollback a commit."""

    try:
        result = rollback_commit(db, commit_id=commit_id, message=request.message)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    db.commit()
    return result


@router.get("/audit", response_model=list[AuditEventRead])
def get_audit(limit: int = 100, db: Session = Depends(get_db)) -> list[object]:
    """Return recent audit events."""

    return crud.list_audit_events(db, limit=limit)
