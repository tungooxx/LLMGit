"""Branch endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud
from app.commit_engine import create_branch, ensure_main_branch, merge_branch
from app.db import get_db
from app.schemas import BranchCreate, BranchRead, CommitResultRead, MergeRequest

router = APIRouter(prefix="/branches", tags=["branches"])


@router.get("", response_model=list[BranchRead])
def list_branches(db: Session = Depends(get_db)) -> list[object]:
    """List branches."""

    ensure_main_branch(db)
    db.commit()
    return crud.list_branches(db)


@router.post("", response_model=BranchRead)
def post_branch(request: BranchCreate, db: Session = Depends(get_db)) -> object:
    """Create a branch."""

    try:
        branch = create_branch(
            db,
            name=request.name,
            description=request.description,
            parent_branch_id=request.parent_branch_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    db.commit()
    return branch


@router.post("/{branch_id}/merge", response_model=CommitResultRead)
def post_merge_branch(
    branch_id: int,
    request: MergeRequest,
    db: Session = Depends(get_db),
) -> object:
    """Merge a branch into a target branch."""

    target = (
        crud.get_branch(db, request.target_branch_id)
        if request.target_branch_id is not None
        else crud.get_branch_by_name(db, "main")
    )
    if target is None:
        raise HTTPException(status_code=404, detail="Target branch not found")
    try:
        result = merge_branch(
            db,
            source_branch_id=branch_id,
            target_branch_id=target.id,
            message=request.message,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    db.commit()
    return result
