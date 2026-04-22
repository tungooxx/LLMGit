"""Belief query endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import crud, models
from app.db import get_db
from app.schemas import BeliefVersionRead, BeliefWithVersions

router = APIRouter(prefix="/beliefs", tags=["beliefs"])


@router.get("/search", response_model=list[BeliefVersionRead])
def search_beliefs(
    q: str = Query(..., min_length=1),
    branch_id: int | None = None,
    include_inactive: bool = False,
    db: Session = Depends(get_db),
) -> list[dict[str, object]]:
    """Search belief versions."""

    return [
        _version_response(db, version)
        for version in crud.search_beliefs(db, query=q, branch_id=branch_id, include_inactive=include_inactive)
    ]


@router.get("/active", response_model=list[BeliefVersionRead])
def get_active_belief(
    subject: str,
    predicate: str,
    branch_id: int | None = None,
    db: Session = Depends(get_db),
) -> list[dict[str, object]]:
    """Return active belief versions for a subject and predicate."""

    branch = crud.get_branch(db, branch_id) if branch_id is not None else crud.get_branch_by_name(db, "main")
    if branch is None:
        raise HTTPException(status_code=404, detail="Branch not found")
    belief = crud.get_belief_by_subject_predicate(db, subject=subject, predicate=predicate)
    if belief is None:
        return []
    return [
        _version_response(db, version)
        for version in crud.get_current_versions(db, belief_id=belief.id, branch_id=branch.id)
    ]


@router.get("/{belief_id}", response_model=BeliefWithVersions)
def get_belief(belief_id: int, db: Session = Depends(get_db)) -> dict[str, object]:
    """Return a belief and all versions."""

    belief = crud.get_belief(db, belief_id)
    if belief is None:
        raise HTTPException(status_code=404, detail="Belief not found")
    return {
        "belief": belief,
        "versions": [_version_response(db, version) for version in crud.list_belief_versions(db, belief_id)],
    }


@router.get("/{belief_id}/timeline", response_model=list[BeliefVersionRead])
def get_belief_timeline(belief_id: int, db: Session = Depends(get_db)) -> list[dict[str, object]]:
    """Return chronological belief versions."""

    if crud.get_belief(db, belief_id) is None:
        raise HTTPException(status_code=404, detail="Belief not found")
    return [_version_response(db, version) for version in crud.list_belief_versions(db, belief_id)]


def _version_response(db: Session, version: models.BeliefVersion) -> dict[str, object]:
    payload = {
        "id": version.id,
        "belief_id": version.belief_id,
        "commit_id": version.commit_id,
        "branch_id": version.branch_id,
        "object_value": version.object_value,
        "normalized_object_value": version.normalized_object_value,
        "confidence": version.confidence,
        "valid_from": version.valid_from,
        "valid_to": version.valid_to,
        "status": version.status,
        "source_id": version.source_id,
        "supersedes_version_id": version.supersedes_version_id,
        "contradiction_group": version.contradiction_group,
        "metadata_json": version.metadata_json,
        "created_at": version.created_at,
    }
    payload.update(crud.support_graph_payload(db, version))
    return payload
