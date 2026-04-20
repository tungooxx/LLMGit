"""Chat and ingest endpoints."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud
from app.commit_engine import ensure_main_branch
from app.config import get_settings
from app.db import get_db
from app.llm import LLMClient
from app.normalization import normalize_extracted_claim
from app.schemas import (
    BranchRead,
    ChatRequest,
    ChatResponse,
    Citation,
    ExtractedClaim,
    IngestRequest,
    IngestResponse,
    SourceCreate,
)
from app.tools import apply_staged_commit, stage_belief_changes

router = APIRouter(tags=["chat"])


def get_llm_client() -> LLMClient:
    """FastAPI dependency for the LLM client."""

    return LLMClient()


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    llm: LLMClient = Depends(get_llm_client),
) -> ChatResponse:
    """Chat with TruthGit memory while preserving belief lineage."""

    branch = _resolve_branch(db, request.branch_id)
    extracted = llm.extract_claims(request.message)
    source = SourceCreate(
        source_type="user_message",
        source_ref="chat",
        excerpt=request.message,
        trust_score=0.7,
    )
    staged = stage_belief_changes(claims=extracted.claims, branch_id=branch.id, source=source)
    created_commit_id: int | None = None
    introduced_versions: list[object] = []
    warnings = list(staged.warnings)

    if request.auto_commit and extracted.claims:
        try:
            plan = llm.plan_answer(request.message, [])
            result = apply_staged_commit(
                db,
                staged_commit_id=staged.staged_commit_id,
                commit_message=plan.proposed_commit_message or "Update belief memory from chat",
                created_by="agent",
                model_name=get_settings().openai_model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        db.commit()
        created_commit_id = result.commit.id
        introduced_versions = list(result.introduced_versions)
        warnings.extend(result.warnings)

    citations = _citations_for_versions(db, introduced_versions)
    if not citations:
        relevant_versions = _retrieve_relevant_versions(db, request.message, branch.id)
        citations = _citations_for_versions(db, relevant_versions)
    answer = _compose_answer(
        db=db,
        message=request.message,
        branch_name=branch.name,
        introduced_versions=introduced_versions,
        citations=citations,
    )
    return ChatResponse(
        answer=answer,
        citations=citations,
        memory_updated=bool(introduced_versions),
        created_commit_id=created_commit_id,
        branch=BranchRead.model_validate(branch),
        warnings=sorted(set(warnings)),
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    db: Session = Depends(get_db),
    llm: LLMClient = Depends(get_llm_client),
) -> IngestResponse:
    """Extract and optionally commit claims from raw text."""

    branch = _resolve_branch(db, request.branch_id)
    extracted = llm.extract_claims(request.raw_text)
    source = SourceCreate(
        source_type=request.source_type,
        source_ref=request.source_ref,
        excerpt=request.raw_text,
        trust_score=request.trust_score,
    )
    staged = stage_belief_changes(claims=extracted.claims, branch_id=branch.id, source=source)
    warnings = list(staged.warnings)
    commit_id: int | None = None
    memory_updated = False
    if request.auto_commit and extracted.claims:
        try:
            result = apply_staged_commit(
                db,
                staged_commit_id=staged.staged_commit_id,
                commit_message=f"Ingest {request.source_type}: {request.source_ref or 'raw text'}",
                created_by="agent",
                model_name=get_settings().openai_model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        db.commit()
        commit_id = result.commit.id
        memory_updated = bool(result.introduced_versions)
        warnings.extend(result.warnings)

    return IngestResponse(
        extracted_claims=extracted.claims,
        staged_commit_id=staged.staged_commit_id,
        memory_updated=memory_updated,
        created_commit_id=commit_id,
        warnings=sorted(set(warnings)),
    )


def _resolve_branch(db: Session, branch_id: int | None) -> object:
    branch = crud.get_branch(db, branch_id) if branch_id is not None else ensure_main_branch(db)
    if branch is None:
        raise HTTPException(status_code=404, detail="Branch not found")
    db.flush()
    return branch


def _retrieve_relevant_versions(db: Session, message: str, branch_id: int) -> list[object]:
    terms = _query_terms(message)
    seen: set[int] = set()
    versions: list[object] = []
    for term in terms:
        for version in crud.search_beliefs(db, query=term, branch_id=None, include_inactive=True, limit=10):
            if version.id in seen:
                continue
            belief = crud.get_belief(db, version.belief_id)
            if belief is None:
                continue
            current_ids = {
                current.id
                for current in crud.get_current_versions(db, belief_id=belief.id, branch_id=branch_id)
            }
            if version.id in current_ids or "why" in message.lower() or "previously" in message.lower():
                seen.add(version.id)
                versions.append(version)
    return versions[:8]


def _query_terms(message: str) -> list[str]:
    capitals = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", message)
    terms = capitals + [word for word in re.findall(r"\b[a-zA-Z]{4,}\b", message.lower()) if word not in {"what", "where", "when", "with", "that", "this", "from"}]
    return terms or [message]


def _citations_for_versions(db: Session, versions: list[object]) -> list[Citation]:
    citations: list[Citation] = []
    for version in versions:
        belief = crud.get_belief(db, version.belief_id)
        if belief is None:
            continue
        citations.append(
            Citation(
                belief_id=belief.id,
                belief_version_id=version.id,
                subject=belief.subject,
                predicate=belief.predicate,
                object_value=version.object_value,
                status=version.status,
                source_id=version.source_id,
                commit_id=version.commit_id,
            )
        )
    return citations


def _compose_answer(
    *,
    db: Session,
    message: str,
    branch_name: str,
    introduced_versions: list[object],
    citations: list[Citation],
) -> str:
    if introduced_versions:
        pieces = []
        for version in introduced_versions:
            belief = crud.get_belief(db, version.belief_id)
            if belief is None:
                continue
            lineage = (
                f", superseding version {version.supersedes_version_id}"
                if version.supersedes_version_id
                else ""
            )
            pieces.append(
                f"{belief.subject} {belief.predicate} {version.object_value} "
                f"as version {version.id} on branch '{branch_name}'{lineage}"
            )
        return "Recorded: " + "; ".join(pieces) + "."

    lower = message.lower()
    if "why" in lower and citations:
        grouped = {}
        for citation in citations:
            grouped.setdefault((citation.subject, citation.predicate), []).append(citation)
        lines = []
        for (subject, predicate), items in grouped.items():
            timeline = sorted(items, key=lambda item: item.belief_version_id)
            rendered = " -> ".join(
                f"v{item.belief_version_id}:{item.object_value}({item.status})" for item in timeline
            )
            active = [item for item in timeline if item.status in {"active", "hypothetical"}]
            current = active[-1].object_value if active else "no active value"
            lines.append(
                f"For {subject} {predicate}, the lineage is {rendered}. "
                f"The current branch value is {current} because later validated versions supersede earlier ones."
            )
        return " ".join(lines)

    if citations:
        active = [citation for citation in citations if citation.status in {"active", "hypothetical"}]
        if active:
            facts = "; ".join(
                f"{item.subject} {item.predicate} {item.object_value} (v{item.belief_version_id})"
                for item in active
            )
            return f"On branch '{branch_name}', TruthGit has evidence for: {facts}."
    return "I do not have an evidence-backed TruthGit belief for that request yet."
