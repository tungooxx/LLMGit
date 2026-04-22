"""Chat and ingest endpoints."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import crud, models
from app.commit_engine import create_branch, ensure_main_branch
from app.config import get_settings
from app.db import get_db
from app.llm import LLMClient
from app.schemas import (
    BranchRead,
    ChatRequest,
    ChatResponse,
    Citation,
    IngestRequest,
    IngestResponse,
    SourceCreate,
)
from app.tools import apply_staged_commit, stage_belief_changes
from app.write_policy import enforce_write_policy, safe_branch_name

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

    context_branch = _resolve_branch(db, request.branch_id, "main")
    if _is_read_only_memory_request(request.message):
        answer_context = _chat_answer_context(db, context_branch, request.message)
        relevant_versions = _retrieve_relevant_versions(db, request.message, context_branch.id)
        if not relevant_versions:
            relevant_versions = [
                version
                for row in answer_context["current_beliefs"]
                if (version := db.get(models.BeliefVersion, int(row["id"]))) is not None
            ]
        return ChatResponse(
            answer=llm.answer_from_memory(request.message, answer_context),
            citations=_citations_for_versions(db, relevant_versions),
            memory_updated=False,
            created_commit_id=None,
            staged_commit_id=None,
            review_required=False,
            branch=BranchRead.model_validate(context_branch),
            warnings=[],
        )

    plan = llm.plan_memory_write(
        request.message,
        fallback_branch_name=context_branch.name,
        memory_context=_chat_memory_context(db, context_branch, request.message),
    )
    policy = enforce_write_policy(
        db,
        plan=plan,
        source_excerpt=request.message,
        fallback_branch_name=context_branch.name,
    )
    branch = context_branch if request.branch_id is not None and policy.branch_name == context_branch.name else _resolve_branch(db, None, policy.branch_name)
    source = SourceCreate(
        source_type="user_message",
        source_ref="chat",
        excerpt=request.message,
        trust_score=policy.trust_score,
    )
    should_stage = bool(plan.claims) and policy.write_action != "reject"
    staged = (
        stage_belief_changes(
            db,
            claims=plan.claims,
            branch_id=branch.id,
            source=source,
            proposed_commit_message="Update belief memory from chat",
            created_by="agent",
            model_name=get_settings().openai_model,
            review_required=policy.review_required,
            risk_reasons=policy.risk_reasons,
            warnings=policy.warnings,
        )
        if should_stage
        else None
    )
    created_commit_id: int | None = None
    introduced_versions: list[object] = []
    warnings = list(staged.warnings_json) if staged else [*policy.warnings, *policy.risk_reasons]

    if request.auto_commit and staged and not staged.review_required:
        try:
            result = apply_staged_commit(
                db,
                staged_commit_id=staged.id,
                commit_message=staged.proposed_commit_message,
                created_by="agent",
                model_name=get_settings().openai_model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        db.commit()
        created_commit_id = result.commit.id
        introduced_versions = list(result.introduced_versions)
        warnings.extend(result.warnings)
    elif staged and staged.review_required:
        warnings.append(f"TruthGit policy staged commit {staged.id} for review.")

    if staged and staged.status not in {"applied", "rejected"}:
        db.commit()

    citations = _citations_for_versions(db, introduced_versions)
    if not citations:
        relevant_versions = _retrieve_relevant_versions(db, request.message, branch.id)
        citations = _citations_for_versions(db, relevant_versions)
    if policy.write_action == "reject" and plan.claims:
        answer = plan.assistant_reply
    elif staged and staged.status not in {"applied", "rejected"}:
        answer = f"{plan.assistant_reply} Staged {len(plan.claims)} claim(s) as {staged.id}."
    else:
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
        staged_commit_id=staged.id if staged else None,
        review_required=bool(staged.review_required) if staged else False,
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

    context_branch = _resolve_branch(db, request.branch_id, "main")
    plan = llm.plan_memory_write(
        request.raw_text,
        fallback_branch_name=context_branch.name,
        fallback_trust_score=request.trust_score,
        memory_context=_chat_memory_context(db, context_branch, request.raw_text),
    )
    policy = enforce_write_policy(
        db,
        plan=plan,
        source_excerpt=request.raw_text,
        fallback_branch_name=context_branch.name,
    )
    branch = context_branch if request.branch_id is not None and policy.branch_name == context_branch.name else _resolve_branch(db, None, policy.branch_name)
    source = SourceCreate(
        source_type=request.source_type,
        source_ref=request.source_ref,
        excerpt=request.raw_text,
        trust_score=policy.trust_score,
    )
    staged = (
        stage_belief_changes(
            db,
            claims=plan.claims,
            branch_id=branch.id,
            source=source,
            proposed_commit_message=f"Ingest {request.source_type}: {request.source_ref or 'raw text'}",
            created_by="agent",
            model_name=get_settings().openai_model,
            review_required=policy.review_required,
            risk_reasons=policy.risk_reasons,
            warnings=policy.warnings,
        )
        if plan.claims and policy.write_action != "reject"
        else None
    )
    warnings = list(staged.warnings_json) if staged else [*policy.warnings, *policy.risk_reasons]
    commit_id: int | None = None
    memory_updated = False
    if request.auto_commit and staged and plan.claims and not staged.review_required:
        try:
            result = apply_staged_commit(
                db,
                staged_commit_id=staged.id,
                commit_message=staged.proposed_commit_message,
                created_by="agent",
                model_name=get_settings().openai_model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        db.commit()
        commit_id = result.commit.id
        memory_updated = bool(result.introduced_versions)
        warnings.extend(result.warnings)
    elif staged and staged.review_required:
        warnings.append(f"TruthGit policy staged commit {staged.id} for review.")

    if staged and staged.status not in {"applied", "rejected"}:
        db.commit()

    return IngestResponse(
        extracted_claims=plan.claims,
        staged_commit_id=staged.id if staged else None,
        staged_status=staged.status if staged else None,
        review_required=staged.review_required if staged else False,
        memory_updated=memory_updated,
        created_commit_id=commit_id,
        warnings=sorted(set(warnings)),
    )


def _resolve_branch(db: Session, branch_id: int | None, branch_name: str = "main") -> object:
    if branch_id is not None:
        branch = crud.get_branch(db, branch_id)
    else:
        clean_name = _safe_branch_name(branch_name)
        if clean_name == "main":
            branch = ensure_main_branch(db)
        else:
            branch = crud.get_branch_by_name(db, clean_name)
            if branch is None:
                branch = create_branch(db, name=clean_name, description=f"Model-selected branch {clean_name}")
    if branch is None:
        raise HTTPException(status_code=404, detail="Branch not found")
    db.flush()
    return branch


def _safe_branch_name(value: str) -> str:
    return safe_branch_name(value)


def _write_plan_risk_reasons(plan: object) -> list[str]:
    reasons = list(getattr(plan, "risk_reasons", []) or [])
    write_action = getattr(plan, "write_action", "commit_now")
    if write_action in {"stage_for_review", "reject"}:
        reasons.append(f"model_write_action:{write_action}")
    return reasons


def _chat_memory_context(db: Session, branch: object, message: str) -> dict[str, object]:
    """Build compact current memory context for model write planning."""

    relevant_versions = _retrieve_relevant_versions(db, message, branch.id)
    current_versions = crud.search_beliefs(db, query="", branch_id=branch.id, include_inactive=False, limit=30)
    combined: list[object] = []
    seen: set[int] = set()
    for version in [*relevant_versions, *current_versions]:
        if version.id in seen:
            continue
        seen.add(version.id)
        combined.append(version)
    return {
        "branch": {"id": branch.id, "name": branch.name},
        "belief_versions": [_memory_version_row(db, version) for version in combined[:30]],
    }


def _chat_answer_context(db: Session, branch: object, message: str) -> dict[str, object]:
    """Build read-only memory context for answering direct chat commands."""

    current: list[dict[str, object]] = []
    all_beliefs = list(db.scalars(select(models.Belief).order_by(models.Belief.id)))
    for belief in all_beliefs:
        current.extend(_memory_version_row(db, version) for version in crud.get_current_versions(db, belief_id=belief.id, branch_id=branch.id))
    timelines = crud.search_beliefs(db, query="", include_inactive=True, limit=80)
    staged = list(
        db.scalars(
            select(models.StagedCommit)
            .order_by(models.StagedCommit.created_at.desc())
            .limit(30)
        )
    )
    audit = crud.list_audit_events(db, limit=40)
    return {
        "branch": {"id": branch.id, "name": branch.name},
        "current_beliefs": current,
        "timelines": [_memory_version_row(db, version) for version in timelines],
        "staged_commits": [_staged_context_row(db, item) for item in staged],
        "pending_staged_commits": [
            _staged_context_row(db, item)
            for item in staged
            if item.status in {"pending", "proposed", "checked", "review_required", "quarantined"}
        ],
        "audit_events": [
            {
                "id": event.id,
                "event_type": event.event_type,
                "entity_type": event.entity_type,
                "entity_id": event.entity_id,
                "entity_key": event.entity_key,
                "payload_json": event.payload_json,
                "created_at": event.created_at.isoformat() if event.created_at else None,
            }
            for event in audit
        ],
    }


def _memory_version_row(db: Session, version: object) -> dict[str, object]:
    belief = crud.get_belief(db, version.belief_id)
    row = {
        "belief_version_id": version.id,
        "id": version.id,
        "belief_id": version.belief_id,
        "subject": belief.subject if belief else None,
        "predicate": belief.predicate if belief else None,
        "object_value": version.object_value,
        "status": version.status,
        "confidence": version.confidence,
        "valid_from": version.valid_from.isoformat() if version.valid_from else None,
        "valid_to": version.valid_to.isoformat() if version.valid_to else None,
        "supersedes_version_id": version.supersedes_version_id,
        "contradiction_group": version.contradiction_group,
    }
    row.update(crud.support_graph_payload(db, version))
    return row


def _staged_context_row(db: Session, staged: models.StagedCommit) -> dict[str, object]:
    return {
        "id": staged.id,
        "status": staged.status,
        "branch_id": staged.branch_id,
        "claims_json": staged.claims_json,
        "review_required": staged.review_required,
        "risk_reasons": staged.risk_reasons,
        "warnings": staged.warnings_json,
        "source_ref": staged.source_ref,
        "source_excerpt": staged.source_excerpt,
        "source_trust_score": staged.source_trust_score,
        "quarantine_reason_summary": staged.quarantine_reason_summary,
        "quarantine_release_status": staged.quarantine_release_status,
        "applied_commit_id": staged.applied_commit_id,
        "checks": {},
    }


def _is_read_only_memory_request(message: str) -> bool:
    stripped = message.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if stripped.endswith("?"):
        return True
    return bool(
        re.match(
            r"^(why|what|where|when|how|who|which|show|list|display|explain|summarize|tell me)\b",
            lower,
        )
    )


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
    if any(term in lower for term in ("source", "provenance", "support", "justify", "justifies")) and citations:
        lines = []
        for citation in citations:
            if citation.status not in {"active", "hypothetical"}:
                continue
            version = db.get(models.BeliefVersion, citation.belief_version_id)
            if version is None:
                continue
            graph = crud.support_graph_payload(db, version)
            active_supports = [
                _source_label(source)
                for source in graph["support_sources"]
                if source.get("status") == "active"
            ]
            rolled_back = [
                _source_label(source)
                for source in graph["support_sources"]
                if source.get("status") == "rolled_back"
            ]
            quarantined = [
                _source_label(source)
                for source in graph["support_sources"]
                if source.get("status") == "quarantined"
            ]
            active_opposition = [
                _source_label(source)
                for source in graph["opposition_sources"]
                if source.get("status") == "active"
            ]
            support_text = ", ".join(active_supports) if active_supports else "no active support sources"
            extra = []
            if active_opposition:
                extra.append(f"active opposition: {', '.join(active_opposition)}")
            if rolled_back:
                extra.append(f"rolled back: {', '.join(rolled_back)}")
            if quarantined:
                extra.append(f"quarantined: {', '.join(quarantined)}")
            suffix = f" ({'; '.join(extra)})" if extra else ""
            lines.append(
                f"{citation.subject} {citation.predicate} {citation.object_value} "
                f"is currently supported by {support_text}{suffix}."
            )
        if lines:
            return " ".join(lines)

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


def _source_label(source: dict[str, object]) -> str:
    source_ref = str(source.get("source_ref") or "")
    excerpt = str(source.get("excerpt") or "")
    if source_ref and not source_ref.startswith(("demo-ui:", "chat", "source-")):
        return source_ref
    quote = _short_source_excerpt(excerpt)
    if quote:
        return quote
    return source_ref or f"source-{source.get('source_id')}"


def _short_source_excerpt(excerpt: str) -> str:
    compact = re.sub(r"\s+", " ", excerpt).strip()
    if not compact:
        return ""
    return compact[:140] + ("..." if len(compact) > 140 else "")
