"""Validated tool registry used by the chat loop."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app import crud, models
from app.commit_engine import apply_claims
from app.memory_ci import run_memory_ci
from app.normalization import normalize_extracted_claim
from app.schemas import ExtractedClaim, SourceCreate
from app.write_policy import review_requirements_for_claims


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "search_beliefs",
        "description": "Search belief versions by text query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "branch_id": {"type": ["integer", "null"]},
                "include_inactive": {"type": "boolean"},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "get_belief_timeline",
        "description": "Get historical versions for a belief.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": ["string", "null"]},
                "predicate": {"type": ["string", "null"]},
                "belief_id": {"type": ["integer", "null"]},
            },
        },
    },
    {
        "type": "function",
        "name": "get_active_belief",
        "description": "Get active belief versions for subject and predicate.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "branch_id": {"type": ["integer", "null"]},
            },
            "required": ["subject", "predicate"],
        },
    },
    {
        "type": "function",
        "name": "stage_belief_changes",
        "description": "Stage model-approved belief changes without writing durable versions.",
        "parameters": {
            "type": "object",
            "properties": {
                "claims": {"type": "array", "items": {"type": "object"}},
                "review_required": {"type": "boolean"},
                "risk_reasons": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "commit_message": {"type": "string"},
            },
            "required": ["claims"],
        },
    },
    {
        "type": "function",
        "name": "apply_staged_commit",
        "description": "Apply a previously staged commit through deterministic validation.",
        "parameters": {
            "type": "object",
            "properties": {
                "staged_commit_id": {"type": "string"},
                "commit_message": {"type": "string"},
            },
            "required": ["staged_commit_id", "commit_message"],
        },
    },
]


def serialize_version(db: Session, version: Any) -> dict[str, Any]:
    """Serialize a belief version with its belief identity."""

    belief = crud.get_belief(db, version.belief_id)
    payload = {
        "id": version.id,
        "belief_id": version.belief_id,
        "subject": belief.subject if belief else None,
        "predicate": belief.predicate if belief else None,
        "object_value": version.object_value,
        "status": version.status,
        "branch_id": version.branch_id,
        "commit_id": version.commit_id,
        "source_id": version.source_id,
        "supersedes_version_id": version.supersedes_version_id,
        "contradiction_group": version.contradiction_group,
        "valid_from": version.valid_from,
        "valid_to": version.valid_to,
    }
    payload.update(crud.support_graph_payload(db, version))
    return payload


def search_beliefs(
    db: Session,
    *,
    query: str,
    branch_id: int | None = None,
    include_inactive: bool = False,
) -> list[dict[str, Any]]:
    """Tool: search beliefs."""

    return [
        serialize_version(db, version)
        for version in crud.search_beliefs(
            db,
            query=query,
            branch_id=branch_id,
            include_inactive=include_inactive,
        )
    ]


def get_belief_timeline(
    db: Session,
    *,
    subject: str | None = None,
    predicate: str | None = None,
    belief_id: int | None = None,
) -> list[dict[str, Any]]:
    """Tool: fetch belief timeline."""

    belief = crud.get_belief(db, belief_id) if belief_id is not None else None
    if belief is None and subject and predicate:
        belief = crud.get_belief_by_subject_predicate(db, subject=subject, predicate=predicate)
    if belief is None:
        return []
    return [serialize_version(db, version) for version in crud.list_belief_versions(db, belief.id)]


def get_active_belief(
    db: Session,
    *,
    subject: str,
    predicate: str,
    branch_id: int | None = None,
) -> list[dict[str, Any]]:
    """Tool: get active belief."""

    branch = crud.get_branch(db, branch_id) if branch_id is not None else crud.get_branch_by_name(db, "main")
    belief = crud.get_belief_by_subject_predicate(db, subject=subject, predicate=predicate)
    if branch is None or belief is None:
        return []
    return [
        serialize_version(db, version)
        for version in crud.get_current_versions(db, belief_id=belief.id, branch_id=branch.id)
    ]


def get_commit_history(
    db: Session,
    *,
    commit_id: int | None = None,
    belief_id: int | None = None,
    branch_id: int | None = None,
) -> list[dict[str, Any]]:
    """Tool: get commit history."""

    return [
        {
            "id": commit.id,
            "branch_id": commit.branch_id,
            "parent_commit_id": commit.parent_commit_id,
            "operation_type": commit.operation_type,
            "message": commit.message,
            "created_by": commit.created_by,
            "created_at": commit.created_at,
        }
        for commit in crud.list_commits(
            db,
            commit_id=commit_id,
            belief_id=belief_id,
            branch_id=branch_id,
        )
    ]


def get_branch_info(db: Session, *, branch_id: int) -> dict[str, Any] | None:
    """Tool: get branch information."""

    branch = crud.get_branch(db, branch_id)
    if branch is None:
        return None
    return {
        "id": branch.id,
        "name": branch.name,
        "description": branch.description,
        "parent_branch_id": branch.parent_branch_id,
        "status": branch.status,
        "created_at": branch.created_at,
    }


def explain_conflict_context(db: Session, *, belief_id: int) -> dict[str, Any]:
    """Tool: explain conflict context as structured lineage."""

    versions = crud.list_belief_versions(db, belief_id)
    groups = sorted({version.contradiction_group for version in versions if version.contradiction_group})
    return {
        "belief_id": belief_id,
        "contradiction_groups": groups,
        "timeline": [serialize_version(db, version) for version in versions],
    }


def stage_belief_changes(
    db: Session,
    *,
    claims: list[ExtractedClaim],
    branch_id: int,
    source: SourceCreate,
    proposed_commit_message: str = "Update belief memory",
    created_by: str = "agent",
    model_name: str | None = None,
    review_required: bool = False,
    risk_reasons: list[str] | None = None,
    warnings: list[str] | None = None,
) -> models.StagedCommit:
    """Tool: persist proposed belief changes for review or auto-approval."""

    staged_id = str(uuid4())
    risk_reason_values = list(risk_reasons or [])
    warning_values = list(warnings or [])
    branch = crud.get_branch(db, branch_id)
    if branch is None:
        raise ValueError(f"Branch does not exist: {branch_id}")
    policy = review_requirements_for_claims(
        db,
        claims=claims,
        branch_id=branch_id,
        branch_name=branch.name,
        source_trust=source.trust_score,
        source_excerpt=source.excerpt,
    )
    review_required = review_required or policy.review_required
    risk_reason_values.extend(policy.risk_reasons)
    warning_values.extend(policy.warnings)
    if not claims:
        warning_values.append("No explicit atomic claims were staged.")
    staged = models.StagedCommit(
        id=staged_id,
        branch_id=branch_id,
        status="proposed",
        claims_json=[claim.model_dump(mode="json", by_alias=False) for claim in claims],
        source_type=source.source_type,
        source_ref=source.source_ref,
        source_excerpt=source.excerpt,
        source_trust_score=source.trust_score,
        proposed_commit_message=proposed_commit_message,
        created_by=created_by,
        model_name=model_name,
        review_required=review_required,
        risk_reasons=risk_reason_values,
        warnings_json=warning_values,
    )
    db.add(staged)
    db.flush()
    crud.add_audit_event(
        db,
        event_type="staged_commit.created",
        entity_type="staged_commit",
        entity_id=0,
        entity_key=staged.id,
        payload={
            "staged_commit_id": staged.id,
            "branch_id": branch_id,
            "review_required": review_required,
            "risk_reasons": risk_reason_values,
        },
    )
    run_memory_ci(db, staged.id)
    return staged


def apply_staged_commit(
    db: Session,
    *,
    staged_commit_id: str,
    commit_message: str,
    created_by: str = "agent",
    model_name: str | None = None,
) -> Any:
    """Tool: apply a staged commit when the stored model decision allows it."""

    staged = db.get(models.StagedCommit, staged_commit_id)
    if staged is None:
        raise ValueError(f"Unknown staged commit: {staged_commit_id}")
    if staged.latest_check_run_id is None:
        run_memory_ci(db, staged.id)
        db.flush()
    if staged.status == "quarantined":
        raise ValueError(f"Staged commit {staged_commit_id} is quarantined and cannot auto-apply")
    if staged.status == "rejected":
        raise ValueError(f"Staged commit {staged_commit_id} was rejected")
    if staged.review_required:
        raise ValueError(f"Staged commit {staged_commit_id} requires human review")
    return approve_staged_commit(
        db,
        staged_commit_id=staged_commit_id,
        reviewer=created_by,
        notes="Auto-approved by model write decision.",
        commit_message=commit_message,
        model_name=model_name,
    )


def approve_staged_commit(
    db: Session,
    *,
    staged_commit_id: str,
    reviewer: str = "user",
    notes: str | None = None,
    commit_message: str | None = None,
    model_name: str | None = None,
    override_quarantine: bool = False,
) -> Any:
    """Approve and apply a pending staged commit."""

    staged = db.get(models.StagedCommit, staged_commit_id)
    if staged is None:
        raise ValueError(f"Unknown staged commit: {staged_commit_id}")
    if staged.latest_check_run_id is None and staged.status not in {"rejected", "applied"}:
        run_memory_ci(db, staged.id)
        db.flush()
    if staged.status == "quarantined" and not override_quarantine:
        raise ValueError(f"Staged commit {staged_commit_id} is quarantined and must be released or overridden")
    if staged.status in {"applied", "rejected"}:
        raise ValueError(f"Staged commit {staged_commit_id} is {staged.status}, not reviewable")
    if staged.status not in {"pending", "proposed", "checked", "review_required", "approved", "quarantined"}:
        raise ValueError(f"Staged commit {staged_commit_id} is {staged.status}, not reviewable")
    if staged.status == "quarantined" and override_quarantine:
        if not notes or not notes.strip():
            raise ValueError("Overriding quarantine requires reviewer notes")
        staged.quarantine_release_status = "approved_override"
        staged.quarantine_reviewer = reviewer
        staged.quarantine_notes = notes
        crud.add_audit_event(
            db,
            event_type="staged_commit.quarantine_override_approved",
            entity_type="staged_commit",
            entity_id=0,
            entity_key=staged.id,
            payload={"staged_commit_id": staged.id, "reviewer": reviewer, "notes": notes},
        )
    staged.status = "approved"
    crud.add_audit_event(
        db,
        event_type="staged_commit.approval_recorded",
        entity_type="staged_commit",
        entity_id=0,
        entity_key=staged.id,
        payload={"staged_commit_id": staged.id, "reviewer": reviewer, "notes": notes},
    )
    source = crud.create_source(
        db,
        source_type=staged.source_type,
        source_ref=staged.source_ref,
        excerpt=staged.source_excerpt,
        trust_score=staged.source_trust_score,
    )
    claims = [ExtractedClaim.model_validate(claim) for claim in staged.claims_json]
    result = apply_claims(
        db,
        claims=[normalize_extracted_claim(claim) for claim in claims],
        branch_id=staged.branch_id,
        source=source,
        message=commit_message or staged.proposed_commit_message,
        created_by=reviewer,
        model_name=model_name or staged.model_name,
    )
    staged.status = "applied"
    staged.review_required = False
    staged.reviewer = reviewer
    staged.review_notes = notes
    staged.reviewed_at = models.utc_now()
    staged.applied_commit_id = result.commit.id
    crud.add_audit_event(
        db,
        event_type="staged_commit.approved",
        entity_type="staged_commit",
        entity_id=0,
        entity_key=staged.id,
        payload={
            "staged_commit_id": staged.id,
            "applied_commit_id": result.commit.id,
            "reviewer": reviewer,
            "notes": notes,
        },
    )
    return result


def reject_staged_commit(
    db: Session,
    *,
    staged_commit_id: str,
    reviewer: str = "user",
    notes: str | None = None,
) -> models.StagedCommit:
    """Reject a pending staged commit without mutating belief memory."""

    staged = db.get(models.StagedCommit, staged_commit_id)
    if staged is None:
        raise ValueError(f"Unknown staged commit: {staged_commit_id}")
    if staged.status in {"applied", "rejected"}:
        raise ValueError(f"Staged commit {staged_commit_id} is {staged.status}, not reviewable")
    staged.status = "rejected"
    staged.reviewer = reviewer
    staged.review_notes = notes
    staged.reviewed_at = models.utc_now()
    if staged.quarantined_at is not None:
        staged.quarantine_release_status = "rejected"
        staged.quarantine_reviewer = reviewer
        staged.quarantine_notes = notes
    crud.add_audit_event(
        db,
        event_type="staged_commit.rejected",
        entity_type="staged_commit",
        entity_id=0,
        entity_key=staged.id,
        payload={"staged_commit_id": staged.id, "reviewer": reviewer, "notes": notes},
    )
    db.flush()
    return staged


class ToolExecutor:
    """Session-bound tool dispatcher."""

    def __init__(self, db: Session, branch_id: int, source: SourceCreate | None = None) -> None:
        self.db = db
        self.branch_id = branch_id
        self.source = source or SourceCreate(source_type="system", excerpt="tool loop")

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a registered tool by name."""

        if name == "search_beliefs":
            return search_beliefs(self.db, **arguments)
        if name == "get_belief_timeline":
            return get_belief_timeline(self.db, **arguments)
        if name == "get_active_belief":
            arguments.setdefault("branch_id", self.branch_id)
            return get_active_belief(self.db, **arguments)
        if name == "get_commit_history":
            return get_commit_history(self.db, **arguments)
        if name == "get_branch_info":
            return get_branch_info(self.db, **arguments)
        if name == "explain_conflict_context":
            return explain_conflict_context(self.db, **arguments)
        if name == "stage_belief_changes":
            claims = [ExtractedClaim.model_validate(item) for item in arguments.get("claims", [])]
            staged = stage_belief_changes(
                self.db,
                claims=claims,
                branch_id=self.branch_id,
                source=self.source,
                proposed_commit_message=arguments.get("commit_message", "Update belief memory"),
                review_required=bool(arguments.get("review_required", False)),
                risk_reasons=list(arguments.get("risk_reasons", []) or []),
                warnings=list(arguments.get("warnings", []) or []),
            )
            return {
                "staged_commit_id": staged.id,
                "status": staged.status,
                "review_required": staged.review_required,
                "risk_reasons": staged.risk_reasons,
                "warnings": staged.warnings_json,
            }
        if name == "apply_staged_commit":
            result = apply_staged_commit(self.db, **arguments)
            return {
                "commit_id": result.commit.id,
                "introduced_version_ids": [version.id for version in result.introduced_versions],
                "warnings": result.warnings,
            }
        raise ValueError(f"Unknown tool: {name}")
