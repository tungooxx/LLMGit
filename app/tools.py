"""Validated tool registry used by the chat loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app import crud
from app.commit_engine import apply_claims
from app.normalization import normalize_extracted_claim
from app.schemas import ExtractedClaim, SourceCreate


@dataclass
class StagedCommit:
    """In-memory staged commit for MVP use."""

    staged_commit_id: str
    branch_id: int
    claims: list[ExtractedClaim]
    source: SourceCreate
    warnings: list[str]


STAGED_COMMITS: dict[str, StagedCommit] = {}


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
        "description": "Stage validated belief changes without writing durable versions.",
        "parameters": {
            "type": "object",
            "properties": {"claims": {"type": "array", "items": {"type": "object"}}},
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
    return {
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
    *,
    claims: list[ExtractedClaim],
    branch_id: int,
    source: SourceCreate,
) -> StagedCommit:
    """Tool: stage claims in memory."""

    staged_id = str(uuid4())
    warnings = []
    normalized = [normalize_extracted_claim(claim) for claim in claims]
    if not normalized:
        warnings.append("No explicit atomic claims were staged.")
    if source.trust_score < 0.4:
        warnings.append("Low-trust source was staged.")
    staged = StagedCommit(
        staged_commit_id=staged_id,
        branch_id=branch_id,
        claims=claims,
        source=source,
        warnings=warnings,
    )
    STAGED_COMMITS[staged_id] = staged
    return staged


def apply_staged_commit(
    db: Session,
    *,
    staged_commit_id: str,
    commit_message: str,
    created_by: str = "agent",
    model_name: str | None = None,
) -> Any:
    """Tool: validate and apply a staged commit."""

    staged = STAGED_COMMITS.get(staged_commit_id)
    if staged is None:
        raise ValueError(f"Unknown staged commit: {staged_commit_id}")
    source = crud.create_source(
        db,
        source_type=staged.source.source_type,
        source_ref=staged.source.source_ref,
        excerpt=staged.source.excerpt,
        trust_score=staged.source.trust_score,
    )
    result = apply_claims(
        db,
        claims=[normalize_extracted_claim(claim) for claim in staged.claims],
        branch_id=staged.branch_id,
        source=source,
        message=commit_message,
        created_by=created_by,
        model_name=model_name,
    )
    STAGED_COMMITS.pop(staged_commit_id, None)
    return result


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
            return stage_belief_changes(claims=claims, branch_id=self.branch_id, source=self.source).__dict__
        if name == "apply_staged_commit":
            result = apply_staged_commit(self.db, **arguments)
            return {
                "commit_id": result.commit.id,
                "introduced_version_ids": [version.id for version in result.introduced_versions],
                "warnings": result.warnings,
            }
        raise ValueError(f"Unknown tool: {name}")
