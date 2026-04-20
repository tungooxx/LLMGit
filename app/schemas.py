"""Pydantic schemas for the TruthGit API and LLM structured outputs."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SourceType = Literal["user_message", "document", "api", "manual", "system"]
BranchStatus = Literal["active", "merged", "archived"]
CommitOperation = Literal["add", "update", "retract", "merge", "rollback"]
BeliefVersionStatus = Literal["active", "superseded", "retracted", "hypothetical"]
AnswerMode = Literal[
    "direct_answer",
    "ask_clarification",
    "memory_update_then_answer",
    "historical_answer",
    "branch_answer",
]


class ExtractedClaim(BaseModel):
    """Atomic claim extracted from natural language."""

    subject: str
    predicate: str
    object_value: str = Field(alias="object")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    valid_from: date | None = None
    valid_to: date | None = None
    is_negation: bool = False
    source_quote: str | None = None
    notes: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class ExtractedClaimList(BaseModel):
    """Structured output wrapper for claim extraction."""

    claims: list[ExtractedClaim] = Field(default_factory=list)


class AnswerPlan(BaseModel):
    """Structured answer strategy selected by the model."""

    answer_mode: AnswerMode
    relevant_belief_ids: list[int] = Field(default_factory=list)
    requires_memory_update: bool = False
    proposed_commit_message: str | None = None
    explanation_style: str = "concise"


class MemoryWritePlan(BaseModel):
    """Structured write plan for reviewable memory ingestion."""

    claims: list[ExtractedClaim] = Field(default_factory=list)
    branch_name: str = "main"
    trust_score: float = Field(default=0.7, ge=0.0, le=1.0)
    rationale: str = "Use the default branch and trust policy."


class SourceCreate(BaseModel):
    source_type: SourceType = "user_message"
    source_ref: str | None = None
    excerpt: str
    trust_score: float = Field(default=0.7, ge=0.0, le=1.0)


class SourceRead(SourceCreate):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BranchCreate(BaseModel):
    name: str
    description: str | None = None
    parent_branch_id: int | None = None


class BranchRead(BaseModel):
    id: int
    name: str
    description: str | None
    parent_branch_id: int | None
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CommitRead(BaseModel):
    id: int
    branch_id: int
    parent_commit_id: int | None
    operation_type: str
    message: str
    created_by: str
    model_name: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BeliefRead(BaseModel):
    id: int
    subject: str
    predicate: str
    canonical_key: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BeliefVersionRead(BaseModel):
    id: int
    belief_id: int
    commit_id: int
    branch_id: int
    object_value: str
    normalized_object_value: str
    confidence: float
    valid_from: date | None
    valid_to: date | None
    status: str
    source_id: int
    supersedes_version_id: int | None
    contradiction_group: str | None
    metadata_json: dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BeliefWithVersions(BaseModel):
    belief: BeliefRead
    versions: list[BeliefVersionRead]


class AuditEventRead(BaseModel):
    id: int
    event_type: str
    entity_type: str
    entity_id: int
    entity_key: str | None = None
    payload_json: dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatRequest(BaseModel):
    message: str
    branch_id: int | None = None
    auto_commit: bool = True


class IngestRequest(BaseModel):
    raw_text: str
    source_type: SourceType = "document"
    source_ref: str | None = None
    branch_id: int | None = None
    trust_score: float = Field(default=0.7, ge=0.0, le=1.0)
    auto_commit: bool = True


class Citation(BaseModel):
    belief_id: int
    belief_version_id: int
    subject: str
    predicate: str
    object_value: str
    status: str
    source_id: int
    commit_id: int


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    memory_updated: bool = False
    created_commit_id: int | None = None
    staged_commit_id: str | None = None
    review_required: bool = False
    branch: BranchRead
    warnings: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    extracted_claims: list[ExtractedClaim]
    staged_commit_id: str
    staged_status: str = "pending"
    review_required: bool = False
    memory_updated: bool
    created_commit_id: int | None
    warnings: list[str] = Field(default_factory=list)


class MergeRequest(BaseModel):
    target_branch_id: int | None = None
    message: str = "Merge branch"


class RollbackRequest(BaseModel):
    message: str | None = None


class StagedCommitRead(BaseModel):
    id: str
    branch_id: int
    status: str
    claims_json: list[dict[str, Any]]
    source_type: str
    source_ref: str | None
    source_excerpt: str
    source_trust_score: float
    proposed_commit_message: str
    created_by: str
    model_name: str | None
    review_required: bool
    risk_reasons: list[str]
    warnings_json: list[str]
    reviewer: str | None
    review_notes: str | None
    applied_commit_id: int | None
    created_at: datetime
    reviewed_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class StagedReviewRequest(BaseModel):
    reviewer: str = "user"
    notes: str | None = None
    commit_message: str | None = None


class StagedRejectRequest(BaseModel):
    reviewer: str = "user"
    notes: str | None = None


class CommitResultRead(BaseModel):
    commit: CommitRead
    introduced_versions: list[BeliefVersionRead]
    restored_versions: list[BeliefVersionRead] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
