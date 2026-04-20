"""Run TruthGit as a real memory system on LongMemEval.

This adapter is separate from the prompt-only LongMemEval runner. It ingests
LongMemEval chat sessions into a fresh TruthGit store, uses reviewable staged
commits for extracted claims, answers from TruthGit belief versions plus source
excerpts, and writes official LongMemEval hypothesis JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, Protocol

from openai import OpenAI, OpenAIError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import crud, models
from app.commit_engine import ensure_main_branch
from app.db import Base
from app.llm import openai_strict_json_schema
from app.normalization import canonical_text
from app.schemas import ExtractedClaim, ExtractedClaimList, SourceCreate
from app.tools import approve_staged_commit, stage_belief_changes
from experiments.public_benchmarks.longmemeval import (
    LongMemEvalRecord,
    _with_retries,
    load_records,
    select_records,
)

ExtractionMode = Literal["per_session", "record_batch"]


@dataclass(frozen=True)
class TruthGitMemoryContext:
    """Memory context exposed to the LongMemEval answerer."""

    question_id: str
    question: str
    question_type: str
    question_date: str | None
    belief_versions: list[dict[str, Any]]
    source_excerpts: list[dict[str, Any]]
    audit_event_count: int
    staged_commit_count: int
    commit_count: int
    claim_count: int


@dataclass(frozen=True)
class TruthGitRecordStats:
    """Per-record ingestion and answer statistics."""

    question_id: str
    selected_session_count: int
    claim_count: int
    staged_commit_count: int
    commit_count: int
    audit_event_count: int
    hypothesis: str


@dataclass(frozen=True)
class TruthGitGenerationStats:
    """Summary for a TruthGit LongMemEval hypothesis generation run."""

    output_jsonl: str
    requested_count: int
    generated_count: int
    skipped_existing_count: int
    answer_model: str
    extraction_model: str
    extraction_mode: str
    max_sessions: int | None


class ClaimExtractor(Protocol):
    """Extract claims from LongMemEval session payloads."""

    def extract(
        self,
        *,
        record: LongMemEvalRecord,
        session_payloads: list[dict[str, Any]],
    ) -> list[ExtractedClaim]:
        """Return extracted claims."""


class Answerer(Protocol):
    """Answer a LongMemEval question from TruthGit memory context."""

    def answer(self, *, record: LongMemEvalRecord, context: TruthGitMemoryContext) -> str:
        """Return a hypothesis string."""


class OpenAITruthGitClaimExtractor:
    """Question-aware OpenAI claim extractor for TruthGit LongMemEval ingestion."""

    def __init__(
        self,
        *,
        model: str,
        max_retries: int = 5,
        sleep_seconds: float = 0.0,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for TruthGit LongMemEval extraction")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds

    def extract(
        self,
        *,
        record: LongMemEvalRecord,
        session_payloads: list[dict[str, Any]],
    ) -> list[ExtractedClaim]:
        prompt = {
            "question": record.question,
            "question_type": record.question_type,
            "question_date": record.question_date,
            "sessions": session_payloads,
        }
        response = _with_retries(
            lambda: self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": _truthgit_extraction_system_prompt(),
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "ExtractedClaimList",
                        "schema": openai_strict_json_schema(ExtractedClaimList),
                        "strict": True,
                    }
                },
            ),
            max_retries=self.max_retries,
        )
        output_text = getattr(response, "output_text", None) or _response_text(response)
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        payload = json.loads(output_text)
        _coerce_extracted_claim_dates(payload)
        return ExtractedClaimList.model_validate(payload).claims


class OpenAITruthGitAnswerer:
    """OpenAI answerer that uses TruthGit belief memory and source excerpts."""

    def __init__(
        self,
        *,
        model: str,
        max_output_tokens: int = 256,
        max_retries: int = 5,
        sleep_seconds: float = 0.0,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for TruthGit LongMemEval answering")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds

    def answer(self, *, record: LongMemEvalRecord, context: TruthGitMemoryContext) -> str:
        prompt = {
            "question": record.question,
            "question_type": record.question_type,
            "question_date": record.question_date,
            "truthgit_memory": asdict(context),
        }
        response = _with_retries(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": _truthgit_answer_system_prompt(),
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, default=str)},
                ],
                temperature=0,
                max_tokens=self.max_output_tokens,
            ),
            max_retries=self.max_retries,
        )
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return (response.choices[0].message.content or "").strip()


def generate_truthgit_hypotheses(
    *,
    records: list[LongMemEvalRecord],
    output_jsonl: Path,
    answer_model: str,
    extraction_model: str,
    extraction_mode: ExtractionMode = "per_session",
    max_sessions: int | None = 12,
    max_versions: int = 160,
    max_source_excerpts: int = 12,
    extractor: ClaimExtractor | None = None,
    answerer: Answerer | None = None,
    resume: bool = True,
    trace_dir: Path | None = None,
    sleep_seconds: float = 0.0,
) -> TruthGitGenerationStats:
    """Generate official hypotheses using a fresh TruthGit memory per record."""

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
    existing_ids = _existing_hypothesis_ids(output_jsonl) if resume else set()
    claim_extractor = extractor or OpenAITruthGitClaimExtractor(
        model=extraction_model,
        sleep_seconds=sleep_seconds,
    )
    memory_answerer = answerer or OpenAITruthGitAnswerer(
        model=answer_model,
        sleep_seconds=sleep_seconds,
    )
    generated_count = 0
    skipped_count = 0
    with output_jsonl.open("a" if resume else "w", encoding="utf-8") as handle:
        for idx, record in enumerate(records, start=1):
            if record.question_id in existing_ids:
                skipped_count += 1
                continue
            stats = run_truthgit_record(
                record=record,
                extractor=claim_extractor,
                answerer=memory_answerer,
                extraction_mode=extraction_mode,
                max_sessions=max_sessions,
                max_versions=max_versions,
                max_source_excerpts=max_source_excerpts,
                trace_dir=trace_dir,
            )
            row = {
                "question_id": record.question_id,
                "question_type": record.question_type,
                "hypothesis": stats.hypothesis,
                "model": answer_model,
                "memory_system": "truthgit",
                "truthgit": {
                    "extraction_model": extraction_model,
                    "extraction_mode": extraction_mode,
                    "max_sessions": max_sessions,
                    "claim_count": stats.claim_count,
                    "staged_commit_count": stats.staged_commit_count,
                    "commit_count": stats.commit_count,
                    "audit_event_count": stats.audit_event_count,
                    "selected_session_count": stats.selected_session_count,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            generated_count += 1
            print(
                f"[{idx}/{len(records)}] truthgit wrote {record.question_id} "
                f"claims={stats.claim_count} commits={stats.commit_count}",
                flush=True,
            )
    return TruthGitGenerationStats(
        output_jsonl=str(output_jsonl),
        requested_count=len(records),
        generated_count=generated_count,
        skipped_existing_count=skipped_count,
        answer_model=answer_model,
        extraction_model=extraction_model,
        extraction_mode=extraction_mode,
        max_sessions=max_sessions,
    )


def run_truthgit_record(
    *,
    record: LongMemEvalRecord,
    extractor: ClaimExtractor,
    answerer: Answerer,
    extraction_mode: ExtractionMode,
    max_sessions: int | None,
    max_versions: int,
    max_source_excerpts: int,
    trace_dir: Path | None = None,
) -> TruthGitRecordStats:
    """Ingest one LongMemEval record into TruthGit and answer it."""

    selected_indexes = select_session_indexes(record, max_sessions=max_sessions)
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    Base.metadata.create_all(bind=engine)
    db = session_factory()
    try:
        branch = ensure_main_branch(db)
        db.commit()
        if extraction_mode == "per_session":
            selected_payloads = [session_payload(record, index) for index in selected_indexes]
            for payload in selected_payloads:
                claims = extractor.extract(record=record, session_payloads=[payload])
                claims = [_claim_with_session_date(claim, payload["date"]) for claim in claims]
                _stage_and_approve_claims(
                    db,
                    branch_id=branch.id,
                    record=record,
                    claims=claims,
                    source_ref=f"longmemeval:{record.question_id}:session:{payload['session_index']}",
                    source_excerpt=session_text(payload),
                )
        elif extraction_mode == "record_batch":
            selected_payloads = [session_payload(record, index) for index in selected_indexes]
            claims = extractor.extract(record=record, session_payloads=selected_payloads)
            _stage_and_approve_claims(
                db,
                branch_id=branch.id,
                record=record,
                claims=claims,
                source_ref=f"longmemeval:{record.question_id}:selected-history",
                source_excerpt="\n\n".join(session_text(payload) for payload in selected_payloads),
            )
        else:
            raise ValueError("extraction_mode must be 'per_session' or 'record_batch'")

        context = build_truthgit_context(
            db,
            record=record,
            max_versions=max_versions,
            max_source_excerpts=max_source_excerpts,
            selected_session_payloads=selected_payloads,
        )
        hypothesis = answerer.answer(record=record, context=context)
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"{_safe_filename(record.question_id)}.truthgit-trace.json"
            trace_path.write_text(
                json.dumps({"context": asdict(context), "hypothesis": hypothesis}, indent=2, default=str),
                encoding="utf-8",
            )
        return TruthGitRecordStats(
            question_id=record.question_id,
            selected_session_count=len(selected_indexes),
            claim_count=context.claim_count,
            staged_commit_count=context.staged_commit_count,
            commit_count=context.commit_count,
            audit_event_count=context.audit_event_count,
            hypothesis=hypothesis,
        )
    finally:
        db.close()
        engine.dispose()


def build_truthgit_context(
    db: Session,
    *,
    record: LongMemEvalRecord,
    max_versions: int,
    max_source_excerpts: int,
    selected_session_payloads: list[dict[str, Any]] | None = None,
) -> TruthGitMemoryContext:
    """Serialize ranked TruthGit memory context for answering."""

    versions = list(db.scalars(select(models.BeliefVersion).order_by(models.BeliefVersion.id.desc())))
    version_rows = [_version_context_row(db, version) for version in versions]
    version_rows = _rank_rows(record.question, version_rows, max_items=max_versions)
    sources = list(db.scalars(select(models.Source).order_by(models.Source.id.desc())))
    source_rows = [
        {
            "source_id": source.id,
            "source_type": source.source_type,
            "source_ref": source.source_ref,
            "trust_score": source.trust_score,
            "excerpt": _focused_excerpt(source.excerpt, record.question, max_chars=2600),
        }
        for source in sources
    ]
    source_rows.extend(
        _selected_session_source_rows(
            record,
            selected_session_payloads or [],
        )
    )
    source_rows = _dedupe_source_rows(source_rows)
    source_rows = _rank_rows(record.question, source_rows, max_items=max_source_excerpts)
    staged_count = len(list(db.scalars(select(models.StagedCommit.id))))
    commit_count = len(list(db.scalars(select(models.Commit.id))))
    audit_count = len(list(db.scalars(select(models.AuditEvent.id))))
    return TruthGitMemoryContext(
        question_id=record.question_id,
        question=record.question,
        question_type=record.question_type,
        question_date=record.question_date,
        belief_versions=version_rows,
        source_excerpts=source_rows,
        audit_event_count=audit_count,
        staged_commit_count=staged_count,
        commit_count=commit_count,
        claim_count=len(versions),
    )


def _selected_session_source_rows(
    record: LongMemEvalRecord,
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        session_index = payload.get("session_index")
        rows.append(
            {
                "source_id": None,
                "source_type": "selected_session",
                "source_ref": f"longmemeval:{record.question_id}:session:{session_index}",
                "session_index": session_index,
                "session_date": payload.get("date"),
                "trust_score": 0.75,
                "excerpt": _focused_session_excerpt(payload, record.question, max_chars=2400),
            }
        )
    return rows


def _dedupe_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("source_ref") or row.get("excerpt") or row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _truthgit_extraction_system_prompt() -> str:
    return (
        "Extract durable atomic memory claims from chat history for TruthGit. "
        "Use only explicit information in the sessions. Extract facts, preferences, "
        "plans, updates, and user-specific state that could help answer the question. "
        "Preserve exact answer-bearing details: names, titles, store names, venue names, "
        "studio names, playlist names, dates, durations, locations, product names, and "
        "relationship labels. Choose concise model-generated snake_case predicates that "
        "describe the relation in the source text. Examples include lives_in, works_at, "
        "prefers, has_pet, appointment_date, event_title, playlist_name, and commute_duration, "
        "but these are examples, not a closed predicate list. "
        "When later sessions update an earlier fact, keep the same subject and predicate "
        "and set valid_from to the session date when possible. Return claims in chronological "
        "order. Return no claims for unsupported guesses."
    )


def _truthgit_answer_system_prompt() -> str:
    return (
        "You answer LongMemEval personal-memory questions from TruthGit memory. "
        "Use only the provided TruthGit belief versions and source excerpts; do not use "
        "outside knowledge or generic recommendations. Prefer active belief versions for "
        "current truth, superseded versions for historical truth, and never use retracted "
        "versions as current justification. Use source excerpts as exact evidence when "
        "extracted beliefs are incomplete or too generic. Resolve updates by dates and "
        "question date. Answer the user's remembered fact with the exact value when it is "
        "supported, such as the store, title, venue, studio, duration, date, or name. If "
        "the memory does not support an exact personal-memory answer, say the answer is "
        "unknown. Return one concise final answer, not advice."
    )


def select_session_indexes(record: LongMemEvalRecord, *, max_sessions: int | None) -> list[int]:
    """Select session indexes without using answer labels."""

    total = len(record.haystack_sessions)
    if max_sessions is None or max_sessions >= total:
        return list(range(total))
    query_tokens = _tokens(record.question)
    scored: list[tuple[float, int]] = []
    for index, session in enumerate(record.haystack_sessions):
        session_joined = " ".join(str(turn.get("content", "")) for turn in session)
        session_tokens = _tokens(session_joined)
        overlap = len(query_tokens.intersection(session_tokens))
        recency = index / max(1, total)
        scored.append((overlap + 0.05 * recency, index))
    selected = [index for _, index in sorted(scored, reverse=True)[:max_sessions]]
    return sorted(selected)


def session_payload(record: LongMemEvalRecord, index: int) -> dict[str, Any]:
    """Return a non-leaking session payload."""

    session = record.haystack_sessions[index]
    return {
        "session_index": index,
        "session_id": record.haystack_session_ids[index] if index < len(record.haystack_session_ids) else str(index),
        "date": record.haystack_dates[index] if index < len(record.haystack_dates) else None,
        "turns": [
            {"role": str(turn.get("role", "unknown")), "content": str(turn.get("content", ""))}
            for turn in session
        ],
    }


def session_text(payload: dict[str, Any]) -> str:
    """Render one session payload as compact text."""

    lines = [f"[Session {payload['session_index']} | {payload.get('date') or 'unknown'}]"]
    for turn in payload["turns"]:
        lines.append(f"{turn['role']}: {turn['content']}")
    return "\n".join(lines)


def _stage_and_approve_claims(
    db: Session,
    *,
    branch_id: int,
    record: LongMemEvalRecord,
    claims: list[ExtractedClaim],
    source_ref: str,
    source_excerpt: str,
) -> None:
    if not claims:
        return
    staged = stage_belief_changes(
        db,
        claims=claims,
        branch_id=branch_id,
        source=SourceCreate(
            source_type="document",
            source_ref=source_ref,
            excerpt=source_excerpt,
            trust_score=0.75,
        ),
        proposed_commit_message=f"LongMemEval memory ingest for {record.question_id}",
        created_by="longmemeval-truthgit",
        model_name="longmemeval-truthgit",
    )
    approve_staged_commit(
        db,
        staged_commit_id=staged.id,
        reviewer="longmemeval-truthgit",
        notes="Automated benchmark ingestion approval.",
        commit_message=staged.proposed_commit_message,
        model_name="longmemeval-truthgit",
    )
    db.commit()


def _claim_with_session_date(claim: ExtractedClaim, session_date: str | None) -> ExtractedClaim:
    if claim.valid_from is not None:
        return claim
    parsed = _parse_session_date(session_date)
    if parsed is None:
        return claim
    return claim.model_copy(update={"valid_from": parsed})


def _version_context_row(db: Session, version: models.BeliefVersion) -> dict[str, Any]:
    belief = crud.get_belief(db, version.belief_id)
    source = db.get(models.Source, version.source_id)
    return {
        "belief_version_id": version.id,
        "belief_id": version.belief_id,
        "subject": belief.subject if belief else None,
        "predicate": belief.predicate if belief else None,
        "object_value": version.object_value,
        "status": version.status,
        "confidence": version.confidence,
        "valid_from": version.valid_from.isoformat() if version.valid_from else None,
        "valid_to": version.valid_to.isoformat() if version.valid_to else None,
        "source_ref": source.source_ref if source else None,
        "source_trust_score": source.trust_score if source else None,
        "supersedes_version_id": version.supersedes_version_id,
        "contradiction_group": version.contradiction_group,
        "source_quote": version.metadata_json.get("source_quote") if version.metadata_json else None,
    }


def _rank_rows(question: str, rows: list[dict[str, Any]], *, max_items: int) -> list[dict[str, Any]]:
    query_tokens = _tokens(question)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        text = json.dumps(row, ensure_ascii=False, default=str)
        overlap = len(query_tokens.intersection(_tokens(text)))
        status_boost = 0.25 if row.get("status") in {"active", "hypothetical"} else 0.0
        scored.append((overlap + status_boost, index, row))
    ranked = [row for _, _, row in sorted(scored, key=lambda item: (item[0], -item[1]), reverse=True)]
    return ranked[:max_items]


def _response_text(response: Any) -> str:
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


def _tokens(text: str) -> set[str]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "is",
        "are",
        "was",
        "were",
        "what",
        "when",
        "where",
        "who",
        "which",
        "does",
        "do",
        "did",
        "my",
        "me",
        "i",
        "you",
    }
    return {token for token in re.findall(r"[a-z0-9]+", canonical_text(text)) if token not in stop}


def _parse_session_date(value: str | None) -> date | None:
    if not value:
        return None
    match = re.search(r"\d{4}-\d{2}-\d{2}", value)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(0))
    except ValueError:
        return None


def _coerce_extracted_claim_dates(payload: dict[str, Any]) -> None:
    claims = payload.get("claims")
    if not isinstance(claims, list):
        return
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        for key in ("valid_from", "valid_to"):
            claim[key] = _coerce_schema_date(claim.get(key))


def _coerce_schema_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value.isoformat()
    text = str(value)
    match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _compact_text(value: str, *, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def _focused_excerpt(value: str, question: str, *, max_chars: int) -> str:
    """Return source snippets centered on question terms instead of a prefix."""

    cleaned = value.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    query_tokens = _tokens(question)
    if not query_tokens:
        return _compact_text(cleaned, max_chars=max_chars)
    snippets: list[tuple[int, int, str]] = []
    for line_index, line in enumerate(cleaned.splitlines()):
        line_tokens = _tokens(line)
        overlap = query_tokens.intersection(line_tokens)
        if not overlap:
            continue
        snippets.append((len(overlap), -line_index, _line_window(line, overlap, max_chars=900)))
    if not snippets:
        return _compact_text(cleaned, max_chars=max_chars)
    selected: list[str] = []
    used_chars = 0
    for _score, _line_rank, snippet in sorted(snippets, reverse=True):
        if snippet in selected:
            continue
        projected = used_chars + len(snippet) + (4 if selected else 0)
        if projected > max_chars:
            remaining = max_chars - used_chars - (4 if selected else 0)
            if remaining > 160:
                selected.append(_compact_text(snippet, max_chars=remaining))
            break
        selected.append(snippet)
        used_chars = projected
    return "\n...\n".join(selected)


def _focused_session_excerpt(payload: dict[str, Any], question: str, *, max_chars: int) -> str:
    """Return a compact session excerpt that preserves user-stated memory facts."""

    header = f"[Session {payload.get('session_index')} | {payload.get('date') or 'unknown'}]"
    user_blocks = [
        f"{turn.get('role', 'unknown')}: {str(turn.get('content', '')).strip()}"
        for turn in payload.get("turns", [])
        if str(turn.get("role", "")).lower() == "user" and str(turn.get("content", "")).strip()
    ]
    if user_blocks:
        user_only = "\n".join([header, *user_blocks])
        if len(user_only) <= max_chars:
            return user_only
    return _focused_excerpt(session_text(payload), question, max_chars=max_chars)


def _line_window(line: str, overlap: set[str], *, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", line).strip()
    if len(compact) <= max_chars:
        return compact
    lowered = compact.lower()
    positions = [
        lowered.find(token)
        for token in sorted(overlap, key=len, reverse=True)
        if lowered.find(token) >= 0
    ]
    center = min(positions) if positions else 0
    start = max(0, center - max_chars // 3)
    end = min(len(compact), start + max_chars)
    start = max(0, end - max_chars)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(compact) else ""
    return prefix + compact[start:end].strip() + suffix


def _existing_hypothesis_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(json.loads(line).get("question_id", ""))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:160]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--data", required=True, type=Path)
    generate_parser.add_argument(
        "--output-jsonl",
        default=Path("experiments/public_results/longmemeval/truthgit.hypotheses.jsonl"),
        type=Path,
    )
    generate_parser.add_argument("--answer-model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    generate_parser.add_argument("--extraction-model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    generate_parser.add_argument("--extraction-mode", choices=["per_session", "record_batch"], default="per_session")
    generate_parser.add_argument("--max-sessions", type=int, default=12)
    generate_parser.add_argument("--max-versions", type=int, default=160)
    generate_parser.add_argument("--max-source-excerpts", type=int, default=12)
    generate_parser.add_argument("--limit", type=int, default=None)
    generate_parser.add_argument("--sample-size", type=int, default=None)
    generate_parser.add_argument("--sample-seed", type=int, default=0)
    generate_parser.add_argument("--no-resume", action="store_true")
    generate_parser.add_argument("--trace-dir", default=None, type=Path)
    generate_parser.add_argument("--sleep-seconds", type=float, default=0.0)

    args = parser.parse_args()
    records = select_records(
        load_records(args.data),
        limit=args.limit,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    if args.command == "generate":
        stats = generate_truthgit_hypotheses(
            records=records,
            output_jsonl=args.output_jsonl,
            answer_model=args.answer_model,
            extraction_model=args.extraction_model,
            extraction_mode=args.extraction_mode,
            max_sessions=args.max_sessions,
            max_versions=args.max_versions,
            max_source_excerpts=args.max_source_excerpts,
            resume=not args.no_resume,
            trace_dir=args.trace_dir,
            sleep_seconds=args.sleep_seconds,
        )
        print(json.dumps(asdict(stats), indent=2))
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
