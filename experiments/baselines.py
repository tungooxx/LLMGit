"""Comparable systems for changing-world memory experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import crud, models
from app.commit_engine import apply_claims, create_branch, ensure_main_branch, merge_branch, rollback_commit
from app.db import Base
from app.normalization import NormalizedClaim, normalize_object_value

from experiments.benchmark import BenchmarkEvent, BenchmarkQuestion


@dataclass
class SystemAnswer:
    """Structured answer emitted by a benchmarked memory system."""

    question_id: str
    system_name: str
    answer_text: str
    object_value: str | None = None
    historical_objects: list[str] = field(default_factory=list)
    source_ref: str | None = None
    had_low_trust_warning: bool = False
    conflict_resolved: bool = False
    branch_name: str = "main"


class MemorySystem(Protocol):
    """Protocol implemented by all benchmark memory systems."""

    name: str

    def reset(self) -> None:
        """Reset all memory state."""

    def ingest_event(self, event: BenchmarkEvent) -> None:
        """Apply one benchmark event."""

    def answer(self, question: BenchmarkQuestion) -> SystemAnswer:
        """Answer one structured benchmark question."""


@dataclass
class MemoryFact:
    """Flat baseline fact record."""

    event_id: str
    subject: str
    predicate: str
    object_value: str
    branch_name: str
    source_ref: str
    trust_score: float
    active: bool = True


class NaiveChatHistoryBaseline:
    """Baseline that treats memory as unversioned chat history."""

    name = "naive_chat_history"

    def __init__(self) -> None:
        self.facts: list[MemoryFact] = []

    def reset(self) -> None:
        self.facts.clear()

    def ingest_event(self, event: BenchmarkEvent) -> None:
        if event.event_type in {"fact", "bad_fact", "branch_fact"} and event.subject and event.predicate and event.object_value:
            self.facts.append(
                MemoryFact(
                    event_id=event.event_id,
                    subject=event.subject,
                    predicate=event.predicate,
                    object_value=event.object_value,
                    branch_name=event.branch_name,
                    source_ref=event.source_ref,
                    trust_score=event.trust_score,
                )
            )
        # This baseline does not understand rollback or merge as state operations.

    def answer(self, question: BenchmarkQuestion) -> SystemAnswer:
        matches = [
            fact
            for fact in self.facts
            if fact.subject == question.subject and fact.predicate == question.predicate and fact.active
        ]
        chosen = matches[-1] if matches else None
        historical = [fact.object_value for fact in matches]
        return SystemAnswer(
            question_id=question.question_id,
            system_name=self.name,
            answer_text=chosen.object_value if chosen else "unknown",
            object_value=chosen.object_value if chosen else None,
            historical_objects=historical,
            source_ref=chosen.source_ref if chosen else None,
            had_low_trust_warning=False,
            conflict_resolved=False,
            branch_name=question.branch_name,
        )


class SimpleRagBaseline:
    """Baseline that retrieves the highest lexical-overlap memory chunk."""

    name = "simple_rag"

    def __init__(self) -> None:
        self.facts: list[MemoryFact] = []

    def reset(self) -> None:
        self.facts.clear()

    def ingest_event(self, event: BenchmarkEvent) -> None:
        if event.event_type in {"fact", "bad_fact", "branch_fact"} and event.subject and event.predicate and event.object_value:
            self.facts.append(
                MemoryFact(
                    event_id=event.event_id,
                    subject=event.subject,
                    predicate=event.predicate,
                    object_value=event.object_value,
                    branch_name=event.branch_name,
                    source_ref=event.source_ref,
                    trust_score=event.trust_score,
                )
            )

    def answer(self, question: BenchmarkQuestion) -> SystemAnswer:
        matches = [
            fact
            for fact in self.facts
            if fact.subject.lower() in question.prompt.lower() and fact.predicate == question.predicate
        ]
        if not matches:
            matches = [
                fact
                for fact in self.facts
                if fact.subject == question.subject and fact.predicate == question.predicate
            ]
        chosen = max(matches, key=lambda fact: (fact.trust_score, fact.event_id), default=None)
        historical = [fact.object_value for fact in matches]
        return SystemAnswer(
            question_id=question.question_id,
            system_name=self.name,
            answer_text=chosen.object_value if chosen else "unknown",
            object_value=chosen.object_value if chosen else None,
            historical_objects=historical,
            source_ref=chosen.source_ref if chosen else None,
            had_low_trust_warning=False,
            conflict_resolved=False,
            branch_name=question.branch_name,
        )


class TruthGitSystem:
    """Benchmark adapter that uses the real TruthGit commit engine."""

    name = "truthgit"

    def __init__(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        self.session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
        self.db: Session | None = None
        self.branch_ids: dict[str, int] = {}
        self.commit_ids_by_event: dict[str, int] = {}
        self.warnings_by_event: dict[str, list[str]] = {}

    def reset(self) -> None:
        if self.db is not None:
            self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        self.db = self.session_factory()
        main = ensure_main_branch(self.db)
        self.db.commit()
        self.branch_ids = {"main": main.id}
        self.commit_ids_by_event.clear()
        self.warnings_by_event.clear()

    def ingest_event(self, event: BenchmarkEvent) -> None:
        db = self._db()
        if event.event_type in {"fact", "bad_fact", "branch_fact"}:
            branch_id = self._ensure_branch(event.branch_name)
            source = crud.create_source(
                db,
                source_type="document",
                source_ref=event.source_ref,
                excerpt=event.text,
                trust_score=event.trust_score,
            )
            claim = NormalizedClaim(
                subject=event.subject or "",
                predicate=event.predicate or "",
                object_value=event.object_value or "",
                normalized_object_value=normalize_object_value(event.object_value or ""),
                confidence=event.confidence,
                valid_from=event.valid_from,
                valid_to=event.valid_to,
            )
            result = apply_claims(
                db,
                claims=[claim],
                branch_id=branch_id,
                source=source,
                message=event.text,
                created_by="benchmark",
            )
            self.commit_ids_by_event[event.event_id] = result.commit.id
            self.warnings_by_event[event.event_id] = result.warnings
            db.commit()
            return

        if event.event_type == "rollback" and event.rollback_target_event_id:
            target_commit_id = self.commit_ids_by_event[event.rollback_target_event_id]
            result = rollback_commit(
                db,
                commit_id=target_commit_id,
                message=event.text,
                created_by="benchmark",
            )
            self.commit_ids_by_event[event.event_id] = result.commit.id
            self.warnings_by_event[event.event_id] = result.warnings
            db.commit()
            return

        if event.event_type == "merge" and event.merge_source_branch:
            source_branch_id = self._ensure_branch(event.merge_source_branch)
            target_branch_id = self._ensure_branch(event.merge_target_branch)
            result = merge_branch(
                db,
                source_branch_id=source_branch_id,
                target_branch_id=target_branch_id,
                message=event.text,
                created_by="benchmark",
            )
            self.commit_ids_by_event[event.event_id] = result.commit.id
            self.warnings_by_event[event.event_id] = result.warnings
            db.commit()

    def answer(self, question: BenchmarkQuestion) -> SystemAnswer:
        db = self._db()
        branch_id = self._ensure_branch(question.branch_name)
        belief = crud.get_belief_by_subject_predicate(
            db,
            subject=question.subject,
            predicate=question.predicate,
        )
        if belief is None:
            return SystemAnswer(
                question_id=question.question_id,
                system_name=self.name,
                answer_text="unknown",
                branch_name=question.branch_name,
            )
        current = crud.get_current_versions(db, belief_id=belief.id, branch_id=branch_id)
        chosen = self._choose_current(db, current)
        historical_versions = crud.list_belief_versions(db, belief.id)
        source_ref = self._source_ref(chosen.source_id) if chosen is not None else None
        related_warnings = self.warnings_by_event.get(question.related_event_id or "", [])
        answer_text = chosen.object_value if chosen else "unknown"
        return SystemAnswer(
            question_id=question.question_id,
            system_name=self.name,
            answer_text=answer_text,
            object_value=chosen.object_value if chosen else None,
            historical_objects=[version.object_value for version in historical_versions],
            source_ref=source_ref,
            had_low_trust_warning=any("Low-trust" in warning for warning in related_warnings),
            conflict_resolved=bool(chosen and chosen.normalized_object_value == normalize_object_value(question.expected_object_value or "")),
            branch_name=question.branch_name,
        )

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None

    def _db(self) -> Session:
        if self.db is None:
            self.reset()
        assert self.db is not None
        return self.db

    def _ensure_branch(self, name: str) -> int:
        if name in self.branch_ids:
            return self.branch_ids[name]
        db = self._db()
        branch = crud.get_branch_by_name(db, name)
        if branch is None:
            branch = create_branch(db, name=name, description=f"Benchmark branch {name}")
            db.commit()
        self.branch_ids[name] = branch.id
        return branch.id

    def _source_ref(self, source_id: int) -> str | None:
        source = self._db().get(models.Source, source_id)
        return source.source_ref if source else None

    @staticmethod
    def _choose_current(db: Session, versions: list[object]) -> object | None:
        if not versions:
            return None
        return max(
            versions,
            key=lambda version: (
                crud.source_trust(db, version.source_id) * version.confidence,
                version.id,
            ),
        )


def default_systems() -> list[MemorySystem]:
    """Return the systems compared by the benchmark script."""

    return [NaiveChatHistoryBaseline(), SimpleRagBaseline(), TruthGitSystem()]
