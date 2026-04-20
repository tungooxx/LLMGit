"""Comparable systems for changing-world memory experiments."""

from __future__ import annotations

import math
import re
from collections import Counter
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
    unresolved_conflict: bool = False
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

    def memory_context(self, question: BenchmarkQuestion, *, max_items: int = 20) -> dict[str, object]:
        """Return the context this memory system would expose to a shared reader."""


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
    confidence: float
    text: str
    valid_from: object | None = None
    active: bool = True

    @property
    def document(self) -> str:
        return (
            f"{self.subject} {self.predicate} {self.object_value}. "
            f"Source {self.source_ref}. Branch {self.branch_name}. {self.text}"
        )


class NaiveChatHistoryBaseline:
    """Baseline that treats memory as unversioned chat history."""

    name = "naive_chat_history"

    def __init__(self) -> None:
        self.facts: list[MemoryFact] = []

    def reset(self) -> None:
        self.facts.clear()

    def ingest_event(self, event: BenchmarkEvent) -> None:
        if _is_fact_event(event):
            self.facts.append(_fact_from_event(event))
        # This baseline does not understand rollback, merge, branch isolation, or warnings.

    def answer(self, question: BenchmarkQuestion) -> SystemAnswer:
        matches = [
            fact
            for fact in self.facts
            if fact.subject == question.subject and fact.predicate == question.predicate and fact.active
        ]
        chosen = matches[-1] if matches else None
        historical = [fact.object_value for fact in matches]
        return _answer_from_fact(self.name, question, chosen, historical)

    def memory_context(self, question: BenchmarkQuestion, *, max_items: int = 20) -> dict[str, object]:
        matches = [
            fact
            for fact in self.facts
            if fact.subject == question.subject and fact.predicate == question.predicate
        ]
        return {
            "memory_system": self.name,
            "retrieval_policy": "flat chronological subject+predicate history; no rollback, merge, branch, or warning state",
            "facts": [_fact_context_row(fact) for fact in matches[-max_items:]],
        }


class SimpleRagBaseline:
    """Baseline that retrieves by subject/predicate and source trust only."""

    name = "simple_rag"

    def __init__(self) -> None:
        self.facts: list[MemoryFact] = []

    def reset(self) -> None:
        self.facts.clear()

    def ingest_event(self, event: BenchmarkEvent) -> None:
        if _is_fact_event(event):
            self.facts.append(_fact_from_event(event))

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
        return _answer_from_fact(self.name, question, chosen, [fact.object_value for fact in matches])

    def memory_context(self, question: BenchmarkQuestion, *, max_items: int = 20) -> dict[str, object]:
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
        ranked = sorted(matches, key=lambda fact: (fact.trust_score, fact.event_id), reverse=True)
        return {
            "memory_system": self.name,
            "retrieval_policy": "lexical subject/predicate retrieval ranked by source trust",
            "facts": [_fact_context_row(fact) for fact in ranked[:max_items]],
        }


class EmbeddingRagBaseline:
    """TF-IDF embedding RAG baseline with cosine retrieval over memory chunks."""

    name = "embedding_rag"

    def __init__(self, top_k: int = 5) -> None:
        self.facts: list[MemoryFact] = []
        self.top_k = top_k

    def reset(self) -> None:
        self.facts.clear()

    def ingest_event(self, event: BenchmarkEvent) -> None:
        if _is_fact_event(event):
            self.facts.append(_fact_from_event(event))

    def answer(self, question: BenchmarkQuestion) -> SystemAnswer:
        if not self.facts:
            return _answer_from_fact(self.name, question, None, [])
        query = (
            f"{question.prompt} {question.subject} {question.predicate} "
            f"{question.branch_name}"
        )
        ranked = self._rank(query)
        matches = [
            fact
            for fact in ranked[: self.top_k]
            if fact.subject == question.subject and fact.predicate == question.predicate
        ]
        if not matches:
            matches = [
                fact
                for fact in self.facts
                if fact.subject == question.subject and fact.predicate == question.predicate
            ]
        chosen = max(
            matches,
            key=lambda fact: (_date_rank(fact.valid_from), fact.trust_score * fact.confidence, fact.event_id),
            default=None,
        )
        return _answer_from_fact(self.name, question, chosen, [fact.object_value for fact in matches])

    def memory_context(self, question: BenchmarkQuestion, *, max_items: int = 20) -> dict[str, object]:
        query = (
            f"{question.prompt} {question.subject} {question.predicate} "
            f"{question.branch_name}"
        )
        ranked = self._rank(query) if self.facts else []
        return {
            "memory_system": self.name,
            "retrieval_policy": "local TF-IDF cosine retrieval over flat memory chunks",
            "facts": [_fact_context_row(fact) for fact in ranked[: min(max_items, self.top_k)]],
        }

    def _rank(self, query: str) -> list[MemoryFact]:
        docs = [fact.document for fact in self.facts]
        doc_tokens = [_tokens(doc) for doc in docs]
        query_tokens = _tokens(query)
        vocabulary = sorted(set(query_tokens).union(*(set(tokens) for tokens in doc_tokens)))
        idf = _idf(vocabulary, doc_tokens)
        query_vec = _tfidf(query_tokens, vocabulary, idf)
        scored = []
        for fact, tokens in zip(self.facts, doc_tokens, strict=True):
            scored.append((_cosine(query_vec, _tfidf(tokens, vocabulary, idf)), fact))
        return [fact for _, fact in sorted(scored, key=lambda item: item[0], reverse=True)]


class TruthGitSystem:
    """Benchmark adapter that uses the real TruthGit commit engine."""

    def __init__(
        self,
        *,
        name: str = "truthgit",
        support_branches: bool = True,
        support_rollback: bool = True,
        trust_aware: bool = True,
        review_gate: bool = True,
    ) -> None:
        self.name = name
        self.support_branches = support_branches
        self.support_rollback = support_rollback
        self.trust_aware = trust_aware
        self.review_gate = review_gate
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
        if _is_fact_event(event):
            branch_id = self._ensure_branch(event.branch_name if self.support_branches else "main")
            source = crud.create_source(
                db,
                source_type="document",
                source_ref=event.source_ref,
                excerpt=event.text,
                trust_score=event.trust_score if self.trust_aware else 0.5,
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
            self.warnings_by_event[event.event_id] = result.warnings if self.review_gate else []
            db.commit()
            return

        if event.event_type == "rollback" and event.rollback_target_event_id:
            if not self.support_rollback:
                self.warnings_by_event[event.event_id] = []
                return
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
            if not self.support_branches:
                return
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
        branch_id = self._ensure_branch(question.branch_name if self.support_branches else "main")
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
        historical_versions = [
            version
            for version in sorted(
                crud.list_belief_versions(db, belief.id),
                key=lambda version: (_date_rank(version.valid_from), version.id),
            )
            if version.status != "retracted"
        ]
        chosen = (
            self._choose_as_of(db, historical_versions, question.as_of)
            if question.as_of is not None
            else self._choose_current(db, current)
        )
        unresolved_conflict = self._has_unresolved_conflict(current)
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
            conflict_resolved=bool(
                chosen
                and chosen.normalized_object_value
                == normalize_object_value(question.expected_object_value or "")
            ),
            unresolved_conflict=unresolved_conflict,
            branch_name=question.branch_name,
        )

    def memory_context(self, question: BenchmarkQuestion, *, max_items: int = 20) -> dict[str, object]:
        db = self._db()
        branch_id = self._ensure_branch(question.branch_name if self.support_branches else "main")
        belief = crud.get_belief_by_subject_predicate(
            db,
            subject=question.subject,
            predicate=question.predicate,
        )
        if belief is None:
            versions: list[object] = []
            current: list[object] = []
        else:
            versions = sorted(
                crud.list_belief_versions(db, belief.id),
                key=lambda version: (_date_rank(version.valid_from), version.id),
            )
            current = crud.get_current_versions(db, belief_id=belief.id, branch_id=branch_id)
        return {
            "memory_system": self.name,
            "retrieval_policy": "TruthGit belief versions with branch, status, provenance, rollback, conflict, and warning state",
            "branch_name": question.branch_name,
            "current_versions": [
                _version_context_row(db, version)
                for version in current[:max_items]
            ],
            "belief_timeline": [
                _version_context_row(db, version)
                for version in versions[:max_items]
            ],
            "warnings_by_event": dict(self.warnings_by_event),
        }

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

    def _choose_current(self, db: Session, versions: list[object]) -> object | None:
        if not versions:
            return None
        if self.trust_aware:
            return max(
                versions,
                key=lambda version: (
                    _date_rank(version.valid_from),
                    crud.source_trust(db, version.source_id) * version.confidence,
                    version.id,
                ),
            )
        return max(versions, key=lambda version: (_date_rank(version.valid_from), version.id))

    def _has_unresolved_conflict(self, versions: list[object]) -> bool:
        conflict_objects = {
            getattr(version, "normalized_object_value", "")
            for version in versions
            if getattr(version, "contradiction_group", None)
        }
        return len(conflict_objects) > 1

    def _choose_as_of(self, db: Session, versions: list[object], as_of: object | None) -> object | None:
        eligible = [
            version
            for version in versions
            if _date_rank(version.valid_from) <= _date_rank(as_of)
            and (version.valid_to is None or _date_rank(version.valid_to) >= _date_rank(as_of))
        ]
        if not eligible:
            return None
        return self._choose_current(db, eligible)


def primary_systems() -> list[MemorySystem]:
    """Return the first paper-table systems."""

    return [
        NaiveChatHistoryBaseline(),
        SimpleRagBaseline(),
        EmbeddingRagBaseline(),
        TruthGitSystem(),
    ]


def ablation_systems() -> list[MemorySystem]:
    """Return TruthGit ablations for structural attribution."""

    return [
        TruthGitSystem(name="truthgit_no_branches", support_branches=False),
        TruthGitSystem(name="truthgit_no_rollback", support_rollback=False),
        TruthGitSystem(name="truthgit_no_review_gate", review_gate=False),
        TruthGitSystem(name="truthgit_no_trust_scoring", trust_aware=False, review_gate=False),
    ]


def default_systems(*, include_ablations: bool = False) -> list[MemorySystem]:
    """Return systems compared by the benchmark script."""

    systems = primary_systems()
    if include_ablations:
        systems.extend(ablation_systems())
    return systems


def _is_fact_event(event: BenchmarkEvent) -> bool:
    return (
        event.event_type in {"fact", "bad_fact", "branch_fact"}
        and event.subject is not None
        and event.predicate is not None
        and event.object_value is not None
    )


def _fact_from_event(event: BenchmarkEvent) -> MemoryFact:
    return MemoryFact(
        event_id=event.event_id,
        subject=event.subject or "",
        predicate=event.predicate or "",
        object_value=event.object_value or "",
        branch_name=event.branch_name,
        source_ref=event.source_ref,
        trust_score=event.trust_score,
        confidence=event.confidence,
        text=event.text,
        valid_from=event.valid_from,
    )


def _answer_from_fact(
    system_name: str,
    question: BenchmarkQuestion,
    chosen: MemoryFact | None,
    historical_objects: list[str],
) -> SystemAnswer:
    return SystemAnswer(
        question_id=question.question_id,
        system_name=system_name,
        answer_text=chosen.object_value if chosen else "unknown",
        object_value=chosen.object_value if chosen else None,
        historical_objects=historical_objects,
        source_ref=chosen.source_ref if chosen else None,
        had_low_trust_warning=False,
        conflict_resolved=bool(
            chosen and chosen.object_value.lower() == (question.expected_object_value or "").lower()
        ),
        branch_name=question.branch_name,
    )


def _fact_context_row(fact: MemoryFact) -> dict[str, object]:
    return {
        "event_id": fact.event_id,
        "subject": fact.subject,
        "predicate": fact.predicate,
        "object_value": fact.object_value,
        "branch_name": fact.branch_name,
        "source_ref": fact.source_ref,
        "trust_score": fact.trust_score,
        "confidence": fact.confidence,
        "valid_from": fact.valid_from,
        "active": fact.active,
        "text": fact.text,
    }


def _version_context_row(db: Session, version: object) -> dict[str, object]:
    belief = crud.get_belief(db, version.belief_id)
    source = db.get(models.Source, version.source_id)
    branch = db.get(models.Branch, version.branch_id)
    return {
        "belief_version_id": version.id,
        "belief_id": version.belief_id,
        "subject": belief.subject if belief else None,
        "predicate": belief.predicate if belief else None,
        "object_value": version.object_value,
        "status": version.status,
        "branch_id": version.branch_id,
        "branch_name": branch.name if branch else None,
        "commit_id": version.commit_id,
        "source_ref": source.source_ref if source else None,
        "source_trust_score": source.trust_score if source else None,
        "confidence": version.confidence,
        "valid_from": version.valid_from,
        "valid_to": version.valid_to,
        "supersedes_version_id": version.supersedes_version_id,
        "contradiction_group": version.contradiction_group,
    }


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _idf(vocabulary: list[str], docs: list[list[str]]) -> dict[str, float]:
    total = len(docs)
    return {
        term: math.log((1 + total) / (1 + sum(1 for doc in docs if term in doc))) + 1.0
        for term in vocabulary
    }


def _tfidf(tokens: list[str], vocabulary: list[str], idf: dict[str, float]) -> list[float]:
    counts = Counter(tokens)
    total = max(1, len(tokens))
    return [(counts[term] / total) * idf[term] for term in vocabulary]


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _date_rank(value: object | None) -> int:
    if value is None:
        return 0
    return int(getattr(value, "toordinal", lambda: 0)())
