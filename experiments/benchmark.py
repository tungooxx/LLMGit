"""Synthetic changing-world benchmark generation for TruthGit research."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal


EventType = Literal["fact", "bad_fact", "branch_fact", "rollback", "merge"]
MetricName = Literal[
    "current_truth_accuracy",
    "historical_truth_accuracy",
    "provenance_accuracy",
    "rollback_recovery_rate",
    "branch_isolation_score",
    "merge_conflict_resolution_score",
    "low_trust_warning_rate",
]


@dataclass(frozen=True)
class BenchmarkEvent:
    """One world-changing event to feed into a memory system."""

    event_id: str
    event_type: EventType
    text: str
    subject: str | None = None
    predicate: str | None = None
    object_value: str | None = None
    branch_name: str = "main"
    source_ref: str = "synthetic"
    trust_score: float = 0.75
    confidence: float = 0.8
    valid_from: date | None = None
    valid_to: date | None = None
    rollback_target_event_id: str | None = None
    merge_source_branch: str | None = None
    merge_target_branch: str = "main"

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("valid_from", "valid_to"):
            if data[key] is not None:
                data[key] = data[key].isoformat()
        return data


@dataclass(frozen=True)
class BenchmarkQuestion:
    """One evaluation question with exact structured expectations."""

    question_id: str
    metric: MetricName
    prompt: str
    subject: str
    predicate: str
    branch_name: str = "main"
    expected_object_value: str | None = None
    forbidden_object_value: str | None = None
    expected_historical_objects: list[str] = field(default_factory=list)
    expected_source_ref: str | None = None
    expected_low_trust_warning: bool = False
    expected_conflict_resolution: bool = False
    related_event_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkCase:
    """A self-contained synthetic changing-world memory scenario."""

    case_id: str
    description: str
    events: list[BenchmarkEvent]
    questions: list[BenchmarkQuestion]

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "description": self.description,
            "events": [event.to_json() for event in self.events],
            "questions": [question.to_json() for question in self.questions],
        }


class SyntheticBenchmarkGenerator:
    """Generate deterministic benchmark cases covering TruthGit research behaviors."""

    def generate(self) -> list[BenchmarkCase]:
        """Return the default benchmark suite."""

        return [
            self._supersession_case(),
            self._conflicting_sources_case(),
            self._branch_isolation_case(),
            self._rollback_case(),
            self._merge_conflict_case(),
        ]

    def _supersession_case(self) -> BenchmarkCase:
        events = [
            BenchmarkEvent(
                event_id="supersede-seoul",
                event_type="fact",
                text="Alice lives in Seoul.",
                subject="Alice",
                predicate="lives_in",
                object_value="Seoul",
                source_ref="city-register-2025",
                trust_score=0.82,
                confidence=0.84,
            ),
            BenchmarkEvent(
                event_id="supersede-busan",
                event_type="fact",
                text="Alice moved to Busan in March 2026.",
                subject="Alice",
                predicate="lives_in",
                object_value="Busan",
                source_ref="city-register-2026",
                trust_score=0.9,
                confidence=0.86,
                valid_from=date(2026, 3, 1),
            ),
        ]
        questions = [
            BenchmarkQuestion(
                question_id="q-current-alice",
                metric="current_truth_accuracy",
                prompt="Where does Alice live now?",
                subject="Alice",
                predicate="lives_in",
                expected_object_value="Busan",
            ),
            BenchmarkQuestion(
                question_id="q-history-alice",
                metric="historical_truth_accuracy",
                prompt="What is Alice's residence timeline?",
                subject="Alice",
                predicate="lives_in",
                expected_historical_objects=["Seoul", "Busan"],
            ),
            BenchmarkQuestion(
                question_id="q-provenance-alice",
                metric="provenance_accuracy",
                prompt="Which source supports Alice living in Busan?",
                subject="Alice",
                predicate="lives_in",
                expected_object_value="Busan",
                expected_source_ref="city-register-2026",
            ),
        ]
        return BenchmarkCase(
            case_id="superseded-facts",
            description="A newer residence claim should supersede an older one while preserving history.",
            events=events,
            questions=questions,
        )

    def _conflicting_sources_case(self) -> BenchmarkCase:
        events = [
            BenchmarkEvent(
                event_id="conflict-paris",
                event_type="fact",
                text="Noah lives in Paris.",
                subject="Noah",
                predicate="lives_in",
                object_value="Paris",
                source_ref="government-record",
                trust_score=0.92,
                confidence=0.9,
            ),
            BenchmarkEvent(
                event_id="conflict-berlin",
                event_type="bad_fact",
                text="A rumor says Noah lives in Berlin.",
                subject="Noah",
                predicate="lives_in",
                object_value="Berlin",
                source_ref="anonymous-rumor",
                trust_score=0.2,
                confidence=0.65,
            ),
        ]
        questions = [
            BenchmarkQuestion(
                question_id="q-conflict-current-noah",
                metric="current_truth_accuracy",
                prompt="Where should the system say Noah lives?",
                subject="Noah",
                predicate="lives_in",
                expected_object_value="Paris",
                forbidden_object_value="Berlin",
            ),
            BenchmarkQuestion(
                question_id="q-low-trust-warning-noah",
                metric="low_trust_warning_rate",
                prompt="Did the system warn about the low-trust Berlin source?",
                subject="Noah",
                predicate="lives_in",
                expected_low_trust_warning=True,
                related_event_id="conflict-berlin",
            ),
        ]
        return BenchmarkCase(
            case_id="conflicting-sources",
            description="A high-trust current fact receives a conflicting low-trust claim.",
            events=events,
            questions=questions,
        )

    def _branch_isolation_case(self) -> BenchmarkCase:
        events = [
            BenchmarkEvent(
                event_id="branch-main-home",
                event_type="fact",
                text="Mira lives in Madrid.",
                subject="Mira",
                predicate="lives_in",
                object_value="Madrid",
                source_ref="profile-main",
                trust_score=0.82,
            ),
            BenchmarkEvent(
                event_id="branch-trip-stay",
                event_type="branch_fact",
                text="During the conference week, Mira will stay in Tokyo.",
                subject="Mira",
                predicate="stays_in",
                object_value="Tokyo",
                branch_name="trip-plan",
                source_ref="conference-itinerary",
                trust_score=0.78,
            ),
        ]
        questions = [
            BenchmarkQuestion(
                question_id="q-branch-main-isolated",
                metric="branch_isolation_score",
                prompt="On main, where will Mira stay during the conference week?",
                subject="Mira",
                predicate="stays_in",
                branch_name="main",
                expected_object_value=None,
                forbidden_object_value="Tokyo",
            ),
            BenchmarkQuestion(
                question_id="q-branch-trip-visible",
                metric="branch_isolation_score",
                prompt="On trip-plan, where will Mira stay during the conference week?",
                subject="Mira",
                predicate="stays_in",
                branch_name="trip-plan",
                expected_object_value="Tokyo",
            ),
        ]
        return BenchmarkCase(
            case_id="branch-only-hypothetical",
            description="A hypothetical branch claim must not leak into main.",
            events=events,
            questions=questions,
        )

    def _rollback_case(self) -> BenchmarkCase:
        events = [
            BenchmarkEvent(
                event_id="rollback-rome",
                event_type="fact",
                text="Omar lives in Rome.",
                subject="Omar",
                predicate="lives_in",
                object_value="Rome",
                source_ref="verified-profile",
                trust_score=0.86,
            ),
            BenchmarkEvent(
                event_id="rollback-atlantis",
                event_type="bad_fact",
                text="A corrupted import says Omar lives in Atlantis.",
                subject="Omar",
                predicate="lives_in",
                object_value="Atlantis",
                source_ref="corrupted-import",
                trust_score=0.18,
                confidence=0.7,
            ),
            BenchmarkEvent(
                event_id="rollback-bad-commit",
                event_type="rollback",
                text="Rollback the corrupted Omar memory.",
                rollback_target_event_id="rollback-atlantis",
            ),
        ]
        questions = [
            BenchmarkQuestion(
                question_id="q-rollback-omar",
                metric="rollback_recovery_rate",
                prompt="After rollback, where does Omar live?",
                subject="Omar",
                predicate="lives_in",
                expected_object_value="Rome",
                forbidden_object_value="Atlantis",
                related_event_id="rollback-bad-commit",
            )
        ]
        return BenchmarkCase(
            case_id="rollback-needed-bad-commit",
            description="A bad commit should be retracted without deleting historical records.",
            events=events,
            questions=questions,
        )

    def _merge_conflict_case(self) -> BenchmarkCase:
        events = [
            BenchmarkEvent(
                event_id="merge-lab-a",
                event_type="fact",
                text="Pia works at Lab A.",
                subject="Pia",
                predicate="works_at",
                object_value="Lab A",
                source_ref="old-hr-record",
                trust_score=0.6,
                confidence=0.7,
            ),
            BenchmarkEvent(
                event_id="merge-lab-b",
                event_type="branch_fact",
                text="Pia works at Lab B in the reorg branch.",
                subject="Pia",
                predicate="works_at",
                object_value="Lab B",
                branch_name="reorg",
                source_ref="signed-reorg-plan",
                trust_score=0.96,
                confidence=0.9,
            ),
            BenchmarkEvent(
                event_id="merge-reorg-main",
                event_type="merge",
                text="Merge the reorg branch into main.",
                merge_source_branch="reorg",
                merge_target_branch="main",
            ),
        ]
        questions = [
            BenchmarkQuestion(
                question_id="q-merge-pia",
                metric="merge_conflict_resolution_score",
                prompt="After merging reorg, where does Pia work?",
                subject="Pia",
                predicate="works_at",
                expected_object_value="Lab B",
                forbidden_object_value="Lab A",
                expected_source_ref="signed-reorg-plan",
                expected_conflict_resolution=True,
            )
        ]
        return BenchmarkCase(
            case_id="merge-conflict-resolution",
            description="A high-trust branch update should resolve an older main-branch value on merge.",
            events=events,
            questions=questions,
        )


def default_benchmark() -> list[BenchmarkCase]:
    """Convenience wrapper for the default synthetic benchmark."""

    return SyntheticBenchmarkGenerator().generate()
