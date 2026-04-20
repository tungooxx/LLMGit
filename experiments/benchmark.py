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
    as_of: date | None = None
    expected_source_ref: str | None = None
    expected_low_trust_warning: bool = False
    expected_conflict_resolution: bool = False
    expected_unresolved_conflict: bool = False
    related_event_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        if data["as_of"] is not None:
            data["as_of"] = data["as_of"].isoformat()
        return data


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

    people = [
        "Alice",
        "Noah",
        "Mira",
        "Omar",
        "Pia",
        "Lena",
        "Kai",
        "Iris",
        "Theo",
        "Nina",
        "Ravi",
        "Zara",
        "Hana",
        "Jon",
        "Uma",
    ]
    cities = [
        "Seoul",
        "Busan",
        "Tokyo",
        "Madrid",
        "Rome",
        "Berlin",
        "Paris",
        "Lisbon",
        "Dublin",
        "Oslo",
        "Prague",
        "Vienna",
        "Lima",
        "Quito",
        "Hanoi",
    ]
    labs = ["Lab A", "Lab B", "Lab C", "Lab D", "Lab E", "Lab F"]

    def generate(self) -> list[BenchmarkCase]:
        """Return the benchmark-v3 suite with deterministic changing-world cases."""

        cases: list[BenchmarkCase] = []
        cases.extend(self._temporal_supersession_cases(12))
        cases.extend(self._poisoning_cases(10))
        cases.extend(self._branch_leakage_cases(10))
        cases.extend(self._rollback_cases(8))
        cases.extend(self._merge_conflict_cases(5))
        cases.extend(self._provenance_cases(5))
        cases.extend(self._provenance_hardening_cases(9))
        cases.extend(self._merge_hardening_cases(6))
        return cases

    def _temporal_supersession_cases(self, count: int) -> list[BenchmarkCase]:
        cases: list[BenchmarkCase] = []
        for idx in range(count):
            person = self.people[idx % len(self.people)]
            first = self.cities[idx % len(self.cities)]
            second = self.cities[(idx + 4) % len(self.cities)]
            third = self.cities[(idx + 8) % len(self.cities)]
            y1 = 2024 + (idx % 2)
            y2 = 2025 + (idx % 2)
            y3 = y2 + 1
            case_id = f"temporal-supersession-{idx:02d}"
            events = [
                BenchmarkEvent(
                    event_id=f"{case_id}-initial",
                    event_type="fact",
                    text=f"{person} lives in {first} as of January {y1}.",
                    subject=person,
                    predicate="lives_in",
                    object_value=first,
                    source_ref=f"residence-ledger-{idx}-a",
                    trust_score=0.82,
                    confidence=0.82,
                    valid_from=date(y1, 1, 1),
                ),
                BenchmarkEvent(
                    event_id=f"{case_id}-middle",
                    event_type="fact",
                    text=f"{person} moved to {second} in June {y2}.",
                    subject=person,
                    predicate="lives_in",
                    object_value=second,
                    source_ref=f"residence-ledger-{idx}-b",
                    trust_score=0.86,
                    confidence=0.84,
                    valid_from=date(y2, 6, 1),
                ),
                BenchmarkEvent(
                    event_id=f"{case_id}-latest",
                    event_type="fact",
                    text=f"{person} moved to {third} in February {y3}.",
                    subject=person,
                    predicate="lives_in",
                    object_value=third,
                    source_ref=f"residence-ledger-{idx}-c",
                    trust_score=0.9,
                    confidence=0.86,
                    valid_from=date(y3, 2, 1),
                ),
            ]
            questions = [
                BenchmarkQuestion(
                    question_id=f"{case_id}-current",
                    metric="current_truth_accuracy",
                    prompt=f"Where does {person} live now after the latest move?",
                    subject=person,
                    predicate="lives_in",
                    expected_object_value=third,
                    forbidden_object_value=first,
                ),
                BenchmarkQuestion(
                    question_id=f"{case_id}-history",
                    metric="historical_truth_accuracy",
                    prompt=f"What is {person}'s full residence timeline?",
                    subject=person,
                    predicate="lives_in",
                    expected_historical_objects=[first, second, third],
                ),
                BenchmarkQuestion(
                    question_id=f"{case_id}-provenance",
                    metric="provenance_accuracy",
                    prompt=f"Which exact source supports {person}'s current residence?",
                    subject=person,
                    predicate="lives_in",
                    expected_object_value=third,
                    expected_source_ref=f"residence-ledger-{idx}-c",
                ),
            ]
            if idx < 6:
                questions.append(
                    BenchmarkQuestion(
                        question_id=f"{case_id}-slice-middle",
                        metric="historical_truth_accuracy",
                        prompt=f"Where did {person} live in December {y2}?",
                        subject=person,
                        predicate="lives_in",
                        expected_object_value=second,
                        forbidden_object_value=third,
                        as_of=date(y2, 12, 1),
                    )
                )
            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    description="Three-step temporal supersession with current, history, and provenance checks.",
                    events=events,
                    questions=questions,
                )
            )
        return cases

    def _poisoning_cases(self, count: int) -> list[BenchmarkCase]:
        cases: list[BenchmarkCase] = []
        poison_targets = ["Atlantis", "El Dorado", "Gotham", "Metropolis", "Narnia"]
        for idx in range(count):
            person = f"Poison{idx}"
            trusted_city = self.cities[(idx + 2) % len(self.cities)]
            poisoned_city = poison_targets[idx % len(poison_targets)]
            case_id = f"poisoning-low-trust-{idx:02d}"
            events = [
                BenchmarkEvent(
                    event_id=f"{case_id}-trusted",
                    event_type="fact",
                    text=f"Verified registry says {person} lives in {trusted_city}.",
                    subject=person,
                    predicate="lives_in",
                    object_value=trusted_city,
                    source_ref=f"verified-registry-{idx}",
                    trust_score=0.93,
                    confidence=0.9,
                ),
                BenchmarkEvent(
                    event_id=f"{case_id}-poison",
                    event_type="bad_fact",
                    text=f"Unverified web scrape says {person} lives in {poisoned_city}.",
                    subject=person,
                    predicate="lives_in",
                    object_value=poisoned_city,
                    source_ref=f"untrusted-scrape-{idx}",
                    trust_score=0.12 + (idx % 3) * 0.05,
                    confidence=0.7,
                ),
            ]
            questions = [
                BenchmarkQuestion(
                    question_id=f"{case_id}-current",
                    metric="current_truth_accuracy",
                    prompt=f"Where should the system say {person} lives after the untrusted scrape?",
                    subject=person,
                    predicate="lives_in",
                    expected_object_value=trusted_city,
                    forbidden_object_value=poisoned_city,
                ),
                BenchmarkQuestion(
                    question_id=f"{case_id}-warning",
                    metric="low_trust_warning_rate",
                    prompt=f"Did the system flag the low-trust source for {person}?",
                    subject=person,
                    predicate="lives_in",
                    expected_low_trust_warning=True,
                    related_event_id=f"{case_id}-poison",
                ),
            ]
            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    description="Low-trust poisoning attempt should not displace trusted current truth.",
                    events=events,
                    questions=questions,
                )
            )
        return cases

    def _branch_leakage_cases(self, count: int) -> list[BenchmarkCase]:
        cases: list[BenchmarkCase] = []
        for idx in range(count):
            person = f"Branch{idx}"
            home = self.cities[idx % len(self.cities)]
            trip_city = self.cities[(idx + 7) % len(self.cities)]
            branch_name = f"trip-plan-{idx}"
            case_id = f"branch-leakage-{idx:02d}"
            events = [
                BenchmarkEvent(
                    event_id=f"{case_id}-home",
                    event_type="fact",
                    text=f"{person} lives in {home}.",
                    subject=person,
                    predicate="lives_in",
                    object_value=home,
                    source_ref=f"profile-main-{idx}",
                    trust_score=0.82,
                ),
                BenchmarkEvent(
                    event_id=f"{case_id}-branch-stay",
                    event_type="branch_fact",
                    text=f"During the conference week, {person} will stay in {trip_city}.",
                    subject=person,
                    predicate="stays_in",
                    object_value=trip_city,
                    branch_name=branch_name,
                    source_ref=f"conference-itinerary-{idx}",
                    trust_score=0.78,
                ),
            ]
            questions = [
                BenchmarkQuestion(
                    question_id=f"{case_id}-main-isolation",
                    metric="branch_isolation_score",
                    prompt=f"On main, where will {person} stay during the conference week?",
                    subject=person,
                    predicate="stays_in",
                    branch_name="main",
                    expected_object_value=None,
                    forbidden_object_value=trip_city,
                ),
                BenchmarkQuestion(
                    question_id=f"{case_id}-branch-visible",
                    metric="branch_isolation_score",
                    prompt=f"On {branch_name}, where will {person} stay during the conference week?",
                    subject=person,
                    predicate="stays_in",
                    branch_name=branch_name,
                    expected_object_value=trip_city,
                ),
                BenchmarkQuestion(
                    question_id=f"{case_id}-branch-source",
                    metric="provenance_accuracy",
                    prompt=f"Which itinerary source supports {person}'s branch-only stay?",
                    subject=person,
                    predicate="stays_in",
                    branch_name=branch_name,
                    expected_object_value=trip_city,
                    expected_source_ref=f"conference-itinerary-{idx}",
                ),
            ]
            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    description="Branch-only hypothetical facts should be visible only on their branch.",
                    events=events,
                    questions=questions,
                )
            )
        return cases

    def _rollback_cases(self, count: int) -> list[BenchmarkCase]:
        cases: list[BenchmarkCase] = []
        for idx in range(count):
            person = f"Rollback{idx}"
            correct = self.cities[(idx + 3) % len(self.cities)]
            corrupt = self.cities[(idx + 10) % len(self.cities)]
            case_id = f"rollback-bad-commit-{idx:02d}"
            events = [
                BenchmarkEvent(
                    event_id=f"{case_id}-correct",
                    event_type="fact",
                    text=f"Verified profile says {person} lives in {correct}.",
                    subject=person,
                    predicate="lives_in",
                    object_value=correct,
                    source_ref=f"verified-profile-{idx}",
                    trust_score=0.82,
                    confidence=0.82,
                    valid_from=date(2025, 1, 1),
                ),
                BenchmarkEvent(
                    event_id=f"{case_id}-bad",
                    event_type="bad_fact",
                    text=f"Corrupted admin import says {person} lives in {corrupt}.",
                    subject=person,
                    predicate="lives_in",
                    object_value=corrupt,
                    source_ref=f"corrupted-admin-import-{idx}",
                    trust_score=0.96,
                    confidence=0.92,
                    valid_from=date(2026, 1, 1),
                ),
                BenchmarkEvent(
                    event_id=f"{case_id}-rollback",
                    event_type="rollback",
                    text=f"Rollback corrupted memory for {person}.",
                    rollback_target_event_id=f"{case_id}-bad",
                ),
            ]
            questions = [
                BenchmarkQuestion(
                    question_id=f"{case_id}-recovery",
                    metric="rollback_recovery_rate",
                    prompt=f"After rollback, where does {person} live?",
                    subject=person,
                    predicate="lives_in",
                    expected_object_value=correct,
                    forbidden_object_value=corrupt,
                    related_event_id=f"{case_id}-rollback",
                )
            ]
            if idx < 4:
                questions.append(
                    BenchmarkQuestion(
                        question_id=f"{case_id}-history-after-rollback",
                        metric="historical_truth_accuracy",
                        prompt=f"What is {person}'s rollback-cleaned residence timeline?",
                        subject=person,
                        predicate="lives_in",
                        expected_historical_objects=[correct],
                    )
                )
            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    description="A high-trust bad import should require rollback recovery.",
                    events=events,
                    questions=questions,
                )
            )
        return cases

    def _merge_conflict_cases(self, count: int) -> list[BenchmarkCase]:
        cases: list[BenchmarkCase] = []
        for idx in range(count):
            person = f"Merge{idx}"
            old_lab = self.labs[idx % len(self.labs)]
            new_lab = self.labs[(idx + 2) % len(self.labs)]
            branch_name = f"reorg-{idx}"
            case_id = f"merge-conflict-{idx:02d}"
            events = [
                BenchmarkEvent(
                    event_id=f"{case_id}-old",
                    event_type="fact",
                    text=f"{person} works at {old_lab}.",
                    subject=person,
                    predicate="works_at",
                    object_value=old_lab,
                    source_ref=f"old-hr-record-{idx}",
                    trust_score=0.6,
                    confidence=0.7,
                ),
                BenchmarkEvent(
                    event_id=f"{case_id}-new",
                    event_type="branch_fact",
                    text=f"{person} works at {new_lab} in the reorg branch.",
                    subject=person,
                    predicate="works_at",
                    object_value=new_lab,
                    branch_name=branch_name,
                    source_ref=f"signed-reorg-plan-{idx}",
                    trust_score=0.96,
                    confidence=0.9,
                ),
                BenchmarkEvent(
                    event_id=f"{case_id}-merge",
                    event_type="merge",
                    text=f"Merge {branch_name} into main.",
                    merge_source_branch=branch_name,
                    merge_target_branch="main",
                ),
            ]
            questions = [
                BenchmarkQuestion(
                    question_id=f"{case_id}-merged-current",
                    metric="merge_conflict_resolution_score",
                    prompt=f"After merging {branch_name}, where does {person} work?",
                    subject=person,
                    predicate="works_at",
                    expected_object_value=new_lab,
                    forbidden_object_value=old_lab,
                    expected_source_ref=f"signed-reorg-plan-{idx}",
                    expected_conflict_resolution=True,
                )
            ]
            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    description="High-trust branch update should resolve older main value after merge.",
                    events=events,
                    questions=questions,
                )
            )
        return cases

    def _provenance_cases(self, count: int) -> list[BenchmarkCase]:
        cases: list[BenchmarkCase] = []
        for idx in range(count):
            person = f"Source{idx}"
            city = self.cities[(idx + 5) % len(self.cities)]
            old_city = self.cities[(idx + 1) % len(self.cities)]
            case_id = f"exact-provenance-{idx:02d}"
            events = [
                BenchmarkEvent(
                    event_id=f"{case_id}-profile",
                    event_type="fact",
                    text=f"Profile snapshot says {person} lives in {old_city}.",
                    subject=person,
                    predicate="lives_in",
                    object_value=old_city,
                    source_ref=f"profile-snapshot-{idx}",
                    trust_score=0.7,
                    confidence=0.74,
                    valid_from=date(2025, 1, 1),
                ),
                BenchmarkEvent(
                    event_id=f"{case_id}-registry",
                    event_type="fact",
                    text=f"Signed registry confirms {person} moved to {city}.",
                    subject=person,
                    predicate="lives_in",
                    object_value=city,
                    source_ref=f"signed-registry-{idx}",
                    trust_score=0.96,
                    confidence=0.9,
                    valid_from=date(2026, 5, 1),
                ),
            ]
            questions = [
                BenchmarkQuestion(
                    question_id=f"{case_id}-source",
                    metric="provenance_accuracy",
                    prompt=f"Which exact source should be cited for {person}'s current residence?",
                    subject=person,
                    predicate="lives_in",
                    expected_object_value=city,
                    expected_source_ref=f"signed-registry-{idx}",
                )
            ]
            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    description="Same-object corroboration where exact best source tracking matters.",
                    events=events,
                    questions=questions,
                )
            )
        return cases

    def _provenance_hardening_cases(self, count: int) -> list[BenchmarkCase]:
        cases: list[BenchmarkCase] = []
        for idx in range(count):
            mode = idx % 3
            if mode == 0:
                cases.append(self._same_fact_current_justification_case(idx))
            elif mode == 1:
                cases.append(self._rollback_source_case(idx))
            else:
                cases.append(self._branch_specific_source_case(idx))
        return cases

    def _same_fact_current_justification_case(self, idx: int) -> BenchmarkCase:
        person = f"Corroborated{idx}"
        city = self.cities[(idx + 6) % len(self.cities)]
        case_id = f"provenance-current-justification-{idx:02d}"
        events = [
            BenchmarkEvent(
                event_id=f"{case_id}-early-blog",
                event_type="fact",
                text=f"Community blog says {person} lives in {city}.",
                subject=person,
                predicate="lives_in",
                object_value=city,
                source_ref=f"community-blog-{idx}",
                trust_score=0.46,
                confidence=0.72,
            ),
            BenchmarkEvent(
                event_id=f"{case_id}-official-registry",
                event_type="fact",
                text=f"Official registry confirms {person} lives in {city}.",
                subject=person,
                predicate="lives_in",
                object_value=city,
                source_ref=f"official-current-registry-{idx}",
                trust_score=0.98,
                confidence=0.94,
            ),
        ]
        questions = [
            BenchmarkQuestion(
                question_id=f"{case_id}-source",
                metric="provenance_accuracy",
                prompt=f"Which exact current source should justify {person}'s residence?",
                subject=person,
                predicate="lives_in",
                expected_object_value=city,
                expected_source_ref=f"official-current-registry-{idx}",
            )
        ]
        return BenchmarkCase(
            case_id=case_id,
            description="Multiple sources mention the same current fact; only the stronger current source is correct.",
            events=events,
            questions=questions,
        )

    def _rollback_source_case(self, idx: int) -> BenchmarkCase:
        person = f"RollbackSource{idx}"
        correct = self.cities[(idx + 2) % len(self.cities)]
        bad = self.cities[(idx + 11) % len(self.cities)]
        case_id = f"provenance-rollback-source-{idx:02d}"
        events = [
            BenchmarkEvent(
                event_id=f"{case_id}-verified",
                event_type="fact",
                text=f"Verified residency file says {person} lives in {correct}.",
                subject=person,
                predicate="lives_in",
                object_value=correct,
                source_ref=f"verified-residency-file-{idx}",
                trust_score=0.9,
                confidence=0.88,
                valid_from=date(2025, 4, 1),
            ),
            BenchmarkEvent(
                event_id=f"{case_id}-bad-import",
                event_type="bad_fact",
                text=f"Bad bulk import says {person} lives in {bad}.",
                subject=person,
                predicate="lives_in",
                object_value=bad,
                source_ref=f"bad-bulk-import-source-{idx}",
                trust_score=0.97,
                confidence=0.91,
                valid_from=date(2026, 4, 1),
            ),
            BenchmarkEvent(
                event_id=f"{case_id}-rollback",
                event_type="rollback",
                text=f"Rollback the bad bulk import for {person}.",
                rollback_target_event_id=f"{case_id}-bad-import",
            ),
        ]
        questions = [
            BenchmarkQuestion(
                question_id=f"{case_id}-source",
                metric="provenance_accuracy",
                prompt=f"After rollback, which source should be cited for {person}'s current residence?",
                subject=person,
                predicate="lives_in",
                expected_object_value=correct,
                expected_source_ref=f"verified-residency-file-{idx}",
                forbidden_object_value=bad,
                related_event_id=f"{case_id}-rollback",
            )
        ]
        return BenchmarkCase(
            case_id=case_id,
            description="After rollback, the retracted bad source must no longer justify current truth.",
            events=events,
            questions=questions,
        )

    def _branch_specific_source_case(self, idx: int) -> BenchmarkCase:
        person = f"BranchSource{idx}"
        home_lab = self.labs[idx % len(self.labs)]
        branch_lab = self.labs[(idx + 3) % len(self.labs)]
        branch_name = f"grant-branch-{idx}"
        case_id = f"provenance-branch-context-{idx:02d}"
        events = [
            BenchmarkEvent(
                event_id=f"{case_id}-main-contract",
                event_type="fact",
                text=f"Main employment contract says {person} works at {home_lab}.",
                subject=person,
                predicate="works_at",
                object_value=home_lab,
                source_ref=f"main-employment-contract-{idx}",
                trust_score=0.88,
                confidence=0.87,
            ),
            BenchmarkEvent(
                event_id=f"{case_id}-branch-grant",
                event_type="branch_fact",
                text=f"Grant planning branch says {person} would work at {branch_lab}.",
                subject=person,
                predicate="works_at",
                object_value=branch_lab,
                branch_name=branch_name,
                source_ref=f"branch-grant-plan-{idx}",
                trust_score=0.82,
                confidence=0.83,
            ),
        ]
        questions = [
            BenchmarkQuestion(
                question_id=f"{case_id}-main-source",
                metric="provenance_accuracy",
                prompt=f"On main, which source supports where {person} works?",
                subject=person,
                predicate="works_at",
                branch_name="main",
                expected_object_value=home_lab,
                expected_source_ref=f"main-employment-contract-{idx}",
            ),
            BenchmarkQuestion(
                question_id=f"{case_id}-branch-source",
                metric="provenance_accuracy",
                prompt=f"On {branch_name}, which source supports where {person} works?",
                subject=person,
                predicate="works_at",
                branch_name=branch_name,
                expected_object_value=branch_lab,
                expected_source_ref=f"branch-grant-plan-{idx}",
            ),
        ]
        return BenchmarkCase(
            case_id=case_id,
            description="The correct provenance source differs between main and a hypothetical branch.",
            events=events,
            questions=questions,
        )

    def _merge_hardening_cases(self, count: int) -> list[BenchmarkCase]:
        cases: list[BenchmarkCase] = []
        for idx in range(count):
            if idx % 2 == 0:
                cases.append(self._manual_review_merge_case(idx))
            else:
                cases.append(self._concurrent_update_merge_case(idx))
        return cases

    def _manual_review_merge_case(self, idx: int) -> BenchmarkCase:
        person = f"ManualMerge{idx}"
        old_lab = self.labs[idx % len(self.labs)]
        proposed_lab = self.labs[(idx + 1) % len(self.labs)]
        branch_name = f"uncertain-reorg-{idx}"
        case_id = f"merge-manual-review-{idx:02d}"
        events = [
            BenchmarkEvent(
                event_id=f"{case_id}-main-record",
                event_type="fact",
                text=f"Audited HR file says {person} works at {old_lab}.",
                subject=person,
                predicate="works_at",
                object_value=old_lab,
                source_ref=f"audited-hr-file-{idx}",
                trust_score=0.95,
                confidence=0.9,
            ),
            BenchmarkEvent(
                event_id=f"{case_id}-branch-rumor",
                event_type="branch_fact",
                text=f"Unconfirmed reorg draft says {person} works at {proposed_lab}.",
                subject=person,
                predicate="works_at",
                object_value=proposed_lab,
                branch_name=branch_name,
                source_ref=f"unconfirmed-reorg-draft-{idx}",
                trust_score=0.5,
                confidence=0.7,
            ),
            BenchmarkEvent(
                event_id=f"{case_id}-merge",
                event_type="merge",
                text=f"Merge {branch_name} into main for review.",
                merge_source_branch=branch_name,
                merge_target_branch="main",
            ),
        ]
        questions = [
            BenchmarkQuestion(
                question_id=f"{case_id}-unresolved",
                metric="merge_conflict_resolution_score",
                prompt=f"After merging {branch_name}, should {person}'s workplace conflict require manual review?",
                subject=person,
                predicate="works_at",
                expected_unresolved_conflict=True,
                related_event_id=f"{case_id}-merge",
            )
        ]
        return BenchmarkCase(
            case_id=case_id,
            description="Lower-trust branch merge should preserve an unresolved conflict for manual review.",
            events=events,
            questions=questions,
        )

    def _concurrent_update_merge_case(self, idx: int) -> BenchmarkCase:
        person = f"ConcurrentMerge{idx}"
        old_lab = self.labs[idx % len(self.labs)]
        branch_lab = self.labs[(idx + 2) % len(self.labs)]
        main_lab = self.labs[(idx + 4) % len(self.labs)]
        branch_name = f"parallel-reorg-{idx}"
        case_id = f"merge-concurrent-update-{idx:02d}"
        events = [
            BenchmarkEvent(
                event_id=f"{case_id}-initial",
                event_type="fact",
                text=f"Baseline HR file says {person} works at {old_lab}.",
                subject=person,
                predicate="works_at",
                object_value=old_lab,
                source_ref=f"baseline-hr-file-{idx}",
                trust_score=0.72,
                confidence=0.75,
            ),
            BenchmarkEvent(
                event_id=f"{case_id}-branch-update",
                event_type="branch_fact",
                text=f"Parallel branch says {person} works at {branch_lab}.",
                subject=person,
                predicate="works_at",
                object_value=branch_lab,
                branch_name=branch_name,
                source_ref=f"parallel-branch-update-{idx}",
                trust_score=0.78,
                confidence=0.8,
            ),
            BenchmarkEvent(
                event_id=f"{case_id}-main-update",
                event_type="fact",
                text=f"Signed main update says {person} now works at {main_lab}.",
                subject=person,
                predicate="works_at",
                object_value=main_lab,
                source_ref=f"signed-main-concurrent-update-{idx}",
                trust_score=0.97,
                confidence=0.92,
            ),
            BenchmarkEvent(
                event_id=f"{case_id}-merge",
                event_type="merge",
                text=f"Merge {branch_name} into main after concurrent main update.",
                merge_source_branch=branch_name,
                merge_target_branch="main",
            ),
        ]
        questions = [
            BenchmarkQuestion(
                question_id=f"{case_id}-unresolved",
                metric="merge_conflict_resolution_score",
                prompt=f"After concurrent main and branch updates, should {person}'s merge be unresolved?",
                subject=person,
                predicate="works_at",
                expected_unresolved_conflict=True,
                related_event_id=f"{case_id}-merge",
            )
        ]
        return BenchmarkCase(
            case_id=case_id,
            description="Concurrent main and branch updates should surface an unresolved merge conflict.",
            events=events,
            questions=questions,
        )


def default_benchmark() -> list[BenchmarkCase]:
    """Convenience wrapper for the default synthetic benchmark."""

    return SyntheticBenchmarkGenerator().generate()
