"""Evaluation metrics for changing-world memory experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.baselines import SystemAnswer
from experiments.benchmark import BenchmarkQuestion, MetricName


@dataclass(frozen=True)
class QuestionScore:
    """Per-question score record."""

    system_name: str
    question_id: str
    metric: MetricName
    score: float
    expected: str | None
    observed: str | None
    answer_text: str


def score_answer(question: BenchmarkQuestion, answer: SystemAnswer) -> float:
    """Score one structured answer against exact benchmark expectations."""

    if question.metric == "current_truth_accuracy":
        return _truth_match(question, answer)
    if question.metric == "historical_truth_accuracy":
        expected = {value.lower() for value in question.expected_historical_objects}
        observed = {value.lower() for value in answer.historical_objects}
        return 1.0 if expected.issubset(observed) else 0.0
    if question.metric == "provenance_accuracy":
        return 1.0 if answer.source_ref == question.expected_source_ref else 0.0
    if question.metric == "rollback_recovery_rate":
        return _truth_match(question, answer)
    if question.metric == "branch_isolation_score":
        return _truth_match(question, answer)
    if question.metric == "merge_conflict_resolution_score":
        return 1.0 if _truth_match(question, answer) == 1.0 and answer.conflict_resolved else 0.0
    if question.metric == "low_trust_warning_rate":
        return 1.0 if answer.had_low_trust_warning == question.expected_low_trust_warning else 0.0
    raise ValueError(f"Unknown metric: {question.metric}")


def score_questions(
    *,
    system_name: str,
    questions: list[BenchmarkQuestion],
    answers: list[SystemAnswer],
) -> list[QuestionScore]:
    """Score all answers from one system."""

    answer_by_id = {answer.question_id: answer for answer in answers}
    scores: list[QuestionScore] = []
    for question in questions:
        answer = answer_by_id[question.question_id]
        expected = question.expected_object_value or question.expected_source_ref
        scores.append(
            QuestionScore(
                system_name=system_name,
                question_id=question.question_id,
                metric=question.metric,
                score=score_answer(question, answer),
                expected=expected,
                observed=answer.object_value or answer.source_ref,
                answer_text=answer.answer_text,
            )
        )
    return scores


def aggregate_scores(scores: list[QuestionScore]) -> list[dict[str, float | str]]:
    """Aggregate per-question scores by system and metric."""

    buckets: dict[tuple[str, str], list[float]] = {}
    for score in scores:
        buckets.setdefault((score.system_name, score.metric), []).append(score.score)
    rows: list[dict[str, float | str]] = []
    for (system_name, metric), values in sorted(buckets.items()):
        rows.append(
            {
                "system_name": system_name,
                "metric": metric,
                "score": sum(values) / len(values),
                "n": float(len(values)),
            }
        )
    return rows


def scores_to_dicts(scores: list[QuestionScore]) -> list[dict[str, object]]:
    """Serialize question scores."""

    return [asdict(score) for score in scores]


def _truth_match(question: BenchmarkQuestion, answer: SystemAnswer) -> float:
    if question.expected_object_value is not None:
        if (answer.object_value or "").lower() != question.expected_object_value.lower():
            return 0.0
    if question.expected_object_value is None and answer.object_value is not None:
        return 0.0
    if question.forbidden_object_value:
        forbidden = question.forbidden_object_value.lower()
        text = f"{answer.object_value or ''} {answer.answer_text}".lower()
        if forbidden in text:
            return 0.0
    return 1.0
