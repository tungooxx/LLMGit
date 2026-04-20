from __future__ import annotations

from experiments.benchmark import default_benchmark
from experiments.baselines import SystemAnswer
from experiments.metrics import aggregate_scores, score_answer
from experiments.run_benchmark import run_benchmark


def test_default_benchmark_covers_required_metrics() -> None:
    cases = default_benchmark()
    metrics = {question.metric for case in cases for question in case.questions}

    assert len(cases) >= 80
    assert "current_truth_accuracy" in metrics
    assert "historical_truth_accuracy" in metrics
    assert "provenance_accuracy" in metrics
    assert "rollback_recovery_rate" in metrics
    assert "branch_isolation_score" in metrics
    assert "merge_conflict_resolution_score" in metrics
    assert "low_trust_warning_rate" in metrics
    time_slice_questions = [
        question
        for case in cases
        for question in case.questions
        if question.metric == "historical_truth_accuracy" and question.as_of is not None
    ]
    assert 5 <= len(time_slice_questions) <= 10
    unresolved_merge_questions = [
        question
        for case in cases
        for question in case.questions
        if question.metric == "merge_conflict_resolution_score"
        and question.expected_unresolved_conflict
    ]
    branch_provenance_questions = [
        question
        for case in cases
        for question in case.questions
        if question.metric == "provenance_accuracy" and question.branch_name != "main"
    ]
    rollback_provenance_questions = [
        question
        for case in cases
        for question in case.questions
        if question.metric == "provenance_accuracy"
        and "rollback-source" in question.question_id
    ]
    assert len(unresolved_merge_questions) >= 9
    assert branch_provenance_questions
    assert rollback_provenance_questions
    assert sum(1 for case in cases for question in case.questions if question.metric == "provenance_accuracy") >= 50


def test_benchmark_run_exports_all_system_scores() -> None:
    results = run_benchmark(default_benchmark())
    summary = results["metric_summary"]
    systems = {row["system_name"] for row in summary}
    question_count = sum(len(case.questions) for case in default_benchmark())

    assert systems == {"naive_chat_history", "simple_rag", "embedding_rag", "truthgit"}
    assert len(results["predictions"]) == question_count * 4
    assert len(results["question_scores"]) == question_count * 4
    assert results["metadata"]["backbone"] == "gpt-4o-mini"


def test_benchmark_can_include_truthgit_ablations() -> None:
    results = run_benchmark(default_benchmark(), include_ablations=True)
    systems = {row["system_name"] for row in results["metric_summary"]}

    assert "truthgit_no_branches" in systems
    assert "truthgit_no_rollback" in systems
    assert "truthgit_no_review_gate" in systems
    assert "truthgit_no_trust_scoring" in systems


def test_historical_scoring_requires_exact_order() -> None:
    question = next(
        question
        for case in default_benchmark()
        for question in case.questions
        if question.question_id == "temporal-supersession-00-history"
    )

    assert score_answer(
        question,
        SystemAnswer(
            question_id=question.question_id,
            system_name="test",
            answer_text="",
            historical_objects=list(question.expected_historical_objects),
        ),
    ) == 1.0
    assert score_answer(
        question,
        SystemAnswer(
            question_id=question.question_id,
            system_name="test",
            answer_text="",
            historical_objects=list(reversed(question.expected_historical_objects)),
        ),
    ) == 0.0


def test_historical_scoring_supports_time_slice_questions() -> None:
    question = next(
        question
        for case in default_benchmark()
        for question in case.questions
        if question.question_id == "temporal-supersession-00-slice-middle"
    )

    assert question.as_of is not None
    assert score_answer(
        question,
        SystemAnswer(
            question_id=question.question_id,
            system_name="test",
            answer_text="",
            object_value=question.expected_object_value,
        ),
    ) == 1.0
    assert score_answer(
        question,
        SystemAnswer(
            question_id=question.question_id,
            system_name="test",
            answer_text="",
            object_value=question.forbidden_object_value,
        ),
    ) == 0.0


def test_merge_scoring_supports_unresolved_conflicts() -> None:
    question = next(
        question
        for case in default_benchmark()
        for question in case.questions
        if question.expected_unresolved_conflict
    )

    assert score_answer(
        question,
        SystemAnswer(
            question_id=question.question_id,
            system_name="test",
            answer_text="manual review required",
            unresolved_conflict=True,
        ),
    ) == 1.0
    assert score_answer(
        question,
        SystemAnswer(
            question_id=question.question_id,
            system_name="test",
            answer_text="resolved",
            object_value="Lab A",
            unresolved_conflict=False,
        ),
    ) == 0.0


def test_truthgit_scores_branch_and_rollback() -> None:
    results = run_benchmark(default_benchmark())
    summary = aggregate_scores(
        [
            type("Score", (), row)
            for row in results["question_scores"]
        ]
    )
    truthgit = {
        row["metric"]: row["score"]
        for row in summary
        if row["system_name"] == "truthgit"
    }

    assert truthgit["branch_isolation_score"] == 1.0
    assert truthgit["rollback_recovery_rate"] == 1.0
    assert truthgit["provenance_accuracy"] == 1.0
    assert truthgit["merge_conflict_resolution_score"] == 1.0
