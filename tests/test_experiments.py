from __future__ import annotations

from experiments.benchmark import default_benchmark
from experiments.metrics import aggregate_scores
from experiments.run_benchmark import run_benchmark


def test_default_benchmark_covers_required_metrics() -> None:
    cases = default_benchmark()
    metrics = {question.metric for case in cases for question in case.questions}

    assert "current_truth_accuracy" in metrics
    assert "historical_truth_accuracy" in metrics
    assert "provenance_accuracy" in metrics
    assert "rollback_recovery_rate" in metrics
    assert "branch_isolation_score" in metrics
    assert "merge_conflict_resolution_score" in metrics
    assert "low_trust_warning_rate" in metrics


def test_benchmark_run_exports_all_system_scores() -> None:
    results = run_benchmark(default_benchmark())
    summary = results["metric_summary"]
    systems = {row["system_name"] for row in summary}
    question_count = sum(len(case.questions) for case in default_benchmark())

    assert systems == {"naive_chat_history", "simple_rag", "truthgit"}
    assert len(results["predictions"]) == question_count * 3
    assert len(results["question_scores"]) == question_count * 3


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
