from __future__ import annotations

from experiments.benchmark import default_benchmark
from experiments.reader_benchmark import (
    HeuristicContextReader,
    question_payload_for_reader,
    run_reader_benchmark,
)


def test_reader_question_payload_does_not_include_expected_answers() -> None:
    question = default_benchmark()[0].questions[0]

    payload = question_payload_for_reader(question)

    assert "expected_object_value" not in payload
    assert "expected_source_ref" not in payload
    assert payload["question_id"] == question.question_id
    assert payload["prompt"] == question.prompt


def test_model_in_loop_runner_uses_shared_reader_contexts() -> None:
    cases = default_benchmark()[:2]

    results = run_reader_benchmark(cases, reader=HeuristicContextReader())
    systems = {row["system_name"] for row in results["metric_summary"]}
    question_count = sum(len(case.questions) for case in cases)

    assert results["metadata"]["evaluation_mode"] == "model_in_loop_reader"
    assert results["metadata"]["reader_name"] == "heuristic_context_reader"
    assert systems == {"naive_chat_history", "simple_rag", "embedding_rag", "truthgit"}
    assert len(results["predictions"]) == question_count * 4
    assert len(results["contexts"]) == question_count * 4
    assert all("context" in row for row in results["contexts"])
