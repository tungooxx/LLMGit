from __future__ import annotations

import json
from pathlib import Path

from experiments.public_benchmarks.longmemeval import (
    aggregate_eval_logs,
    build_prompt,
    build_answer_check_prompt,
    evaluate_hypotheses,
    generate_hypotheses,
    inspect_records,
    load_records,
    select_records,
    summarize_eval_log,
    write_prompt_jsonl,
)


def test_longmemeval_loader_and_inspector(tmp_path: Path) -> None:
    data_path = tmp_path / "longmemeval_sample.json"
    data_path.write_text(json.dumps([_sample_record()]), encoding="utf-8")

    records = load_records(data_path)
    stats = inspect_records(records)

    assert len(records) == 1
    assert records[0].question_id == "q1"
    assert stats["record_count"] == 1
    assert stats["question_type_counts"] == {"knowledge-update": 1}
    assert stats["abstention_count"] == 0


def test_longmemeval_prompt_jsonl(tmp_path: Path) -> None:
    data_path = tmp_path / "longmemeval_sample.json"
    output_path = tmp_path / "prompts.jsonl"
    data_path.write_text(json.dumps([_sample_record()]), encoding="utf-8")
    records = load_records(data_path)

    prompt = build_prompt(records[0])
    write_prompt_jsonl(records=records, output_jsonl=output_path)
    written = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

    assert "Question date: 2026-04-20" in prompt
    assert "The review moved to May 19." in prompt
    assert "has_answer" not in prompt
    assert "[evidence]" not in prompt
    assert "answer" not in written
    assert "has_answer" not in written["prompt"]
    assert written["question_id"] == "q1"
    assert "Answer:" in written["prompt"]


def test_longmemeval_optional_evidence_markers_for_debug_only(tmp_path: Path) -> None:
    data_path = tmp_path / "longmemeval_sample.json"
    data_path.write_text(json.dumps([_sample_record()]), encoding="utf-8")
    records = load_records(data_path)

    prompt = build_prompt(records[0], history_format="nl", include_evidence_markers=True)

    assert "user [evidence]: The review moved to May 19." in prompt


def test_longmemeval_select_records_supports_seeded_heldout_samples(tmp_path: Path) -> None:
    data_path = tmp_path / "longmemeval_sample.json"
    payload = [_sample_record(question_id=f"q{i}") for i in range(8)]
    data_path.write_text(json.dumps(payload), encoding="utf-8")
    records = load_records(data_path)

    first = select_records(records, sample_size=3, sample_seed=13)
    second = select_records(records, sample_size=3, sample_seed=13)

    assert [record.question_id for record in first] == [record.question_id for record in second]
    assert [record.question_id for record in first] != ["q0", "q1", "q2"]


def test_longmemeval_select_records_supports_deterministic_shards(tmp_path: Path) -> None:
    data_path = tmp_path / "longmemeval_sample.json"
    payload = [_sample_record(question_id=f"q{i}") for i in range(8)]
    data_path.write_text(json.dumps(payload), encoding="utf-8")
    records = load_records(data_path)

    shard = select_records(records, start_index=3, limit=2)

    assert [record.question_id for record in shard] == ["q3", "q4"]


def test_longmemeval_generate_hypotheses_with_fake_answerer(tmp_path: Path) -> None:
    data_path = tmp_path / "longmemeval_sample.json"
    output_path = tmp_path / "hypotheses.jsonl"
    data_path.write_text(json.dumps([_sample_record()]), encoding="utf-8")
    records = load_records(data_path)

    stats = generate_hypotheses(
        records=records,
        output_jsonl=output_path,
        model="fake-model",
        answer_fn=lambda _prompt, _record: "May 19",
        resume=False,
    )
    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

    assert stats.generated_count == 1
    assert row["question_id"] == "q1"
    assert row["hypothesis"] == "May 19"
    assert row["prompt_mode"]["evidence_markers"] is False


def test_longmemeval_evaluate_and_summarize_with_fake_judge(tmp_path: Path) -> None:
    data_path = tmp_path / "longmemeval_sample.json"
    hypotheses_path = tmp_path / "hypotheses.jsonl"
    eval_log = tmp_path / "hypotheses.eval"
    data_path.write_text(json.dumps([_sample_record()]), encoding="utf-8")
    records = load_records(data_path)
    hypotheses_path.write_text(
        json.dumps({"question_id": "q1", "hypothesis": "May 19"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    stats = evaluate_hypotheses(
        records=records,
        hypotheses_jsonl=hypotheses_path,
        output_log=eval_log,
        judge_model="gpt-4o",
        judge_fn=lambda _prompt, _record, hypothesis: hypothesis == "May 19",
    )
    summarized = summarize_eval_log(eval_log=eval_log, records=records)
    prompt = build_answer_check_prompt(records[0], "May 19")

    assert "Correct Answer: May 19" in prompt
    assert stats.overall_accuracy == 1.0
    assert stats.by_task["knowledge-update"]["accuracy"] == 1.0
    assert summarized.task_averaged_accuracy == 1.0


def test_longmemeval_aggregate_shards_validates_full_coverage(tmp_path: Path) -> None:
    data_path = tmp_path / "longmemeval_sample.json"
    payload = [_sample_record(question_id=f"q{i}") for i in range(3)]
    data_path.write_text(json.dumps(payload), encoding="utf-8")
    records = load_records(data_path)
    shard_a = tmp_path / "shard_a.eval.jsonl"
    shard_b = tmp_path / "shard_b.eval.jsonl"
    output_log = tmp_path / "aggregate.eval.jsonl"
    output_json = tmp_path / "aggregate.summary.json"
    shard_a.write_text(
        "\n".join(
            [
                json.dumps({"question_id": "q0", "autoeval_label": {"label": True}}),
                json.dumps({"question_id": "q1", "autoeval_label": {"label": False}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    shard_b.write_text(
        json.dumps({"question_id": "q2", "autoeval_label": {"label": True}}) + "\n",
        encoding="utf-8",
    )

    payload = aggregate_eval_logs(
        eval_logs=[shard_b, shard_a],
        records=records,
        output_log=output_log,
        output_json=output_json,
    )

    assert payload["evaluated_count"] == 3
    assert payload["overall_accuracy"] == 0.6667
    assert payload["coverage"]["missing_count"] == 0
    assert [json.loads(line)["question_id"] for line in output_log.read_text(encoding="utf-8").splitlines()] == [
        "q0",
        "q1",
        "q2",
    ]


def _sample_record(question_id: str = "q1") -> dict[str, object]:
    return {
        "question_id": question_id,
        "question_type": "knowledge-update",
        "question": "When is the launch review?",
        "answer": "May 19",
        "question_date": "2026-04-20",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2026-04-18"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "The review moved to May 19.", "has_answer": True},
                {"role": "assistant", "content": "Noted."},
            ]
        ],
        "answer_session_ids": ["s1"],
    }
