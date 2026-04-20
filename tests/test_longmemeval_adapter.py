from __future__ import annotations

import json
from pathlib import Path

from experiments.public_benchmarks.longmemeval import (
    build_prompt,
    inspect_records,
    load_records,
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
    assert "user [evidence]: The review moved to May 19." in prompt
    assert written["question_id"] == "q1"
    assert "Answer:" in written["prompt"]


def _sample_record() -> dict[str, object]:
    return {
        "question_id": "q1",
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
