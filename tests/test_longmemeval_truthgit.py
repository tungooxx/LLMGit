from __future__ import annotations

import json
from pathlib import Path

from app.schemas import ExtractedClaim
from experiments.public_benchmarks.longmemeval import load_records
from experiments.public_benchmarks.longmemeval_truthgit import (
    TruthGitMemoryContext,
    _coerce_extracted_claim_dates,
    _focused_excerpt,
    _focused_session_excerpt,
    _truthgit_answer_system_prompt,
    _truthgit_extraction_system_prompt,
    generate_truthgit_hypotheses,
    run_truthgit_record,
    select_session_indexes,
)


class FakeExtractor:
    def extract(self, *, record: object, session_payloads: list[dict[str, object]]) -> list[ExtractedClaim]:
        del record
        text = json.dumps(session_payloads)
        if "Seoul" in text:
            return [
                ExtractedClaim.model_validate(
                    {
                        "subject": "Alice",
                        "predicate": "lives_in",
                        "object": "Seoul",
                        "confidence": 0.82,
                        "source_quote": "Alice lives in Seoul.",
                    }
                )
            ]
        if "Busan" in text:
            return [
                ExtractedClaim.model_validate(
                    {
                        "subject": "Alice",
                        "predicate": "lives_in",
                        "object": "Busan",
                        "confidence": 0.88,
                        "source_quote": "Alice moved to Busan.",
                    }
                )
            ]
        return []


class FakeAnswerer:
    def answer(self, *, record: object, context: TruthGitMemoryContext) -> str:
        del record
        assert context.staged_commit_count == 2
        assert context.commit_count == 2
        assert context.audit_event_count > 0
        active_objects = {
            row["object_value"]
            for row in context.belief_versions
            if row["status"] == "active"
        }
        assert "Busan" in active_objects
        return "Busan"


class EmptyExtractor:
    def extract(self, *, record: object, session_payloads: list[dict[str, object]]) -> list[ExtractedClaim]:
        del record, session_payloads
        return []


class ExactSourceAnswerer:
    def answer(self, *, record: object, context: TruthGitMemoryContext) -> str:
        del record
        joined_sources = json.dumps(context.source_excerpts)
        assert context.claim_count == 0
        assert context.staged_commit_count == 0
        assert "Summer Vibes" in joined_sources
        return "Summer Vibes"


def test_truthgit_longmemeval_record_uses_staged_commit_pipeline(tmp_path: Path) -> None:
    records = _sample_records(tmp_path)
    stats = run_truthgit_record(
        record=records[0],
        extractor=FakeExtractor(),
        answerer=FakeAnswerer(),
        extraction_mode="per_session",
        max_sessions=None,
        max_versions=20,
        max_source_excerpts=4,
        trace_dir=tmp_path / "traces",
    )

    assert stats.hypothesis == "Busan"
    assert stats.claim_count == 2
    assert stats.staged_commit_count == 2
    assert stats.commit_count == 2
    assert (tmp_path / "traces" / "q1.truthgit-trace.json").exists()


def test_truthgit_batch_context_keeps_selected_source_snippets(tmp_path: Path) -> None:
    data_path = tmp_path / "source_only.json"
    data_path.write_text(json.dumps([_source_only_record()]), encoding="utf-8")
    record = load_records(data_path)[0]

    stats = run_truthgit_record(
        record=record,
        extractor=EmptyExtractor(),
        answerer=ExactSourceAnswerer(),
        extraction_mode="record_batch",
        max_sessions=None,
        max_versions=20,
        max_source_excerpts=4,
    )

    assert stats.hypothesis == "Summer Vibes"
    assert stats.claim_count == 0
    assert stats.commit_count == 0


def test_truthgit_longmemeval_generate_writes_official_hypotheses(tmp_path: Path) -> None:
    output = tmp_path / "truthgit.hypotheses.jsonl"
    stats = generate_truthgit_hypotheses(
        records=_sample_records(tmp_path),
        output_jsonl=output,
        answer_model="fake-answer",
        extraction_model="fake-extract",
        extraction_mode="per_session",
        max_sessions=None,
        extractor=FakeExtractor(),
        answerer=FakeAnswerer(),
        resume=False,
    )
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

    assert stats.generated_count == 1
    assert row["question_id"] == "q1"
    assert row["hypothesis"] == "Busan"
    assert row["memory_system"] == "truthgit"
    assert row["truthgit"]["staged_commit_count"] == 2


def test_truthgit_session_selection_does_not_use_answer_labels(tmp_path: Path) -> None:
    record = _sample_records(tmp_path)[0]

    selected = select_session_indexes(record, max_sessions=1)

    assert selected == [1]


def test_truthgit_extractor_date_coercion_accepts_slash_datetime() -> None:
    payload = {
        "claims": [
            {
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Busan",
                "confidence": 0.8,
                "valid_from": "2023/05/30T09:50:00Z",
                "valid_to": "",
            }
        ]
    }

    _coerce_extracted_claim_dates(payload)

    assert payload["claims"][0]["valid_from"] == "2023-05-30"
    assert payload["claims"][0]["valid_to"] is None


def test_truthgit_prompts_push_exact_memory_not_generic_advice() -> None:
    extraction_prompt = _truthgit_extraction_system_prompt()
    answer_prompt = _truthgit_answer_system_prompt()

    assert "store names" in extraction_prompt
    assert "venue names" in extraction_prompt
    assert "studio names" in extraction_prompt
    assert "model-generated snake_case predicates" in extraction_prompt
    assert "not a closed predicate list" in extraction_prompt
    assert "not advice" in answer_prompt
    assert "exact personal-memory answer" in answer_prompt


def test_focused_excerpt_keeps_question_relevant_later_session() -> None:
    source = "\n".join(
        [
            "[Session 0]\nuser: " + ("irrelevant boot care advice " * 120),
            "[Session 7]\nuser: I listen during my daily commute, which is 45 minutes each way.",
            "[Session 8]\nuser: " + ("irrelevant plant advice " * 120),
        ]
    )

    excerpt = _focused_excerpt(source, "How long is my daily commute to work?", max_chars=400)

    assert "45 minutes each way" in excerpt
    assert "daily commute" in excerpt


def test_focused_session_excerpt_preserves_user_memory_values() -> None:
    payload = {
        "session_index": 42,
        "date": "2023/05/29",
        "turns": [
            {"role": "user", "content": "I've been using the Cartwheel app from Target."},
            {"role": "assistant", "content": "Here are many coupon organization tips. " * 80},
            {"role": "user", "content": "I redeemed a $5 coupon on coffee creamer last Sunday."},
        ],
    }

    excerpt = _focused_session_excerpt(
        payload,
        "Where did I redeem a $5 coupon on coffee creamer?",
        max_chars=600,
    )

    assert "Target" in excerpt
    assert "$5 coupon on coffee creamer" in excerpt
    assert "coupon organization tips" not in excerpt


def _sample_records(tmp_path: Path):
    data_path = tmp_path / "longmemeval_sample.json"
    data_path.write_text(json.dumps([_sample_record()]), encoding="utf-8")
    return load_records(data_path)


def _sample_record() -> dict[str, object]:
    return {
        "question_id": "q1",
        "question_type": "knowledge-update",
        "question": "Where does Alice live now?",
        "answer": "Busan",
        "question_date": "2026-04-20",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2026-03-01", "2026-04-01"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Alice lives in Seoul.", "has_answer": False},
                {"role": "assistant", "content": "Noted."},
            ],
            [
                {"role": "user", "content": "Alice moved to Busan.", "has_answer": True},
                {"role": "assistant", "content": "Updated."},
            ],
        ],
        "answer_session_ids": ["s2"],
    }


def _source_only_record() -> dict[str, object]:
    return {
        "question_id": "q-source",
        "question_type": "single-session-user",
        "question": "What is the name of the playlist I created on Spotify?",
        "answer": "Summer Vibes",
        "question_date": "2026-04-20",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2026-03-01", "2026-03-08"],
        "haystack_sessions": [
            [
                {
                    "role": "assistant",
                    "content": "There are many yoga apps and class styles you can try.",
                }
            ],
            [
                {
                    "role": "user",
                    "content": "I created a Spotify playlist called Summer Vibes for my weekend trips.",
                    "has_answer": True,
                },
                {"role": "assistant", "content": "Good to know."},
            ],
        ],
        "answer_session_ids": ["s2"],
    }
