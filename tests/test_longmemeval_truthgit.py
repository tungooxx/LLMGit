from __future__ import annotations

import json
from pathlib import Path

from app.schemas import ExtractedClaim
from experiments.public_benchmarks.longmemeval import load_records
from experiments.public_benchmarks.longmemeval_truthgit import (
    ModelEvidenceCandidate,
    ModelEvidencePlan,
    TruthGitMemoryContext,
    _coerce_extracted_claim_dates,
    _focused_excerpt,
    _focused_session_excerpt,
    _legacy_answer_hints,
    _parse_session_date,
    _reduce_model_evidence_plan,
    _model_evidence_planner_system_prompt,
    _truthgit_answer_system_prompt,
    _truthgit_extraction_payload,
    _truthgit_extraction_system_prompt,
    generate_truthgit_hypotheses,
    run_truthgit_record,
    session_payload,
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


class PoisonExtractor:
    def extract(self, *, record: object, session_payloads: list[dict[str, object]]) -> list[ExtractedClaim]:
        del record
        text = json.dumps(session_payloads)
        if "Busan" in text:
            return [
                ExtractedClaim.model_validate(
                    {
                        "subject": "Alice",
                        "predicate": "lives_in",
                        "object": "Busan",
                        "confidence": 0.9,
                        "source_quote": "Verified registry says Alice lives in Busan.",
                    }
                )
            ]
        if "Tokyo" in text:
            return [
                ExtractedClaim.model_validate(
                    {
                        "subject": "Alice",
                        "predicate": "stays_in",
                        "object": "Tokyo",
                        "confidence": 0.7,
                        "source_quote": "During the conference week, Alice will stay in Tokyo.",
                    }
                )
            ]
        return []


class PoisonAnswerer:
    def answer(self, *, record: object, context: TruthGitMemoryContext) -> str:
        del record
        active_objects = {
            row["object_value"]
            for row in context.belief_versions
            if row["status"] == "active"
        }
        assert "Busan" in active_objects
        assert "Tokyo" not in active_objects
        assert context.staged_commit_count == 2
        assert context.commit_count == 1
        return "Busan"


class ExactSourceAnswerer:
    def answer(self, *, record: object, context: TruthGitMemoryContext) -> str:
        del record
        joined_sources = json.dumps(context.source_excerpts)
        assert context.claim_count == 0
        assert context.staged_commit_count == 0
        assert "Summer Vibes" in joined_sources
        return "Summer Vibes"


class ExactSourceAnswererExpectingNoSources:
    def answer(self, *, record: object, context: TruthGitMemoryContext) -> str:
        del record
        assert context.claim_count == 0
        assert context.staged_commit_count == 0
        assert context.source_excerpts == []
        return "unknown"


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
        context_mode="beliefs_and_excerpts",
    )

    assert stats.hypothesis == "Summer Vibes"
    assert stats.claim_count == 0
    assert stats.commit_count == 0


def test_truthgit_belief_only_context_excludes_source_snippets(tmp_path: Path) -> None:
    data_path = tmp_path / "source_only.json"
    data_path.write_text(json.dumps([_source_only_record()]), encoding="utf-8")
    record = load_records(data_path)[0]

    stats = run_truthgit_record(
        record=record,
        extractor=EmptyExtractor(),
        answerer=ExactSourceAnswererExpectingNoSources(),
        extraction_mode="record_batch",
        max_sessions=None,
        max_versions=20,
        max_source_excerpts=4,
        context_mode="beliefs_only",
    )

    assert stats.hypothesis == "unknown"
    assert stats.claim_count == 0
    assert stats.commit_count == 0


def test_truthgit_longmemeval_skips_quarantined_ingest_without_crashing(tmp_path: Path) -> None:
    data_path = tmp_path / "poison_record.json"
    data_path.write_text(json.dumps([_poison_record()]), encoding="utf-8")
    record = load_records(data_path)[0]

    stats = run_truthgit_record(
        record=record,
        extractor=PoisonExtractor(),
        answerer=PoisonAnswerer(),
        extraction_mode="per_session",
        max_sessions=None,
        max_versions=20,
        max_source_excerpts=4,
        context_mode="beliefs_and_excerpts",
    )

    assert stats.hypothesis == "Busan"
    assert stats.staged_commit_count == 2
    assert stats.commit_count == 1
    assert stats.claim_count == 1


def test_truthgit_longmemeval_generate_writes_official_hypotheses(tmp_path: Path) -> None:
    output = tmp_path / "truthgit.hypotheses.jsonl"
    stats = generate_truthgit_hypotheses(
        records=_sample_records(tmp_path),
        output_jsonl=output,
        answer_model="fake-answer",
        extraction_model="fake-extract",
        extraction_mode="per_session",
        max_sessions=None,
        context_mode="beliefs_and_excerpts",
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


def test_truthgit_session_selection_zero_means_full_history(tmp_path: Path) -> None:
    record = _sample_records(tmp_path)[0]

    selected = select_session_indexes(record, max_sessions=0)

    assert selected == [0, 1]


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
    assert "transferable preferences" in answer_prompt
    assert "do not recommend a specific prior item" in answer_prompt
    assert "recommended criteria rather than saying unknown" in answer_prompt
    assert "legacy answer_hints only as fallback" in answer_prompt
    assert "Prefer model_evidence_plan" in answer_prompt
    assert "Do not apologize" in answer_prompt
    assert "analogous local features" in answer_prompt
    assert "named venue from a prior city is evidence" in answer_prompt
    assert "count distinct user-stated actions" in answer_prompt
    assert "repeated purchase/download statements" in answer_prompt
    assert "vinyl records" in answer_prompt
    assert "signed-record" in answer_prompt
    assert "compute user_age - other_person_age" in answer_prompt
    assert "compute older_age - younger_age" in answer_prompt
    assert "use that source's session_date as the event date" in answer_prompt
    assert "quantity times unit price" in answer_prompt
    assert "matches that target date" in answer_prompt
    assert "money_total_candidates" not in answer_prompt
    assert "relative_date_candidates" not in answer_prompt
    assert "temporal_event_candidates" not in answer_prompt
    assert "deduplicate named items" in answer_prompt
    assert "session_date" in answer_prompt
    assert "exact remembered value" in answer_prompt
    planner_prompt = _model_evidence_planner_system_prompt()
    assert "Do not answer the question" in planner_prompt
    assert "temporal_order" in planner_prompt
    assert "date_bounded_count" in planner_prompt
    assert "role='anchor'" in planner_prompt
    assert "scan every provided source excerpt" in planner_prompt
    assert "all distinct candidate target events" in planner_prompt
    assert "mark other expenses as" in planner_prompt


def test_truthgit_extraction_payload_is_question_blind(tmp_path: Path) -> None:
    record = _sample_records(tmp_path)[0]
    payload = _truthgit_extraction_payload([session_payload(record, 1)])
    encoded = json.dumps(payload)

    assert "Where does Alice live now?" not in encoded
    assert "knowledge-update" not in encoded
    assert "2026-04-20" not in encoded
    assert "answer_session_ids" not in encoded
    assert "has_answer" not in encoded
    assert "Busan" in encoded


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


def test_focused_session_excerpt_preserves_age_cues_for_arithmetic_questions() -> None:
    payload = {
        "session_index": 12,
        "date": "2023/05/24 (Wed) 10:39",
        "turns": [
            {
                "role": "user",
                "content": "I'm considering a career change for someone my age. I just turned 32 last month.",
            },
            {"role": "assistant", "content": "Happy belated birthday."},
            {"role": "user", "content": "I'm also interested in data science and analytics."},
        ],
    }

    excerpt = _focused_session_excerpt(
        payload,
        "How old was I when Alex was born?",
        max_chars=260,
    )

    assert "turned 32" in excerpt


def test_focused_session_excerpt_preserves_named_person_and_age_cues() -> None:
    payload = {
        "session_index": 9,
        "date": "2023/05/21 (Sun) 13:21",
        "turns": [
            {"role": "user", "content": "Alex is just 21 and started his first job."},
            {"role": "assistant", "content": "That is a big milestone."},
            {"role": "user", "content": "I just turned 32 last month."},
        ],
    }

    excerpt = _focused_session_excerpt(
        payload,
        "How old was I when Alex was born?",
        max_chars=320,
    )

    assert "Alex is just 21" in excerpt
    assert "turned 32" in excerpt


def test_legacy_answer_hints_surface_count_and_preference_evidence() -> None:
    source_rows = [
        {
            "source_ref": "session:music",
            "excerpt": (
                'user: I downloaded the album "Blue Hour" on Spotify.\n'
                'user: I got my vinyl signed after the show.\n'
            ),
        },
        {
            "source_ref": "session:hotel",
            "excerpt": (
                "user: I like hotels with great views, a rooftop pool, and a hot tub on the balcony."
            ),
        },
    ]

    count_hints = _legacy_answer_hints(
        "How many music albums or EPs have I purchased or downloaded?",
        source_rows,
    )
    preference_hints = _legacy_answer_hints(
        "Can you suggest a hotel for my upcoming trip?",
        source_rows,
    )

    assert "Blue Hour" in json.dumps(count_hints)
    assert "vinyl signed" in json.dumps(count_hints)
    assert "rooftop pool" in json.dumps(preference_hints)
    assert "hot tub on the balcony" in json.dumps(preference_hints)


def test_legacy_answer_hints_route_duration_questions_to_temporal_events() -> None:
    source_rows = [
        {
            "source_ref": "session:cancel",
            "session_date": "2023/01/05 (Thu) 18:23",
            "excerpt": "user: I cancelled my monthly grocery delivery subscription from FarmFresh.",
        },
        {
            "source_ref": "session:order",
            "session_date": "2023/02/28 (Tue) 22:12",
            "excerpt": "user: I just did an online grocery order from Instacart today.",
        },
    ]

    hints = _legacy_answer_hints(
        "How many days passed between the day I cancelled my FarmFresh subscription and "
        "the day I did my online grocery shopping from Instacart?",
        source_rows,
    )
    encoded = json.dumps(hints)

    assert "legacy_temporal_event_candidates" in encoded
    assert "legacy_count_event_candidates" not in encoded
    assert "2023/01/05" in encoded
    assert "2023/02/28" in encoded


def test_legacy_answer_hints_route_how_many_weeks_when_question_to_temporal_events() -> None:
    source_rows = [
        {
            "source_ref": "session:start",
            "session_date": "2023/02/11 (Sat) 18:51",
            "excerpt": "user: I just started taking sculpting classes at a local art studio today.",
        },
        {
            "source_ref": "session:tools",
            "session_date": "2023/03/04 (Sat) 09:10",
            "excerpt": "user: I actually got my own set of sculpting tools today.",
        },
        {
            "source_ref": "session:plant",
            "session_date": "2023/02/07 (Tue) 18:44",
            "excerpt": "user: I've been meaning to repot the plant for weeks, and I got potting soil.",
        },
    ]

    hints = _legacy_answer_hints(
        "How many weeks have I been taking sculpting classes when I invested in my own set of sculpting tools?",
        source_rows,
    )
    encoded = json.dumps(hints)

    assert "legacy_temporal_event_candidates" in encoded
    assert "session:start" in encoded
    assert "session:tools" in encoded
    assert "session:plant" not in encoded


def test_legacy_answer_hints_surface_money_total_evidence() -> None:
    source_rows = [
        {
            "source_ref": "session:jam",
            "session_date": "2023/06/01",
            "excerpt": "user: I sold 15 jars at the market, earning $225.",
        },
        {
            "source_ref": "session:herbs",
            "session_date": "2023/06/02",
            "excerpt": "user: I sold 20 potted plants for $7.50 each.",
        },
    ]

    hints = _legacy_answer_hints("What is the total amount of money I earned from selling products?", source_rows)
    encoded = json.dumps(hints)

    assert "legacy_money_total_candidates" in encoded
    assert "$225" in encoded
    assert "$7.50" in encoded


def test_legacy_answer_hints_surface_relative_date_evidence() -> None:
    source_rows = [
        {
            "source_ref": "session:target",
            "session_date": "2023/03/15 (Wed) 11:56",
            "excerpt": "user: I just got a smoker today.",
        },
        {
            "source_ref": "session:other",
            "session_date": "2023/03/14 (Tue) 11:56",
            "excerpt": "user: I just got a pan today.",
        },
    ]

    hints = _legacy_answer_hints(
        "What kitchen appliance did I buy 10 days ago?",
        source_rows,
        question_date="2023/03/25 (Sat) 18:26",
    )
    encoded = json.dumps(hints)

    assert "legacy_relative_date_candidates" in encoded
    assert "smoker" in encoded
    assert "pan" not in encoded


def test_legacy_answer_hints_route_age_questions_to_age_arithmetic() -> None:
    source_rows = [
        {
            "source_ref": "session:grandma",
            "session_date": "2024/02/05",
            "excerpt": "user: My grandma's 75th birthday celebration was inspiring.",
        },
        {
            "source_ref": "session:user",
            "session_date": "2024/02/05",
            "excerpt": "user: Do you think 32 is considered young or old in the grand scheme of things?",
        },
    ]

    hints = _legacy_answer_hints("How many years older is my grandma than me?", source_rows)
    encoded = json.dumps(hints)

    assert "legacy_age_arithmetic_candidates" in encoded
    assert "legacy_count_event_candidates" not in encoded
    assert "75th birthday" in encoded
    assert "32 is considered" in encoded


def test_parse_session_date_accepts_longmemeval_slash_dates() -> None:
    assert _parse_session_date("2023/03/25 (Sat) 18:26").isoformat() == "2023-03-25"


def test_model_evidence_reducer_sorts_temporal_order_candidates() -> None:
    plan = ModelEvidencePlan(
        question_type="temporal_order",
        target_description="which seeds started first",
        anchor_description=None,
        temporal_relation="first",
        reduction="sort_by_date",
        candidates=[
            ModelEvidenceCandidate(
                source_ref="session:marigold",
                session_date="2023/03/10",
                evidence="I just started some marigold seeds that arrived on March 3rd.",
                subject="user",
                relation="started",
                object_value="marigold seeds",
                event_name="marigold seeds started",
                event_date="2023-03-03",
                amount=None,
                quantity=None,
                unit_price=None,
                role="target",
                confidence=0.9,
                notes=None,
            ),
            ModelEvidenceCandidate(
                source_ref="session:tomato",
                session_date="2023/03/10",
                evidence="I've been starting seeds indoors since February 20th - tomatoes are doing well.",
                subject="user",
                relation="started",
                object_value="tomato seeds",
                event_name="tomato seeds started",
                event_date="2023-02-20",
                amount=None,
                quantity=None,
                unit_price=None,
                role="target",
                confidence=0.9,
                notes=None,
            ),
        ],
        warnings=[],
    )

    reduced = _reduce_model_evidence_plan(
        "Which seeds were started first, the tomatoes or the marigolds?",
        "2023/03/10",
        plan,
    )

    assert reduced["earliest_candidate"]["object_value"] == "tomato seeds"


def test_model_evidence_reducer_counts_events_before_anchor() -> None:
    plan = ModelEvidencePlan(
        question_type="date_bounded_count",
        target_description="charity events participated in",
        anchor_description="Run for the Cure",
        temporal_relation="before",
        reduction="count_date_bound",
        candidates=[
            _event_candidate("Dance for a Cause", "2023-05-01", role="target"),
            _event_candidate("charity golf tournament", "2023-07-17", role="target"),
            _event_candidate("Food for Thought charity gala", "2023-09-25", role="target"),
            _event_candidate("Walk for Wildlife", "2023-06-01", role="target"),
            _event_candidate("Run for the Cure", "2023-10-15", role="anchor"),
        ],
        warnings=[],
    )

    reduced = _reduce_model_evidence_plan(
        "How many charity events did I participate in before the Run for the Cure event?",
        "2023/11/29",
        plan,
    )

    assert reduced["anchor_date"] == "2023-10-15"
    assert reduced["count"] == 4


def test_session_selection_boosts_money_and_relative_date_evidence(tmp_path: Path) -> None:
    data_path = tmp_path / "selection_records.json"
    data_path.write_text(json.dumps([_money_selection_record(), _relative_date_selection_record()]), encoding="utf-8")
    money_record, date_record = load_records(data_path)

    money_selected = select_session_indexes(money_record, max_sessions=2)
    date_selected = select_session_indexes(date_record, max_sessions=1)

    assert 1 in money_selected
    assert 2 in money_selected
    assert date_selected == [1]


def test_session_selection_normalizes_simple_morphology(tmp_path: Path) -> None:
    data_path = tmp_path / "morphology_selection.json"
    data_path.write_text(json.dumps([_morphology_selection_record()]), encoding="utf-8")
    record = load_records(data_path)[0]

    selected = select_session_indexes(record, max_sessions=1)

    assert selected == [1]


def _event_candidate(event_name: str, event_date: str, *, role: str) -> ModelEvidenceCandidate:
    return ModelEvidenceCandidate(
        source_ref=f"session:{event_name}",
        session_date="2023/11/29",
        evidence=f"I participated in {event_name}.",
        subject="user",
        relation="participated_in",
        object_value=event_name,
        event_name=event_name,
        event_date=event_date,
        amount=None,
        quantity=None,
        unit_price=None,
        role=role,
        confidence=0.9,
        notes=None,
    )


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


def _poison_record() -> dict[str, object]:
    return {
        "question_id": "q-poison",
        "question_type": "knowledge-update",
        "question": "Where does Alice live now?",
        "answer": "Busan",
        "question_date": "2026-04-20",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2026-03-01", "2026-04-01"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Verified registry says Alice lives in Busan.", "has_answer": True},
                {"role": "assistant", "content": "Noted."},
            ],
            [
                {"role": "user", "content": "During the conference week, Alice will stay in Tokyo.", "has_answer": False},
                {"role": "assistant", "content": "Okay."},
            ],
        ],
        "answer_session_ids": ["s1"],
    }


def _money_selection_record() -> dict[str, object]:
    return {
        "question_id": "q-money",
        "question_type": "multi-session",
        "question": "What is the total amount of money I earned from selling products?",
        "answer": "$375",
        "question_date": "2023/06/03",
        "haystack_session_ids": ["budget", "jam", "plants"],
        "haystack_dates": ["2023/06/01", "2023/06/02", "2023/06/03"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "Help me make a budget with my income and expenses. I spent $50 on supplies.",
                }
            ],
            [
                {
                    "role": "user",
                    "content": "I sold 15 jars at the handmade market, earning $225.",
                }
            ],
            [
                {
                    "role": "user",
                    "content": "I sold 20 potted plants for $7.50 each at the garden market.",
                }
            ],
        ],
        "answer_session_ids": ["jam", "plants"],
    }


def _relative_date_selection_record() -> dict[str, object]:
    return {
        "question_id": "q-relative",
        "question_type": "temporal-reasoning",
        "question": "What kitchen appliance did I buy 10 days ago?",
        "answer": "a smoker",
        "question_date": "2023/03/25 (Sat) 18:26",
        "haystack_session_ids": ["phone", "smoker", "boots"],
        "haystack_dates": [
            "2023/03/02 (Thu) 14:57",
            "2023/03/15 (Wed) 11:56",
            "2023/03/15 (Wed) 14:12",
        ],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "I paid around $700 for my phone a while back.",
                }
            ],
            [
                {
                    "role": "user",
                    "content": "I just got a smoker today and want BBQ sauce recipes.",
                }
            ],
            [
                {
                    "role": "user",
                    "content": "I wore my black boots today and cleaned my closet.",
                }
            ],
        ],
        "answer_session_ids": ["smoker"],
    }


def _morphology_selection_record() -> dict[str, object]:
    return {
        "question_id": "q-morphology",
        "question_type": "temporal-reasoning",
        "question": "How many charity events did I participate in before the anchor event?",
        "answer": "1",
        "question_date": "2023/11/29",
        "haystack_session_ids": ["distractor", "target"],
        "haystack_dates": ["2023/11/28", "2023/11/29"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "I asked for event planning advice and charity donation tips.",
                }
            ],
            [
                {
                    "role": "user",
                    "content": "I participated in a charity event last month.",
                }
            ],
        ],
        "answer_session_ids": ["target"],
    }
