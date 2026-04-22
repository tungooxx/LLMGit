from __future__ import annotations

from fastapi.testclient import TestClient

from app.schemas import ExtractedClaim, MemoryWritePlan


def test_demo_page_loads(client: TestClient) -> None:
    response = client.get("/demo")

    assert response.status_code == 200
    assert "TruthGit Memory Chat" in response.text
    assert "Memory Chat" in response.text
    assert "Git Graph" in response.text
    assert "use LLM" in response.text
    assert "LLM branch/trust" in response.text
    assert "ask current" in response.text
    assert "Load Benchmark Case" in response.text
    assert "lineageStrip" in response.text
    assert "Benchmark Playback" not in response.text


def test_demo_manual_prompt_supersession_and_rollback(client: TestClient) -> None:
    reset = client.post("/demo/reset", json={"confirm": True})
    assert reset.status_code == 200

    first = client.post(
        "/demo/manual",
        json={
            "message": "Alice lives in Seoul.",
            "branch_name": "main",
            "trust_score": 0.8,
            "auto_approve": True,
        },
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["commit"] is None
    assert first_payload["staged"]["status"] == "review_required"
    assert first_payload["staged"]["review_required"] is True
    assert "staged it" in first_payload["assistant_reply"]
    assert "assistant_reply" in first_payload
    _approve_demo_staged(client, first_payload["staged"]["id"])
    snapshot = client.get("/viz/data").json()
    assert any(version["object_value"] == "Seoul" for version in snapshot["belief_versions"])

    second = client.post(
        "/demo/manual",
        json={
            "message": "Alice moved to Busan in March 2026.",
            "branch_name": "main",
            "trust_score": 0.86,
            "auto_approve": True,
        },
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["commit"] is None
    assert second_payload["staged"]["status"] == "review_required"
    _approve_demo_staged(client, second_payload["staged"]["id"])
    versions = client.get("/viz/data").json()["belief_versions"]
    assert any(version["object_value"] == "Busan" and version["status"] == "active" for version in versions)
    assert any(version["object_value"] == "Seoul" and version["status"] == "superseded" for version in versions)

    bad = client.post(
        "/demo/manual",
        json={
            "message": "Alice lives in Atlantis.",
            "branch_name": "main",
            "trust_score": 0.2,
            "auto_approve": True,
        },
    )
    assert bad.status_code == 200
    bad_payload = bad.json()
    assert bad_payload["commit"] is None
    assert bad_payload["staged"]["status"] == "quarantined"
    assert bad_payload["staged"]["review_required"] is True

    approved = client.post(
        f"/quarantine/{bad_payload['staged']['id']}/approve-and-apply",
        json={"reviewer": "tester", "notes": "Intentional bad-memory injection for rollback demo."},
    )
    assert approved.status_code == 200
    bad_commit_id = approved.json()["commit"]["id"]

    rollback = client.post("/demo/rollback", json={"commit_id": bad_commit_id})
    assert rollback.status_code == 200
    rollback_payload = rollback.json()
    assert any(version["object_value"] == "Atlantis" for version in rollback_payload["retracted_versions"])
    assert any(
        version["object_value"] == "Atlantis" and version["status"] == "retracted"
        for version in rollback_payload["snapshot"]["versions"]
    )


def test_demo_memory_context_includes_support_graph(client: TestClient) -> None:
    reset = client.post("/demo/reset", json={"confirm": True})
    assert reset.status_code == 200

    response = client.post(
        "/demo/manual",
        json={
            "message": "A verified city registry says Alice lives in Busan.",
            "branch_name": "main",
            "trust_score": 0.8,
            "auto_approve": True,
        },
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["commit"] is None
    assert payload["staged"]["status"] == "review_required"
    _approve_demo_staged(client, payload["staged"]["id"])
    active = client.get("/beliefs/active?subject=Alice&predicate=lives_in")
    assert active.status_code == 200
    busan = next(version for version in active.json() if version["object_value"] == "Busan")

    assert busan["active_support_count"] == 1
    assert "support_sources" in busan
    assert "verified city registry" in busan["support_sources"][0]["excerpt"]


def test_demo_manual_prompt_branch_only_hypothetical(client: TestClient) -> None:
    reset = client.post("/demo/reset", json={"confirm": True})
    assert reset.status_code == 200

    response = client.post(
        "/demo/manual",
        json={
            "message": "During the conference week, Alice will stay in Tokyo.",
            "branch_name": "trip-plan",
            "trust_score": 0.78,
            "auto_approve": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["branch"]["name"] == "conference-week"
    assert payload["commit"]["id"] is not None
    assert any(
        version["object_value"] == "Tokyo"
        and version["status"] == "hypothetical"
        and version["branch_name"] == "conference-week"
        for version in payload["versions"]
    )


def test_demo_llm_model_selected_branch_is_honored(client: TestClient, monkeypatch) -> None:
    class FakeLLMClient:
        client = object()

        def __init__(self, settings=None) -> None:
            del settings

        def plan_memory_write(
            self,
            text: str,
            *,
            fallback_branch_name: str = "main",
            fallback_trust_score: float = 0.7,
            memory_context: dict | None = None,
        ) -> MemoryWritePlan:
            del text, fallback_branch_name, fallback_trust_score, memory_context
            return MemoryWritePlan(
                claims=[
                    ExtractedClaim(
                        subject="Alice",
                        predicate="lives_in",
                        object_value="Tokyo",
                        confidence=0.84,
                        source_quote="On the trip-plan branch, during the conference week, Alice will stay in Tokyo.",
                    )
                ],
                branch_name="trip-plan",
                trust_score=0.84,
                write_action="branch_hypothetical",
                rationale="Model selected a branch for future travel.",
                assistant_reply="Okay, I'll keep that conference-week memory separate from main truth.",
            )

    monkeypatch.setattr("app.routes.demo.LLMClient", FakeLLMClient)
    reset = client.post("/demo/reset", json={"confirm": True})
    assert reset.status_code == 200

    response = client.post(
        "/demo/manual",
        json={
            "message": "On the trip-plan branch, during the conference week, Alice will stay in Tokyo.",
            "branch_name": "main",
            "trust_score": 0.8,
            "auto_approve": True,
            "extraction_mode": "llm",
            "auto_metadata": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["branch"]["name"] == "trip-plan"
    assert payload["commit"]["id"] is not None
    assert payload["staged"]["status"] == "applied"
    assert any(
        version["object_value"] == "Tokyo"
        and version["status"] == "hypothetical"
        and version["branch_name"] == "trip-plan"
        for version in payload["versions"]
    )


def test_demo_what_if_scenario_stages_branch_memory(client: TestClient, monkeypatch) -> None:
    class FakeLLMClient:
        client = object()

        def __init__(self, settings=None) -> None:
            del settings

        def plan_memory_write(
            self,
            text: str,
            *,
            fallback_branch_name: str = "main",
            fallback_trust_score: float = 0.7,
            memory_context: dict | None = None,
        ) -> MemoryWritePlan:
            del text, fallback_branch_name, fallback_trust_score, memory_context
            return MemoryWritePlan(
                claims=[
                    ExtractedClaim(
                        subject="Alice",
                        predicate="stays_in",
                        object_value="Kyoto",
                        confidence=0.84,
                        source_quote="During the fellowship month, she will stay in Kyoto.",
                    ),
                    ExtractedClaim(
                        subject="Alice",
                        predicate="works_from",
                        object_value="university lab",
                        confidence=0.8,
                        source_quote="During the fellowship month, she will work from the university lab.",
                    ),
                ],
                branch_name="main",
                trust_score=0.84,
                write_action="branch_hypothetical",
                rationale="What-if fellowship scenario belongs on a branch.",
                assistant_reply="Okay, I'll keep that Kyoto fellowship scenario separate from main truth.",
            )

    monkeypatch.setattr("app.routes.demo.LLMClient", FakeLLMClient)
    reset = client.post("/demo/reset", json={"confirm": True})
    assert reset.status_code == 200

    response = client.post(
        "/demo/manual",
        json={
            "message": "What if Alice accepts the Kyoto fellowship? During the fellowship month, she will stay in Kyoto and work from the university lab.",
            "branch_name": "main",
            "trust_score": 0.8,
            "auto_approve": True,
            "extraction_mode": "llm",
            "auto_metadata": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["branch"]["name"] == "kyoto-fellowship"
    assert payload["commit"]["id"] is not None
    assert any(
        version["object_value"] == "Kyoto"
        and version["status"] == "hypothetical"
        and version["branch_name"] == "kyoto-fellowship"
        for version in payload["versions"]
    )


def test_demo_confirmed_conference_fact_stays_on_main(client: TestClient, monkeypatch) -> None:
    class FakeLLMClient:
        client = object()

        def __init__(self, settings=None) -> None:
            del settings

        def plan_memory_write(
            self,
            text: str,
            *,
            fallback_branch_name: str = "main",
            fallback_trust_score: float = 0.7,
            memory_context: dict | None = None,
        ) -> MemoryWritePlan:
            del text, fallback_branch_name, fallback_trust_score, memory_context
            return MemoryWritePlan(
                claims=[
                    ExtractedClaim(
                        subject="Alice",
                        predicate="stays_in",
                        object_value="Tokyo",
                        confidence=0.86,
                        source_quote="Confirmed: Alice is already in Tokyo for the conference.",
                    )
                ],
                branch_name="main",
                trust_score=0.86,
                write_action="commit_now",
                rationale="Confirmed current conference fact.",
                assistant_reply="Okay, I'll treat that as confirmed current memory.",
            )

    monkeypatch.setattr("app.routes.demo.LLMClient", FakeLLMClient)
    reset = client.post("/demo/reset", json={"confirm": True})
    assert reset.status_code == 200

    response = client.post(
        "/demo/manual",
        json={
            "message": "Confirmed: Alice is already in Tokyo for the conference.",
            "branch_name": "main",
            "trust_score": 0.8,
            "auto_approve": True,
            "extraction_mode": "llm",
            "auto_metadata": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["branch"]["name"] == "main"
    assert payload["commit"]["id"] is not None
    assert any(
        version["object_value"] == "Tokyo"
        and version["status"] == "active"
        and version["branch_name"] == "main"
        for version in payload["versions"]
    )


def test_demo_caps_overtrusted_implausible_location_claim(client: TestClient, monkeypatch) -> None:
    class FakeLLMClient:
        client = object()

        def __init__(self, settings=None) -> None:
            del settings

        def plan_memory_write(
            self,
            text: str,
            *,
            fallback_branch_name: str = "main",
            fallback_trust_score: float = 0.7,
            memory_context: dict | None = None,
        ) -> MemoryWritePlan:
            del text, fallback_branch_name, fallback_trust_score, memory_context
            return MemoryWritePlan(
                claims=[
                    ExtractedClaim(
                        subject="Alice",
                        predicate="lives_in",
                        object_value="Atlantis",
                        confidence=0.95,
                        source_quote="Alice lives in Atlantis.",
                    )
                ],
                branch_name="main",
                trust_score=0.7,
                write_action="commit_now",
                rationale="Model overtrusted a clear but implausible claim.",
                assistant_reply="Okay, I'll remember that Alice lives in Atlantis.",
            )

    monkeypatch.setattr("app.routes.demo.LLMClient", FakeLLMClient)
    reset = client.post("/demo/reset", json={"confirm": True})
    assert reset.status_code == 200

    response = client.post(
        "/demo/manual",
        json={
            "message": "Alice lives in Atlantis.",
            "branch_name": "main",
            "trust_score": 0.8,
            "auto_approve": True,
            "extraction_mode": "llm",
            "auto_metadata": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["commit"] is None
    assert payload["staged"]["status"] == "quarantined"
    assert payload["staged"]["review_required"] is True
    assert payload["extraction"]["trust_score"] == 0.25
    assert any("Implausible real-world claim" in warning for warning in payload["warnings"])


def test_demo_branch_fallback_does_not_capture_scheduled_internship(client: TestClient, monkeypatch) -> None:
    class FakeLLMClient:
        client = object()

        def __init__(self, settings=None) -> None:
            del settings

        def plan_memory_write(
            self,
            text: str,
            *,
            fallback_branch_name: str = "main",
            fallback_trust_score: float = 0.7,
            memory_context: dict | None = None,
        ) -> MemoryWritePlan:
            del text, fallback_branch_name, fallback_trust_score, memory_context
            return MemoryWritePlan(
                claims=[
                    ExtractedClaim(
                        subject="Alice",
                        predicate="internship_starts_in",
                        object_value="new company",
                        confidence=0.82,
                        source_quote="Tomorrow Alice will start internship in her new company.",
                    )
                ],
                branch_name="trip-plan",
                trust_score=0.82,
                write_action="commit_now",
                rationale="Model copied branch fallback.",
                assistant_reply="Okay, I'll remember that scheduled internship.",
            )

    monkeypatch.setattr("app.routes.demo.LLMClient", FakeLLMClient)
    reset = client.post("/demo/reset", json={"confirm": True})
    assert reset.status_code == 200

    response = client.post(
        "/demo/manual",
        json={
            "message": "Tomorrow Alice will start internship in her new company.",
            "branch_name": "trip-plan",
            "trust_score": 0.8,
            "auto_approve": True,
            "extraction_mode": "llm",
            "auto_metadata": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["branch"]["name"] == "main"
    assert payload["commit"]["id"] is not None
    assert any(
        version["object_value"] == "new company"
        and version["status"] == "active"
        and version["branch_name"] == "main"
        for version in payload["versions"]
    )
    assert any("scheduled fact to main" in warning for warning in payload["warnings"])


def test_demo_stale_branch_fallback_does_not_capture_new_scenario(client: TestClient, monkeypatch) -> None:
    class FakeLLMClient:
        client = object()

        def __init__(self, settings=None) -> None:
            del settings

        def plan_memory_write(
            self,
            text: str,
            *,
            fallback_branch_name: str = "main",
            fallback_trust_score: float = 0.7,
            memory_context: dict | None = None,
        ) -> MemoryWritePlan:
            del text, fallback_branch_name, fallback_trust_score, memory_context
            return MemoryWritePlan(
                claims=[
                    ExtractedClaim(
                        subject="Alice",
                        predicate="lives_in",
                        object_value="Berlin",
                        confidence=0.8,
                        source_quote=(
                            "In the Berlin relocation scenario, Alice would live in Berlin from July 2026 "
                            "while testing a remote work plan."
                        ),
                    )
                ],
                branch_name="conference-week",
                trust_score=0.8,
                write_action="commit_now",
                rationale="Model copied stale branch fallback.",
                assistant_reply="Okay, I'll remember that Berlin relocation scenario.",
            )

    monkeypatch.setattr("app.routes.demo.LLMClient", FakeLLMClient)
    reset = client.post("/demo/reset", json={"confirm": True})
    assert reset.status_code == 200

    response = client.post(
        "/demo/manual",
        json={
            "message": (
                "In the Berlin relocation scenario, Alice would live in Berlin from July 2026 "
                "while testing a remote work plan."
            ),
            "branch_name": "conference-week",
            "trust_score": 0.8,
            "auto_approve": True,
            "extraction_mode": "llm",
            "auto_metadata": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["branch"]["name"] == "berlin-relocation"
    assert payload["commit"]["id"] is not None
    assert any(
        version["object_value"] == "Berlin"
        and version["status"] == "hypothetical"
        and version["branch_name"] == "berlin-relocation"
        for version in payload["versions"]
    )
    assert all(branch["name"] != "conference-week" for branch in payload["snapshot"]["branches"])
    assert any("stale branch fallback" in warning for warning in payload["warnings"])


def test_demo_loads_one_benchmark_case_into_graph(client: TestClient) -> None:
    response = client.post("/demo/benchmark-case", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["benchmark_case"]["question_id"] == "demo-alice"
    assert payload["benchmark_case"]["mode"] == "LongMemEval-style record_batch"
    assert "Busan" in payload["benchmark_case"]["answer"]
    assert payload["staged"]["status"] == "applied"
    assert payload["staged"]["source_ref"] == "longmemeval:demo-alice:selected-history"
    assert payload["commit"]["id"] is not None
    assert len(payload["claims"]) == 2
    versions = payload["snapshot"]["versions"]
    assert any(version["object_value"] == "Busan" and version["status"] == "active" for version in versions)
    assert any(version["object_value"] == "Seoul" and version["status"] == "superseded" for version in versions)
    assert payload["snapshot"]["counts"]["commits"] == 1
    assert payload["snapshot"]["counts"]["staged"] == 0


def test_demo_answers_questions_from_memory_without_writing(client: TestClient) -> None:
    reset = client.post("/demo/reset", json={"confirm": True})
    assert reset.status_code == 200

    write = client.post(
        "/demo/manual",
        json={
            "message": "Alice lives in Seoul.",
            "branch_name": "main",
            "trust_score": 0.8,
            "auto_approve": True,
            "extraction_mode": "local",
        },
    )
    assert write.status_code == 200
    write_payload = write.json()
    assert write_payload["commit"] is None
    _approve_demo_staged(client, write_payload["staged"]["id"])

    answer = client.post(
        "/demo/manual",
        json={
            "message": "Where does Alice live now?",
            "branch_name": "main",
            "trust_score": 0.8,
            "auto_approve": True,
            "extraction_mode": "llm",
            "auto_metadata": True,
        },
    )

    assert answer.status_code == 200
    payload = answer.json()
    assert payload["claims"] == []
    assert payload["staged"] is None
    assert payload["commit"] is None
    assert "Seoul" in payload["assistant_reply"]


def test_demo_why_question_does_not_stage_duplicate_memory(client: TestClient) -> None:
    reset = client.post("/demo/reset", json={"confirm": True})
    assert reset.status_code == 200

    for message in ("Alice lives in Seoul.", "Alice moved to Busan in March 2026."):
        response = client.post(
            "/demo/manual",
            json={
                "message": message,
                "branch_name": "main",
                "trust_score": 0.8,
                "auto_approve": True,
                "extraction_mode": "local",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["commit"] is None
        _approve_demo_staged(client, payload["staged"]["id"])

    before = client.get("/viz/data")
    assert before.status_code == 200
    before_counts = before.json()["counts"]

    answer = client.post(
        "/demo/manual",
        json={
            "message": "Why do you think Alice lives in Busan if earlier she lived in Seoul?",
            "branch_name": "main",
            "trust_score": 0.8,
            "auto_approve": True,
            "extraction_mode": "llm",
            "auto_metadata": True,
        },
    )

    assert answer.status_code == 200
    payload = answer.json()
    assert payload["claims"] == []
    assert payload["staged"] is None
    assert payload["commit"] is None
    assert "Busan" in payload["assistant_reply"]

    after = client.get("/viz/data")
    assert after.status_code == 200
    after_counts = after.json()["counts"]
    assert after_counts["commits"] == before_counts["commits"]
    assert after_counts["versions"] == before_counts["versions"]


def test_demo_show_audit_trail_is_answered_without_staging(client: TestClient) -> None:
    reset = client.post("/demo/reset", json={"confirm": True})
    assert reset.status_code == 200

    seed = client.post(
        "/demo/manual",
        json={
            "message": "A verified city registry says Alice lives in Busan.",
            "branch_name": "main",
            "trust_score": 0.8,
            "auto_approve": True,
            "extraction_mode": "local",
        },
    )
    assert seed.status_code == 200
    _approve_demo_staged(client, seed.json()["staged"]["id"])

    bad = client.post(
        "/demo/manual",
        json={
            "message": "An anonymous forum joke says Alice lives in Atlantis now.",
            "branch_name": "main",
            "trust_score": 0.2,
            "auto_approve": True,
            "extraction_mode": "local",
        },
    )
    assert bad.status_code == 200
    bad_payload = bad.json()
    assert bad_payload["staged"]["status"] == "quarantined"
    rejected = client.post(
        f"/quarantine/{bad_payload['staged']['id']}/reject",
        json={"reviewer": "tester", "notes": "Rejected poisoned demo source."},
    )
    assert rejected.status_code == 200

    before = client.get("/viz/data").json()["counts"]
    answer = client.post(
        "/demo/manual",
        json={
            "message": "Show the audit trail for the rejected Atlantis memory.",
            "branch_name": "main",
            "trust_score": 0.8,
            "auto_approve": True,
            "extraction_mode": "llm",
            "auto_metadata": True,
        },
    )

    assert answer.status_code == 200
    payload = answer.json()
    assert payload["claims"] == []
    assert payload["staged"] is None
    assert payload["commit"] is None
    assert "Atlantis" in payload["assistant_reply"]
    assert "rejected" in payload["assistant_reply"]
    after = client.get("/viz/data").json()["counts"]
    assert after["commits"] == before["commits"]
    assert after["versions"] == before["versions"]


def _approve_demo_staged(client: TestClient, staged_id: str) -> dict:
    response = client.post(
        f"/staged/{staged_id}/approve",
        json={"reviewer": "tester", "notes": "approved during demo test"},
    )
    assert response.status_code == 200
    return response.json()


def test_demo_llm_mode_falls_back_and_suggests_metadata_without_api_key(client: TestClient) -> None:
    reset = client.post("/demo/reset", json={"confirm": True})
    assert reset.status_code == 200

    response = client.post(
        "/demo/manual",
        json={
            "message": "During the conference week, Alice will stay in Tokyo.",
            "branch_name": "main",
            "trust_score": 0.7,
            "auto_approve": False,
            "extraction_mode": "llm",
            "auto_metadata": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["branch"]["name"] == "conference-week"
    assert payload["staged"]["status"] == "checked"
    assert payload["extraction"]["mode"] == "llm"
    assert payload["extraction"]["used_fallback"] is True
    assert payload["assistant_reply"]
    assert any("OPENAI_API_KEY" in warning for warning in payload["warnings"])
