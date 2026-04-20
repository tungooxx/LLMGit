from __future__ import annotations

from fastapi.testclient import TestClient


def _approve_staged(client: TestClient, payload: dict) -> dict:
    staged_id = payload["staged"]["id"]
    response = client.post(
        f"/staged/{staged_id}/approve",
        json={"reviewer": "tester", "notes": "approved in demo test"},
    )
    assert response.status_code == 200
    return response.json()


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
    assert first_payload["staged"]["status"] == "pending"
    assert first_payload["staged"]["review_required"] is True
    first_approved = _approve_staged(client, first_payload)
    assert first_approved["commit"]["id"] is not None
    assert "assistant_reply" in first_payload
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
    _approve_staged(client, second_payload)
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
    assert bad_payload["staged"]["status"] == "pending"
    assert bad_payload["staged"]["review_required"] is True
    assert any("Review gate blocked auto-approval" in warning for warning in bad_payload["warnings"])
    assert not any(version["object_value"] == "Atlantis" for version in bad_payload["snapshot"]["versions"])

    bad_approved = _approve_staged(client, bad_payload)
    bad_commit_id = bad_approved["commit"]["id"]

    rollback = client.post("/demo/rollback", json={"commit_id": bad_commit_id})
    assert rollback.status_code == 200
    rollback_payload = rollback.json()
    assert any(version["object_value"] == "Atlantis" for version in rollback_payload["retracted_versions"])
    assert any(
        version["object_value"] == "Atlantis" and version["status"] == "retracted"
        for version in rollback_payload["snapshot"]["versions"]
    )


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
    assert payload["branch"]["name"] == "trip-plan"
    assert "branch-specific hypothetical" in payload["assistant_reply"]
    assert any(
        version["object_value"] == "Tokyo"
        and version["status"] == "hypothetical"
        and version["branch_name"] == "trip-plan"
        for version in payload["versions"]
    )


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
    _approve_staged(client, write.json())

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
        _approve_staged(client, response.json())

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
    assert payload["branch"]["name"] == "trip-plan"
    assert payload["staged"]["status"] == "pending"
    assert payload["extraction"]["mode"] == "llm"
    assert payload["extraction"]["used_fallback"] is True
    assert payload["assistant_reply"]
    assert any("OPENAI_API_KEY" in warning for warning in payload["warnings"])
