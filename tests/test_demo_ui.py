from __future__ import annotations

from fastapi.testclient import TestClient


def test_demo_page_loads(client: TestClient) -> None:
    response = client.get("/demo")

    assert response.status_code == 200
    assert "TruthGit Memory Chat" in response.text
    assert "Memory Chat" in response.text
    assert "Git Graph" in response.text
    assert "use LLM" in response.text
    assert "LLM branch/trust" in response.text
    assert "ask current" in response.text
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
    assert first_payload["commit"]["id"] is not None
    assert "assistant_reply" in first_payload
    assert any(version["object_value"] == "Seoul" for version in first_payload["snapshot"]["versions"])

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
    versions = second.json()["snapshot"]["versions"]
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
    bad_commit_id = bad.json()["commit"]["id"]

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
