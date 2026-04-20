from __future__ import annotations


def test_chat_endpoint_happy_path(client) -> None:
    response = client.post("/chat", json={"message": "Alice lives in Seoul."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["memory_updated"] is False
    assert payload["created_commit_id"] is None
    assert payload["staged_commit_id"] is not None
    assert payload["review_required"] is True

    approval = client.post(
        f"/staged/{payload['staged_commit_id']}/approve",
        json={"reviewer": "tester", "notes": "verified"},
    )

    assert approval.status_code == 200
    applied = approval.json()
    assert applied["commit"]["id"] is not None
    assert applied["introduced_versions"][0]["object_value"] == "Seoul"


def test_chat_endpoint_explains_lineage(client) -> None:
    first = client.post("/chat", json={"message": "Alice lives in Seoul."}).json()
    client.post(f"/staged/{first['staged_commit_id']}/approve", json={"reviewer": "tester"})
    second = client.post("/chat", json={"message": "Alice moved to Busan in March 2026."}).json()
    client.post(f"/staged/{second['staged_commit_id']}/approve", json={"reviewer": "tester"})

    response = client.post(
        "/chat",
        json={"message": "Why do you think Alice is in Busan now if you previously said Seoul?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "lineage" in payload["answer"]
    assert len(payload["citations"]) >= 2


def test_low_trust_ingest_warns(client) -> None:
    first = client.post("/ingest", json={"raw_text": "Alice lives in Seoul.", "trust_score": 0.9}).json()
    client.post(f"/staged/{first['staged_commit_id']}/approve", json={"reviewer": "tester"})
    response = client.post(
        "/ingest",
        json={"raw_text": "Alice lives in Atlantis.", "trust_score": 0.2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["memory_updated"] is False
    assert payload["review_required"] is True
    assert any("Low-trust" in warning for warning in payload["warnings"])


def test_staged_commit_is_db_backed_and_rejectable(client) -> None:
    response = client.post("/chat", json={"message": "Alice lives in Seoul."})
    payload = response.json()
    staged_id = payload["staged_commit_id"]

    fetched = client.get(f"/staged/{staged_id}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "pending"

    rejected = client.post(
        f"/staged/{staged_id}/reject",
        json={"reviewer": "tester", "notes": "bad source"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"

    active = client.get("/beliefs/active?subject=Alice&predicate=lives_in")
    assert active.status_code == 200
    assert active.json() == []
