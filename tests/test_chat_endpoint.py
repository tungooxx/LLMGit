from __future__ import annotations


def test_chat_endpoint_happy_path(client) -> None:
    response = client.post("/chat", json={"message": "Alice lives in Seoul."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["memory_updated"] is True
    assert payload["created_commit_id"] is not None
    assert payload["citations"][0]["subject"] == "Alice"
    assert payload["citations"][0]["object_value"] == "Seoul"


def test_chat_endpoint_explains_lineage(client) -> None:
    client.post("/chat", json={"message": "Alice lives in Seoul."})
    client.post("/chat", json={"message": "Alice moved to Busan in March 2026."})

    response = client.post(
        "/chat",
        json={"message": "Why do you think Alice is in Busan now if you previously said Seoul?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "lineage" in payload["answer"]
    assert len(payload["citations"]) >= 2


def test_low_trust_ingest_warns(client) -> None:
    client.post("/ingest", json={"raw_text": "Alice lives in Seoul.", "trust_score": 0.9})
    response = client.post(
        "/ingest",
        json={"raw_text": "Alice lives in Atlantis.", "trust_score": 0.2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert any("Low-trust" in warning for warning in payload["warnings"])
