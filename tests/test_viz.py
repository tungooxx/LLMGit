from __future__ import annotations

from fastapi.testclient import TestClient


def test_visualization_page_loads(client: TestClient) -> None:
    response = client.get("/viz")

    assert response.status_code == 200
    assert "TruthGit Memory Graph" in response.text


def test_visualization_data_shape(client: TestClient) -> None:
    response = client.get("/viz/data")

    assert response.status_code == 200
    payload = response.json()
    assert "counts" in payload
    assert "branches" in payload
    assert "belief_versions" in payload
    assert "staged_commits" in payload
    assert "audit_events" in payload
    assert payload["counts"]["branches"] >= 1
