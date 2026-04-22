from __future__ import annotations

from app.main import app
from app.routes.chat import _source_label, get_llm_client
from app.schemas import ExtractedClaim, MemoryWritePlan


def test_chat_endpoint_happy_path(client) -> None:
    response = client.post("/chat", json={"message": "Alice lives in Seoul."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["memory_updated"] is False
    assert payload["created_commit_id"] is None
    assert payload["staged_commit_id"] is not None
    assert payload["review_required"] is True
    staged = client.get(f"/staged/{payload['staged_commit_id']}")
    assert staged.status_code == 200
    assert staged.json()["status"] == "review_required"


def test_internal_source_label_falls_back_to_excerpt() -> None:
    label = _source_label(
        {
            "source_id": 3,
            "source_ref": "demo-ui:llm:main",
            "excerpt": "A verified city registry says Alice lives in Busan.",
        }
    )

    assert "verified city registry" in label
    assert "demo-ui" not in label


def test_chat_endpoint_explains_lineage(client) -> None:
    first = client.post("/chat", json={"message": "Alice lives in Seoul."}).json()
    first_commit = _approve_staged(client, first["staged_commit_id"])
    assert first_commit["commit"]["id"] is not None
    second = client.post("/chat", json={"message": "Alice moved to Busan in March 2026."}).json()
    second_commit = _approve_staged(client, second["staged_commit_id"])
    assert second_commit["commit"]["id"] is not None

    response = client.post(
        "/chat",
        json={"message": "Why do you think Alice is in Busan now if you previously said Seoul?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "lineage" in payload["answer"]
    assert len(payload["citations"]) >= 2


def test_low_trust_ingest_warns(client) -> None:
    class FakeReviewLLM:
        def plan_memory_write(self, text: str, **kwargs: object) -> MemoryWritePlan:
            del kwargs
            if "Atlantis" in text:
                return MemoryWritePlan(
                    claims=[
                        ExtractedClaim(
                            subject="Alice",
                            predicate="lives_in",
                            object_value="Atlantis",
                            confidence=0.3,
                            source_quote=text,
                        )
                    ],
                    branch_name="main",
                    trust_score=0.25,
                    write_action="stage_for_review",
                    risk_reasons=["model_low_trust_source"],
                    warnings=["Model judged this source low-trust and requested review."],
                    rationale="The text names an unreliable location claim.",
                    assistant_reply="I staged that claim for review instead of committing it.",
                )
            return MemoryWritePlan(
                claims=[
                    ExtractedClaim(
                        subject="Alice",
                        predicate="lives_in",
                        object_value="Seoul",
                        confidence=0.9,
                        source_quote=text,
                    )
                ],
                branch_name="main",
                trust_score=0.9,
                write_action="commit_now",
                rationale="Direct explicit claim.",
                assistant_reply="Okay, I'll remember that.",
            )

    app.dependency_overrides[get_llm_client] = lambda: FakeReviewLLM()
    first = client.post("/ingest", json={"raw_text": "Alice lives in Seoul.", "trust_score": 0.9}).json()
    assert first["memory_updated"] is False
    _approve_staged(client, first["staged_commit_id"])
    response = client.post(
        "/ingest",
        json={"raw_text": "Alice lives in Atlantis.", "trust_score": 0.2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["memory_updated"] is False
    assert payload["review_required"] is True
    assert any("low-trust" in warning for warning in payload["warnings"])


def test_model_commit_now_conflict_is_staged_by_policy(client) -> None:
    class FakeOverconfidentConflictLLM:
        def plan_memory_write(self, text: str, **kwargs: object) -> MemoryWritePlan:
            del kwargs
            if "Atlantis" in text:
                return MemoryWritePlan(
                    claims=[
                        ExtractedClaim(
                            subject="Alice",
                            predicate="lives_in",
                            object_value="Atlantis",
                            confidence=0.6,
                            source_quote=text,
                        )
                    ],
                    branch_name="main",
                    trust_score=0.6,
                    write_action="commit_now",
                    rationale="Model incorrectly wanted to commit a weak conflict.",
                    assistant_reply="Okay, I'll remember that.",
                )
            return MemoryWritePlan(
                claims=[
                    ExtractedClaim(
                        subject="Alice",
                        predicate="lives_in",
                        object_value="Seoul",
                        confidence=0.95,
                        source_quote=text,
                    )
                ],
                branch_name="main",
                trust_score=0.95,
                write_action="commit_now",
                rationale="Direct explicit claim.",
                assistant_reply="Okay, I'll remember that.",
            )

    app.dependency_overrides[get_llm_client] = lambda: FakeOverconfidentConflictLLM()
    first = client.post("/chat", json={"message": "Alice lives in Seoul."}).json()
    assert first["memory_updated"] is False
    _approve_staged(client, first["staged_commit_id"])

    response = client.post("/chat", json={"message": "Alice lives in Atlantis."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["memory_updated"] is False
    assert payload["review_required"] is True
    assert any("stronger active memory" in warning for warning in payload["warnings"])


def test_implausible_location_claim_caps_trust_and_requires_review(client) -> None:
    class FakeOvertrustingLLM:
        def plan_memory_write(self, text: str, **kwargs: object) -> MemoryWritePlan:
            del kwargs
            return MemoryWritePlan(
                claims=[
                    ExtractedClaim(
                        subject="Alice",
                        predicate="lives_in",
                        object_value="Atlantis",
                        confidence=0.95,
                        source_quote=text,
                    )
                ],
                branch_name="main",
                trust_score=0.7,
                write_action="commit_now",
                rationale="Model overtrusted a clear but implausible claim.",
                assistant_reply="Okay, I'll remember that.",
            )

    app.dependency_overrides[get_llm_client] = lambda: FakeOvertrustingLLM()
    response = client.post("/chat", json={"message": "Alice lives in Atlantis."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["memory_updated"] is False
    assert payload["review_required"] is True
    assert any("Implausible real-world claim" in warning for warning in payload["warnings"])


def test_model_branch_hypothetical_action_writes_non_main_branch(client) -> None:
    class FakeBranchLLM:
        def plan_memory_write(self, text: str, **kwargs: object) -> MemoryWritePlan:
            del kwargs
            return MemoryWritePlan(
                claims=[
                    ExtractedClaim(
                        subject="Alice",
                        predicate="stays_in",
                        object_value="Tokyo",
                        confidence=0.82,
                        source_quote=text,
                    )
                ],
                branch_name="main",
                trust_score=0.82,
                write_action="branch_hypothetical",
                rationale="Branch-only conference-week fact.",
                assistant_reply="Okay, I'll keep that separate from main truth.",
            )

    app.dependency_overrides[get_llm_client] = lambda: FakeBranchLLM()
    response = client.post("/chat", json={"message": "During the conference week, Alice will stay in Tokyo."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["memory_updated"] is True
    assert payload["review_required"] is False
    assert payload["branch"]["name"] == "conference-week"
    assert payload["citations"][0]["status"] == "hypothetical"


def test_branch_fallback_does_not_capture_unrelated_scheduled_fact(client) -> None:
    class FakeScheduledFactLLM:
        def plan_memory_write(self, text: str, **kwargs: object) -> MemoryWritePlan:
            del kwargs
            return MemoryWritePlan(
                claims=[
                    ExtractedClaim(
                        subject="Alice",
                        predicate="internship_starts_in",
                        object_value="new company",
                        confidence=0.82,
                        source_quote=text,
                    )
                ],
                branch_name="trip-plan",
                trust_score=0.82,
                write_action="commit_now",
                rationale="Model copied the UI branch fallback.",
                assistant_reply="Okay, I'll remember that.",
            )

    app.dependency_overrides[get_llm_client] = lambda: FakeScheduledFactLLM()
    response = client.post(
        "/chat",
        json={"message": "Tomorrow Alice will start internship in her new company."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["memory_updated"] is True
    assert payload["branch"]["name"] == "main"
    assert payload["citations"][0]["status"] == "active"
    assert any("scheduled fact to main" in warning for warning in payload["warnings"])


def test_branch_fallback_does_not_capture_unrelated_branch_scenario(client) -> None:
    class FakeStaleBranchLLM:
        def plan_memory_write(self, text: str, **kwargs: object) -> MemoryWritePlan:
            del kwargs
            return MemoryWritePlan(
                claims=[
                    ExtractedClaim(
                        subject="Alice",
                        predicate="lives_in",
                        object_value="Berlin",
                        confidence=0.8,
                        source_quote=text,
                    )
                ],
                branch_name="conference-week",
                trust_score=0.8,
                write_action="commit_now",
                rationale="Model copied a stale branch fallback.",
                assistant_reply="Okay, I'll remember that scenario.",
            )

    app.dependency_overrides[get_llm_client] = lambda: FakeStaleBranchLLM()
    response = client.post(
        "/chat",
        json={
            "message": "In the Berlin relocation scenario, Alice would live in Berlin from July 2026 while testing a remote work plan."
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["memory_updated"] is True
    assert payload["branch"]["name"] == "berlin-relocation"
    assert payload["citations"][0]["status"] == "hypothetical"
    assert any("stale branch fallback" in warning for warning in payload["warnings"])


def test_staged_commit_is_db_backed_and_rejectable(client) -> None:
    response = client.post("/chat", json={"message": "Alice lives in Seoul.", "auto_commit": False})
    payload = response.json()
    staged_id = payload["staged_commit_id"]

    fetched = client.get(f"/staged/{staged_id}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "review_required"

    rejected = client.post(
        f"/staged/{staged_id}/reject",
        json={"reviewer": "tester", "notes": "bad source"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"

    active = client.get("/beliefs/active?subject=Alice&predicate=lives_in")
    assert active.status_code == 200
    assert active.json() == []


def _approve_staged(client, staged_id: str) -> dict:
    response = client.post(
        f"/staged/{staged_id}/approve",
        json={"reviewer": "tester", "notes": "approved in test"},
    )
    assert response.status_code == 200
    return response.json()
