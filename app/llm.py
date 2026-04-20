"""OpenAI integration for claim extraction and answer generation."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from openai import OpenAI, OpenAIError

from app.config import Settings, get_settings
from app.normalization import deterministic_extract_simple_claims
from app.schemas import AnswerPlan, ExtractedClaim, ExtractedClaimList, MemoryWritePlan

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI Responses API client with deterministic local fallbacks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key) if self.settings.openai_api_key else None

    def extract_claims(self, text: str) -> ExtractedClaimList:
        """Extract atomic claims using structured outputs, falling back locally without an API key."""

        if self.client is None:
            return ExtractedClaimList.model_validate(
                {"claims": deterministic_extract_simple_claims(text)}
            )
        schema = openai_strict_json_schema(ExtractedClaimList)
        try:
            response = self.client.responses.create(
                model=self.settings.openai_model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "Extract only explicit atomic factual claims. "
                            "Do not infer answers from questions. Return no claims for pure questions."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "ExtractedClaimList",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
            output_text = getattr(response, "output_text", None) or _response_text(response)
            return ExtractedClaimList.model_validate_json(output_text)
        except (OpenAIError, ValueError, TypeError) as exc:
            if self.settings.app_env == "production":
                raise
            logger.warning("Falling back to deterministic claim extraction: %s", exc)
            return ExtractedClaimList.model_validate(
                {"claims": deterministic_extract_simple_claims(text)}
            )

    def plan_answer(self, message: str, candidate_belief_ids: list[int]) -> AnswerPlan:
        """Return a structured answer plan."""

        if self.client is None:
            requires_update = bool(deterministic_extract_simple_claims(message))
            return AnswerPlan(
                answer_mode="memory_update_then_answer" if requires_update else "direct_answer",
                relevant_belief_ids=candidate_belief_ids,
                requires_memory_update=requires_update,
                proposed_commit_message="Update belief memory from user message" if requires_update else None,
                explanation_style="concise",
            )
        try:
            response = self.client.responses.create(
                model=self.settings.openai_model,
                input=[
                    {
                        "role": "system",
                        "content": "Choose a compact answer plan for a version-controlled belief memory system.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"message": message, "candidate_belief_ids": candidate_belief_ids}
                        ),
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "AnswerPlan",
                        "schema": openai_strict_json_schema(AnswerPlan),
                        "strict": True,
                    }
                },
            )
            output_text = getattr(response, "output_text", None) or _response_text(response)
            return AnswerPlan.model_validate_json(output_text)
        except (OpenAIError, ValueError, TypeError) as exc:
            if self.settings.app_env == "production":
                raise
            logger.warning("Falling back to deterministic answer planning: %s", exc)
            return AnswerPlan(
                answer_mode="direct_answer",
                relevant_belief_ids=candidate_belief_ids,
                requires_memory_update=False,
                explanation_style="concise",
            )

    def plan_memory_write(
        self,
        text: str,
        *,
        fallback_branch_name: str = "main",
        fallback_trust_score: float = 0.7,
    ) -> MemoryWritePlan:
        """Extract claims and propose bounded memory-write metadata."""

        if self.client is None:
            return _fallback_memory_write_plan(
                text,
                fallback_branch_name=fallback_branch_name,
                fallback_trust_score=fallback_trust_score,
            )
        try:
            response = self.client.responses.create(
                model=self.settings.openai_model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "Extract explicit atomic factual claims for TruthGit memory. "
                            "Also choose safe metadata for reviewable staging. "
                            "Use branch_name='main' for current durable facts. "
                            "Use a short branch name such as 'trip-plan' or 'what-if' for hypothetical, future, "
                            "temporary, planning, or counterfactual claims. "
                            "Choose trust_score from source wording only: around 0.85 for clear direct statements, "
                            "around 0.55 for uncertain or hearsay claims, and around 0.25 for suspicious, joke, "
                            "impossible, or low-trust claims. "
                            "Return no claims for pure questions. Do not mutate memory."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "MemoryWritePlan",
                        "schema": openai_strict_json_schema(MemoryWritePlan),
                        "strict": True,
                    }
                },
            )
            output_text = getattr(response, "output_text", None) or _response_text(response)
            return MemoryWritePlan.model_validate_json(output_text)
        except (OpenAIError, ValueError, TypeError) as exc:
            if self.settings.app_env == "production":
                raise
            logger.warning("Falling back to deterministic memory-write planning: %s", exc)
            return _fallback_memory_write_plan(
                text,
                fallback_branch_name=fallback_branch_name,
                fallback_trust_score=fallback_trust_score,
            )

    def run_tool_loop(
        self,
        *,
        message: str,
        tool_definitions: list[dict[str, Any]],
        execute_tool: Callable[[str, dict[str, Any]], Any],
        max_turns: int = 4,
    ) -> str:
        """Run an OpenAI tool loop for read-only memory lookup and staged mutations."""

        if self.client is None:
            return "OpenAI tool loop unavailable; answered with deterministic local memory operations."

        input_items: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are TruthGit. Use tools for memory lookup. "
                    "You may stage or apply changes only through provided validated tools."
                ),
            },
            {"role": "user", "content": message},
        ]
        for _ in range(max_turns):
            response = self.client.responses.create(
                model=self.settings.openai_model,
                input=input_items,
                tools=tool_definitions,
            )
            calls = _function_calls(response)
            if not calls:
                return getattr(response, "output_text", None) or _response_text(response)
            input_items.extend(_output_items(response))
            for call in calls:
                result = execute_tool(call["name"], call["arguments"])
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": json.dumps(result, default=str),
                    }
                )
        return "I reached the tool-call limit while querying TruthGit memory."


def _response_text(response: Any) -> str:
    """Best-effort text extraction from a Responses API object."""

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


def _output_items(response: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        if hasattr(item, "model_dump"):
            items.append(item.model_dump())
        elif isinstance(item, dict):
            items.append(item)
    return items


def _function_calls(response: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        item_type = getattr(item, "type", None) if not isinstance(item, dict) else item.get("type")
        if item_type != "function_call":
            continue
        name = getattr(item, "name", None) if not isinstance(item, dict) else item.get("name")
        raw_arguments = getattr(item, "arguments", "{}") if not isinstance(item, dict) else item.get("arguments", "{}")
        call_id = getattr(item, "call_id", None) if not isinstance(item, dict) else item.get("call_id")
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
        calls.append({"name": name, "arguments": arguments, "call_id": call_id})
    return calls


def parse_extracted_claims_json(payload: str) -> ExtractedClaimList:
    """Parse structured extraction JSON for tests and offline validation."""

    return ExtractedClaimList.model_validate_json(payload)


def claims_from_dicts(claims: list[dict[str, Any]]) -> list[ExtractedClaim]:
    """Build extracted claims from dictionaries."""

    return [ExtractedClaim.model_validate(claim) for claim in claims]


def _fallback_memory_write_plan(
    text: str,
    *,
    fallback_branch_name: str,
    fallback_trust_score: float,
) -> MemoryWritePlan:
    """Build a local write plan when OpenAI is unavailable."""

    claims = claims_from_dicts(deterministic_extract_simple_claims(text))
    return MemoryWritePlan(
        claims=claims,
        branch_name=_fallback_branch_name(text, fallback_branch_name),
        trust_score=_fallback_trust_score(text, fallback_trust_score),
        rationale="Local fallback chose metadata from simple wording heuristics.",
    )


def _fallback_branch_name(text: str, fallback_branch_name: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ("conference", "trip", "vacation", "itinerary")):
        return "trip-plan"
    if any(token in lower for token in ("what if", "hypothetical", "suppose", "maybe later", "will stay")):
        return "what-if"
    return fallback_branch_name.strip() or "main"


def _fallback_trust_score(text: str, fallback_trust_score: float) -> float:
    lower = text.lower()
    if any(token in lower for token in ("atlantis", "fake", "joke", "impossible", "poison")):
        return 0.25
    if any(token in lower for token in ("rumor", "heard", "maybe", "not sure", "unverified")):
        return 0.45
    if any(token in lower for token in ("confirmed", "official", "verified")):
        return 0.85
    return max(0.0, min(1.0, fallback_trust_score))


def openai_strict_json_schema(model: Any) -> dict[str, Any]:
    """Return an OpenAI strict-structured-output compatible JSON schema."""

    schema = model.model_json_schema()
    _stricten_schema_node(schema)
    return schema


def _stricten_schema_node(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("default", None)
        node.pop("format", None)
        if node.get("type") == "object" or "properties" in node:
            properties = node.get("properties") or {}
            node["additionalProperties"] = False
            node["required"] = list(properties.keys())
        for value in node.values():
            _stricten_schema_node(value)
    elif isinstance(node, list):
        for item in node:
            _stricten_schema_node(item)
