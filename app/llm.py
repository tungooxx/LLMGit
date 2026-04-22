"""OpenAI integration for claim extraction and answer generation."""

from __future__ import annotations

import json
import logging
import re
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
        memory_context: dict[str, Any] | None = None,
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
                            "Choose concise model-generated snake_case predicates without using a closed label set. "
                            "Extract claims only from the message field; default_branch_name and default_trust_score "
                            "are controls, not facts to remember. Choose branch_name, trust_score, and write_action "
                            "from the user's wording and context. Use branch_name='main' for normal durable facts. "
                            "Only choose a non-main branch when the message explicitly describes a hypothetical, "
                            "counterfactual, plan, temporary scenario, or named branch/workspace. Never invent generic "
                            "branch names such as 'userprovidedinformation' for ordinary factual statements. "
                            "Separate claim confidence from trust_score: confidence means the sentence clearly states "
                            "an extractable claim; trust_score means the claim/source is credible as real-world memory. "
                            "A clear but implausible statement can have high claim confidence and low trust_score. "
                            "For example, 'Alice lives in Seoul' can be confidence about 0.9 and trust_score about 0.8; "
                            "'Alice lives in Atlantis' should be confidence about 0.9 but trust_score about 0.2 to 0.3 "
                            "unless the user explicitly frames it as fiction or a game. "
                            "Branch policy is strict: stable current facts and scheduled real-world facts belong "
                            "on main, even when their valid_from is in the future. Temporary, speculative, "
                            "conference-week, itinerary, trip, planning, what-if, or branch-local facts must use "
                            "a non-main branch name. If the user says someone will stay somewhere during a "
                            "conference, trip, visit, or other bounded scenario, choose a short branch such as "
                            "'trip-plan' or 'conference-week' and do not choose main. A branch-local plausible claim may still use "
                            "write_action='branch_hypothetical' for branch-only claims, or commit_now if you already "
                            "selected a non-main branch and judge the claim safe to write there. "
                            "Use truthgit_memory only to decide branch, trust, and write_action; never extract new "
                            "claims from truthgit_memory. Direct dated updates such as moved/changed/started/stopped "
                            "with a month, date, tomorrow, or later timestamp are normal scheduled or supersession "
                            "events on main; choose commit_now when they are clear and plausible so TruthGit can "
                            "preserve the old version as superseded or schedule the new fact with valid_from. "
                            "If the new claim conflicts with active memory but is undated, very low trust, impossible, "
                            "joking, rumored, or would overwrite current truth without enough evidence, choose "
                            "stage_for_review or reject and explain why in warnings/risk_reasons. "
                            "write_action must be commit_now when you judge the memory should be written now on the "
                            "chosen branch, branch_hypothetical when the write belongs only on a non-main branch, "
                            "stage_for_review when you judge a human should review first, and reject when the text "
                            "should not become memory. Python will still enforce generic safety invariants before "
                            "anything becomes durable. Include risk_reasons and warnings only when useful. "
                            "Return no claims for pure questions unless the user explicitly asks you to remember "
                            "the question itself. "
                            "Write assistant_reply as a concise, natural chat response. Start with an acknowledgement "
                            "such as 'Okay, I'll remember that' when write_action is commit_now. Explain briefly "
                            "when you choose review or reject. Do not claim Python has written the memory yet."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "message": text,
                                "default_branch_name": fallback_branch_name,
                                "default_trust_score": fallback_trust_score,
                                "truthgit_memory": memory_context or {},
                            },
                            ensure_ascii=False,
                        ),
                    },
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

    def answer_from_memory(self, message: str, memory_context: dict[str, Any]) -> str:
        """Answer a chat turn from explicit TruthGit memory context."""

        if self.client is None:
            return _fallback_answer_from_memory(message, memory_context)
        try:
            response = self.client.responses.create(
                model=self.settings.openai_model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are the chat face of TruthGit. Answer naturally and concisely using only "
                            "the supplied version-controlled memory context. For current-truth questions, "
                            "prefer current_beliefs from the selected branch. For history or why questions, "
                            "use timelines and mention superseded, retracted, or hypothetical status when relevant. "
                            "For provenance or support questions, use each belief's support_sources and "
                            "opposition_sources. Prefer human-readable source excerpts or source_ref values; "
                            "do not summarize internal refs like demo-ui as if they were the actual evidence. "
                            "Say which sources actively support the belief and which sources are opposition, "
                            "rolled back, or pending when that information is present. "
                            "For audit-trail questions, use audit_events and staged_commits. Explain staged, "
                            "checked, quarantined, approved, applied, released, or rejected transitions in order. "
                            "If the memory context does not support an answer, say that TruthGit has no evidence yet. "
                            "Do not mutate memory and do not invent facts."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"message": message, "truthgit_memory": memory_context},
                            ensure_ascii=False,
                            default=str,
                        ),
                    },
                ],
            )
            return (getattr(response, "output_text", None) or _response_text(response)).strip()
        except (OpenAIError, ValueError, TypeError) as exc:
            if self.settings.app_env == "production":
                raise
            logger.warning("Falling back to deterministic memory answer: %s", exc)
            return _fallback_answer_from_memory(message, memory_context)

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
        branch_name=fallback_branch_name.strip() or "main",
        trust_score=max(0.0, min(1.0, fallback_trust_score)),
        write_action="commit_now" if claims else "reject",
        risk_reasons=[] if claims else ["no_explicit_claim"],
        warnings=[] if claims else ["No explicit memory claim was found."],
        rationale="Local fallback chose metadata from simple wording heuristics.",
        assistant_reply=(
            "Okay, I'll remember that in TruthGit memory."
            if claims
            else "I did not find an explicit memory claim to remember."
        ),
    )


def _fallback_answer_from_memory(message: str, memory_context: dict[str, Any]) -> str:
    """Answer from memory context without an LLM."""

    current = memory_context.get("current_beliefs") or []
    timelines = memory_context.get("timelines") or []
    staged_commits = memory_context.get("staged_commits") or memory_context.get("pending_staged_commits") or []
    audit_events = memory_context.get("audit_events") or []
    branch_name = (memory_context.get("branch") or {}).get("name", "main")
    lower = message.lower()
    subject_hint = _subject_hint(message)

    if any(token in lower for token in ("audit", "trail", "staged", "quarantine", "quarantined", "rejected")):
        return _fallback_audit_answer(staged_commits, audit_events, subject_hint)

    if any(token in lower for token in ("why", "previous", "previously", "history", "timeline")):
        relevant_timeline = [
            item
            for item in timelines
            if subject_hint is None or str(item.get("subject", "")).lower() == subject_hint
        ]
        if relevant_timeline:
            rendered = " -> ".join(
                f"v{item.get('id')}:{item.get('object_value')}({item.get('status')})"
                for item in sorted(relevant_timeline, key=lambda row: int(row.get("id") or 0))
            )
            return f"TruthGit's lineage on branch '{branch_name}' is {rendered}."

    relevant_current = [
        item
        for item in current
        if subject_hint is None or str(item.get("subject", "")).lower() == subject_hint
    ]
    if any(token in lower for token in ("source", "sources", "support", "supports", "provenance", "justify", "justifies")):
        lines = []
        for item in relevant_current:
            supports = [
                _memory_source_label(source)
                for source in item.get("support_sources", [])
                if source.get("status") == "active"
            ]
            oppositions = [
                _memory_source_label(source)
                for source in item.get("opposition_sources", [])
                if source.get("status") == "active"
            ]
            if not supports and item.get("source_ref"):
                supports = [str(item["source_ref"])]
            support_text = ", ".join(supports) if supports else "no active support sources"
            opposition_text = f"; opposition: {', '.join(oppositions)}" if oppositions else ""
            lines.append(
                f"{item.get('subject')} {item.get('predicate')} {item.get('object_value')} "
                f"is supported by {support_text}{opposition_text}."
            )
        if lines:
            return " ".join(lines)

    if any(token in lower for token in ("where", "live", "lives", "residence", "current")):
        location = [
            item
            for item in relevant_current
            if item.get("predicate") in {"lives_in", "stays_in"}
        ]
        if location:
            facts = "; ".join(
                f"{item.get('subject')} {item.get('predicate')} {item.get('object_value')} "
                f"(v{item.get('id')}, {item.get('status')})"
                for item in location
            )
            return f"On branch '{branch_name}', TruthGit currently has: {facts}."

    if relevant_current:
        facts = "; ".join(
            f"{item.get('subject')} {item.get('predicate')} {item.get('object_value')} "
            f"(v{item.get('id')}, {item.get('status')})"
            for item in relevant_current[:5]
        )
        return f"On branch '{branch_name}', TruthGit has evidence for: {facts}."
    return f"TruthGit has no evidence-backed memory for that on branch '{branch_name}' yet."


def _fallback_audit_answer(
    staged_commits: list[dict[str, Any]],
    audit_events: list[dict[str, Any]],
    subject_hint: str | None,
) -> str:
    """Summarize staged-write and audit state without an LLM."""

    relevant_staged = []
    for staged in staged_commits:
        text = " ".join(
            [
                str(staged.get("source_excerpt") or ""),
                str(staged.get("source_ref") or ""),
                json.dumps(staged.get("claims_json") or [], default=str),
            ]
        ).lower()
        if subject_hint is None or subject_hint in text:
            relevant_staged.append(staged)
    relevant_events = []
    staged_ids = {str(item.get("id")) for item in relevant_staged if item.get("id")}
    for event in audit_events:
        entity_key = str(event.get("entity_key") or "")
        payload = json.dumps(event.get("payload_json") or {}, default=str).lower()
        if entity_key in staged_ids or subject_hint is None or (subject_hint and subject_hint in payload):
            relevant_events.append(event)

    if not relevant_staged and not relevant_events:
        return "TruthGit has no staged or audit trail evidence matching that request yet."

    lines: list[str] = []
    for staged in relevant_staged[:5]:
        claim_text = _staged_claim_text(staged)
        reason = staged.get("quarantine_reason_summary") or ", ".join(staged.get("risk_reasons") or [])
        reason_suffix = f" Reason: {reason}." if reason else ""
        lines.append(
            f"Staged write {staged.get('id')} is {staged.get('status')}"
            f"{f' for {claim_text}' if claim_text else ''}.{reason_suffix}"
        )
    if relevant_events:
        ordered = sorted(relevant_events, key=lambda row: int(row.get("id") or 0))
        events = " -> ".join(
            f"#{event.get('id')} {event.get('event_type')}" for event in ordered[-12:]
        )
        lines.append(f"Audit sequence: {events}.")
    return " ".join(lines)


def _staged_claim_text(staged: dict[str, Any]) -> str:
    claims = staged.get("claims_json") or []
    pieces = []
    for claim in claims[:3]:
        if not isinstance(claim, dict):
            continue
        subject = claim.get("subject")
        predicate = claim.get("predicate")
        object_value = claim.get("object") or claim.get("object_value")
        pieces.append(" ".join(str(value) for value in (subject, predicate, object_value) if value))
    return "; ".join(pieces)


def _memory_source_label(source: dict[str, Any]) -> str:
    source_ref = str(source.get("source_ref") or "")
    if source_ref and not source_ref.startswith(("demo-ui:", "chat", "source-")):
        return source_ref
    excerpt = re.sub(r"\s+", " ", str(source.get("excerpt") or "")).strip()
    return excerpt[:140] + ("..." if len(excerpt) > 140 else "")


def _subject_hint(message: str) -> str | None:
    ignored = {
        "what",
        "where",
        "when",
        "why",
        "how",
        "who",
        "which",
        "show",
        "list",
        "display",
        "explain",
        "summarize",
        "tell",
        "truthgit",
    }
    for match in re.finditer(r"\b([A-Z][a-zA-Z0-9_-]+)\b", message):
        candidate = match.group(1).lower()
        if candidate not in ignored:
            return candidate
    return None


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
