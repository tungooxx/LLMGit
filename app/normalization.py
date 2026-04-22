"""Deterministic claim normalization utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any


# Predicates are open, model-generated relation labels. This alias map is only a
# stability layer for obvious lexical variants, not a closed ontology.
PREDICATE_ALIASES: dict[str, str] = {
    "live in": "lives_in",
    "lives in": "lives_in",
    "living in": "lives_in",
    "resides in": "lives_in",
    "resides_in": "lives_in",
    "residence": "lives_in",
    "current residence": "lives_in",
    "current_residence": "lives_in",
    "current location": "lives_in",
    "current_location": "lives_in",
    "home": "lives_in",
    "home city": "lives_in",
    "home_city": "lives_in",
    "based in": "lives_in",
    "based_in": "lives_in",
    "location": "lives_in",
    "moved to": "lives_in",
    "move to": "lives_in",
    "will stay in": "stays_in",
    "stay in": "stays_in",
    "stays in": "stays_in",
    "stays_in": "stays_in",
}

MONTHS: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


@dataclass(frozen=True)
class NormalizedClaim:
    """Canonical claim form used by deterministic memory operations."""

    subject: str
    predicate: str
    object_value: str
    normalized_object_value: str
    confidence: float
    valid_from: date | None
    valid_to: date | None
    is_negation: bool = False
    source_quote: str | None = None
    notes: str | None = None


def canonical_text(value: str) -> str:
    """Normalize text for stable comparisons."""

    cleaned = re.sub(r"\s+", " ", value.strip().lower())
    cleaned = cleaned.strip(" .,!?:;\"'")
    return cleaned


def normalize_predicate(predicate: str) -> str:
    """Normalize an open model-generated predicate label."""

    key = canonical_text(predicate).replace("_", " ")
    if key in PREDICATE_ALIASES:
        return PREDICATE_ALIASES[key]
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_")


def canonical_key(subject: str, predicate: str) -> str:
    """Build stable belief identity from subject and predicate."""

    return f"{canonical_text(subject)}::{normalize_predicate(predicate)}"


def normalize_object_value(object_value: str) -> str:
    """Normalize object values for conflict comparisons."""

    return canonical_text(object_value)


def parse_month_year(text: str) -> date | None:
    """Parse month/year phrases such as 'March 2026'."""

    match = re.search(
        r"\b("
        + "|".join(MONTHS.keys())
        + r")\s+(\d{4})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    month = MONTHS[match.group(1).lower()]
    year = int(match.group(2))
    return date(year, month, 1)


def parse_iso_date(value: str | None) -> date | None:
    """Parse an ISO date string, returning None for invalid or empty values."""

    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def normalize_extracted_claim(claim: Any) -> NormalizedClaim:
    """Normalize a Pydantic claim or dict-like claim."""

    data = claim.model_dump(by_alias=False) if hasattr(claim, "model_dump") else dict(claim)
    subject = str(data["subject"]).strip()
    predicate = normalize_predicate(str(data["predicate"]))
    object_value = str(data.get("object_value") or data.get("object") or "").strip()
    confidence = float(data.get("confidence", 0.7))
    valid_from = data.get("valid_from")
    valid_to = data.get("valid_to")
    if isinstance(valid_from, str):
        valid_from = parse_iso_date(valid_from)
    if isinstance(valid_to, str):
        valid_to = parse_iso_date(valid_to)
    return NormalizedClaim(
        subject=subject,
        predicate=predicate,
        object_value=object_value,
        normalized_object_value=normalize_object_value(object_value),
        confidence=max(0.0, min(1.0, confidence)),
        valid_from=valid_from,
        valid_to=valid_to,
        is_negation=bool(data.get("is_negation", False)),
        source_quote=data.get("source_quote"),
        notes=data.get("notes"),
    )


def windows_overlap(
    left_from: date | None,
    left_to: date | None,
    right_from: date | None,
    right_to: date | None,
) -> bool:
    """Return True if two validity windows overlap."""

    min_date = date.min
    max_date = date.max
    lf = left_from or min_date
    lt = left_to or max_date
    rf = right_from or min_date
    rt = right_to or max_date
    return lf <= rt and rf <= lt


def deterministic_extract_simple_claims(text: str) -> list[dict[str, Any]]:
    """Extract demo-friendly claims without an LLM.

    This intentionally handles only transparent MVP patterns and keeps all
    database mutation deterministic after extraction.
    """

    stripped = text.strip()
    if not stripped or stripped.endswith("?") or re.match(r"^(why|what|where|when|how|who)\b", stripped, re.I):
        return []

    claims: list[dict[str, Any]] = []
    for sentence in re.split(r"(?<=[.!])\s+", stripped):
        sentence = sentence.strip()
        if not sentence:
            continue
        claim_sentence = _strip_source_attribution(sentence)

        moved = re.search(
            r"^(?P<subject>[A-Z][\w\s.'-]*?)\s+moved\s+to\s+(?P<object>[A-Z][\w\s.'-]*?)(?:\s+in\s+(?P<when>[A-Za-z]+\s+\d{4}))?[.!]?$",
            claim_sentence,
        )
        if moved:
            when_text = moved.group("when") or claim_sentence
            claims.append(
                {
                    "subject": moved.group("subject").strip(),
                    "predicate": "lives_in",
                    "object": moved.group("object").strip(),
                    "confidence": 0.82,
                    "valid_from": parse_month_year(when_text),
                    "valid_to": None,
                    "is_negation": False,
                    "source_quote": sentence,
                    "notes": "fallback extractor: moved-to pattern",
                }
            )
            continue

        lives = re.search(
            r"^(?P<subject>[A-Z][\w\s.'-]*?)\s+(?:lives|resides)\s+in\s+(?P<object>[A-Z][\w\s.'-]*?)[.!]?$",
            claim_sentence,
        )
        if lives:
            claims.append(
                {
                    "subject": lives.group("subject").strip(),
                    "predicate": "lives_in",
                    "object": lives.group("object").strip(),
                    "confidence": 0.75,
                    "valid_from": None,
                    "valid_to": None,
                    "is_negation": False,
                    "source_quote": sentence,
                    "notes": "fallback extractor: lives-in pattern",
                }
            )
            continue

        stay = re.search(
            r"^(?:During\s+(?P<context>.*?),\s+)?(?P<subject>[A-Z][\w\s.'-]*?)\s+(?:will\s+)?stay\s+in\s+(?P<object>[A-Z][\w\s.'-]*?)[.!]?$",
            claim_sentence,
        )
        if stay:
            claims.append(
                {
                    "subject": stay.group("subject").strip(),
                    "predicate": "stays_in",
                    "object": stay.group("object").strip(),
                    "confidence": 0.68,
                    "valid_from": None,
                    "valid_to": None,
                    "is_negation": False,
                    "source_quote": sentence,
                    "notes": stay.group("context") or "fallback extractor: stay-in pattern",
                }
            )
            continue

    return claims


def _strip_source_attribution(sentence: str) -> str:
    """Return the quoted claim part from simple "source says X" wording."""

    attribution = re.search(
        r"\b(?:also\s+)?(?:says|said|confirms|confirmed|reports|reported|states|stated|claims|claimed)\s+"
        r"(?P<claim>[A-Z][^.!?]*[.!?]?)$",
        sentence,
        flags=re.IGNORECASE,
    )
    if attribution:
        return attribution.group("claim").strip()
    return sentence
