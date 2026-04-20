from __future__ import annotations

from datetime import date

from app.normalization import (
    deterministic_extract_simple_claims,
    normalize_extracted_claim,
    normalize_predicate,
)
from app.schemas import ExtractedClaim


def test_normalizes_moved_to_busan_claim() -> None:
    extracted = deterministic_extract_simple_claims("Alice moved to Busan in March 2026.")

    assert extracted[0]["predicate"] == "lives_in"
    assert extracted[0]["object"] == "Busan"
    assert extracted[0]["valid_from"] == date(2026, 3, 1)

    claim = ExtractedClaim.model_validate(extracted[0])
    normalized = normalize_extracted_claim(claim)
    assert normalized.subject == "Alice"
    assert normalized.predicate == "lives_in"
    assert normalized.normalized_object_value == "busan"


def test_predicate_aliases_are_stable() -> None:
    assert normalize_predicate("moved to") == "lives_in"
    assert normalize_predicate("resides_in") == "lives_in"
    assert normalize_predicate("will stay in") == "stays_in"
