from __future__ import annotations

from app.llm import parse_extracted_claims_json


def test_structured_extraction_parsing() -> None:
    parsed = parse_extracted_claims_json(
        """
        {
          "claims": [
            {
              "subject": "Alice",
              "predicate": "lives_in",
              "object": "Busan",
              "confidence": 0.8,
              "valid_from": "2026-03-01",
              "valid_to": null,
              "is_negation": false,
              "source_quote": "Alice moved to Busan in March 2026.",
              "notes": "explicit"
            }
          ]
        }
        """
    )

    assert parsed.claims[0].object_value == "Busan"
    assert parsed.claims[0].valid_from.isoformat() == "2026-03-01"
