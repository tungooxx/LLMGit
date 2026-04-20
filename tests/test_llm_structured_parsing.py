from __future__ import annotations

from app.llm import openai_strict_json_schema, parse_extracted_claims_json
from app.schemas import ExtractedClaimList


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


def test_openai_strict_schema_adds_required_and_blocks_extra_properties() -> None:
    schema = openai_strict_json_schema(ExtractedClaimList)
    claim_schema = schema["$defs"]["ExtractedClaim"]

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["claims"]
    assert claim_schema["additionalProperties"] is False
    assert set(claim_schema["required"]) == set(claim_schema["properties"])
    assert "default" not in claim_schema["properties"]["confidence"]
    assert "format" not in claim_schema["properties"]["valid_from"]["anyOf"][0]
