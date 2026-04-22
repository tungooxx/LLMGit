"""Generic validation policy for model-proposed TruthGit memory writes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from sqlalchemy.orm import Session

from app import crud
from app.conflict_engine import classify_claim_against_current
from app.normalization import NormalizedClaim, normalize_extracted_claim
from app.schemas import ExtractedClaim, MemoryWritePlan


LOW_TRUST_THRESHOLD = 0.5
LOW_CONFIDENCE_THRESHOLD = 0.5
IMPLAUSIBLE_REAL_WORLD_TRUST_CAP = 0.25


@dataclass(frozen=True)
class WritePolicyDecision:
    """Effective write metadata after deterministic safety checks."""

    branch_name: str
    trust_score: float
    write_action: str
    review_required: bool
    risk_reasons: list[str]
    warnings: list[str]


def enforce_write_policy(
    db: Session,
    *,
    plan: MemoryWritePlan,
    source_excerpt: str,
    fallback_branch_name: str = "main",
) -> WritePolicyDecision:
    """Validate a model write plan without using a closed predicate list."""

    branch_name = safe_branch_name(plan.branch_name or fallback_branch_name)
    trust_score = _clamp_score(plan.trust_score)
    write_action = plan.write_action
    risk_reasons = _dedupe([*plan.risk_reasons])
    warnings = _dedupe([*plan.warnings])
    if _has_implausible_real_world_claim(plan.claims, source_excerpt):
        trust_score = min(trust_score, IMPLAUSIBLE_REAL_WORLD_TRUST_CAP)
        risk_reasons.append("implausible_real_world_claim")
        warnings.append(
            "Implausible real-world claim detected; TruthGit capped source trust and requires review."
        )

    if write_action == "reject":
        if "model_write_action:reject" not in risk_reasons:
            risk_reasons.append("model_write_action:reject")
        return WritePolicyDecision(
            branch_name=branch_name,
            trust_score=trust_score,
            write_action=write_action,
            review_required=False,
            risk_reasons=risk_reasons,
            warnings=warnings,
        )

    if write_action == "branch_hypothetical" and _is_main_branch_name(branch_name):
        branch_name = _derive_branch_name(source_excerpt)
        warnings.append(
            f"Model selected branch_hypothetical; TruthGit will write this on branch '{branch_name}'."
        )

    branch_only = _has_branch_only_cue(source_excerpt, plan.claims)
    if branch_only:
        derived_branch_name = _derive_branch_name(source_excerpt)
    else:
        derived_branch_name = branch_name

    if _is_main_branch_name(branch_name) and branch_only:
        branch_name = derived_branch_name
        write_action = "branch_hypothetical"
        risk_reasons.append("branch_only_claim")
        warnings.append(
            f"TruthGit routed this future or branch-local claim to branch '{branch_name}' instead of main."
        )
    elif (
        not _is_main_branch_name(branch_name)
        and branch_only
        and branch_name != derived_branch_name
        and not _source_mentions_branch_name(source_excerpt, branch_name)
    ):
        previous_branch_name = branch_name
        branch_name = derived_branch_name
        if write_action == "commit_now":
            write_action = "branch_hypothetical"
        warnings.append(
            f"TruthGit routed this branch-local scenario to '{branch_name}' instead of stale branch fallback "
            f"'{previous_branch_name}'."
        )
    elif not _is_main_branch_name(branch_name) and not branch_only and not _has_explicit_branch_reference(source_excerpt):
        branch_name = "main"
        if write_action == "branch_hypothetical":
            write_action = "commit_now"
        warnings.append("TruthGit routed this stable or scheduled fact to main instead of the branch fallback.")

    branch = crud.get_branch_by_name(db, branch_name)
    branch_id = branch.id if branch is not None else None
    review = review_requirements_for_claims(
        db,
        claims=plan.claims,
        branch_id=branch_id,
        branch_name=branch_name,
        source_trust=trust_score,
        source_excerpt=source_excerpt,
    )
    risk_reasons = _dedupe([*risk_reasons, *review.risk_reasons])
    warnings = _dedupe([*warnings, *review.warnings])
    review_required = review.review_required

    if review_required and write_action == "commit_now":
        write_action = "stage_for_review"
        warnings.append("TruthGit policy overrode model commit_now because review is required.")

    return WritePolicyDecision(
        branch_name=branch_name,
        trust_score=trust_score,
        write_action=write_action,
        review_required=review_required,
        risk_reasons=_dedupe(risk_reasons),
        warnings=_dedupe(warnings),
    )


def review_requirements_for_claims(
    db: Session,
    *,
    claims: list[ExtractedClaim],
    branch_id: int | None,
    branch_name: str,
    source_trust: float,
    source_excerpt: str,
) -> WritePolicyDecision:
    """Return generic review requirements for already-routed claims."""

    trust_score = _clamp_score(source_trust)
    risk_reasons: list[str] = []
    warnings: list[str] = []
    review_required = False
    branch_is_main = _is_main_branch_name(branch_name)

    if trust_score < LOW_TRUST_THRESHOLD:
        review_required = True
        risk_reasons.append("low_trust_source")
        warnings.append("Low-trust source requires review before durable memory update.")

    if branch_is_main and _has_branch_only_cue(source_excerpt, claims):
        review_required = True
        risk_reasons.append("branch_only_claim_on_main")
        warnings.append("Future, temporary, or hypothetical claims must not be committed directly to main.")

    for claim in claims:
        normalized = normalize_extracted_claim(claim)
        if normalized.confidence < LOW_CONFIDENCE_THRESHOLD:
            review_required = True
            risk_reasons.append("low_claim_confidence")
            warnings.append("Low-confidence claim requires review before durable memory update.")
        if branch_id is None:
            continue
        if branch_is_main and _claim_conflicts_with_stronger_current(
            db,
            claim=normalized,
            branch_id=branch_id,
            source_trust=trust_score,
        ):
            review_required = True
            risk_reasons.append("conflicts_with_stronger_current_memory")
            warnings.append("New claim conflicts with stronger active memory; review is required.")

    return WritePolicyDecision(
        branch_name=branch_name,
        trust_score=trust_score,
        write_action="stage_for_review" if review_required else "commit_now",
        review_required=review_required,
        risk_reasons=_dedupe(risk_reasons),
        warnings=_dedupe(warnings),
    )


def safe_branch_name(value: str) -> str:
    """Normalize model-proposed branch names."""

    clean = value.strip().lower().replace("_", "-")
    clean = "".join(character for character in clean if character.isalnum() or character == "-")
    clean = "-".join(part for part in clean.split("-") if part)
    return clean[:40] or "main"


def _claim_conflicts_with_stronger_current(
    db: Session,
    *,
    claim: NormalizedClaim,
    branch_id: int,
    source_trust: float,
) -> bool:
    belief = crud.get_belief_by_subject_predicate(db, subject=claim.subject, predicate=claim.predicate)
    if belief is None:
        return False
    decision = classify_claim_against_current(
        db,
        claim=claim,
        current_versions=crud.get_current_versions(db, belief_id=belief.id, branch_id=branch_id),
        source_trust=source_trust,
        branch_is_hypothetical=False,
    )
    return decision.action == "unresolved_conflict"


def _has_branch_only_cue(source_excerpt: str, claims: list[ExtractedClaim]) -> bool:
    text = source_excerpt.lower()
    confirmed_current = re.search(r"\b(confirmed|already|currently|right now|now)\b", text)
    speculative_or_temporary = re.search(
        r"\b(would|could|might|hypothetical|counterfactual|what if|temporary|planning|plan|scenario)\b",
        text,
    )
    bounded_scenario = re.search(
        r"\b(during|itinerary|trip|conference|conference week|travel|visit|vacation|relocation|remote work)\b",
        text,
    )
    if (speculative_or_temporary or bounded_scenario) and not confirmed_current:
        return True
    today = date.today()
    for claim in claims:
        if claim.valid_to is not None and not confirmed_current:
            return True
        if claim.valid_from is not None and claim.valid_from > today and not confirmed_current:
            return True
    return False


def _has_explicit_branch_reference(source_excerpt: str) -> bool:
    text = source_excerpt.lower()
    return bool(
        re.search(
            r"\b(branch|workspace|scenario|hypothetical|what-if|what if|trip-plan|conference-week|relocation)\b",
            text,
        )
    )


def _has_implausible_real_world_claim(claims: list[ExtractedClaim], source_excerpt: str) -> bool:
    """Detect clear real-world memory claims about implausible or fictional places."""

    if _is_fictional_context(source_excerpt):
        return False
    for claim in claims:
        if not _is_location_like_claim(claim):
            continue
        if _object_names_implausible_place(claim.object_value):
            return True
    return False


def _is_fictional_context(source_excerpt: str) -> bool:
    text = source_excerpt.lower()
    return bool(
        re.search(
            r"\b(fiction|fictional|story|novel|game|roleplay|role-play|fantasy|worldbuilding|imaginary)\b",
            text,
        )
    )


def _is_location_like_claim(claim: ExtractedClaim) -> bool:
    predicate = safe_branch_name(claim.predicate).replace("-", "_")
    return bool(
        re.search(
            r"(live|lives|resid|stay|stays|location|located|address|city|country|move|moved|base|work_from|visit)",
            predicate,
        )
    )


def _object_names_implausible_place(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", value.lower())
    tokens = set(normalized.split())
    phrases = {
        "atlantis",
        "narnia",
        "hogwarts",
        "mordor",
        "middle earth",
        "neverland",
        "wakanda",
        "gotham",
        "metropolis",
        "asgard",
    }
    if tokens.intersection({"atlantis", "narnia", "hogwarts", "mordor", "neverland", "wakanda", "gotham", "metropolis", "asgard"}):
        return True
    return any(phrase in normalized for phrase in phrases)


def _derive_branch_name(source_excerpt: str) -> str:
    text = source_excerpt.lower()
    relocation = re.search(
        r"\b(?:in\s+)?(?:the\s+)?([a-z][a-z0-9-]{2,40})\s+relocation\s+scenario\b",
        text,
    )
    if relocation:
        return safe_branch_name(f"{relocation.group(1)}-relocation")
    named_fellowship = re.search(r"\b([a-z][a-z0-9-]{2,40})\s+fellowship\b", text)
    if named_fellowship:
        return safe_branch_name(f"{named_fellowship.group(1)}-fellowship")
    named_scenario = re.search(
        r"\b(?:in\s+)?(?:the\s+)?([a-z][a-z0-9-]{2,40}(?:\s+[a-z][a-z0-9-]{2,40}){0,2})\s+scenario\b",
        text,
    )
    if named_scenario:
        return safe_branch_name(f"{named_scenario.group(1)}-scenario")
    if "kyoto" in text and "fellowship" in text:
        return "kyoto-fellowship"
    if "fellowship" in text:
        return "fellowship-plan"
    if "relocation" in text:
        return "relocation-scenario"
    if "remote work" in text:
        return "remote-work-plan"
    if "conference" in text:
        return "conference-week"
    if "trip" in text or "itinerary" in text or "travel" in text:
        return "trip-plan"
    if "what if" in text or "hypothetical" in text or "counterfactual" in text:
        return "hypothetical"
    return "branch-hypothesis"


def _source_mentions_branch_name(source_excerpt: str, branch_name: str) -> bool:
    """Return True when the user text explicitly names the proposed branch."""

    clean_branch_name = safe_branch_name(branch_name)
    text = source_excerpt.lower()
    spaced_name = clean_branch_name.replace("-", " ")
    normalized_text = re.sub(r"[^a-z0-9-]+", "-", text.replace("_", "-")).strip("-")
    return clean_branch_name in normalized_text or spaced_name in text


def _is_main_branch_name(value: str) -> bool:
    return safe_branch_name(value) == "main"


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
