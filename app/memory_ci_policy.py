"""Policy configuration for deterministic TruthGit Memory CI."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PredicatePolicyClass:
    """Governance class applied to model-generated predicate labels."""

    name: str
    description: str
    requires_review_on_main: bool = False
    low_trust_fail_threshold: float = 0.35
    low_trust_warn_threshold: float = 0.65
    min_auto_apply_trust: float = 0.75
    min_support_count_for_auto_apply: int = 1
    suspicious_corroboration_trust_jump: float = 0.97
    suspicious_corroboration_max_existing_support: float = 0.65


@dataclass(frozen=True)
class PredicatePolicyRule:
    """Map an open predicate label onto a policy class."""

    name: str
    policy_class_name: str
    exact_predicates: frozenset[str] = frozenset()
    predicate_patterns: tuple[str, ...] = ()

    def matches(self, predicate: str) -> bool:
        """Return True when this rule applies to a normalized predicate."""

        if predicate in self.exact_predicates:
            return True
        return any(re.search(pattern, predicate) for pattern in self.predicate_patterns)


@dataclass(frozen=True)
class MemoryCIPolicyConfig:
    """Configurable Memory CI policy suite."""

    suite_version: str = "memory-ci-v1"
    predicate_classes: dict[str, PredicatePolicyClass] = field(default_factory=dict)
    predicate_rules: tuple[PredicatePolicyRule, ...] = ()
    enabled_checks: tuple[str, ...] = (
        "low_trust_source_check",
        "protected_predicate_review_check",
        "contradiction_spike_check",
        "temporal_overlap_check",
        "branch_leakage_risk_check",
        "rollback_regression_check",
        "suspicious_same_object_corroboration_check",
        "merge_conflict_policy_check",
        "duplicate_source_anomaly_check",
        "support_gap_check",
    )
    default_policy_class_name: str = "low_risk"

    def policy_for_predicate(self, predicate: str) -> PredicatePolicyClass:
        """Return the policy class for a normalized predicate label."""

        for rule in self.predicate_rules:
            if rule.matches(predicate):
                return self.predicate_classes[rule.policy_class_name]
        return self.predicate_classes[self.default_policy_class_name]


def default_memory_ci_policy() -> MemoryCIPolicyConfig:
    """Return the default policy suite used by the app."""

    classes = {
        "low_risk": PredicatePolicyClass(
            name="low_risk",
            description="Ordinary preferences and low-impact facts.",
            requires_review_on_main=False,
            min_auto_apply_trust=0.75,
            min_support_count_for_auto_apply=1,
        ),
        "identity_state": PredicatePolicyClass(
            name="identity_state",
            description="Residence, workplace, ownership, or other durable user state.",
            requires_review_on_main=True,
            min_auto_apply_trust=0.9,
            min_support_count_for_auto_apply=1,
        ),
        "financial": PredicatePolicyClass(
            name="financial",
            description="Financial or compensation facts.",
            requires_review_on_main=True,
            low_trust_fail_threshold=0.45,
            low_trust_warn_threshold=0.75,
            min_auto_apply_trust=0.95,
            min_support_count_for_auto_apply=2,
        ),
        "operational_deadline": PredicatePolicyClass(
            name="operational_deadline",
            description="Deadlines, launches, cancellations, and scheduling state.",
            requires_review_on_main=True,
            low_trust_fail_threshold=0.4,
            low_trust_warn_threshold=0.7,
            min_auto_apply_trust=0.92,
            min_support_count_for_auto_apply=1,
        ),
    }
    rules = (
        PredicatePolicyRule(
            name="identity_state_exact",
            policy_class_name="identity_state",
            exact_predicates=frozenset({"lives_in", "works_at", "owns"}),
        ),
        PredicatePolicyRule(
            name="identity_state_pattern",
            policy_class_name="identity_state",
            predicate_patterns=(
                r"(residence|resident|address|location|located|workplace|employer|employment|owner|owns)",
            ),
        ),
        PredicatePolicyRule(
            name="financial_exact",
            policy_class_name="financial",
            exact_predicates=frozenset({"salary", "compensation", "bank_account", "payment_method"}),
        ),
        PredicatePolicyRule(
            name="financial_pattern",
            policy_class_name="financial",
            predicate_patterns=(r"(salary|compensation|bank|payment|invoice|budget|revenue)",),
        ),
        PredicatePolicyRule(
            name="operational_deadline_exact",
            policy_class_name="operational_deadline",
            exact_predicates=frozenset({"due_date", "launch_date", "cancellation_state"}),
        ),
        PredicatePolicyRule(
            name="operational_deadline_pattern",
            policy_class_name="operational_deadline",
            predicate_patterns=(r"(due|deadline|launch|cancel|cancellation)",),
        ),
    )
    return MemoryCIPolicyConfig(predicate_classes=classes, predicate_rules=rules)
