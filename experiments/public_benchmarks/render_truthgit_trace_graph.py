"""Render TruthGit LongMemEval trace files as showcase git-style graphs.

The renderer is intentionally offline and deterministic. It reads the trace
JSON emitted by ``longmemeval_truthgit.py`` and writes a self-contained HTML
page that highlights session commits, extracted belief versions, model evidence
planning, and the final answer.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SESSION_REF_RE = re.compile(r":session:(\d+)$")


@dataclass(frozen=True)
class TraceGraph:
    """Normalized data needed to render one trace graph."""

    case_label: str
    question_id: str
    question: str
    question_type: str
    hypothesis: str
    claim_count: int
    staged_commit_count: int
    commit_count: int
    audit_event_count: int
    branch_name: str
    source_refs: list[str]
    source_excerpts: dict[str, dict[str, Any]]
    belief_versions: list[dict[str, Any]]
    lineage_versions: list[dict[str, Any]]
    status_counts: dict[str, int]
    supersession_count: int
    quarantine_count: int
    state_note: str
    evidence_refs: set[str]
    evidence_candidates: list[dict[str, Any]]
    relevant_beliefs: list[dict[str, Any]]


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _session_index(source_ref: str) -> int:
    match = SESSION_REF_RE.search(source_ref)
    if match is None:
        return 10**9
    return int(match.group(1))


def _short_source_ref(source_ref: str) -> str:
    match = SESSION_REF_RE.search(source_ref)
    if match is not None:
        return f"s{match.group(1)}"
    return source_ref.rsplit(":", 1)[-1]


def _node_payload(graph: TraceGraph, source_ref: str) -> dict[str, Any]:
    source = graph.source_excerpts.get(source_ref) or {}
    evidence = [
        candidate
        for candidate in graph.evidence_candidates
        if str(candidate.get("source_ref")) == source_ref
    ]
    beliefs = [
        belief
        for belief in graph.belief_versions
        if str(belief.get("source_ref")) == source_ref
    ]
    has_lineage = any(
        belief.get("supersedes_version_id") or belief.get("status") in {"superseded", "retracted"}
        for belief in beliefs
    )
    is_quarantined = source.get("ci_status") == "quarantined" or any(
        belief.get("status") == "quarantined" for belief in beliefs
    )
    return {
        "question_id": graph.question_id,
        "source_ref": source_ref,
        "short_ref": _short_source_ref(source_ref),
        "session_index": _session_index(source_ref),
        "is_evidence": source_ref in graph.evidence_refs,
        "has_lineage": has_lineage,
        "is_quarantined": is_quarantined,
        "ci_status": source.get("ci_status"),
        "ci_reason": source.get("ci_reason"),
        "trust_score": source.get("trust_score"),
        "source_type": source.get("source_type"),
        "excerpt": source.get("excerpt"),
        "evidence": evidence,
        "beliefs": beliefs,
    }


def _json_attr(payload: dict[str, Any]) -> str:
    return _escape(json.dumps(payload, ensure_ascii=False, default=str))


def _load_trace(path: Path) -> TraceGraph:
    data = json.loads(path.read_text(encoding="utf-8"))
    context = data["context"]
    model_evidence = data.get("model_evidence") or {}
    plan = model_evidence.get("model_evidence_plan") or {}
    candidates = list(plan.get("candidates") or [])
    evidence_refs = {
        str(candidate.get("source_ref"))
        for candidate in candidates
        if candidate.get("source_ref")
    }
    source_excerpts = {
        str(source["source_ref"]): source for source in context.get("source_excerpts", [])
    }
    source_refs = sorted(source_excerpts.keys(), key=_session_index)
    belief_versions = list(context.get("full_belief_versions") or context.get("belief_versions", []))
    lineage_versions = list(context.get("lineage_versions") or [])
    if not lineage_versions:
        lineage_versions = [
            belief
            for belief in belief_versions
            if belief.get("supersedes_version_id") or belief.get("status") in {"superseded", "retracted"}
        ]
    status_counts: dict[str, int] = {}
    for belief in belief_versions:
        status = str(belief.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    lineage_beliefs = lineage_versions
    relevant_beliefs = [
        belief for belief in belief_versions if str(belief.get("source_ref")) in evidence_refs
    ]
    for belief in lineage_beliefs:
        if belief not in relevant_beliefs:
            relevant_beliefs.append(belief)
    if not relevant_beliefs:
        relevant_beliefs = belief_versions[:12]
    return TraceGraph(
        case_label="LongMemEval ID",
        question_id=str(context.get("question_id", path.stem.split(".")[0])),
        question=str(context.get("question", "")),
        question_type=str(context.get("question_type", "")),
        hypothesis=str(data.get("hypothesis", "")),
        claim_count=int(context.get("claim_count") or 0),
        staged_commit_count=int(context.get("staged_commit_count") or 0),
        commit_count=int(context.get("commit_count") or 0),
        audit_event_count=int(context.get("audit_event_count") or 0),
        branch_name="main",
        source_refs=source_refs,
        source_excerpts=source_excerpts,
        belief_versions=belief_versions,
        lineage_versions=lineage_versions,
        status_counts=status_counts,
        supersession_count=len(lineage_beliefs),
        quarantine_count=status_counts.get("quarantined", 0),
        state_note=(
            "This public LongMemEval adapter ingests records on the main branch. "
            "Quarantine and branch-only hypothetical writes show up in the live demo/API "
            "or in the governance demo when the input triggers those states."
        ),
        evidence_refs=evidence_refs,
        evidence_candidates=candidates,
        relevant_beliefs=relevant_beliefs[:12],
    )


def _governance_demo_graph() -> TraceGraph:
    """Create a compact professor-facing graph for CI/quarantine behavior."""

    source_refs = [
        "demo:apples:session:1",
        "demo:apples:session:2",
        "demo:apples:session:3",
    ]
    source_excerpts: dict[str, dict[str, Any]] = {
        source_refs[0]: {
            "source_id": "demo-source-1",
            "source_type": "manual",
            "source_ref": source_refs[0],
            "trust_score": 0.82,
            "ci_status": "passed",
            "excerpt": "Initial pantry count: the user has 3 apples.",
        },
        source_refs[1]: {
            "source_id": "demo-source-2",
            "source_type": "user_message",
            "source_ref": source_refs[1],
            "trust_score": 0.9,
            "ci_status": "passed",
            "excerpt": "Correction after recounting: I have 4 apples now.",
        },
        source_refs[2]: {
            "source_id": "demo-source-3",
            "source_type": "anonymous_post",
            "source_ref": source_refs[2],
            "trust_score": 0.1,
            "ci_status": "quarantined",
            "ci_reason": "low_trust_source_check and contradiction_spike_check failed",
            "excerpt": "Anonymous spam says the user has 400 apples now.",
        },
    }
    belief_versions: list[dict[str, Any]] = [
        {
            "belief_version_id": 1,
            "belief_id": 1,
            "subject": "user",
            "predicate": "has_apple_count",
            "object_value": "3",
            "status": "superseded",
            "confidence": 0.82,
            "valid_from": "2026-04-20",
            "valid_to": "2026-04-21",
            "source_ref": source_refs[0],
            "source_trust_score": 0.82,
            "supersedes_version_id": None,
            "source_quote": "Initial pantry count: the user has 3 apples.",
            "metadata_json": {"superseded_by_version_id": 2},
        },
        {
            "belief_version_id": 2,
            "belief_id": 1,
            "subject": "user",
            "predicate": "has_apple_count",
            "object_value": "4",
            "status": "active",
            "confidence": 0.9,
            "valid_from": "2026-04-21",
            "valid_to": None,
            "source_ref": source_refs[1],
            "source_trust_score": 0.9,
            "supersedes_version_id": 1,
            "source_quote": "Correction after recounting: I have 4 apples now.",
            "metadata_json": {"support_role": "current_governing_version"},
        },
        {
            "belief_version_id": "staged-q1",
            "belief_id": 1,
            "subject": "user",
            "predicate": "has_apple_count",
            "object_value": "400",
            "status": "quarantined",
            "confidence": 0.2,
            "valid_from": "2026-04-22",
            "valid_to": None,
            "source_ref": source_refs[2],
            "source_trust_score": 0.1,
            "supersedes_version_id": None,
            "source_quote": "Anonymous spam says the user has 400 apples now.",
            "metadata_json": {
                "ci_decision": "quarantine",
                "failed_checks": ["low_trust_source_check", "contradiction_spike_check"],
            },
        },
    ]
    evidence_candidates: list[dict[str, Any]] = [
        {
            "source_ref": source_refs[1],
            "session_date": "2026-04-21",
            "evidence": "The recount says the user has 4 apples now.",
            "subject": "user",
            "relation": "has_apple_count",
            "object_value": "4",
            "event_name": "recount",
            "event_date": "2026-04-21",
            "role": "current_support",
            "confidence": 0.9,
            "notes": "This active version supersedes the earlier count of 3.",
        },
        {
            "source_ref": source_refs[0],
            "session_date": "2026-04-20",
            "evidence": "The earlier count was 3 apples.",
            "subject": "user",
            "relation": "has_apple_count",
            "object_value": "3",
            "event_name": "initial_count",
            "event_date": "2026-04-20",
            "role": "superseded_history",
            "confidence": 0.82,
            "notes": "Preserved for lineage but not current truth.",
        },
        {
            "source_ref": source_refs[2],
            "session_date": "2026-04-22",
            "evidence": "The anonymous 400-apple update was quarantined.",
            "subject": "user",
            "relation": "has_apple_count",
            "object_value": "400",
            "event_name": "poison_attempt",
            "event_date": "2026-04-22",
            "role": "quarantined_opposition",
            "confidence": 0.2,
            "notes": "Blocked by Memory CI; it cannot become active truth until reviewed.",
        },
    ]
    return TraceGraph(
        case_label="Governance Demo",
        question_id="apples-supersession-quarantine",
        question="How many apples does the user have now?",
        question_type="truth-maintenance-demo",
        hypothesis=(
            "TruthGit answers 4 apples. The 4-apple version supersedes the old 3-apple "
            "version, and the anonymous 400-apple update is quarantined so it cannot "
            "poison current truth."
        ),
        claim_count=3,
        staged_commit_count=3,
        commit_count=2,
        audit_event_count=9,
        branch_name="main",
        source_refs=source_refs,
        source_excerpts=source_excerpts,
        belief_versions=belief_versions,
        lineage_versions=belief_versions,
        status_counts={"active": 1, "superseded": 1, "quarantined": 1},
        supersession_count=1,
        quarantine_count=1,
        state_note=(
            "Illustrative governance scenario, not a LongMemEval benchmark row: "
            "a normal update supersedes stale truth, while a low-trust contradictory "
            "update is quarantined before it can affect the answer."
        ),
        evidence_refs={source_refs[1]},
        evidence_candidates=evidence_candidates,
        relevant_beliefs=belief_versions,
    )


def _render_commit_svg(graph: TraceGraph) -> str:
    refs = graph.source_refs
    if not refs:
        return '<div class="empty">No source commits in trace.</div>'
    gap = 28
    left = 48
    width = max(980, left * 2 + gap * max(1, len(refs) - 1))
    height = 148
    y = 58
    nodes: list[str] = [
        f'<line x1="{left}" y1="{y}" x2="{width - left}" y2="{y}" class="rail" />'
    ]
    for idx, ref in enumerate(refs):
        x = left + idx * gap
        is_evidence = ref in graph.evidence_refs
        payload = _node_payload(graph, ref)
        class_parts = ["node"]
        if is_evidence:
            class_parts.append("evidence")
        if payload["is_quarantined"]:
            class_parts.append("quarantined")
        if payload["has_lineage"]:
            class_parts.append("lineage")
        node_class = " ".join(class_parts)
        label_class = "node-label"
        if is_evidence:
            label_class += " evidence-label"
        if payload["is_quarantined"]:
            label_class += " quarantine-label"
        radius = 9 if is_evidence or payload["is_quarantined"] else 5
        nodes.append(
            f'<circle cx="{x}" cy="{y}" r="{radius}" class="{node_class}" '
            f'tabindex="0" role="button" data-node="{_json_attr(payload)}" '
            f'data-evidence="{str(is_evidence).lower()}" '
            f'aria-label="Session commit {_escape(_short_source_ref(ref))}" />'
        )
        if is_evidence or idx % 5 == 0 or idx == len(refs) - 1:
            nodes.append(
                f'<text x="{x}" y="{y + 28}" text-anchor="middle" '
                f'class="{label_class}">{_escape(_short_source_ref(ref))}</text>'
            )
    evidence_labels = ", ".join(_short_source_ref(ref) for ref in sorted(graph.evidence_refs, key=_session_index))
    if evidence_labels:
        nodes.append(
            f'<text x="{left}" y="24" class="branch-label">{_escape(graph.branch_name)} branch: '
            f'{len(refs)} session commits, evidence highlighted at {evidence_labels}</text>'
        )
    else:
        nodes.append(
            f'<text x="{left}" y="24" class="branch-label">{_escape(graph.branch_name)} branch: '
            f'{len(refs)} session commits</text>'
        )
    return (
        '<div class="graph-shell">'
        '<div class="graph-toolbar">'
        '<button type="button" class="toggle-evidence">Evidence only</button>'
        '<button type="button" class="reset-view">Reset view</button>'
        '<span>Click any commit dot to inspect its source, extracted beliefs, and selected evidence.</span>'
        '</div>'
        '<div class="svg-scroll">'
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="TruthGit session commit graph for {_escape(graph.question_id)}">'
        + "".join(nodes)
        + "</svg></div>"
        '<aside class="inspector">'
        '<div class="inspector-empty">Select a commit dot to inspect this memory write.</div>'
        "</aside></div>"
    )


def _render_stat(label: str, value: object) -> str:
    return f'<div class="stat"><span>{_escape(label)}</span><strong>{_escape(value)}</strong></div>'


def _render_state_summary(graph: TraceGraph) -> str:
    status_items = "".join(
        f'<span class="state-pill { _escape(status) }">{_escape(status)}: {_escape(count)}</span>'
        for status, count in sorted(graph.status_counts.items())
    )
    if not status_items:
        status_items = '<span class="state-pill">no belief versions</span>'
    if graph.quarantine_count:
        quarantine = f'<span class="state-pill quarantined">quarantine: {_escape(graph.quarantine_count)}</span>'
    else:
        quarantine = '<span class="state-pill none">quarantine: none in this trace</span>'
    branch_note = (
        f'<span class="state-pill branch">branch: {_escape(graph.branch_name)}</span>'
    )
    lineage = (
        f'<span class="state-pill lineage-pill">supersession links: {_escape(graph.supersession_count)}</span>'
    )
    return (
        '<div class="state-summary">'
        f"{branch_note}{status_items}{lineage}{quarantine}"
        f'<p>{_escape(graph.state_note)}</p>'
        "</div>"
    )


def _render_evidence(graph: TraceGraph) -> str:
    if not graph.evidence_candidates:
        return '<div class="empty">No model evidence plan was captured.</div>'
    cards = []
    for idx, candidate in enumerate(graph.evidence_candidates, start=1):
        role = candidate.get("role") or "evidence"
        cards.append(
            f'<article class="evidence-card" data-source-ref="{_escape(candidate.get("source_ref"))}">'
            f'<div class="pill">{_escape(role)}</div>'
            f'<h4>Evidence #{idx}: {_escape(candidate.get("source_ref"))}</h4>'
            f'<p>{_escape(candidate.get("evidence"))}</p>'
            '<dl>'
            f'<dt>relation</dt><dd>{_escape(candidate.get("relation"))}</dd>'
            f'<dt>object</dt><dd>{_escape(candidate.get("object_value"))}</dd>'
            f'<dt>confidence</dt><dd>{_escape(candidate.get("confidence"))}</dd>'
            '</dl>'
            '</article>'
        )
    return '<div class="evidence-grid">' + "".join(cards) + "</div>"


def _render_beliefs(graph: TraceGraph) -> str:
    if not graph.relevant_beliefs:
        return '<div class="empty">No belief versions available.</div>'
    rows = []
    for belief in graph.relevant_beliefs:
        status = str(belief.get("status") or "unknown")
        rows.append(
            f'<tr data-belief-row data-search="{_escape(json.dumps(belief, ensure_ascii=False, default=str).lower())}">'
            f'<td>#{_escape(belief.get("belief_version_id"))}</td>'
            f'<td><strong>{_escape(belief.get("subject"))}</strong><br>'
            f'<span>{_escape(belief.get("predicate"))}</span></td>'
            f'<td>{_escape(belief.get("object_value"))}</td>'
            f'<td><span class="status { _escape(status) }">{_escape(status)}</span></td>'
            f'<td>{_escape(belief.get("source_ref"))}</td>'
            f'<td>{_escape(belief.get("source_quote"))}</td>'
            "</tr>"
        )
    return (
        '<div class="belief-toolbar">'
        '<input type="search" class="belief-search" placeholder="Search belief versions, objects, source refs..." />'
        "</div>"
        '<table class="beliefs"><thead><tr>'
        "<th>Version</th><th>Belief</th><th>Object</th><th>Status</th><th>Source</th><th>Quote</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_supersession_links(graph: TraceGraph) -> str:
    links = list(graph.lineage_versions)
    if not links:
        return '<div class="empty">No supersession or retraction links in this trace.</div>'
    by_id = {str(belief.get("belief_version_id")): belief for belief in links}
    rows = []
    for belief in links:
        status = str(belief.get("status") or "unknown")
        supersedes = belief.get("supersedes_version_id")
        if not supersedes and status not in {"superseded", "retracted"}:
            continue
        predecessor = by_id.get(str(supersedes)) if supersedes is not None else None
        if supersedes:
            relation = f"supersedes #{supersedes}"
        elif status == "superseded":
            relation = "superseded predecessor"
        else:
            relation = "retracted predecessor"
        replacement_cell = (
            f'<strong>#{_escape(belief.get("belief_version_id"))} '
            f'{_escape(belief.get("subject"))}.{_escape(belief.get("predicate"))}</strong>'
            f'<br><span class="lineage-object">{_escape(belief.get("object_value"))}</span>'
            f'<br><span class="status { _escape(status) }">{_escape(status)}</span>'
            f'<br><small>{_escape(belief.get("source_ref"))}</small>'
            f'<blockquote>{_escape(belief.get("source_quote"))}</blockquote>'
        )
        if predecessor is not None:
            predecessor_status = str(predecessor.get("status") or "unknown")
            predecessor_cell = (
                f'<strong>#{_escape(predecessor.get("belief_version_id"))} '
                f'{_escape(predecessor.get("subject"))}.{_escape(predecessor.get("predicate"))}</strong>'
                f'<br><span class="lineage-object">{_escape(predecessor.get("object_value"))}</span>'
                f'<br><span class="status { _escape(predecessor_status) }">{_escape(predecessor_status)}</span>'
                f'<br><small>{_escape(predecessor.get("source_ref"))}</small>'
                f'<blockquote>{_escape(predecessor.get("source_quote"))}</blockquote>'
            )
        else:
            predecessor_cell = (
                f'<span class="muted">Predecessor #{_escape(supersedes)} is not present in this trace.</span>'
                if supersedes
                else '<span class="muted">No predecessor pointer.</span>'
            )
        search_payload = {"replacement": belief, "predecessor": predecessor}
        rows.append(
            f'<tr data-lineage-row data-search="{_escape(json.dumps(search_payload, ensure_ascii=False, default=str).lower())}">'
            f'<td>{_escape(relation)}</td>'
            f"<td>{replacement_cell}</td>"
            f"<td>{predecessor_cell}</td>"
            "</tr>"
        )
    return (
        '<div class="lineage-note">'
        '<strong>How to read this:</strong> the left cell is the newer replacement version. '
        'The right cell is the older predecessor version from the full TruthGit database when available.'
        "</div>"
        '<div class="belief-toolbar">'
        '<input type="search" class="lineage-search" placeholder="Search supersession links, source refs, predicates..." />'
        "</div>"
        '<table class="beliefs lineage-table"><thead><tr>'
        "<th>Lineage</th><th>New replacement quote</th><th>Old superseded quote</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_graph(graph: TraceGraph) -> str:
    stats = "".join(
        [
            _render_stat("claims", graph.claim_count),
            _render_stat("staged commits", graph.staged_commit_count),
            _render_stat("applied commits", graph.commit_count),
            _render_stat("audit events", graph.audit_event_count),
        ]
    )
    evidence_json = _escape(
        json.dumps(
            [
                {
                    "source_ref": candidate.get("source_ref"),
                    "short_ref": _short_source_ref(str(candidate.get("source_ref", ""))),
                    "role": candidate.get("role"),
                    "evidence": candidate.get("evidence"),
                    "object_value": candidate.get("object_value"),
                    "confidence": candidate.get("confidence"),
                }
                for candidate in graph.evidence_candidates
            ],
            ensure_ascii=False,
            default=str,
        )
    )
    return (
        f'<section class="case" data-case-id="{_escape(graph.question_id)}" '
        f"data-evidence-summary=\"{evidence_json}\">"
        f'<header><div><p class="kicker">{_escape(graph.case_label)} {_escape(graph.question_id)}</p>'
        f'<h2>{_escape(graph.question)}</h2>'
        f'<p class="type">{_escape(graph.question_type)}</p></div>'
        f'<div class="answer"><span>TruthGit answer</span>{_escape(graph.hypothesis)}</div></header>'
        f'<div class="stats">{stats}</div>'
        f"{_render_state_summary(graph)}"
        f'<h3>Git-Style Memory Flow</h3>{_render_commit_svg(graph)}'
        f'<h3>Model Evidence Plan</h3>{_render_evidence(graph)}'
        f'<h3>Supersession Links</h3>{_render_supersession_links(graph)}'
        f'<h3>Belief Versions Used For Showcase</h3>{_render_beliefs(graph)}'
        "</section>"
    )


def render_html(graphs: list[TraceGraph]) -> str:
    body = "\n".join(_render_graph(graph) for graph in graphs)
    script = """
  <script>
    const escapeHtml = (value) => String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

    const shortText = (value, maxLength = 900) => {
      const text = String(value ?? "");
      return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
    };

    const renderInspector = (payload) => {
      const evidence = payload.evidence || [];
      const beliefs = payload.beliefs || [];
      const evidenceHtml = evidence.length
        ? `<h4>Selected Evidence</h4><ul>${evidence.map(item => `
            <li><strong>${escapeHtml(item.role || "evidence")}</strong>: ${escapeHtml(item.evidence)}
              <br><span class="meta">${escapeHtml(item.relation)} -> ${escapeHtml(item.object_value)} | confidence ${escapeHtml(item.confidence)}</span>
            </li>`).join("")}</ul>`
        : `<p class="meta">This commit was ingested into memory but was not selected as final evidence for this question.</p>`;
      const beliefsHtml = beliefs.length
        ? `<h4>Belief Versions From This Commit</h4><ul>${beliefs.map(item => `
            <li><strong>#${escapeHtml(item.belief_version_id)} ${escapeHtml(item.subject)}.${escapeHtml(item.predicate)}</strong>
              = ${escapeHtml(item.object_value)} <span class="status ${escapeHtml(item.status)}">${escapeHtml(item.status)}</span>
              <br><span class="meta">${escapeHtml(item.source_quote)}</span>
            </li>`).join("")}</ul>`
        : `<p class="meta">No highlighted belief rows for this source in the showcase subset.</p>`;
      const ciHtml = payload.ci_status
        ? `<p class="meta">Memory CI: ${escapeHtml(payload.ci_status)}${payload.ci_reason ? ` | ${escapeHtml(payload.ci_reason)}` : ""}</p>`
        : "";
      return `
        <h4>${escapeHtml(payload.short_ref)}: ${payload.is_evidence ? "evidence commit" : "memory commit"}</h4>
        <p class="meta">${escapeHtml(payload.source_ref)} | trust ${escapeHtml(payload.trust_score)} | ${escapeHtml(payload.source_type)}</p>
        ${ciHtml}
        <p>${escapeHtml(shortText(payload.excerpt))}</p>
        ${evidenceHtml}
        ${beliefsHtml}
      `;
    };

    document.querySelectorAll(".case").forEach((section) => {
      const inspector = section.querySelector(".inspector");
      const nodes = [...section.querySelectorAll(".node")];
      const evidenceCards = [...section.querySelectorAll(".evidence-card")];
      const beliefRows = [...section.querySelectorAll("[data-belief-row]")];
      const lineageRows = [...section.querySelectorAll("[data-lineage-row]")];
      const toggle = section.querySelector(".toggle-evidence");
      const reset = section.querySelector(".reset-view");
      const search = section.querySelector(".belief-search");
      const lineageSearch = section.querySelector(".lineage-search");

      const selectNode = (node) => {
        nodes.forEach(item => item.classList.remove("selected"));
        evidenceCards.forEach(item => item.classList.remove("selected"));
        beliefRows.forEach(item => item.classList.remove("selected"));
        lineageRows.forEach(item => item.classList.remove("selected"));
        node.classList.add("selected");
        const payload = JSON.parse(node.dataset.node || "{}");
        inspector.innerHTML = renderInspector(payload);
        evidenceCards
          .filter(card => card.dataset.sourceRef === payload.source_ref)
          .forEach(card => card.classList.add("selected"));
        beliefRows
          .filter(row => row.textContent.includes(payload.source_ref))
          .forEach(row => row.classList.add("selected"));
        lineageRows
          .filter(row => row.textContent.includes(payload.source_ref))
          .forEach(row => row.classList.add("selected"));
      };

      nodes.forEach((node) => {
        node.addEventListener("click", () => selectNode(node));
        node.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            selectNode(node);
          }
        });
      });

      toggle?.addEventListener("click", () => {
        section.classList.toggle("evidence-only");
        toggle.classList.toggle("active");
      });

      reset?.addEventListener("click", () => {
        section.classList.remove("evidence-only");
        toggle?.classList.remove("active");
        nodes.forEach(item => item.classList.remove("selected"));
        evidenceCards.forEach(item => item.classList.remove("selected"));
        beliefRows.forEach(item => {
          item.classList.remove("selected");
          item.hidden = false;
        });
        lineageRows.forEach(item => {
          item.classList.remove("selected");
          item.hidden = false;
        });
        if (search) search.value = "";
        if (lineageSearch) lineageSearch.value = "";
        inspector.innerHTML = '<div class="inspector-empty">Select a commit dot to inspect this memory write.</div>';
      });

      search?.addEventListener("input", () => {
        const query = search.value.trim().toLowerCase();
        beliefRows.forEach(row => {
          row.hidden = Boolean(query) && !row.dataset.search.includes(query);
        });
      });

      lineageSearch?.addEventListener("input", () => {
        const query = lineageSearch.value.trim().toLowerCase();
        lineageRows.forEach(row => {
          row.hidden = Boolean(query) && !row.dataset.search.includes(query);
        });
      });

      const firstEvidence = nodes.find(node => node.dataset.evidence === "true");
      if (firstEvidence) selectNode(firstEvidence);
    });
  </script>
"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TruthGit LongMemEval Git Graph Showcase</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17212b;
      --muted: #5c6778;
      --line: #d7dee8;
      --panel: #ffffff;
      --soft: #f4f7fb;
      --green: #12805c;
      --amber: #b7791f;
      --blue: #1d4ed8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #edf2f7;
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      width: min(1440px, calc(100vw - 40px));
      margin: 28px auto 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, #0f3d2e, #13293d);
      color: white;
      padding: 28px;
      border-radius: 8px;
      margin-bottom: 18px;
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .hero p {{ margin: 0; color: #d8e9e0; max-width: 900px; }}
    .case {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
      margin-top: 18px;
      box-shadow: 0 16px 36px rgba(23, 33, 43, 0.08);
    }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 440px);
      gap: 18px;
      align-items: start;
    }}
    .kicker {{
      margin: 0 0 8px;
      color: var(--green);
      font-weight: 800;
      letter-spacing: .06em;
      text-transform: uppercase;
      font-size: 12px;
    }}
    h2 {{ margin: 0; font-size: 24px; }}
    h3 {{ margin: 24px 0 12px; font-size: 16px; }}
    .type {{ margin: 8px 0 0; color: var(--muted); }}
    .answer {{
      border-left: 4px solid var(--green);
      background: #eef8f3;
      padding: 14px 16px;
      border-radius: 6px;
      font-weight: 650;
    }}
    .answer span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .05em;
      margin-bottom: 6px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .stat {{
      background: var(--soft);
      border: 1px solid var(--line);
      padding: 12px;
      border-radius: 6px;
    }}
    .stat span {{ display: block; color: var(--muted); font-size: 13px; }}
    .stat strong {{ display: block; margin-top: 4px; font-size: 24px; }}
    .state-summary {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      margin: 14px 0 2px;
      color: var(--muted);
    }}
    .state-summary p {{
      flex-basis: 100%;
      margin: 2px 0 0;
      font-size: 13px;
    }}
    .state-pill {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #f8fafc;
      padding: 5px 9px;
      font-size: 13px;
      font-weight: 750;
      color: var(--muted);
    }}
    .state-pill.active {{ background: #e7f6ef; border-color: #9bd8ba; color: var(--green); }}
    .state-pill.superseded, .state-pill.lineage-pill {{ background: #fff7e6; border-color: #eac16c; color: #9a5b00; }}
    .state-pill.quarantined {{ background: #fde1df; border-color: #f4a09a; color: #b42318; }}
    .state-pill.branch {{ background: #eaf1ff; border-color: #b9cdfd; color: var(--blue); }}
    .state-pill.none {{ background: #f8fafc; }}
    .svg-scroll {{
      overflow-x: auto;
      border: 1px solid var(--line);
      background: #fbfcfe;
      border-radius: 8px;
      padding: 8px;
    }}
    svg {{ width: 100%; min-width: 980px; display: block; }}
    .rail {{ stroke: #9ca8b8; stroke-width: 3; }}
    .node {{
      fill: white;
      stroke: #6b7280;
      stroke-width: 2;
      cursor: pointer;
      transition: transform .15s ease, opacity .15s ease, stroke-width .15s ease;
      transform-box: fill-box;
      transform-origin: center;
    }}
    .node:hover, .node:focus {{ transform: scale(1.45); stroke-width: 3; outline: none; }}
    .node.selected {{ fill: var(--amber); stroke: #7c2d12; stroke-width: 4; transform: scale(1.6); }}
    .case.evidence-only .node:not(.evidence) {{ opacity: .18; pointer-events: none; }}
    .node.evidence {{ fill: var(--green); stroke: #064e3b; stroke-width: 3; }}
    .node.quarantined {{ fill: #fde1df; stroke: #b42318; stroke-width: 4; }}
    .node.lineage:not(.evidence) {{ fill: #fff7e6; stroke: var(--amber); stroke-width: 3; }}
    .node-label {{ fill: var(--muted); font-size: 11px; font-weight: 650; }}
    .evidence-label {{ fill: var(--green); font-size: 13px; }}
    .quarantine-label {{ fill: #b42318; font-size: 13px; }}
    .branch-label {{ fill: var(--ink); font-size: 14px; font-weight: 750; }}
    .graph-shell {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 12px;
      align-items: stretch;
    }}
    .graph-toolbar {{
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
      flex-wrap: wrap;
    }}
    button {{
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 6px;
      padding: 8px 10px;
      font-weight: 750;
      cursor: pointer;
    }}
    button:hover {{ border-color: #94a3b8; background: #f8fafc; }}
    .toggle-evidence.active {{ background: #e7f6ef; border-color: var(--green); color: var(--green); }}
    .inspector {{
      border: 1px solid var(--line);
      background: #ffffff;
      border-radius: 8px;
      padding: 14px;
      min-height: 172px;
      overflow: auto;
    }}
    .inspector h4 {{ margin: 0 0 8px; font-size: 15px; }}
    .inspector .meta {{ color: var(--muted); margin: 0 0 10px; }}
    .inspector p {{ margin: 8px 0; }}
    .inspector ul {{ margin: 8px 0 0 18px; padding: 0; }}
    .inspector li {{ margin-bottom: 6px; }}
    .inspector-empty {{ color: var(--muted); }}
    .evidence-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }}
    .evidence-card {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 14px;
    }}
    .evidence-card.selected {{ border-color: var(--green); box-shadow: 0 0 0 3px #d8f3e6; }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      color: white;
      background: var(--blue);
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 800;
      margin-bottom: 10px;
    }}
    .evidence-card h4 {{ margin: 0 0 8px; font-size: 14px; }}
    .evidence-card p {{ margin: 0 0 10px; }}
    dl {{ display: grid; grid-template-columns: 90px 1fr; gap: 4px 8px; margin: 0; }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    .belief-toolbar {{
      display: flex;
      justify-content: flex-end;
      margin-bottom: 10px;
    }}
    .belief-search, .lineage-search {{
      width: min(100%, 420px);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
    }}
    .lineage-note {{
      border-left: 4px solid var(--amber);
      background: #fff7e6;
      padding: 10px 12px;
      border-radius: 6px;
      color: #684000;
      margin-bottom: 10px;
    }}
    .beliefs th, .beliefs td {{
      border-bottom: 1px solid var(--line);
      padding: 10px;
      text-align: left;
      vertical-align: top;
    }}
    .beliefs th {{ color: var(--muted); font-size: 13px; }}
    .beliefs span {{ color: var(--muted); }}
    .beliefs tr.selected td {{ background: #fff7e6; }}
    .lineage-table tr.selected td {{ background: #fff1d6; }}
    .lineage-object {{
      display: inline-block;
      color: var(--ink) !important;
      font-weight: 800;
      margin: 4px 0;
    }}
    .lineage-table blockquote {{
      margin: 8px 0 0;
      padding-left: 10px;
      border-left: 3px solid var(--line);
      color: var(--ink);
    }}
    .muted {{ color: var(--muted); }}
    .status {{
      display: inline-block;
      background: #e7f6ef;
      color: var(--green) !important;
      border-radius: 999px;
      padding: 3px 8px;
      font-weight: 750;
    }}
    .status.superseded {{ background: #fff7e6; color: #9a5b00 !important; }}
    .status.quarantined {{ background: #fde1df; color: #b42318 !important; }}
    .status.retracted {{ background: #f4f4f5; color: #52525b !important; }}
    .status.hypothetical {{ background: #f4f0ff; color: #6d3fc4 !important; }}
    .empty {{
      border: 1px dashed var(--line);
      color: var(--muted);
      padding: 16px;
      border-radius: 8px;
    }}
    @media (max-width: 900px) {{
      main {{ width: calc(100vw - 20px); }}
      header {{ grid-template-columns: 1fr; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .graph-shell {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>TruthGit Git Graph Showcase</h1>
      <p>Each dot is a staged memory write flowing through TruthGit. Green dots are current-answer evidence, amber marks supersession lineage, and red marks quarantined writes blocked by Memory CI.</p>
    </section>
    {body}
  </main>
  {script}
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default=None, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ids", nargs="*", default=None, help="Optional question IDs to include.")
    parser.add_argument(
        "--include-governance-demo",
        action="store_true",
        help="Append an illustrative supersession/quarantine demo graph.",
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Render only the illustrative supersession/quarantine demo graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graphs: list[TraceGraph] = []
    if not args.demo_only:
        if args.trace_dir is None:
            raise SystemExit("--trace-dir is required unless --demo-only is set")
        ids = set(args.ids or [])
        traces = sorted(args.trace_dir.glob("*.truthgit-trace.json"))
        if ids:
            traces = [path for path in traces if path.name.split(".", 1)[0] in ids]
        graphs.extend(_load_trace(path) for path in traces)
    if args.include_governance_demo or args.demo_only:
        graphs.append(_governance_demo_graph())
    if not graphs:
        raise SystemExit(f"No trace files found in {args.trace_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(graphs), encoding="utf-8")
    print(f"Wrote {args.output} with {len(graphs)} graph(s).")


if __name__ == "__main__":
    main()
