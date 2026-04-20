"""Visual inspection routes for TruthGit memory state."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models
from app.db import get_db

router = APIRouter(tags=["visualization"])


@router.get("/viz", response_class=HTMLResponse)
def visualization_page() -> HTMLResponse:
    """Serve a small self-contained dashboard for local demos."""

    return HTMLResponse(_HTML)


@router.get("/viz/data")
def visualization_data(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return a compact graph-friendly snapshot of TruthGit state."""

    branches = list(db.scalars(select(models.Branch).order_by(models.Branch.id)))
    commits = list(db.scalars(select(models.Commit).order_by(models.Commit.id.desc()).limit(30)))
    beliefs = list(db.scalars(select(models.Belief).order_by(models.Belief.id)))
    versions = list(
        db.scalars(select(models.BeliefVersion).order_by(models.BeliefVersion.id.desc()).limit(80))
    )
    staged = list(db.scalars(select(models.StagedCommit).order_by(models.StagedCommit.created_at.desc()).limit(30)))
    audit = list(db.scalars(select(models.AuditEvent).order_by(models.AuditEvent.id.desc()).limit(40)))
    sources = {
        source.id: source
        for source in db.scalars(select(models.Source).order_by(models.Source.id))
    }
    belief_by_id = {belief.id: belief for belief in beliefs}
    branch_by_id = {branch.id: branch for branch in branches}

    active_versions = [
        version
        for version in versions
        if version.status in {"active", "hypothetical"}
    ]
    conflict_versions = [
        version
        for version in active_versions
        if version.contradiction_group
    ]
    return {
        "counts": {
            "branches": len(branches),
            "commits": len(commits),
            "beliefs": len(beliefs),
            "versions": len(versions),
            "staged": len(staged),
            "active_versions": len(active_versions),
            "conflict_versions": len(conflict_versions),
            "audit_events": len(audit),
        },
        "branches": [
            {
                "id": branch.id,
                "name": branch.name,
                "status": branch.status,
                "parent_branch_id": branch.parent_branch_id,
                "created_at": _iso(branch.created_at),
            }
            for branch in branches
        ],
        "commits": [
            {
                "id": commit.id,
                "branch_id": commit.branch_id,
                "branch_name": branch_by_id.get(commit.branch_id).name
                if branch_by_id.get(commit.branch_id)
                else str(commit.branch_id),
                "parent_commit_id": commit.parent_commit_id,
                "operation_type": commit.operation_type,
                "message": commit.message,
                "created_by": commit.created_by,
                "created_at": _iso(commit.created_at),
            }
            for commit in commits
        ],
        "belief_versions": [
            _version_payload(version, belief_by_id, branch_by_id, sources)
            for version in versions
        ],
        "staged_commits": [
            {
                "id": item.id,
                "branch_id": item.branch_id,
                "branch_name": branch_by_id.get(item.branch_id).name
                if branch_by_id.get(item.branch_id)
                else str(item.branch_id),
                "status": item.status,
                "review_required": item.review_required,
                "risk_reasons": item.risk_reasons,
                "source_ref": item.source_ref,
                "source_trust_score": item.source_trust_score,
                "applied_commit_id": item.applied_commit_id,
                "created_at": _iso(item.created_at),
            }
            for item in staged
        ],
        "audit_events": [
            {
                "id": event.id,
                "event_type": event.event_type,
                "entity_type": event.entity_type,
                "entity_id": event.entity_id,
                "entity_key": event.entity_key,
                "payload_json": event.payload_json,
                "created_at": _iso(event.created_at),
            }
            for event in audit
        ],
    }


def _version_payload(
    version: models.BeliefVersion,
    belief_by_id: dict[int, models.Belief],
    branch_by_id: dict[int, models.Branch],
    sources: dict[int, models.Source],
) -> dict[str, Any]:
    belief = belief_by_id.get(version.belief_id)
    branch = branch_by_id.get(version.branch_id)
    source = sources.get(version.source_id)
    return {
        "id": version.id,
        "belief_id": version.belief_id,
        "subject": belief.subject if belief else str(version.belief_id),
        "predicate": belief.predicate if belief else "",
        "object_value": version.object_value,
        "status": version.status,
        "branch_id": version.branch_id,
        "branch_name": branch.name if branch else str(version.branch_id),
        "commit_id": version.commit_id,
        "source_id": version.source_id,
        "source_ref": source.source_ref if source else None,
        "source_trust_score": source.trust_score if source else None,
        "supersedes_version_id": version.supersedes_version_id,
        "contradiction_group": version.contradiction_group,
        "valid_from": _iso(version.valid_from),
        "valid_to": _iso(version.valid_to),
        "created_at": _iso(version.created_at),
    }


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TruthGit Memory Graph</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #202124;
      --muted: #5f6368;
      --line: #d9dee3;
      --panel: #ffffff;
      --page: #f4f7f6;
      --green: #1e8e5a;
      --blue: #1a73e8;
      --amber: #b06000;
      --red: #c5221f;
      --violet: #6f42c1;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--page);
      color: var(--ink);
    }
    header {
      min-height: 148px;
      padding: 28px clamp(20px, 4vw, 48px);
      background: #10241c;
      color: #f8fbf9;
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 20px;
    }
    h1 {
      margin: 0;
      font-size: 34px;
      line-height: 1.05;
      letter-spacing: 0;
    }
    .subtitle {
      margin: 10px 0 0;
      color: #c6d7d0;
      max-width: 760px;
      line-height: 1.5;
    }
    button {
      border: 1px solid #91a39b;
      background: #f8fbf9;
      color: #10241c;
      border-radius: 6px;
      padding: 10px 14px;
      font-weight: 650;
      cursor: pointer;
      min-width: 112px;
    }
    main {
      padding: 24px clamp(16px, 4vw, 48px) 42px;
      display: grid;
      gap: 18px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
    }
    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .metric {
      padding: 14px 16px;
      min-height: 88px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 13px;
    }
    .metric strong {
      display: block;
      margin-top: 8px;
      font-size: 30px;
      line-height: 1;
    }
    .pipeline {
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 12px;
      align-items: stretch;
    }
    .step {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      padding: 14px;
      min-height: 96px;
      position: relative;
    }
    .step:not(:last-child)::after {
      content: ">";
      position: absolute;
      right: -10px;
      top: 38px;
      color: var(--muted);
      font-weight: 800;
    }
    .step b { display: block; margin-bottom: 6px; }
    .step small { color: var(--muted); line-height: 1.35; }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(360px, .8fr);
      gap: 18px;
    }
    .panel {
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      padding: 14px 16px;
      font-size: 17px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
      letter-spacing: 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 650;
      background: #fbfcfd;
    }
    .pill {
      display: inline-block;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 650;
      background: #edf2f7;
      color: #25313b;
      white-space: nowrap;
    }
    .active { background: #dff3e8; color: var(--green); }
    .hypothetical { background: #e9ddfb; color: var(--violet); }
    .superseded { background: #fff1d6; color: var(--amber); }
    .retracted { background: #fde1df; color: var(--red); }
    .conflict { color: var(--red); font-weight: 700; }
    .muted { color: var(--muted); }
    .timeline {
      max-height: 460px;
      overflow: auto;
    }
    .event {
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
    }
    .event b {
      display: block;
      font-size: 13px;
    }
    .event small {
      color: var(--muted);
      display: block;
      margin-top: 4px;
    }
    @media (max-width: 920px) {
      header { align-items: flex-start; flex-direction: column; }
      .grid { grid-template-columns: 1fr; }
      .pipeline { grid-template-columns: 1fr; }
      .step:not(:last-child)::after { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>TruthGit Memory Graph</h1>
      <p class="subtitle">Live view of staged writes, branch state, belief versions, provenance, conflicts, and audit events.</p>
    </div>
    <button id="refresh">Refresh</button>
  </header>
  <main>
    <section class="metrics" id="metrics"></section>
    <section class="pipeline">
      <div class="step"><b>Extract</b><small>Claims enter as normalized subject, predicate, object records.</small></div>
      <div class="step"><b>Stage</b><small>Risk gates keep proposed writes reviewable before commit.</small></div>
      <div class="step"><b>Approve</b><small>Reviewed staged commits become deterministic memory operations.</small></div>
      <div class="step"><b>Version</b><small>Belief versions preserve supersession, branch, rollback, and source lineage.</small></div>
      <div class="step"><b>Audit</b><small>Every durable operation is visible in the append-only event log.</small></div>
    </section>
    <section class="grid">
      <div class="panel">
        <h2>Belief Versions</h2>
        <div class="timeline">
          <table>
            <thead>
              <tr>
                <th>ID</th><th>Belief</th><th>Object</th><th>Status</th><th>Branch</th><th>Source</th><th>Lineage</th>
              </tr>
            </thead>
            <tbody id="versions"></tbody>
          </table>
        </div>
      </div>
      <div class="panel">
        <h2>Audit Timeline</h2>
        <div class="timeline" id="audit"></div>
      </div>
    </section>
    <section class="grid">
      <div class="panel">
        <h2>Staged Writes</h2>
        <div class="timeline">
          <table>
            <thead>
              <tr><th>ID</th><th>Status</th><th>Branch</th><th>Review</th><th>Source</th></tr>
            </thead>
            <tbody id="staged"></tbody>
          </table>
        </div>
      </div>
      <div class="panel">
        <h2>Branches</h2>
        <div class="timeline">
          <table>
            <thead>
              <tr><th>ID</th><th>Name</th><th>Status</th><th>Parent</th></tr>
            </thead>
            <tbody id="branches"></tbody>
          </table>
        </div>
      </div>
    </section>
  </main>
  <script>
    const metricLabels = [
      ["branches", "Branches"],
      ["staged", "Staged"],
      ["commits", "Commits"],
      ["beliefs", "Beliefs"],
      ["active_versions", "Active versions"],
      ["conflict_versions", "Conflicts"],
      ["audit_events", "Audit events"]
    ];

    async function loadData() {
      const response = await fetch("/viz/data", {headers: {"Accept": "application/json"}});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      renderMetrics(data.counts);
      renderVersions(data.belief_versions);
      renderAudit(data.audit_events);
      renderStaged(data.staged_commits);
      renderBranches(data.branches);
    }

    function renderMetrics(counts) {
      document.getElementById("metrics").innerHTML = metricLabels.map(([key, label]) => `
        <div class="metric"><span>${escapeHtml(label)}</span><strong>${counts[key] ?? 0}</strong></div>
      `).join("");
    }

    function renderVersions(rows) {
      document.getElementById("versions").innerHTML = rows.map(row => `
        <tr>
          <td>#${row.id}</td>
          <td><b>${escapeHtml(row.subject)}</b><br><span class="muted">${escapeHtml(row.predicate)}</span></td>
          <td>${escapeHtml(row.object_value)}</td>
          <td><span class="pill ${escapeHtml(row.status)}">${escapeHtml(row.status)}</span>${row.contradiction_group ? '<br><span class="conflict">conflict</span>' : ''}</td>
          <td>${escapeHtml(row.branch_name)}</td>
          <td>${escapeHtml(row.source_ref || "source-" + row.source_id)}<br><span class="muted">trust ${formatTrust(row.source_trust_score)}</span></td>
          <td>${row.supersedes_version_id ? "supersedes #" + row.supersedes_version_id : "root"}<br><span class="muted">commit #${row.commit_id}</span></td>
        </tr>
      `).join("") || `<tr><td colspan="7" class="muted">No belief versions yet.</td></tr>`;
    }

    function renderAudit(rows) {
      document.getElementById("audit").innerHTML = rows.map(row => `
        <div class="event">
          <b>#${row.id} ${escapeHtml(row.event_type)}</b>
          <small>${escapeHtml(row.entity_type)}:${escapeHtml(row.entity_key || row.entity_id)} at ${escapeHtml(row.created_at || "")}</small>
        </div>
      `).join("") || `<div class="event muted">No audit events yet.</div>`;
    }

    function renderStaged(rows) {
      document.getElementById("staged").innerHTML = rows.map(row => `
        <tr>
          <td>${escapeHtml(row.id.slice(0, 8))}</td>
          <td><span class="pill">${escapeHtml(row.status)}</span></td>
          <td>${escapeHtml(row.branch_name)}</td>
          <td>${row.review_required ? "required" : "optional"}<br><span class="muted">${escapeHtml((row.risk_reasons || []).join(", "))}</span></td>
          <td>${escapeHtml(row.source_ref || "inline")}<br><span class="muted">trust ${formatTrust(row.source_trust_score)}</span></td>
        </tr>
      `).join("") || `<tr><td colspan="5" class="muted">No staged writes.</td></tr>`;
    }

    function renderBranches(rows) {
      document.getElementById("branches").innerHTML = rows.map(row => `
        <tr>
          <td>#${row.id}</td>
          <td>${escapeHtml(row.name)}</td>
          <td><span class="pill">${escapeHtml(row.status)}</span></td>
          <td>${row.parent_branch_id ?? ""}</td>
        </tr>
      `).join("") || `<tr><td colspan="4" class="muted">No branches.</td></tr>`;
    }

    function formatTrust(value) {
      if (value === null || value === undefined) return "n/a";
      return Number(value).toFixed(2);
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, character => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[character]));
    }

    document.getElementById("refresh").addEventListener("click", loadData);
    loadData().catch(error => {
      document.getElementById("audit").innerHTML = `<div class="event conflict">${escapeHtml(error.message)}</div>`;
    });
  </script>
</body>
</html>
"""
