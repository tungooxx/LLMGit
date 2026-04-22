"""Generate an interactive TruthGit Memory CI showcase for demos.

This script runs a small, real TruthGit workflow in an in-memory SQLite
database:

1. A trusted source says the user has 3 apples.
2. A later trusted source says the user now has 4 apples, superseding 3.
3. A low-trust source claims the user has 400 apples and Memory CI quarantines it.

The generated HTML is meant for professor-facing explanation. It is not a
benchmark scorer and does not use benchmark IDs or expected outputs.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import models
from app.commit_engine import ensure_main_branch
from app.db import Base
from app.schemas import ExtractedClaim, SourceCreate
from app.tools import approve_staged_commit, stage_belief_changes


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _json_default(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _create_session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    return session_factory()


def _stage_apple_claim(
    db: Session,
    *,
    branch_id: int,
    object_value: str,
    confidence: float,
    trust_score: float,
    valid_from: date,
    source_ref: str,
    source_excerpt: str,
) -> models.StagedCommit:
    staged = stage_belief_changes(
        db,
        claims=[
            ExtractedClaim(
                subject="user",
                predicate="has_apples",
                object=object_value,
                confidence=confidence,
                valid_from=valid_from,
                source_quote=source_excerpt,
            )
        ],
        branch_id=branch_id,
        source=SourceCreate(
            source_type="document",
            source_ref=source_ref,
            excerpt=source_excerpt,
            trust_score=trust_score,
        ),
        proposed_commit_message=f"{source_ref}: user has {object_value} apples",
        created_by="professor-showcase",
        model_name="deterministic-showcase",
    )
    db.flush()
    if staged.status != "quarantined":
        approve_staged_commit(
            db,
            staged_commit_id=staged.id,
            reviewer="professor-demo-reviewer",
            notes="Approved for the apple showcase after deterministic Memory CI.",
            commit_message=staged.proposed_commit_message,
            model_name="deterministic-showcase",
        )
    db.commit()
    return staged


def _check_results_for_staged(db: Session, staged_id: str) -> list[dict[str, Any]]:
    runs = db.scalars(
        select(models.MemoryCheckRun)
        .where(models.MemoryCheckRun.staged_commit_id == staged_id)
        .order_by(models.MemoryCheckRun.created_at)
    ).all()
    output: list[dict[str, Any]] = []
    for run in runs:
        results = db.scalars(
            select(models.MemoryCheckResult)
            .where(models.MemoryCheckResult.run_id == run.id)
            .order_by(models.MemoryCheckResult.id)
        ).all()
        output.append(
            {
                "run_id": run.id,
                "overall_status": run.overall_status,
                "decision": run.decision,
                "score": run.score,
                "results": [
                    {
                        "check_name": result.check_name,
                        "severity": result.severity,
                        "passed": result.passed,
                        "reason_code": result.reason_code,
                        "message": result.message,
                        "payload": result.payload_json,
                    }
                    for result in results
                ],
            }
        )
    return output


def _source_payload(db: Session, source_id: int) -> dict[str, Any]:
    source = db.get(models.Source, source_id)
    if source is None:
        return {}
    return {
        "id": source.id,
        "source_ref": source.source_ref,
        "source_type": source.source_type,
        "trust_score": source.trust_score,
        "excerpt": source.excerpt,
    }


def build_showcase_payload() -> dict[str, Any]:
    """Run the real TruthGit workflow and return serializable showcase data."""

    db = _create_session()
    try:
        branch = ensure_main_branch(db)
        db.commit()
        staged_initial = _stage_apple_claim(
            db,
            branch_id=branch.id,
            object_value="3",
            confidence=0.9,
            trust_score=0.9,
            valid_from=date(2026, 4, 20),
            source_ref="showcase:pantry-count-v1",
            source_excerpt="Verified pantry count on April 20: the user has 3 apples.",
        )
        staged_update = _stage_apple_claim(
            db,
            branch_id=branch.id,
            object_value="4",
            confidence=0.92,
            trust_score=0.92,
            valid_from=date(2026, 4, 21),
            source_ref="showcase:pantry-count-v2",
            source_excerpt="Updated pantry count on April 21: the user now has 4 apples.",
        )
        staged_poison = _stage_apple_claim(
            db,
            branch_id=branch.id,
            object_value="400",
            confidence=0.1,
            trust_score=0.1,
            valid_from=date(2026, 4, 22),
            source_ref="showcase:anonymous-forum-poison",
            source_excerpt="Anonymous forum post on April 22 claims the user has 400 apples.",
        )
        staged_items = [staged_initial, staged_update, staged_poison]
        versions = db.scalars(select(models.BeliefVersion).order_by(models.BeliefVersion.id)).all()
        commits = db.scalars(select(models.Commit).order_by(models.Commit.id)).all()
        audits = db.scalars(select(models.AuditEvent).order_by(models.AuditEvent.id)).all()

        active_versions = [version for version in versions if version.status == "active"]
        active_answer = active_versions[-1].object_value if active_versions else "unknown"
        quarantined_claims = [
            {
                "staged_id": staged.id,
                "source_ref": staged.source_ref,
                "object_value": (staged.claims_json[0] or {}).get("object_value") if staged.claims_json else None,
                "status": staged.status,
                "reason": staged.quarantine_reason_summary,
            }
            for staged in staged_items
            if staged.status == "quarantined"
        ]
        return {
            "question": "How many apples does the user have now?",
            "truthgit_answer": f"The user has {active_answer} apples.",
            "naive_poisoned_answer": "A naive last-write memory could answer 400 apples if it trusted the poisoned source.",
            "branch": {"id": branch.id, "name": branch.name},
            "staged": [
                {
                    "id": staged.id,
                    "status": staged.status,
                    "source_ref": staged.source_ref,
                    "source_trust_score": staged.source_trust_score,
                    "review_required": staged.review_required,
                    "quarantine_reason_summary": staged.quarantine_reason_summary,
                    "applied_commit_id": staged.applied_commit_id,
                    "claims": staged.claims_json,
                    "check_runs": _check_results_for_staged(db, staged.id),
                }
                for staged in staged_items
            ],
            "versions": [
                {
                    "id": version.id,
                    "belief_id": version.belief_id,
                    "commit_id": version.commit_id,
                    "branch_id": version.branch_id,
                    "object_value": version.object_value,
                    "status": version.status,
                    "confidence": version.confidence,
                    "valid_from": version.valid_from,
                    "valid_to": version.valid_to,
                    "supersedes_version_id": version.supersedes_version_id,
                    "contradiction_group": version.contradiction_group,
                    "source": _source_payload(db, version.source_id),
                }
                for version in versions
            ],
            "quarantined_claims": quarantined_claims,
            "commits": [
                {
                    "id": commit.id,
                    "parent_commit_id": commit.parent_commit_id,
                    "operation_type": commit.operation_type,
                    "message": commit.message,
                    "created_by": commit.created_by,
                    "model_name": commit.model_name,
                    "created_at": commit.created_at,
                }
                for commit in commits
            ],
            "audit_events": [
                {
                    "id": audit.id,
                    "event_type": audit.event_type,
                    "entity_type": audit.entity_type,
                    "entity_id": audit.entity_id,
                    "entity_key": audit.entity_key,
                    "payload": audit.payload_json,
                    "created_at": audit.created_at,
                }
                for audit in audits
            ],
        }
    finally:
        db.close()


def _render_stage_card(stage: dict[str, Any], index: int) -> str:
    claim = stage["claims"][0] if stage["claims"] else {}
    status = stage["status"]
    checks = [
        result
        for run in stage["check_runs"]
        for result in run["results"]
        if result["severity"] != "info" or not result["passed"]
    ]
    if not checks:
        checks = [
            {
                "check_name": "memory_ci",
                "severity": "pass",
                "passed": True,
                "reason_code": "all_checks_passed",
                "message": "All deterministic Memory CI checks passed.",
            }
        ]
    check_html = "".join(
        "<li>"
        f'<strong>{_escape(check["check_name"])}</strong> '
        f'<span class="severity { _escape(check["severity"]) }">{_escape(check["severity"])}</span>'
        f'<br><span>{_escape(check["message"])}</span>'
        "</li>"
        for check in checks
    )
    action = "applied as truth" if status == "applied" else "blocked in quarantine"
    return (
        f'<article class="stage-card { _escape(status) }" '
        f'data-stage="{_escape(json.dumps(stage, default=_json_default))}">'
        f'<div class="stage-index">{index}</div>'
        f'<h3>{_escape(stage["source_ref"])}</h3>'
        f'<p class="claim">user.has_apples = <strong>{_escape(claim.get("object_value"))}</strong></p>'
        f'<p>trust {_escape(stage["source_trust_score"])} | status <strong>{_escape(status)}</strong> | {action}</p>'
        f'<ul>{check_html}</ul>'
        "</article>"
    )


def _render_version_row(version: dict[str, Any]) -> str:
    source = version["source"]
    lineage = "root"
    if version.get("supersedes_version_id"):
        lineage = f"supersedes #{version['supersedes_version_id']}"
    return (
        f'<tr class="{_escape(version["status"])}">'
        f'<td>#{_escape(version["id"])}</td>'
        f'<td>{_escape(version["object_value"])} apples</td>'
        f'<td><span class="badge { _escape(version["status"]) }">{_escape(version["status"])}</span></td>'
        f'<td>{_escape(version["valid_from"])}</td>'
        f'<td>{_escape(lineage)}</td>'
        f'<td>{_escape(source.get("source_ref"))}<br><small>trust {_escape(source.get("trust_score"))}</small></td>'
        "</tr>"
    )


def _render_audit_rows(events: list[dict[str, Any]]) -> str:
    rows = []
    for event in events:
        if event["event_type"] not in {
            "staged_commit.created",
            "staged_commit.approved",
            "staged_commit.quarantined",
            "commit.created",
            "belief_version.created",
        }:
            continue
        rows.append(
            "<tr>"
            f'<td>#{_escape(event["id"])}</td>'
            f'<td>{_escape(event["event_type"])}</td>'
            f'<td>{_escape(event["entity_type"])}</td>'
            f'<td>{_escape(event["entity_key"] or event["entity_id"])}</td>'
            f'<td><code>{_escape(json.dumps(event["payload"], default=_json_default))}</code></td>'
            "</tr>"
        )
    return "".join(rows)


def render_html(payload: dict[str, Any]) -> str:
    stages = payload["staged"]
    stage_cards = "".join(_render_stage_card(stage, idx + 1) for idx, stage in enumerate(stages))
    version_rows = "".join(_render_version_row(version) for version in payload["versions"])
    audit_rows = _render_audit_rows(payload["audit_events"])
    data_json = _escape(json.dumps(payload, ensure_ascii=False, default=_json_default))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TruthGit Memory CI Showcase</title>
  <style>
    :root {{
      --ink: #15202b;
      --muted: #5c6778;
      --line: #d6dee8;
      --panel: #ffffff;
      --soft: #f5f7fb;
      --green: #12785a;
      --amber: #a76100;
      --red: #b42318;
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
    main {{ width: min(1480px, calc(100vw - 36px)); margin: 24px auto 52px; }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 18px;
      background: #102d24;
      color: white;
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 16px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .hero p {{ margin: 0; color: #dcece5; }}
    .answer-box {{
      background: white;
      color: var(--ink);
      border-radius: 8px;
      padding: 16px;
    }}
    .answer-box strong {{ display: block; color: var(--green); font-size: 26px; margin-top: 6px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 16px;
      box-shadow: 0 12px 28px rgba(21, 32, 43, .08);
    }}
    .flow {{
      display: grid;
      grid-template-columns: repeat(3, minmax(260px, 1fr));
      gap: 12px;
    }}
    .stage-card {{
      position: relative;
      min-height: 260px;
      border: 2px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: white;
      cursor: pointer;
    }}
    .stage-card:hover {{ border-color: #9ca8b8; }}
    .stage-card.selected {{ box-shadow: 0 0 0 4px #dbeafe; border-color: var(--blue); }}
    .stage-card.applied {{ border-color: #8fd0b5; }}
    .stage-card.quarantined {{ border-color: #ee9b96; background: #fff8f7; }}
    .stage-index {{
      position: absolute;
      top: 12px;
      right: 12px;
      width: 34px;
      height: 34px;
      border-radius: 999px;
      display: grid;
      place-items: center;
      background: var(--soft);
      font-weight: 850;
    }}
    .claim {{ font-size: 18px; margin: 12px 0; }}
    ul {{ margin: 10px 0 0 18px; padding: 0; }}
    li {{ margin: 0 0 8px; }}
    .severity, .badge {{
      display: inline-flex;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 800;
      background: #eef2f7;
      color: var(--muted);
    }}
    .severity.pass, .badge.active {{ background: #e7f6ef; color: var(--green); }}
    .severity.warn, .badge.superseded {{ background: #fff1d6; color: var(--amber); }}
    .severity.fail, .badge.quarantined {{ background: #fde1df; color: var(--red); }}
    .split {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 400px;
      gap: 16px;
    }}
    .graph {{
      min-height: 260px;
      border: 1px solid var(--line);
      background: #fbfcfe;
      border-radius: 8px;
      padding: 14px;
    }}
    svg {{ width: 100%; height: 238px; display: block; }}
    .edge {{ stroke: #9ca8b8; stroke-width: 4; fill: none; }}
    .edge.blocked {{ stroke: #d92d20; stroke-dasharray: 7 7; }}
    .node {{ cursor: pointer; }}
    .node circle {{ stroke-width: 3; }}
    .node text {{ font-size: 13px; font-weight: 800; fill: var(--ink); }}
    .node small {{ color: var(--muted); }}
    .inspector {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: white;
      min-height: 260px;
    }}
    .inspector h3 {{ font-size: 18px; }}
    .inspector code {{
      display: block;
      white-space: pre-wrap;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      max-height: 160px;
      overflow: auto;
      font-size: 12px;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); padding: 9px; }}
    th {{ color: var(--muted); font-size: 13px; }}
    tr.superseded td {{ background: #fffbf0; }}
    tr.active td {{ background: #f4fbf8; }}
    .quarantine-callout {{
      border-left: 4px solid var(--red);
      background: #fff8f7;
      padding: 12px;
      border-radius: 6px;
      margin-top: 12px;
    }}
    @media (max-width: 1000px) {{
      .hero, .split, .flow {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div>
      <h1>TruthGit Memory CI/CD Showcase</h1>
      <p>Same belief topic, changing truth: 3 apples is superseded by 4 apples. A poisoned 400-apples update is staged, checked, quarantined, and never becomes active truth.</p>
    </div>
    <div class="answer-box">
      Question: {_escape(payload["question"])}
      <strong>{_escape(payload["truthgit_answer"])}</strong>
      <p>{_escape(payload["naive_poisoned_answer"])}</p>
    </div>
  </section>

  <section class="panel">
    <h2>Reviewable Memory Writes</h2>
    <div class="flow">{stage_cards}</div>
  </section>

  <section class="panel split">
    <div>
      <h2>Git-Style Belief Graph</h2>
      <div class="graph">
        <svg viewBox="0 0 920 238" aria-label="Apple belief graph">
          <path class="edge" d="M120 104 C220 70, 300 70, 400 104" />
          <path class="edge blocked" d="M400 104 C520 70, 620 70, 742 104" />
          <g class="node graph-node" data-index="0" transform="translate(120 104)">
            <circle r="38" fill="#fff1d6" stroke="#a76100"></circle>
            <text text-anchor="middle" y="-4">v1</text>
            <text text-anchor="middle" y="16">3 apples</text>
          </g>
          <g class="node graph-node" data-index="1" transform="translate(400 104)">
            <circle r="42" fill="#e7f6ef" stroke="#12785a"></circle>
            <text text-anchor="middle" y="-4">v2</text>
            <text text-anchor="middle" y="16">4 apples</text>
          </g>
          <g class="node graph-node" data-index="2" transform="translate(742 104)">
            <circle r="42" fill="#fde1df" stroke="#b42318"></circle>
            <text text-anchor="middle" y="-4">quarantine</text>
            <text text-anchor="middle" y="16">400 apples</text>
          </g>
          <text x="178" y="55" fill="#a76100" font-size="14" font-weight="800">superseded</text>
          <text x="505" y="55" fill="#b42318" font-size="14" font-weight="800">blocked by Memory CI</text>
          <text x="390" y="182" fill="#12785a" font-size="15" font-weight="900">current truth</text>
        </svg>
      </div>
    </div>
    <aside class="inspector" id="inspector">
      <h3>Select a memory write</h3>
      <p>Click a card or graph node to inspect the real CI decision, audit link, and belief effect.</p>
    </aside>
  </section>

  <section class="panel">
    <h2>Belief Versions</h2>
    <table>
      <thead><tr><th>Version</th><th>Object</th><th>Status</th><th>Valid From</th><th>Lineage</th><th>Source</th></tr></thead>
      <tbody>{version_rows}</tbody>
    </table>
    <div class="quarantine-callout">
      The quarantined 400-apples proposal has no belief version row because it never became active durable truth. It remains auditable as a staged commit and Memory CI run.
    </div>
  </section>

  <section class="panel">
    <h2>Audit Trail</h2>
    <table>
      <thead><tr><th>ID</th><th>Event</th><th>Entity</th><th>Key</th><th>Payload</th></tr></thead>
      <tbody>{audit_rows}</tbody>
    </table>
  </section>
</main>
<script id="showcase-data" type="application/json">{data_json}</script>
<script>
  const payload = JSON.parse(document.getElementById("showcase-data").textContent);
  const inspector = document.getElementById("inspector");
  const cards = Array.from(document.querySelectorAll(".stage-card"));
  const graphNodes = Array.from(document.querySelectorAll(".graph-node"));

  const escapeHtml = (value) => String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

  function renderStage(stage) {{
    const claim = stage.claims[0] || {{}};
    const checks = stage.check_runs.flatMap(run => run.results);
    const important = checks.filter(check => check.severity !== "info" || check.passed === false);
    const shown = important.length ? important : [{{check_name: "memory_ci", severity: "pass", message: "All deterministic Memory CI checks passed."}}];
    return `
      <h3>${{escapeHtml(stage.source_ref)}}</h3>
      <p><strong>Claim:</strong> user.has_apples = ${{escapeHtml(claim.object_value)}} apples</p>
      <p><strong>Status:</strong> ${{escapeHtml(stage.status)}} | <strong>Trust:</strong> ${{escapeHtml(stage.source_trust_score)}} | <strong>Commit:</strong> ${{escapeHtml(stage.applied_commit_id || "not applied")}}</p>
      ${{stage.quarantine_reason_summary ? `<p class="quarantine-callout"><strong>Quarantine reason:</strong> ${{escapeHtml(stage.quarantine_reason_summary)}}</p>` : ""}}
      <h3>Memory CI checks</h3>
      <ul>${{shown.map(check => `<li><strong>${{escapeHtml(check.check_name)}}</strong> <span class="severity ${{escapeHtml(check.severity)}}">${{escapeHtml(check.severity)}}</span><br>${{escapeHtml(check.message)}}</li>`).join("")}}</ul>
      <h3>Raw staged record</h3>
      <code>${{escapeHtml(JSON.stringify(stage, null, 2))}}</code>
    `;
  }}

  function selectStage(index) {{
    cards.forEach(card => card.classList.remove("selected"));
    const card = cards[index];
    if (card) card.classList.add("selected");
    inspector.innerHTML = renderStage(payload.staged[index]);
  }}

  cards.forEach((card, index) => card.addEventListener("click", () => selectStage(index)));
  graphNodes.forEach((node) => node.addEventListener("click", () => selectStage(Number(node.dataset.index))));
  selectStage(1);
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/public_results/truthgit_professor_apples_ci_showcase.html"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_showcase_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(payload), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(payload["truthgit_answer"])
    print(payload["naive_poisoned_answer"])


if __name__ == "__main__":
    main()
