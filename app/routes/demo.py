"""Professor-facing live demo UI for benchmark playback and manual prompts."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Generator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app import crud, models
from app.commit_engine import create_branch, ensure_main_branch, rollback_commit
from app.db import get_db
from app.normalization import deterministic_extract_simple_claims
from app.schemas import ExtractedClaim, SourceCreate
from app.tools import approve_staged_commit, stage_belief_changes
from experiments.baselines import default_systems
from experiments.benchmark import BenchmarkCase, default_benchmark
from experiments.metrics import score_answer

router = APIRouter(tags=["demo"])


class DemoPromptRequest(BaseModel):
    """Manual prompt request for the demo UI."""

    message: str
    branch_name: str = "main"
    trust_score: float = Field(default=0.7, ge=0.0, le=1.0)
    auto_approve: bool = False


class DemoRollbackRequest(BaseModel):
    """Rollback request for the demo UI."""

    commit_id: int
    message: str | None = None


class DemoResetRequest(BaseModel):
    """Explicit reset request for local demo state."""

    confirm: bool = False


@router.get("/demo", response_class=HTMLResponse)
def demo_page() -> HTMLResponse:
    """Serve the professor-facing demo dashboard."""

    return HTMLResponse(_HTML)


@router.get("/demo/benchmark/events")
def benchmark_events(
    limit: int = Query(default=8, ge=1, le=30),
    delay_ms: int = Query(default=650, ge=0, le=5000),
) -> StreamingResponse:
    """Stream a live benchmark playback as server-sent events."""

    return StreamingResponse(
        _benchmark_event_stream(limit=limit, delay_ms=delay_ms),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/demo/manual")
def manual_prompt(
    request: DemoPromptRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Run deterministic extraction, staging, optional approval, and return memory state."""

    branch = _resolve_demo_branch(db, request.branch_name)
    claim_dicts = deterministic_extract_simple_claims(request.message)
    claims = [ExtractedClaim.model_validate(claim) for claim in claim_dicts]
    source = SourceCreate(
        source_type="manual",
        source_ref=f"demo-ui:{request.branch_name}",
        excerpt=request.message,
        trust_score=request.trust_score,
    )
    if not claims:
        db.commit()
        return {
            "branch": _branch_payload(branch),
            "claims": [],
            "staged": None,
            "commit": None,
            "versions": [],
            "timelines": [],
            "warnings": ["No explicit claim matched the local demo extractor."],
            "snapshot": _demo_snapshot(db),
        }

    staged = stage_belief_changes(
        db,
        claims=claims,
        branch_id=branch.id,
        source=source,
        proposed_commit_message="Demo prompt memory update",
        created_by="demo-ui",
        model_name="deterministic-demo",
    )
    warnings = list(staged.warnings_json)
    result = None
    if request.auto_approve:
        result = approve_staged_commit(
            db,
            staged_commit_id=staged.id,
            reviewer="demo-ui",
            notes="Approved from professor demo panel.",
            commit_message=staged.proposed_commit_message,
            model_name="deterministic-demo",
        )
        warnings.extend(result.warnings)
    db.commit()

    affected_timelines = _affected_timelines(db, claims)
    return {
        "branch": _branch_payload(branch),
        "claims": [claim.model_dump(mode="json") for claim in claims],
        "staged": _staged_payload(staged),
        "commit": _commit_payload(result.commit) if result else None,
        "versions": [_version_payload(db, version) for version in result.introduced_versions] if result else [],
        "timelines": affected_timelines,
        "warnings": sorted(set(warnings)),
        "snapshot": _demo_snapshot(db),
    }


@router.post("/demo/rollback")
def rollback_demo_commit(
    request: DemoRollbackRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Rollback a commit from the demo UI."""

    try:
        result = rollback_commit(
            db,
            commit_id=request.commit_id,
            message=request.message or f"Demo rollback of commit {request.commit_id}",
            created_by="demo-ui",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    db.commit()
    return {
        "commit": _commit_payload(result.commit),
        "retracted_versions": [_version_payload(db, version) for version in result.introduced_versions],
        "restored_versions": [_version_payload(db, version) for version in result.restored_versions],
        "warnings": result.warnings,
        "snapshot": _demo_snapshot(db),
    }


@router.post("/demo/reset")
def reset_demo_state(
    request: DemoResetRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Reset local demo database tables after explicit confirmation."""

    if not request.confirm:
        raise HTTPException(status_code=400, detail="Reset requires confirm=true")
    for model in (
        models.AuditEvent,
        models.StagedCommit,
        models.BeliefVersion,
        models.Belief,
        models.Commit,
        models.Branch,
        models.Source,
    ):
        db.execute(delete(model))
    ensure_main_branch(db)
    db.commit()
    return {"snapshot": _demo_snapshot(db)}


def _benchmark_event_stream(*, limit: int, delay_ms: int) -> Generator[str, None, None]:
    cases = default_benchmark()[:limit]
    systems = default_systems(include_ablations=False)
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    total_questions = sum(len(case.questions) for case in cases)
    processed_questions = 0

    try:
        for system in systems:
            system.reset()
        yield _sse(
            "run_started",
            {
                "case_count": len(cases),
                "question_count": total_questions,
                "systems": [system.name for system in systems],
            },
        )
        for case_index, case in enumerate(cases, start=1):
            yield _sse("case_started", _case_payload(case, case_index, len(cases)))
            _sleep(delay_ms)
            for event in case.events:
                for system in systems:
                    system.ingest_event(event)
                yield _sse(
                    "memory_event",
                    {
                        "case_id": case.case_id,
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "text": event.text,
                        "branch_name": event.branch_name,
                        "source_ref": event.source_ref,
                        "trust_score": event.trust_score,
                    },
                )
                _sleep(delay_ms)
            for question in case.questions:
                processed_questions += 1
                for system in systems:
                    answer = system.answer(question)
                    score = score_answer(question, answer)
                    scores[system.name][question.metric].append(score)
                    yield _sse(
                        "question_scored",
                        {
                            "case_id": case.case_id,
                            "question_id": question.question_id,
                            "metric": question.metric,
                            "prompt": question.prompt,
                            "system_name": system.name,
                            "score": score,
                            "answer": {
                                "object_value": answer.object_value,
                                "source_ref": answer.source_ref,
                                "historical_objects": answer.historical_objects,
                                "had_low_trust_warning": answer.had_low_trust_warning,
                                "conflict_resolved": answer.conflict_resolved,
                                "unresolved_conflict": answer.unresolved_conflict,
                            },
                            "progress": processed_questions / max(1, total_questions),
                            "summary": _score_summary(scores),
                        },
                    )
                    _sleep(max(0, delay_ms // 2))
                _sleep(delay_ms)
        yield _sse("run_complete", {"summary": _score_summary(scores)})
    finally:
        for system in systems:
            close = getattr(system, "close", None)
            if close:
                close()


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


def _sleep(delay_ms: int) -> None:
    if delay_ms:
        time.sleep(delay_ms / 1000)


def _case_payload(case: BenchmarkCase, case_index: int, total_cases: int) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "case_index": case_index,
        "total_cases": total_cases,
        "description": case.description,
        "event_count": len(case.events),
        "question_count": len(case.questions),
    }


def _score_summary(scores: dict[str, dict[str, list[float]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system_name, metric_map in sorted(scores.items()):
        for metric, values in sorted(metric_map.items()):
            rows.append(
                {
                    "system_name": system_name,
                    "metric": metric,
                    "score": round(sum(values) / len(values), 4) if values else 0.0,
                    "n": len(values),
                }
            )
    return rows


def _resolve_demo_branch(db: Session, name: str) -> models.Branch:
    clean_name = name.strip() or "main"
    if clean_name == "main":
        branch = ensure_main_branch(db)
        db.flush()
        return branch
    branch = crud.get_branch_by_name(db, clean_name)
    if branch is not None:
        return branch
    branch = create_branch(db, name=clean_name, description=f"Demo branch {clean_name}")
    db.flush()
    return branch


def _affected_timelines(db: Session, claims: list[ExtractedClaim]) -> list[dict[str, Any]]:
    timelines: list[dict[str, Any]] = []
    seen: set[int] = set()
    for claim in claims:
        belief = crud.get_belief_by_subject_predicate(
            db,
            subject=claim.subject,
            predicate=claim.predicate,
        )
        if belief is None or belief.id in seen:
            continue
        seen.add(belief.id)
        timelines.append(
            {
                "belief": {
                    "id": belief.id,
                    "subject": belief.subject,
                    "predicate": belief.predicate,
                    "canonical_key": belief.canonical_key,
                },
                "versions": [_version_payload(db, version) for version in crud.list_belief_versions(db, belief.id)],
            }
        )
    return timelines


def _demo_snapshot(db: Session) -> dict[str, Any]:
    branches = crud.list_branches(db)
    versions = crud.search_beliefs(db, query="", include_inactive=True, limit=60)
    audit = crud.list_audit_events(db, limit=12)
    return {
        "branches": [_branch_payload(branch) for branch in branches],
        "versions": [_version_payload(db, version) for version in versions],
        "audit": [
            {
                "id": event.id,
                "event_type": event.event_type,
                "entity_type": event.entity_type,
                "entity_id": event.entity_id,
                "entity_key": event.entity_key,
                "created_at": _iso(event.created_at),
            }
            for event in audit
        ],
    }


def _branch_payload(branch: models.Branch) -> dict[str, Any]:
    return {
        "id": branch.id,
        "name": branch.name,
        "status": branch.status,
        "parent_branch_id": branch.parent_branch_id,
        "created_at": _iso(branch.created_at),
    }


def _staged_payload(staged: models.StagedCommit) -> dict[str, Any]:
    return {
        "id": staged.id,
        "status": staged.status,
        "branch_id": staged.branch_id,
        "review_required": staged.review_required,
        "risk_reasons": staged.risk_reasons,
        "warnings": staged.warnings_json,
        "applied_commit_id": staged.applied_commit_id,
    }


def _commit_payload(commit: models.Commit) -> dict[str, Any]:
    return {
        "id": commit.id,
        "branch_id": commit.branch_id,
        "parent_commit_id": commit.parent_commit_id,
        "operation_type": commit.operation_type,
        "message": commit.message,
        "created_by": commit.created_by,
        "created_at": _iso(commit.created_at),
    }


def _version_payload(db: Session, version: models.BeliefVersion) -> dict[str, Any]:
    belief = crud.get_belief(db, version.belief_id)
    source = db.get(models.Source, version.source_id)
    branch = crud.get_branch(db, version.branch_id)
    return {
        "id": version.id,
        "belief_id": version.belief_id,
        "subject": belief.subject if belief else None,
        "predicate": belief.predicate if belief else None,
        "object_value": version.object_value,
        "status": version.status,
        "branch_name": branch.name if branch else str(version.branch_id),
        "commit_id": version.commit_id,
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
  <title>TruthGit Live Demo</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1f2933;
      --muted: #667085;
      --line: #d7dee8;
      --page: #f5f7f9;
      --panel: #ffffff;
      --green: #1b7f5a;
      --blue: #1a73e8;
      --amber: #a65f00;
      --red: #b42318;
      --purple: #6d3fc4;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--page);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 28px clamp(18px, 4vw, 48px);
      background: #10241c;
      color: #f8fbf9;
      display: grid;
      gap: 10px;
    }
    h1 { margin: 0; font-size: 34px; letter-spacing: 0; line-height: 1.05; }
    header p { margin: 0; color: #c6d7d0; max-width: 900px; line-height: 1.45; }
    main {
      padding: 22px clamp(14px, 3vw, 42px) 44px;
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(360px, .85fr);
      gap: 18px;
    }
    .panel, .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .panel { overflow: hidden; min-width: 0; }
    .panel h2 {
      margin: 0;
      padding: 14px 16px;
      font-size: 17px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }
    .section { padding: 14px 16px; }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    button, input, textarea, select {
      font: inherit;
      border: 1px solid #b8c2cc;
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
    }
    button {
      cursor: pointer;
      font-weight: 700;
      padding: 9px 12px;
      min-width: 104px;
    }
    button.primary { background: #1b7f5a; color: white; border-color: #1b7f5a; }
    button.warn { background: #fff7e6; border-color: #e7b35b; color: #6f4400; }
    input, select { padding: 8px 10px; }
    textarea {
      width: 100%;
      min-height: 96px;
      padding: 10px 12px;
      resize: vertical;
      line-height: 1.4;
    }
    label { color: var(--muted); font-size: 13px; display: grid; gap: 4px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(126px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .metric { padding: 12px; min-height: 76px; }
    .metric span { color: var(--muted); font-size: 12px; display: block; }
    .metric strong { font-size: 26px; display: block; margin-top: 6px; }
    .progress {
      height: 12px;
      background: #e8edf2;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 12px;
    }
    .progress > div { height: 100%; width: 0%; background: var(--blue); transition: width .18s ease; }
    .feed, .timeline, .score-grid {
      max-height: 390px;
      overflow: auto;
      border-top: 1px solid var(--line);
    }
    .event {
      padding: 11px 14px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 4px;
    }
    .event b { font-size: 13px; }
    .event small { color: var(--muted); line-height: 1.35; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 9px 10px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }
    th { color: var(--muted); background: #fbfcfd; font-weight: 700; }
    .pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 8px;
      background: #edf2f7;
      color: #344054;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .active { color: var(--green); background: #dff3e8; }
    .hypothetical { color: var(--purple); background: #eee5fb; }
    .superseded { color: var(--amber); background: #fff1d6; }
    .retracted { color: var(--red); background: #fde1df; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .chip { min-width: auto; font-size: 12px; padding: 7px 9px; background: #edf2f7; }
    .result-box {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .stage-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 86px;
      padding: 10px;
      background: #fff;
    }
    .stage-card b { display: block; margin-bottom: 6px; }
    .stage-card small { color: var(--muted); line-height: 1.35; }
    @media (max-width: 1040px) {
      main { grid-template-columns: 1fr; }
      .result-box { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 640px) {
      .grid2, .result-box { grid-template-columns: 1fr; }
      h1 { font-size: 28px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>TruthGit Live Demo</h1>
    <p>Benchmark playback and manual belief-memory prompts with staged writes, commits, supersession, branches, rollback, provenance, and audit state.</p>
  </header>
  <main>
    <section class="panel">
      <h2>Benchmark Playback</h2>
      <div class="section">
        <div class="controls">
          <label>Cases <input id="benchLimit" type="number" min="1" max="30" value="8"></label>
          <label>Pacing ms <input id="benchDelay" type="number" min="0" max="5000" value="650"></label>
          <button class="primary" id="startBench">Start</button>
          <button id="stopBench">Stop</button>
        </div>
        <div class="progress"><div id="benchProgress"></div></div>
        <div class="metrics" id="benchMetrics"></div>
      </div>
      <div class="score-grid">
        <table>
          <thead><tr><th>System</th><th>Metric</th><th>Score</th><th>N</th></tr></thead>
          <tbody id="scoreRows"><tr><td colspan="4">Waiting for run.</td></tr></tbody>
        </table>
      </div>
      <div class="feed" id="benchFeed"></div>
    </section>

    <section class="panel">
      <h2>Manual Prompt Demo</h2>
      <div class="section">
        <textarea id="manualText">Alice lives in Seoul.</textarea>
        <div class="chips">
          <button class="chip" data-example="Alice lives in Seoul." data-branch="main" data-trust="0.8">initial fact</button>
          <button class="chip" data-example="Alice moved to Busan in March 2026." data-branch="main" data-trust="0.86">supersede</button>
          <button class="chip" data-example="Alice lives in Atlantis." data-branch="main" data-trust="0.2">bad update</button>
          <button class="chip" data-example="During the conference week, Alice will stay in Tokyo." data-branch="trip-plan" data-trust="0.78">branch-only</button>
        </div>
        <div class="grid2" style="margin-top: 12px;">
          <label>Branch <input id="manualBranch" value="main"></label>
          <label>Trust score <input id="manualTrust" type="number" min="0" max="1" step="0.01" value="0.8"></label>
        </div>
        <div class="controls" style="margin-top: 12px;">
          <label><input id="autoApprove" type="checkbox" checked> approve after staging</label>
          <button class="primary" id="runManual">Run Prompt</button>
          <button class="warn" id="approveStaged">Approve Staged</button>
          <button class="warn" id="rollbackLast">Rollback Last Commit</button>
          <button id="resetDemo">Reset</button>
        </div>
        <div class="result-box">
          <div class="stage-card"><b>Extract</b><small id="extractState">No prompt run yet.</small></div>
          <div class="stage-card"><b>Stage</b><small id="stageState">No staged commit.</small></div>
          <div class="stage-card"><b>Commit</b><small id="commitState">No commit.</small></div>
          <div class="stage-card"><b>Audit</b><small id="auditState">No audit events loaded.</small></div>
        </div>
      </div>
      <div class="timeline">
        <table>
          <thead><tr><th>Version</th><th>Belief</th><th>Object</th><th>Status</th><th>Lineage</th></tr></thead>
          <tbody id="manualVersions"><tr><td colspan="5">No versions yet.</td></tr></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    let benchSource = null;
    let lastStagedId = null;
    let lastCommitId = null;

    const metrics = [
      ["current_truth_accuracy", "Current"],
      ["historical_truth_accuracy", "History"],
      ["provenance_accuracy", "Provenance"],
      ["rollback_recovery_rate", "Rollback"],
      ["branch_isolation_score", "Branch"],
      ["merge_conflict_resolution_score", "Merge"],
      ["low_trust_warning_rate", "Low-trust"]
    ];

    document.getElementById("startBench").addEventListener("click", startBenchmark);
    document.getElementById("stopBench").addEventListener("click", stopBenchmark);
    document.getElementById("runManual").addEventListener("click", runManualPrompt);
    document.getElementById("approveStaged").addEventListener("click", approveStaged);
    document.getElementById("rollbackLast").addEventListener("click", rollbackLast);
    document.getElementById("resetDemo").addEventListener("click", resetDemo);
    document.querySelectorAll("[data-example]").forEach(button => {
      button.addEventListener("click", () => {
        document.getElementById("manualText").value = button.dataset.example;
        document.getElementById("manualBranch").value = button.dataset.branch;
        document.getElementById("manualTrust").value = button.dataset.trust;
      });
    });

    function startBenchmark() {
      stopBenchmark();
      document.getElementById("benchFeed").innerHTML = "";
      document.getElementById("scoreRows").innerHTML = `<tr><td colspan="4">Starting.</td></tr>`;
      renderBenchMetrics([]);
      const limit = document.getElementById("benchLimit").value || "8";
      const delay = document.getElementById("benchDelay").value || "650";
      benchSource = new EventSource(`/demo/benchmark/events?limit=${encodeURIComponent(limit)}&delay_ms=${encodeURIComponent(delay)}`);
      benchSource.addEventListener("run_started", event => {
        const data = JSON.parse(event.data);
        addFeed("Run started", `${data.case_count} cases, ${data.question_count} questions, ${data.systems.join(", ")}`);
      });
      benchSource.addEventListener("case_started", event => {
        const data = JSON.parse(event.data);
        addFeed(`Case ${data.case_index}/${data.total_cases}`, `${data.case_id}: ${data.description}`);
      });
      benchSource.addEventListener("memory_event", event => {
        const data = JSON.parse(event.data);
        addFeed(`${data.event_type}: ${data.event_id}`, `${data.text} | branch ${data.branch_name} | trust ${Number(data.trust_score).toFixed(2)}`);
      });
      benchSource.addEventListener("question_scored", event => {
        const data = JSON.parse(event.data);
        document.getElementById("benchProgress").style.width = `${Math.round(data.progress * 100)}%`;
        renderScoreRows(data.summary);
        renderBenchMetrics(data.summary);
        if (data.system_name === "truthgit") {
          addFeed(`${data.metric}: ${data.score.toFixed(1)}`, data.prompt);
        }
      });
      benchSource.addEventListener("run_complete", event => {
        const data = JSON.parse(event.data);
        renderScoreRows(data.summary);
        renderBenchMetrics(data.summary);
        addFeed("Run complete", "Final scores rendered.");
        stopBenchmark();
      });
      benchSource.onerror = () => addFeed("Stream warning", "Connection ended or interrupted.");
    }

    function stopBenchmark() {
      if (benchSource) {
        benchSource.close();
        benchSource = null;
      }
    }

    async function runManualPrompt() {
      const payload = {
        message: document.getElementById("manualText").value,
        branch_name: document.getElementById("manualBranch").value,
        trust_score: Number(document.getElementById("manualTrust").value),
        auto_approve: document.getElementById("autoApprove").checked
      };
      const data = await postJson("/demo/manual", payload);
      lastStagedId = data.staged?.status === "pending" ? data.staged.id : null;
      lastCommitId = data.commit?.id || lastCommitId;
      renderManual(data);
    }

    async function approveStaged() {
      if (!lastStagedId) return;
      const approvedId = lastStagedId;
      const data = await postJson(`/staged/${lastStagedId}/approve`, {
        reviewer: "demo-ui",
        notes: "Approved during live demo.",
        commit_message: "Manual approval from demo UI"
      });
      lastCommitId = data.commit.id;
      lastStagedId = null;
      await refreshSnapshot(`Approved staged ${approvedId}`, `commit #${lastCommitId}`);
    }

    async function rollbackLast() {
      if (!lastCommitId) return;
      const data = await postJson("/demo/rollback", {commit_id: lastCommitId});
      lastCommitId = null;
      renderSnapshot(data.snapshot);
      document.getElementById("commitState").textContent = `rollback commit #${data.commit.id}`;
      document.getElementById("stageState").textContent = `${data.retracted_versions.length} version(s) retracted, ${data.restored_versions.length} restored`;
    }

    async function resetDemo() {
      const data = await postJson("/demo/reset", {confirm: true});
      lastStagedId = null;
      lastCommitId = null;
      renderSnapshot(data.snapshot);
      document.getElementById("extractState").textContent = "Reset complete.";
      document.getElementById("stageState").textContent = "No staged commit.";
      document.getElementById("commitState").textContent = "No commit.";
      document.getElementById("auditState").textContent = "Database reset.";
    }

    async function refreshSnapshot(title, text) {
      const response = await fetch("/viz/data");
      const data = await response.json();
      document.getElementById("commitState").textContent = text;
      document.getElementById("stageState").textContent = title;
      document.getElementById("auditState").textContent = `${data.audit_events.length} recent audit event(s)`;
      document.getElementById("manualVersions").innerHTML = data.belief_versions.map(versionRow).join("") || emptyVersionRow();
    }

    function renderManual(data) {
      document.getElementById("extractState").textContent = `${data.claims.length} claim(s): ${data.claims.map(c => `${c.subject} ${c.predicate} ${objectValue(c)}`).join("; ") || "none"}`;
      document.getElementById("stageState").textContent = data.staged ? `${data.staged.status} ${data.staged.review_required ? "review required" : "auto-safe"} ${data.warnings.join(" ")}` : "No staged commit.";
      document.getElementById("commitState").textContent = data.commit ? `commit #${data.commit.id} on branch ${data.branch.name}` : "Pending review.";
      renderSnapshot(data.snapshot);
    }

    function objectValue(claim) {
      return claim.object_value || claim.object || "";
    }

    function renderSnapshot(snapshot) {
      const versions = snapshot?.versions || [];
      const audit = snapshot?.audit || [];
      document.getElementById("auditState").textContent = `${audit.length} recent audit event(s)`;
      document.getElementById("manualVersions").innerHTML = versions.map(versionRow).join("") || emptyVersionRow();
    }

    function versionRow(row) {
      return `<tr>
        <td>#${row.id}</td>
        <td><b>${escapeHtml(row.subject || "")}</b><br><span>${escapeHtml(row.predicate || "")}</span></td>
        <td>${escapeHtml(row.object_value || "")}</td>
        <td><span class="pill ${escapeHtml(row.status || "")}">${escapeHtml(row.status || "")}</span>${row.contradiction_group ? "<br>conflict" : ""}</td>
        <td>${row.supersedes_version_id ? "supersedes #" + row.supersedes_version_id : "root"}<br>commit #${row.commit_id}<br>${escapeHtml(row.branch_name || "")}</td>
      </tr>`;
    }

    function emptyVersionRow() {
      return `<tr><td colspan="5">No belief versions yet.</td></tr>`;
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text);
      }
      return response.json();
    }

    function renderScoreRows(summary) {
      const rows = summary || [];
      document.getElementById("scoreRows").innerHTML = rows.map(row => `
        <tr><td>${escapeHtml(row.system_name)}</td><td>${escapeHtml(row.metric)}</td><td>${Number(row.score).toFixed(3)}</td><td>${row.n}</td></tr>
      `).join("") || `<tr><td colspan="4">No scores yet.</td></tr>`;
    }

    function renderBenchMetrics(summary) {
      const truthgit = new Map((summary || []).filter(row => row.system_name === "truthgit").map(row => [row.metric, row.score]));
      document.getElementById("benchMetrics").innerHTML = metrics.map(([key, label]) => `
        <div class="metric"><span>${escapeHtml(label)}</span><strong>${truthgit.has(key) ? Number(truthgit.get(key)).toFixed(2) : "0.00"}</strong></div>
      `).join("");
    }

    function addFeed(title, text) {
      const feed = document.getElementById("benchFeed");
      const item = document.createElement("div");
      item.className = "event";
      item.innerHTML = `<b>${escapeHtml(title)}</b><small>${escapeHtml(text)}</small>`;
      feed.prepend(item);
      while (feed.children.length > 80) feed.removeChild(feed.lastChild);
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, character => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[character]));
    }
    renderBenchMetrics([]);
  </script>
</body>
</html>
"""
