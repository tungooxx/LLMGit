"""Professor-facing live demo UI for manual TruthGit memory prompts."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app import crud, models
from app.commit_engine import create_branch, ensure_main_branch, rollback_commit
from app.config import get_settings
from app.db import get_db
from app.llm import LLMClient
from app.normalization import deterministic_extract_simple_claims
from app.schemas import ExtractedClaim, MemoryWritePlan, SourceCreate
from app.tools import approve_staged_commit, stage_belief_changes

router = APIRouter(tags=["demo"])


class DemoPromptRequest(BaseModel):
    """Manual prompt request for the demo UI."""

    message: str
    branch_name: str = "main"
    trust_score: float = Field(default=0.7, ge=0.0, le=1.0)
    auto_approve: bool = False
    extraction_mode: Literal["llm", "local"] = "llm"
    auto_metadata: bool = True


class DemoRollbackRequest(BaseModel):
    """Rollback request for the demo UI."""

    commit_id: int
    message: str | None = None


class DemoResetRequest(BaseModel):
    """Explicit reset request for local demo state."""

    confirm: bool = False


@router.get("/demo", response_class=HTMLResponse)
def demo_page() -> HTMLResponse:
    """Serve the professor-facing chat and graph dashboard."""

    return HTMLResponse(_HTML)


@router.post("/demo/manual")
def manual_prompt(
    request: DemoPromptRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Run deterministic extraction, staging, optional approval, and return memory state."""

    write_plan, extraction_info, extraction_warnings = _build_demo_write_plan(request)
    branch = _resolve_demo_branch(db, write_plan.branch_name)
    claims = write_plan.claims
    source = SourceCreate(
        source_type="user_message" if request.extraction_mode == "llm" else "manual",
        source_ref=f"demo-ui:{request.extraction_mode}:{branch.name}",
        excerpt=request.message,
        trust_score=write_plan.trust_score,
    )
    if not claims:
        answer = LLMClient(get_settings()).answer_from_memory(
            request.message,
            _demo_memory_context(db, branch),
        )
        db.commit()
        return {
            "branch": _branch_payload(branch),
            "claims": [],
            "staged": None,
            "commit": None,
            "versions": [],
            "timelines": [],
            "assistant_reply": answer,
            "warnings": sorted(set(extraction_warnings)),
            "extraction": extraction_info,
            "snapshot": _demo_snapshot(db),
        }

    staged = stage_belief_changes(
        db,
        claims=claims,
        branch_id=branch.id,
        source=source,
        proposed_commit_message="Demo prompt memory update",
        created_by="demo-ui",
        model_name=extraction_info["model_name"],
    )
    warnings = [*extraction_warnings, *list(staged.warnings_json)]
    result = None
    if request.auto_approve:
        result = approve_staged_commit(
            db,
            staged_commit_id=staged.id,
            reviewer="demo-ui",
            notes="Approved from professor demo panel.",
            commit_message=staged.proposed_commit_message,
            model_name=extraction_info["model_name"],
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
        "assistant_reply": _assistant_reply_with_outcome(write_plan.assistant_reply, result is not None, staged.id),
        "warnings": sorted(set(warnings)),
        "extraction": extraction_info,
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


def _build_demo_write_plan(request: DemoPromptRequest) -> tuple[MemoryWritePlan, dict[str, Any], list[str]]:
    """Return claims plus branch/trust metadata for the demo prompt."""

    settings = get_settings()
    warnings: list[str] = []
    model_name = "deterministic-demo"
    used_fallback = False

    if request.extraction_mode == "llm":
        llm = LLMClient(settings)
        used_fallback = llm.client is None
        if request.auto_metadata:
            plan = llm.plan_memory_write(
                request.message,
                fallback_branch_name=request.branch_name,
                fallback_trust_score=request.trust_score,
            )
        else:
            extracted = llm.extract_claims(request.message)
            plan = MemoryWritePlan(
                claims=extracted.claims,
                branch_name=request.branch_name,
                trust_score=request.trust_score,
                rationale="LLM extracted claims; branch and trust came from demo controls.",
            )
        model_name = settings.openai_model if not used_fallback else "deterministic-demo-fallback"
        if used_fallback:
            warnings.append("LLM mode requested, but OPENAI_API_KEY is not configured; used local fallback.")
    else:
        claims = [ExtractedClaim.model_validate(claim) for claim in deterministic_extract_simple_claims(request.message)]
        plan = MemoryWritePlan(
            claims=claims,
            branch_name=request.branch_name,
            trust_score=request.trust_score,
            rationale="Local deterministic demo extractor; branch and trust came from demo controls.",
        )

    if not request.auto_metadata:
        plan = plan.model_copy(update={"branch_name": request.branch_name, "trust_score": request.trust_score})
    plan = _sanitize_demo_write_plan(plan, fallback_branch_name=request.branch_name)
    info = {
        "mode": request.extraction_mode,
        "auto_metadata": request.auto_metadata,
        "model_name": model_name,
        "branch_name": plan.branch_name,
        "trust_score": plan.trust_score,
        "rationale": plan.rationale,
        "used_fallback": used_fallback,
    }
    return plan, info, warnings


def _sanitize_demo_write_plan(plan: MemoryWritePlan, *, fallback_branch_name: str) -> MemoryWritePlan:
    """Clamp model-suggested metadata before it can affect durable memory."""

    branch_name = _safe_branch_name(plan.branch_name or fallback_branch_name)
    trust_score = max(0.0, min(1.0, plan.trust_score))
    return plan.model_copy(update={"branch_name": branch_name, "trust_score": trust_score})


def _safe_branch_name(value: str) -> str:
    clean = value.strip().lower().replace("_", "-")
    clean = "".join(character for character in clean if character.isalnum() or character == "-")
    clean = "-".join(part for part in clean.split("-") if part)
    return clean[:40] or "main"


def _assistant_reply_with_outcome(reply: str, committed: bool, staged_id: str) -> str:
    suffix = (
        "The staged memory has been approved into a commit."
        if committed
        else f"I staged it for review as {staged_id}."
    )
    clean_reply = reply.strip() or "I prepared this as a reviewable TruthGit memory update."
    return f"{clean_reply} {suffix}"


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
    commits = crud.list_commits(db)[:40]
    versions = crud.search_beliefs(db, query="", include_inactive=True, limit=80)
    staged = list(db.scalars(select(models.StagedCommit).order_by(models.StagedCommit.created_at.desc()).limit(30)))
    audit = crud.list_audit_events(db, limit=16)
    return {
        "counts": {
            "branches": len(branches),
            "commits": len(commits),
            "versions": len(versions),
            "staged": sum(1 for item in staged if item.status == "pending"),
            "audit_events": len(audit),
        },
        "branches": [_branch_payload(branch) for branch in branches],
        "commits": [_commit_payload(commit) for commit in commits],
        "versions": [_version_payload(db, version) for version in versions],
        "staged_commits": [_staged_payload(item) for item in staged],
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


def _demo_memory_context(db: Session, branch: models.Branch) -> dict[str, Any]:
    """Build compact memory context for read-only demo chat answers."""

    beliefs = list(db.scalars(select(models.Belief).order_by(models.Belief.id)))
    current: list[dict[str, Any]] = []
    for belief in beliefs:
        versions = crud.get_current_versions(db, belief_id=belief.id, branch_id=branch.id)
        current.extend(_version_payload(db, version) for version in versions)
    timelines = crud.search_beliefs(db, query="", include_inactive=True, limit=80)
    return {
        "branch": _branch_payload(branch),
        "current_beliefs": current,
        "timelines": [_version_payload(db, version) for version in timelines],
        "commits": [_commit_payload(commit) for commit in crud.list_commits(db)[:30]],
        "pending_staged_commits": [
            _staged_payload(staged)
            for staged in db.scalars(
                select(models.StagedCommit)
                .where(models.StagedCommit.status == "pending")
                .order_by(models.StagedCommit.created_at.desc())
                .limit(20)
            )
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
        "source_ref": staged.source_ref,
        "source_trust_score": staged.source_trust_score,
        "created_at": _iso(staged.created_at),
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
        "branch_id": version.branch_id,
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
  <title>TruthGit Memory Chat</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1f2933;
      --muted: #667085;
      --line: #d7dee8;
      --page: #f4f6f8;
      --panel: #ffffff;
      --panel-soft: #fbfcfd;
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
      min-height: 100vh;
    }
    header {
      padding: 24px clamp(18px, 4vw, 48px);
      background: #10241c;
      color: #f8fbf9;
      display: grid;
      gap: 8px;
    }
    h1 { margin: 0; font-size: 34px; letter-spacing: 0; line-height: 1.05; }
    header p { margin: 0; color: #c6d7d0; max-width: 980px; line-height: 1.45; }
    main {
      padding: 18px clamp(14px, 3vw, 40px) 36px;
      display: grid;
      grid-template-columns: minmax(360px, 0.9fr) minmax(0, 1.1fr);
      gap: 18px;
      min-height: calc(100vh - 112px);
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      min-width: 0;
    }
    .panel h2 {
      margin: 0;
      padding: 14px 16px;
      font-size: 17px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-soft);
    }
    .chat-panel {
      display: grid;
      grid-template-rows: auto auto minmax(270px, 1fr) auto;
      height: calc(100vh - 156px);
      min-height: 640px;
      max-height: 780px;
    }
    .settings {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 10px;
    }
    .grid3 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
      align-items: end;
    }
    label { color: var(--muted); font-size: 13px; display: grid; gap: 4px; }
    label.check {
      min-height: 37px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border: 1px solid #b8c2cc;
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      white-space: nowrap;
    }
    input, textarea, button {
      font: inherit;
      border: 1px solid #b8c2cc;
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
    }
    input { padding: 8px 10px; min-width: 0; }
    input[type="checkbox"] {
      width: 16px;
      height: 16px;
      min-width: 16px;
      padding: 0;
      accent-color: var(--green);
    }
    button {
      cursor: pointer;
      font-weight: 700;
      padding: 9px 12px;
    }
    button.primary { background: var(--green); color: white; border-color: var(--green); }
    button.warn { background: #fff7e6; border-color: #e7b35b; color: #6f4400; }
    button.danger { background: #fde1df; border-color: #f1a29b; color: var(--red); }
    .quick {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .quick button {
      min-width: auto;
      font-size: 12px;
      padding: 7px 9px;
      background: #edf2f7;
      min-height: 34px;
    }
    .messages {
      padding: 16px;
      overflow: auto;
      display: grid;
      align-content: start;
      gap: 12px;
      background: #f9fbfc;
      min-height: 260px;
    }
    .msg {
      max-width: 88%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: #fff;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    .msg.user {
      justify-self: end;
      color: #fff;
      background: #244d3d;
      border-color: #244d3d;
    }
    .msg.system {
      justify-self: start;
    }
    .msg small {
      display: block;
      margin-top: 6px;
      color: var(--muted);
      white-space: normal;
    }
    .composer {
      border-top: 1px solid var(--line);
      padding: 12px 14px;
      display: grid;
      gap: 10px;
      background: var(--panel);
    }
    textarea {
      width: 100%;
      min-height: 76px;
      resize: vertical;
      line-height: 1.4;
      padding: 10px 12px;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    .graph-shell {
      display: grid;
      grid-template-rows: auto auto minmax(260px, 0.9fr) minmax(280px, 1fr);
      height: calc(100vh - 156px);
      min-height: 640px;
      max-height: 780px;
    }
    .stats {
      padding: 12px 14px;
      display: grid;
      grid-template-columns: repeat(5, minmax(84px, 1fr));
      gap: 10px;
      border-bottom: 1px solid var(--line);
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      min-height: 66px;
      background: #fff;
    }
    .stat span { display: block; color: var(--muted); font-size: 12px; }
    .stat strong { display: block; margin-top: 5px; font-size: 24px; }
    .graph-wrap {
      overflow: auto;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
      min-height: 0;
    }
    svg { display: block; min-width: 760px; }
    .tables {
      display: grid;
      grid-template-columns: 1fr;
      grid-template-rows: minmax(150px, 1fr) minmax(140px, 0.78fr);
      min-height: 0;
      overflow: hidden;
    }
    .table-wrap {
      overflow: auto;
      border-bottom: 1px solid var(--line);
      min-height: 0;
    }
    .table-wrap:last-child { border-bottom: 0; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td {
      padding: 9px 10px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }
    th {
      color: var(--muted);
      background: var(--panel-soft);
      font-weight: 700;
      position: sticky;
      top: 0;
      z-index: 1;
    }
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
    .pending { color: var(--amber); background: #fff1d6; }
    .applied { color: var(--green); background: #dff3e8; }
    .muted { color: var(--muted); }
    @media (max-width: 1120px) {
      main { grid-template-columns: 1fr; }
      .chat-panel, .graph-shell { height: 720px; max-height: none; }
    }
    @media (max-width: 720px) {
      .grid3, .tables, .stats { grid-template-columns: 1fr; }
      h1 { font-size: 28px; }
      .msg { max-width: 96%; }
    }
  </style>
</head>
<body>
  <header>
    <h1>TruthGit Memory Chat</h1>
    <p>Chat normally while TruthGit extracts durable memory updates, answers from versioned beliefs, and refreshes the commit graph from the live SQLite store.</p>
  </header>
  <main>
    <section class="panel chat-panel">
      <h2>Memory Chat</h2>
      <div class="settings">
        <div class="grid3">
          <label>Branch fallback <input id="manualBranch" value="main"></label>
          <label>Trust <input id="manualTrust" type="number" min="0" max="1" step="0.01" value="0.8"></label>
          <label class="check"><input id="useLlm" type="checkbox" checked> use LLM</label>
          <label class="check"><input id="autoMetadata" type="checkbox" checked> LLM branch/trust</label>
          <label class="check"><input id="autoApprove" type="checkbox" checked> approve after staging</label>
        </div>
        <div class="quick">
          <button data-example="Alice lives in Seoul." data-branch="main" data-trust="0.8">initial fact</button>
          <button data-example="Alice moved to Busan in March 2026." data-branch="main" data-trust="0.86">supersede</button>
          <button data-example="Where does Alice live now?" data-branch="main" data-trust="0.8">ask current</button>
          <button data-example="Why do you think Alice lives in Busan if earlier she lived in Seoul?" data-branch="main" data-trust="0.8">ask lineage</button>
          <button data-example="Alice lives in Atlantis." data-branch="main" data-trust="0.2">bad update</button>
          <button data-example="During the conference week, Alice will stay in Tokyo." data-branch="trip-plan" data-trust="0.78">branch-only</button>
        </div>
      </div>
      <div class="messages" id="messages">
        <div class="msg system">Send updates or ask questions. TruthGit will stage new beliefs as commits, answer from current branch memory, and keep old versions visible instead of overwriting them.</div>
      </div>
      <div class="composer">
        <textarea id="manualText" placeholder="Type a memory update, for example: Alice lives in Seoul.">Alice lives in Seoul.</textarea>
        <div class="controls">
          <button class="primary" id="sendPrompt">Send</button>
          <button class="warn" id="approveStaged">Approve Staged</button>
          <button class="warn" id="rollbackLast">Rollback Last Commit</button>
          <button class="danger" id="resetDemo">Reset</button>
        </div>
      </div>
    </section>

    <section class="panel graph-shell">
      <h2>Git Graph</h2>
      <div class="stats" id="stats"></div>
      <div class="graph-wrap">
        <svg id="gitGraph" width="920" height="320" role="img" aria-label="TruthGit commit graph"></svg>
      </div>
      <div class="tables">
        <div class="table-wrap">
          <table>
            <thead><tr><th>Version</th><th>Belief</th><th>Object</th><th>Status</th><th>Lineage</th></tr></thead>
            <tbody id="versionRows"><tr><td colspan="5" class="muted">No belief versions yet.</td></tr></tbody>
          </table>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Staged / Audit</th><th>Status</th></tr></thead>
            <tbody id="auditRows"><tr><td colspan="2" class="muted">No audit events yet.</td></tr></tbody>
          </table>
        </div>
      </div>
    </section>
  </main>
  <script>
    let lastStagedId = null;
    let lastCommitId = null;

    document.getElementById("sendPrompt").addEventListener("click", runManualPrompt);
    document.getElementById("approveStaged").addEventListener("click", approveStaged);
    document.getElementById("rollbackLast").addEventListener("click", rollbackLast);
    document.getElementById("resetDemo").addEventListener("click", resetDemo);
    document.getElementById("manualText").addEventListener("keydown", event => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) runManualPrompt();
    });
    document.querySelectorAll("[data-example]").forEach(button => {
      button.addEventListener("click", () => {
        document.getElementById("manualText").value = button.dataset.example;
        document.getElementById("manualBranch").value = button.dataset.branch;
        document.getElementById("manualTrust").value = button.dataset.trust;
        document.getElementById("manualText").focus();
      });
    });

    async function runManualPrompt() {
      const message = document.getElementById("manualText").value.trim();
      if (!message) return;
      appendMessage("user", message);
      const payload = {
        message,
        branch_name: document.getElementById("manualBranch").value,
        trust_score: Number(document.getElementById("manualTrust").value),
        auto_approve: document.getElementById("autoApprove").checked,
        extraction_mode: document.getElementById("useLlm").checked ? "llm" : "local",
        auto_metadata: document.getElementById("autoMetadata").checked
      };
      try {
        const data = await postJson("/demo/manual", payload);
        lastStagedId = data.staged?.status === "pending" ? data.staged.id : null;
        lastCommitId = data.commit?.id || lastCommitId;
        renderSnapshot(data.snapshot);
        appendMessage("system", data.assistant_reply || summarizeManual(data), detailLine(data));
      } catch (error) {
        appendMessage("system", "Request failed.", String(error));
      }
    }

    async function approveStaged() {
      if (!lastStagedId) {
        appendMessage("system", "No pending staged commit to approve.");
        return;
      }
      const approvedId = lastStagedId;
      try {
        const data = await postJson(`/staged/${lastStagedId}/approve`, {
          reviewer: "demo-ui",
          notes: "Approved during live demo.",
          commit_message: "Manual approval from demo UI"
        });
        lastCommitId = data.commit.id;
        lastStagedId = null;
        const snapshot = await fetchSnapshot();
        renderSnapshot(snapshot);
        appendMessage("system", `Approved staged commit ${approvedId}.`, `Created commit #${lastCommitId}.`);
      } catch (error) {
        appendMessage("system", "Approval failed.", String(error));
      }
    }

    async function rollbackLast() {
      if (!lastCommitId) {
        appendMessage("system", "No last commit is selected for rollback.");
        return;
      }
      try {
        const data = await postJson("/demo/rollback", {commit_id: lastCommitId});
        lastCommitId = data.commit.id;
        renderSnapshot(data.snapshot);
        appendMessage(
          "system",
          `Rollback commit #${data.commit.id} created.`,
          `${data.retracted_versions.length} version(s) retracted, ${data.restored_versions.length} restored.`
        );
      } catch (error) {
        appendMessage("system", "Rollback failed.", String(error));
      }
    }

    async function resetDemo() {
      try {
        const data = await postJson("/demo/reset", {confirm: true});
        lastStagedId = null;
        lastCommitId = null;
        renderSnapshot(data.snapshot);
        document.getElementById("messages").innerHTML = "";
        appendMessage("system", "Demo memory reset. The main branch has been recreated.");
      } catch (error) {
        appendMessage("system", "Reset failed.", String(error));
      }
    }

    async function fetchSnapshot() {
      const response = await fetch("/viz/data", {headers: {"Accept": "application/json"}});
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      const staged = data.staged_commits || [];
      return {
        counts: {
          ...(data.counts || {}),
          staged: staged.filter(row => row.status === "pending").length
        },
        branches: data.branches,
        commits: data.commits,
        versions: data.belief_versions,
        staged_commits: staged,
        audit: data.audit_events
      };
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function summarizeManual(data) {
      const claims = data.claims.map(claim => `${claim.subject} ${claim.predicate} ${claim.object_value || claim.object || ""}`.trim());
      if (!claims.length) return "No atomic claim was extracted from that prompt.";
      const prefix = data.extraction?.mode === "llm" ? "LLM planned" : "Local extractor staged";
      if (data.commit) return `${prefix} and committed ${claims.length} claim(s) on ${data.branch.name}.`;
      return `${prefix} ${claims.length} claim(s) for review on ${data.branch.name}.`;
    }

    function detailLine(data) {
      const parts = [];
      if (data.extraction) {
        parts.push(`${data.extraction.model_name}`);
        parts.push(`trust ${Number(data.extraction.trust_score).toFixed(2)}`);
        if (data.extraction.rationale) parts.push(data.extraction.rationale);
      }
      if (data.staged) parts.push(`staged ${data.staged.id}`);
      if (data.commit) parts.push(`commit #${data.commit.id}`);
      if (data.warnings?.length) parts.push(data.warnings.join(" "));
      return parts.join(" | ");
    }

    function appendMessage(role, text, detail = "") {
      const messages = document.getElementById("messages");
      const item = document.createElement("div");
      item.className = `msg ${role}`;
      item.innerHTML = `${escapeHtml(text)}${detail ? `<small>${escapeHtml(detail)}</small>` : ""}`;
      messages.appendChild(item);
      messages.scrollTop = messages.scrollHeight;
    }

    function renderSnapshot(snapshot) {
      renderStats(snapshot?.counts || {});
      renderGraph(snapshot || {});
      renderVersions(snapshot?.versions || []);
      renderAudit(snapshot?.staged_commits || [], snapshot?.audit || []);
    }

    function renderStats(counts) {
      const entries = [
        ["branches", "Branches"],
        ["commits", "Commits"],
        ["versions", "Versions"],
        ["staged", "Pending"],
        ["audit_events", "Audit"]
      ];
      document.getElementById("stats").innerHTML = entries.map(([key, label]) => `
        <div class="stat"><span>${label}</span><strong>${Number(counts[key] || 0)}</strong></div>
      `).join("");
    }

    function renderGraph(snapshot) {
      const svg = document.getElementById("gitGraph");
      const commits = [...(snapshot.commits || [])].reverse();
      const branches = snapshot.branches || [];
      const branchById = new Map(branches.map(branch => [branch.id, branch]));
      const lanes = new Map(branches.map((branch, index) => [branch.id, 100 + index * 160]));
      const width = Math.max(920, 260 + Math.max(0, branches.length - 1) * 160);
      const height = Math.max(300, 86 + commits.length * 76);
      svg.setAttribute("width", width);
      svg.setAttribute("height", height);
      if (!commits.length) {
        svg.innerHTML = `<text x="40" y="72" fill="#667085" font-size="15">No commits yet. Send a prompt to create the first memory commit.</text>`;
        return;
      }
      const yByCommit = new Map();
      commits.forEach((commit, index) => yByCommit.set(commit.id, 56 + index * 76));
      const lines = [];
      const nodes = [];
      for (const branch of branches) {
        const x = lanes.get(branch.id) || 100;
        lines.push(`<text x="${x - 34}" y="24" fill="#667085" font-size="12">${escapeSvg(branch.name)}</text>`);
      }
      for (const commit of commits) {
        const x = lanes.get(commit.branch_id) || 100;
        const y = yByCommit.get(commit.id);
        if (commit.parent_commit_id && yByCommit.has(commit.parent_commit_id)) {
          const parent = commits.find(item => item.id === commit.parent_commit_id);
          const px = parent ? (lanes.get(parent.branch_id) || 100) : x;
          const py = yByCommit.get(commit.parent_commit_id);
          lines.push(`<line x1="${px}" y1="${py}" x2="${x}" y2="${y}" stroke="#98a2b3" stroke-width="2" />`);
        }
        const color = commit.operation_type === "rollback" ? "#b42318" : commit.operation_type === "merge" ? "#6d3fc4" : "#1b7f5a";
        const branch = branchById.get(commit.branch_id);
        nodes.push(`
          <circle cx="${x}" cy="${y}" r="13" fill="${color}" stroke="#ffffff" stroke-width="3"></circle>
          <text x="${x + 24}" y="${y - 7}" fill="#1f2933" font-size="13" font-weight="700">#${commit.id} ${escapeSvg(commit.operation_type)}</text>
          <text x="${x + 24}" y="${y + 10}" fill="#667085" font-size="12">${escapeSvg(branch?.name || String(commit.branch_id))} ${escapeSvg(shorten(commit.message || "", 48))}</text>
        `);
      }
      svg.innerHTML = lines.join("") + nodes.join("");
    }

    function renderVersions(rows) {
      document.getElementById("versionRows").innerHTML = rows.map(row => `
        <tr>
          <td>#${row.id}</td>
          <td><b>${escapeHtml(row.subject || "")}</b><br><span class="muted">${escapeHtml(row.predicate || "")}</span></td>
          <td>${escapeHtml(row.object_value || "")}</td>
          <td><span class="pill ${escapeHtml(row.status || "")}">${escapeHtml(row.status || "")}</span>${row.contradiction_group ? "<br><span class='muted'>conflict</span>" : ""}</td>
          <td>${row.supersedes_version_id ? "supersedes #" + row.supersedes_version_id : "root"}<br>commit #${row.commit_id}<br>${escapeHtml(row.branch_name || "")}</td>
        </tr>
      `).join("") || `<tr><td colspan="5" class="muted">No belief versions yet.</td></tr>`;
    }

    function renderAudit(staged, audit) {
      const stagedRows = staged.map(row => `
        <tr>
          <td><b>staged</b><br><span class="muted">${escapeHtml(row.id)}</span></td>
          <td><span class="pill ${escapeHtml(row.status)}">${escapeHtml(row.status)}</span><br>${stagedDetail(row)}</td>
        </tr>
      `);
      const auditRows = audit.slice(0, 12).map(row => `
        <tr>
          <td><b>${escapeHtml(row.event_type)}</b><br><span class="muted">${escapeHtml(row.entity_key || row.entity_id || "")}</span></td>
          <td>${escapeHtml(row.entity_type)}<br><span class="muted">${escapeHtml(row.created_at || "")}</span></td>
        </tr>
      `);
      document.getElementById("auditRows").innerHTML = stagedRows.concat(auditRows).join("") || `<tr><td colspan="2" class="muted">No staged writes or audit events yet.</td></tr>`;
    }

    function stagedDetail(row) {
      if (row.status === "pending") return row.review_required ? "review required" : "review clear";
      if (row.applied_commit_id) return `commit #${row.applied_commit_id}`;
      return "reviewed";
    }

    function shorten(value, maxLength) {
      return value.length > maxLength ? `${value.slice(0, maxLength - 1)}...` : value;
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, character => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[character]));
    }

    function escapeSvg(value) {
      return escapeHtml(value);
    }

    renderStats({});
    fetchSnapshot().then(renderSnapshot).catch(() => renderSnapshot({}));
  </script>
</body>
</html>
"""
