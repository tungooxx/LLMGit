"""FastAPI entrypoint for TruthGit."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.commit_engine import ensure_main_branch
from app.db import SessionLocal, init_db
from app.routes import beliefs, branches, chat, commits

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="TruthGit",
    version="0.1.0",
    description="Version-Controlled Belief Memory for LLM Agents",
)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize local database objects and the default branch."""

    init_db()
    db = SessionLocal()
    try:
        ensure_main_branch(db)
        db.commit()
    finally:
        db.close()


@app.get("/")
def root() -> dict[str, str]:
    """Health/info endpoint."""

    return {"name": "TruthGit", "status": "ok"}


app.include_router(chat.router)
app.include_router(beliefs.router)
app.include_router(branches.router)
app.include_router(commits.router)
