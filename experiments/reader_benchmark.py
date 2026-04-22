"""Model-in-the-loop benchmark runner for changing-world memory contexts.

The frozen structural benchmark scores each memory adapter directly. This module
adds the second, stricter table: every memory system ingests the same events,
exposes its retrieved memory context, and the same reader model converts that
context into a structured answer that is scored by the existing metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from app.llm import openai_strict_json_schema
from experiments.baselines import MemorySystem, SystemAnswer, default_systems
from experiments.benchmark import BenchmarkCase, BenchmarkQuestion, default_benchmark
from experiments.final_config import BACKBONE, BENCHMARK_LOGIC_COMMIT_SHA, BENCHMARK_VERSION, PROMPT_TEMPLATE_PATH
from experiments.metrics import aggregate_scores, score_questions, scores_to_dicts


class ReaderStructuredAnswer(BaseModel):
    """Structured answer emitted by a shared reader model."""

    answer_text: str
    object_value: str | None = None
    historical_objects: list[str] = Field(default_factory=list)
    source_ref: str | None = None
    had_low_trust_warning: bool = False
    conflict_resolved: bool = False
    unresolved_conflict: bool = False
    rationale: str | None = None

    model_config = ConfigDict(extra="forbid")


class BenchmarkReader(Protocol):
    """Shared reader over system-provided memory context."""

    name: str
    model: str

    def answer(
        self,
        *,
        system_name: str,
        question: BenchmarkQuestion,
        memory_context: dict[str, object],
    ) -> SystemAnswer:
        """Return a structured benchmark answer."""


class OpenAIContextReader:
    """OpenAI reader that answers from memory context using structured output."""

    name = "openai_context_reader"

    def __init__(
        self,
        *,
        model: str = BACKBONE,
        max_retries: int = 5,
        sleep_seconds: float = 0.0,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for model-in-the-loop reader runs")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds

    def answer(
        self,
        *,
        system_name: str,
        question: BenchmarkQuestion,
        memory_context: dict[str, object],
    ) -> SystemAnswer:
        payload = {
            "question": question_payload_for_reader(question),
            "memory_context": memory_context,
            "output_contract": {
                "object_value": "current or time-slice answer value, or null if unsupported",
                "historical_objects": "ordered timeline values only when the question asks for history",
                "source_ref": "exact source_ref only when the question asks for provenance",
                "had_low_trust_warning": "true only when the context shows a low-trust warning relevant to the question",
                "unresolved_conflict": "true when the context shows unresolved/manual-review conflict state",
            },
        }
        response = _with_retries(
            lambda: self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": _reader_system_prompt(),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "ReaderStructuredAnswer",
                        "schema": openai_strict_json_schema(ReaderStructuredAnswer),
                        "strict": True,
                    }
                },
            ),
            max_retries=self.max_retries,
        )
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        parsed = ReaderStructuredAnswer.model_validate_json(_response_text(response))
        return _system_answer_from_reader(question, system_name, parsed)


class HeuristicContextReader:
    """Deterministic reader for tests and no-network smoke checks.

    This is not the model-in-the-loop paper setting. It exists so the runner can
    be tested without spending API calls.
    """

    name = "heuristic_context_reader"
    model = "deterministic-heuristic"

    def answer(
        self,
        *,
        system_name: str,
        question: BenchmarkQuestion,
        memory_context: dict[str, object],
    ) -> SystemAnswer:
        rows = _context_rows(memory_context)
        current_rows = [row for row in rows if str(row.get("status", "active")) in {"active", "hypothetical"}]
        chosen = _choose_reader_row(current_rows or rows, question)
        source_ref = str(chosen["source_ref"]) if chosen and chosen.get("source_ref") is not None else None
        historical = [
            str(row["object_value"])
            for row in rows
            if row.get("object_value") is not None
            and str(row.get("status", "active")) != "retracted"
        ]
        warnings = memory_context.get("warnings_by_event")
        warning_text = json.dumps(warnings, default=str) if warnings is not None else ""
        unresolved = any(row.get("contradiction_group") for row in current_rows)
        parsed = ReaderStructuredAnswer(
            answer_text=str(chosen["object_value"]) if chosen and chosen.get("object_value") is not None else "unknown",
            object_value=str(chosen["object_value"]) if chosen and chosen.get("object_value") is not None else None,
            historical_objects=historical,
            source_ref=source_ref,
            had_low_trust_warning="Low-trust" in warning_text or "low_trust" in warning_text,
            conflict_resolved=chosen is not None,
            unresolved_conflict=unresolved,
            rationale="Deterministic test reader over exposed context.",
        )
        return _system_answer_from_reader(question, system_name, parsed)


def run_reader_benchmark(
    cases: list[BenchmarkCase],
    *,
    include_ablations: bool = False,
    reader: BenchmarkReader | None = None,
    reader_model: str = BACKBONE,
    context_items: int = 30,
) -> dict[str, object]:
    """Run the model-in-the-loop benchmark over retrieved memory contexts."""

    shared_reader = reader or OpenAIContextReader(model=reader_model)
    all_questions = [question for case in cases for question in case.questions]
    all_scores = []
    predictions = []
    context_records = []

    for system in default_systems(include_ablations=include_ablations):
        system.reset()
        answers: list[SystemAnswer] = []
        try:
            for case in cases:
                for event in case.events:
                    system.ingest_event(event)
                for question in case.questions:
                    memory_context = system.memory_context(question, max_items=context_items)
                    answer = shared_reader.answer(
                        system_name=system.name,
                        question=question,
                        memory_context=memory_context,
                    )
                    answers.append(answer)
                    prediction = dict(answer.__dict__)
                    prediction["reader_name"] = shared_reader.name
                    prediction["reader_model"] = shared_reader.model
                    predictions.append(prediction)
                    context_records.append(
                        {
                            "system_name": system.name,
                            "question_id": question.question_id,
                            "reader_name": shared_reader.name,
                            "context": memory_context,
                        }
                    )
            all_scores.extend(score_questions(system_name=system.name, questions=all_questions, answers=answers))
        finally:
            close = getattr(system, "close", None)
            if close:
                close()

    return {
        "metadata": {
            "benchmark_version": BENCHMARK_VERSION,
            "evaluation_mode": "model_in_loop_reader",
            "reader_name": shared_reader.name,
            "reader_model": shared_reader.model,
            "prompt_template_path": PROMPT_TEMPLATE_PATH.as_posix(),
            "prompt_template_sha256": _sha256_file(PROMPT_TEMPLATE_PATH),
            "benchmark_logic_commit_sha": BENCHMARK_LOGIC_COMMIT_SHA,
            "include_ablations": include_ablations,
            "case_count": len(cases),
            "question_count": len(all_questions),
            "context_items": context_items,
        },
        "benchmark": [case.to_json() for case in cases],
        "predictions": predictions,
        "contexts": context_records,
        "question_scores": scores_to_dicts(all_scores),
        "metric_summary": aggregate_scores(all_scores),
    }


def export_reader_results(results: dict[str, object], output_dir: Path) -> None:
    """Write model-in-the-loop benchmark outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reader_benchmark_results.json").write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )
    _write_csv(output_dir / "reader_metric_summary.csv", results["metric_summary"])
    _write_csv(output_dir / "reader_question_scores.csv", results["question_scores"])
    _write_csv(output_dir / "reader_predictions.csv", results["predictions"])


def question_payload_for_reader(question: BenchmarkQuestion) -> dict[str, object]:
    """Return non-answer question fields for the shared reader prompt."""

    return {
        "question_id": question.question_id,
        "metric": question.metric,
        "prompt": question.prompt,
        "subject": question.subject,
        "predicate": question.predicate,
        "branch_name": question.branch_name,
        "as_of": question.as_of.isoformat() if question.as_of else None,
    }


def _reader_system_prompt() -> str:
    return (
        "You are the same benchmark reader for every memory system. "
        "Use only the provided memory_context, not outside knowledge. "
        "Return the structured answer fields needed by the scoring code. "
        "Do not infer facts that are absent from the context. Preserve exact object values, "
        "ordered timelines, exact source_ref strings, warning state, and unresolved conflict state."
    )


def _system_answer_from_reader(
    question: BenchmarkQuestion,
    system_name: str,
    parsed: ReaderStructuredAnswer,
) -> SystemAnswer:
    return SystemAnswer(
        question_id=question.question_id,
        system_name=system_name,
        answer_text=parsed.answer_text,
        object_value=parsed.object_value,
        historical_objects=parsed.historical_objects,
        source_ref=parsed.source_ref,
        had_low_trust_warning=parsed.had_low_trust_warning,
        conflict_resolved=parsed.conflict_resolved,
        unresolved_conflict=parsed.unresolved_conflict,
        branch_name=question.branch_name,
    )


def _context_rows(memory_context: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for key in ("current_versions", "belief_timeline", "facts"):
        values = memory_context.get(key)
        if isinstance(values, list):
            for row in values:
                if not isinstance(row, dict):
                    continue
                dedupe_key = str(row.get("belief_version_id") or row.get("event_id") or row)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                rows.append(row)
    return rows


def _choose_reader_row(
    rows: list[dict[str, object]],
    question: BenchmarkQuestion,
) -> dict[str, object] | None:
    eligible = [
        row
        for row in rows
        if str(row.get("subject")) == question.subject
        and str(row.get("predicate")) == question.predicate
    ]
    if question.branch_name != "main":
        branch_rows = [row for row in eligible if str(row.get("branch_name")) == question.branch_name]
        if branch_rows:
            eligible = branch_rows
    if not eligible:
        eligible = rows
    if not eligible:
        return None
    return eligible[-1]


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


def _with_retries(fn: Callable[[], Any], *, max_retries: int) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except OpenAIError as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"OpenAI request failed after {max_retries} attempts") from last_error


def _write_csv(path: Path, rows_obj: object) -> None:
    rows = list(rows_obj) if isinstance(rows_obj, list) else []
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--reader-model", default=BACKBONE)
    parser.add_argument("--reader", choices=["openai", "heuristic"], default="openai")
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--context-items", type=int, default=30)
    args = parser.parse_args()

    reader: BenchmarkReader
    if args.reader == "heuristic":
        reader = HeuristicContextReader()
    else:
        reader = OpenAIContextReader(model=args.reader_model)
    results = run_reader_benchmark(
        default_benchmark(),
        include_ablations=args.include_ablations,
        reader=reader,
        reader_model=args.reader_model,
        context_items=args.context_items,
    )
    export_reader_results(results, Path(args.output_dir))
    print(f"Wrote model-in-the-loop benchmark outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
