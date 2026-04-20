"""LongMemEval public benchmark adapter.

This module does not change the frozen TruthGit synthetic benchmark. It adds a
real public-benchmark harness for inspecting official LongMemEval JSON files,
building non-leaking prompts, generating hypothesis JSONL, evaluating with the
official GPT-4o-style judge prompt, and summarizing scores.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlretrieve

from openai import OpenAI, OpenAIError


OFFICIAL_DATASET_REPO = "xiaowu0162/longmemeval-cleaned"
OFFICIAL_CODE_REPO = "https://github.com/xiaowu0162/LongMemEval"
OFFICIAL_ARXIV = "https://arxiv.org/abs/2410.10813"
DATA_URLS = {
    "oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    "s_cleaned": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    "m_cleaned": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json",
}
QUESTION_TYPES = [
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]


@dataclass(frozen=True)
class LongMemEvalRecord:
    """Subset of the official LongMemEval schema needed by this adapter."""

    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str | None
    haystack_session_ids: list[str]
    haystack_dates: list[str]
    haystack_sessions: list[list[dict[str, Any]]]
    answer_session_ids: list[str]

    @property
    def is_abstention(self) -> bool:
        """Return whether the official id marks this as abstention."""

        return self.question_id.endswith("_abs")


def load_records(path: Path, *, limit: int | None = None) -> list[LongMemEvalRecord]:
    """Load official LongMemEval JSON records."""

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        raw_records = payload.get("data") or payload.get("records") or payload.get("examples")
    else:
        raw_records = payload
    if not isinstance(raw_records, list):
        raise ValueError(f"Expected a list of LongMemEval records in {path}")

    selected = raw_records[:limit] if limit is not None else raw_records
    return [_record_from_mapping(item) for item in selected]


def select_records(
    records: list[LongMemEvalRecord],
    *,
    limit: int | None = None,
    sample_size: int | None = None,
    sample_seed: int = 0,
) -> list[LongMemEvalRecord]:
    """Select a deterministic prefix or random sample for benchmark runs."""

    if limit is not None and sample_size is not None:
        raise ValueError("Use either limit or sample_size, not both.")
    if sample_size is None:
        return records[:limit] if limit is not None else records
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative.")
    if sample_size >= len(records):
        return [*records]
    rng = random.Random(sample_seed)
    indexes = sorted(rng.sample(range(len(records)), sample_size))
    return [records[index] for index in indexes]


def inspect_records(records: list[LongMemEvalRecord]) -> dict[str, Any]:
    """Return dataset statistics for a LongMemEval split."""

    session_counts = [len(record.haystack_sessions) for record in records]
    turn_counts = [
        sum(len(session) for session in record.haystack_sessions)
        for record in records
    ]
    question_types = Counter(record.question_type for record in records)
    return {
        "benchmark": "LongMemEval",
        "official_code_repo": OFFICIAL_CODE_REPO,
        "official_dataset_repo": OFFICIAL_DATASET_REPO,
        "official_arxiv": OFFICIAL_ARXIV,
        "record_count": len(records),
        "question_type_counts": dict(sorted(question_types.items())),
        "abstention_count": sum(1 for record in records if record.is_abstention),
        "avg_haystack_sessions": _mean(session_counts),
        "max_haystack_sessions": max(session_counts, default=0),
        "avg_turns": _mean(turn_counts),
        "max_turns": max(turn_counts, default=0),
    }


def write_manifest(
    *,
    records: list[LongMemEvalRecord],
    data_path: Path,
    output_dir: Path,
    split_label: str,
) -> Path:
    """Write a JSON manifest describing a LongMemEval split."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{split_label}_manifest.json"
    manifest = inspect_records(records)
    manifest["split_label"] = split_label
    manifest["data_path"] = str(data_path)
    manifest["output_contract"] = "jsonl lines with question_id and hypothesis for official evaluation"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def write_prompt_jsonl(
    *,
    records: list[LongMemEvalRecord],
    output_jsonl: Path,
    max_sessions: int | None = None,
    history_format: str = "json",
    reader_mode: str = "con",
    include_evidence_markers: bool = False,
) -> Path:
    """Write non-leaking prompts for a LongMemEval answer-generation run."""

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            prompt_record = {
                "question_id": record.question_id,
                "question_type": record.question_type,
                "question": record.question,
                "question_date": record.question_date,
                "prompt": build_prompt(
                    record,
                    max_sessions=max_sessions,
                    history_format=history_format,
                    reader_mode=reader_mode,
                    include_evidence_markers=include_evidence_markers,
                ),
            }
            handle.write(json.dumps(prompt_record, ensure_ascii=False) + "\n")
    return output_jsonl


def build_prompt(
    record: LongMemEvalRecord,
    *,
    max_sessions: int | None = None,
    history_format: str = "json",
    reader_mode: str = "con",
    include_evidence_markers: bool = False,
) -> str:
    """Format one LongMemEval instance as a deterministic answer prompt.

    The default prompt intentionally does not expose `answer`,
    `answer_session_ids`, or `has_answer` evidence labels.
    """

    session_count = len(record.haystack_sessions)
    start = max(0, session_count - max_sessions) if max_sessions is not None else 0
    selected_sessions = record.haystack_sessions[start:]
    selected_dates = record.haystack_dates[start:]
    lines = [
        "You are answering a LongMemEval long-term memory question.",
        "Use only the timestamped chat history below. Do not use outside knowledge.",
        "Resolve changed facts by using the latest supported information before the question date, unless the question asks for older history.",
        "For temporal questions, reason from the session dates and the question date.",
        "If the history does not support an answer, say that the answer is unknown.",
        "Return a concise answer. Include brief reasoning only when it helps disambiguate updates or dates.",
        f"Question date: {record.question_date or 'unknown'}",
        f"Question type: {record.question_type}",
        f"Reader mode: {reader_mode}",
    ]
    if reader_mode == "con":
        lines.append(
            "Reasoning instruction: first identify the relevant memory evidence and any updates, then provide the final answer."
        )
    elif reader_mode == "direct":
        lines.append("Reasoning instruction: answer directly from the most relevant supported memory.")
    else:
        raise ValueError("reader_mode must be 'direct' or 'con'")
    lines.extend(["", "History:"])
    if history_format == "json":
        lines.append(
            json.dumps(
                _history_payload(
                    selected_sessions,
                    selected_dates,
                    start=start,
                    include_evidence_markers=include_evidence_markers,
                ),
                ensure_ascii=False,
            )
        )
    elif history_format == "nl":
        for idx, session in enumerate(selected_sessions):
            session_id = start + idx
            date = selected_dates[idx] if idx < len(selected_dates) else "unknown"
            lines.append(f"[Session {session_id} | {date}]")
            lines.extend(_session_lines(session, include_evidence_markers=include_evidence_markers))
            lines.append("")
    else:
        raise ValueError("history_format must be 'json' or 'nl'")
    lines.extend(["Question:", record.question, "Answer:"])
    return "\n".join(lines)


@dataclass(frozen=True)
class GenerationStats:
    """Summary of a hypothesis generation run."""

    output_jsonl: str
    requested_count: int
    generated_count: int
    skipped_existing_count: int
    model: str


@dataclass(frozen=True)
class EvaluationStats:
    """Summary of a LongMemEval QA evaluation run."""

    eval_log: str
    evaluated_count: int
    overall_accuracy: float
    task_averaged_accuracy: float
    abstention_accuracy: float | None
    by_task: dict[str, dict[str, float | int]]


AnswerFn = Callable[[str, LongMemEvalRecord], str]
JudgeFn = Callable[[str, LongMemEvalRecord, str], bool]


class OpenAIAnswerGenerator:
    """OpenAI-backed LongMemEval hypothesis generator."""

    def __init__(
        self,
        *,
        model: str,
        max_output_tokens: int = 256,
        max_retries: int = 5,
        sleep_seconds: float = 0.0,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LongMemEval hypothesis generation")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds

    def __call__(self, prompt: str, record: LongMemEvalRecord) -> str:
        del record
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful long-term memory QA system. "
                    "Answer only from provided history, resolve updates by date, and abstain when unsupported."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        response = _with_retries(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=self.max_output_tokens,
            ),
            max_retries=self.max_retries,
        )
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return (response.choices[0].message.content or "").strip()


class OpenAIJudge:
    """OpenAI-backed evaluator using the official LongMemEval QA judge prompt."""

    def __init__(self, *, model: str = "gpt-4o", max_retries: int = 5) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LongMemEval evaluation")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def __call__(self, prompt: str, record: LongMemEvalRecord, hypothesis: str) -> bool:
        del record, hypothesis
        response = _with_retries(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            ),
            max_retries=self.max_retries,
        )
        text = (response.choices[0].message.content or "").strip().lower()
        return "yes" in text


def download_dataset(*, split: str, output_path: Path, force: bool = False) -> Path:
    """Download one official cleaned LongMemEval JSON file."""

    if split not in DATA_URLS:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(DATA_URLS)}")
    if output_path.exists() and not force:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(DATA_URLS[split], output_path)
    return output_path


def generate_hypotheses(
    *,
    records: list[LongMemEvalRecord],
    output_jsonl: Path,
    model: str,
    answer_fn: AnswerFn | None = None,
    max_sessions: int | None = None,
    history_format: str = "json",
    reader_mode: str = "con",
    resume: bool = True,
    max_output_tokens: int = 256,
    sleep_seconds: float = 0.0,
) -> GenerationStats:
    """Generate official LongMemEval hypothesis JSONL."""

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = _existing_hypothesis_ids(output_jsonl) if resume else set()
    generator = answer_fn or OpenAIAnswerGenerator(
        model=model,
        max_output_tokens=max_output_tokens,
        sleep_seconds=sleep_seconds,
    )
    generated_count = 0
    skipped_count = 0
    with output_jsonl.open("a" if resume else "w", encoding="utf-8") as handle:
        for idx, record in enumerate(records, start=1):
            if record.question_id in existing_ids:
                skipped_count += 1
                continue
            prompt = build_prompt(
                record,
                max_sessions=max_sessions,
                history_format=history_format,
                reader_mode=reader_mode,
                include_evidence_markers=False,
            )
            hypothesis = generator(prompt, record)
            row = {
                "question_id": record.question_id,
                "question_type": record.question_type,
                "hypothesis": hypothesis,
                "model": model,
                "prompt_mode": {
                    "history_format": history_format,
                    "reader_mode": reader_mode,
                    "max_sessions": max_sessions,
                    "evidence_markers": False,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            generated_count += 1
            print(f"[{idx}/{len(records)}] wrote hypothesis for {record.question_id}", flush=True)
    return GenerationStats(
        output_jsonl=str(output_jsonl),
        requested_count=len(records),
        generated_count=generated_count,
        skipped_existing_count=skipped_count,
        model=model,
    )


def evaluate_hypotheses(
    *,
    records: list[LongMemEvalRecord],
    hypotheses_jsonl: Path,
    output_log: Path,
    judge_model: str = "gpt-4o",
    judge_fn: JudgeFn | None = None,
) -> EvaluationStats:
    """Evaluate hypothesis JSONL with the official LongMemEval judge prompt."""

    references = {record.question_id: record for record in records}
    hypotheses = _load_hypotheses(hypotheses_jsonl)
    judge = judge_fn or OpenAIJudge(model=judge_model)
    output_log.parent.mkdir(parents=True, exist_ok=True)
    with output_log.open("w", encoding="utf-8") as handle:
        for idx, hypothesis_row in enumerate(hypotheses, start=1):
            question_id = str(hypothesis_row.get("question_id", ""))
            record = references.get(question_id)
            if record is None:
                print(f"Warning: skipping {question_id} because it is absent from reference data.", flush=True)
                continue
            hypothesis = str(hypothesis_row.get("hypothesis", ""))
            prompt = build_answer_check_prompt(record, hypothesis)
            label = judge(prompt, record, hypothesis)
            log_row = dict(hypothesis_row)
            log_row["autoeval_label"] = {"model": _judge_model_id(judge_model), "label": label}
            handle.write(json.dumps(log_row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{idx}/{len(hypotheses)}] judged {question_id}: {label}", flush=True)
    return summarize_eval_log(eval_log=output_log, records=records)


def summarize_eval_log(*, eval_log: Path, records: list[LongMemEvalRecord]) -> EvaluationStats:
    """Aggregate a LongMemEval eval log into overall and per-task metrics."""

    references = {record.question_id: record for record in records}
    rows = _load_hypotheses(eval_log)
    by_task_values: dict[str, list[float]] = {task: [] for task in QUESTION_TYPES}
    all_values: list[float] = []
    abstention_values: list[float] = []
    for row in rows:
        record = references.get(str(row.get("question_id", "")))
        if record is None:
            continue
        label = bool((row.get("autoeval_label") or {}).get("label"))
        value = 1.0 if label else 0.0
        by_task_values.setdefault(record.question_type, []).append(value)
        all_values.append(value)
        if record.is_abstention:
            abstention_values.append(value)

    by_task = {
        task: {"accuracy": _float_mean(values), "n": len(values)}
        for task, values in sorted(by_task_values.items())
    }
    task_means = [metrics["accuracy"] for metrics in by_task.values() if metrics["n"]]
    return EvaluationStats(
        eval_log=str(eval_log),
        evaluated_count=len(all_values),
        overall_accuracy=_float_mean(all_values),
        task_averaged_accuracy=_float_mean(task_means),
        abstention_accuracy=_float_mean(abstention_values) if abstention_values else None,
        by_task=by_task,
    )


def build_answer_check_prompt(record: LongMemEvalRecord, hypothesis: str) -> str:
    """Build the official LongMemEval answer-check prompt."""

    if record.is_abstention:
        template = (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is given but the asked information is not."
            "\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )
        return template.format(record.question, record.answer, hypothesis)

    if record.question_type in {"single-session-user", "single-session-assistant", "multi-session"}:
        template = (
            "I will give you a question, a correct answer, and a response from a model.\n"
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
            "If the response only contains a subset of the information required by the answer, answer no."
            "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif record.question_type == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
            "If the response only contains a subset of the information required by the answer, answer no. "
            "In addition, do not penalize off-by-one errors for the number of days. "
            "If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors, the model's response is still correct."
            "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif record.question_type == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, "
            "the response should be considered as correct as long as the updated answer is the required answer."
            "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif record.question_type == "single-session-preference":
        template = (
            "I will give you a question, a rubric for desired personalized response, and a response from a model.\n"
            "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. "
            "The response is correct as long as it recalls and utilizes the user's personal information correctly."
            "\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    else:
        raise NotImplementedError(f"Unsupported LongMemEval question type: {record.question_type}")
    return template.format(record.question, record.answer, hypothesis)


def _record_from_mapping(item: Any) -> LongMemEvalRecord:
    if not isinstance(item, dict):
        raise ValueError("LongMemEval record must be an object")
    return LongMemEvalRecord(
        question_id=str(item.get("question_id", "")),
        question_type=str(item.get("question_type", "")),
        question=str(item.get("question", "")),
        answer=str(item.get("answer", "")),
        question_date=_optional_str(item.get("question_date")),
        haystack_session_ids=[str(value) for value in item.get("haystack_session_ids", [])],
        haystack_dates=[str(value) for value in item.get("haystack_dates", [])],
        haystack_sessions=_normalize_sessions(item.get("haystack_sessions", [])),
        answer_session_ids=[str(value) for value in item.get("answer_session_ids", [])],
    )


def _normalize_sessions(value: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(value, list):
        return []
    sessions: list[list[dict[str, Any]]] = []
    for session in value:
        if not isinstance(session, list):
            sessions.append([])
            continue
        sessions.append([turn for turn in session if isinstance(turn, dict)])
    return sessions


def _history_payload(
    sessions: list[list[dict[str, Any]]],
    dates: list[str],
    *,
    start: int,
    include_evidence_markers: bool,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for idx, session in enumerate(sessions):
        payload.append(
            {
                "session_index": start + idx,
                "date": dates[idx] if idx < len(dates) else "unknown",
                "turns": [
                    _turn_payload(turn, include_evidence_markers=include_evidence_markers)
                    for turn in session
                ],
            }
        )
    return payload


def _turn_payload(turn: dict[str, Any], *, include_evidence_markers: bool) -> dict[str, Any]:
    payload = {
        "role": str(turn.get("role", "unknown")),
        "content": str(turn.get("content", "")),
    }
    if include_evidence_markers and turn.get("has_answer"):
        payload["has_answer"] = True
    return payload


def _session_lines(session: list[dict[str, Any]], *, include_evidence_markers: bool) -> list[str]:
    lines: list[str] = []
    for turn in session:
        role = str(turn.get("role", "unknown"))
        content = str(turn.get("content", ""))
        evidence = " [evidence]" if include_evidence_markers and turn.get("has_answer") else ""
        lines.append(f"{role}{evidence}: {content}")
    return lines


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _float_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _existing_hypothesis_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("question_id", "")) for row in _load_hypotheses(path)}


def _load_hypotheses(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Expected JSONL or a list of objects in {path}")


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


def _judge_model_id(model: str) -> str:
    if model == "gpt-4o":
        return "gpt-4o-2024-08-06"
    if model == "gpt-4o-mini":
        return "gpt-4o-mini-2024-07-18"
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--split", choices=sorted(DATA_URLS), default="s_cleaned")
    download_parser.add_argument("--output", default=Path("data/longmemeval_s_cleaned.json"), type=Path)
    download_parser.add_argument("--force", action="store_true")

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--data", required=True, type=Path)
    inspect_parser.add_argument("--output-dir", default=Path("experiments/public_results/longmemeval"), type=Path)
    inspect_parser.add_argument("--split-label", default="longmemeval_s_cleaned")
    inspect_parser.add_argument("--limit", type=int, default=None)
    inspect_parser.add_argument("--sample-size", type=int, default=None)
    inspect_parser.add_argument("--sample-seed", type=int, default=0)

    prompt_parser = subparsers.add_parser("make-prompts")
    prompt_parser.add_argument("--data", required=True, type=Path)
    prompt_parser.add_argument("--output-jsonl", default=Path("experiments/public_results/longmemeval/prompts.jsonl"), type=Path)
    prompt_parser.add_argument("--max-sessions", type=int, default=None)
    prompt_parser.add_argument("--history-format", choices=["json", "nl"], default="json")
    prompt_parser.add_argument("--reader-mode", choices=["direct", "con"], default="con")
    prompt_parser.add_argument("--include-evidence-markers", action="store_true")
    prompt_parser.add_argument("--limit", type=int, default=None)
    prompt_parser.add_argument("--sample-size", type=int, default=None)
    prompt_parser.add_argument("--sample-seed", type=int, default=0)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--data", required=True, type=Path)
    generate_parser.add_argument("--output-jsonl", default=Path("experiments/public_results/longmemeval/hypotheses.jsonl"), type=Path)
    generate_parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    generate_parser.add_argument("--max-sessions", type=int, default=None)
    generate_parser.add_argument("--history-format", choices=["json", "nl"], default="json")
    generate_parser.add_argument("--reader-mode", choices=["direct", "con"], default="con")
    generate_parser.add_argument("--limit", type=int, default=None)
    generate_parser.add_argument("--sample-size", type=int, default=None)
    generate_parser.add_argument("--sample-seed", type=int, default=0)
    generate_parser.add_argument("--no-resume", action="store_true")
    generate_parser.add_argument("--max-output-tokens", type=int, default=256)
    generate_parser.add_argument("--sleep-seconds", type=float, default=0.0)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--data", required=True, type=Path)
    evaluate_parser.add_argument("--hypotheses", required=True, type=Path)
    evaluate_parser.add_argument("--output-log", default=Path("experiments/public_results/longmemeval/hypotheses.eval-results-gpt-4o"), type=Path)
    evaluate_parser.add_argument("--judge-model", default="gpt-4o")
    evaluate_parser.add_argument("--limit", type=int, default=None)
    evaluate_parser.add_argument("--sample-size", type=int, default=None)
    evaluate_parser.add_argument("--sample-seed", type=int, default=0)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--data", required=True, type=Path)
    summarize_parser.add_argument("--eval-log", required=True, type=Path)
    summarize_parser.add_argument("--output-json", default=None, type=Path)
    summarize_parser.add_argument("--limit", type=int, default=None)
    summarize_parser.add_argument("--sample-size", type=int, default=None)
    summarize_parser.add_argument("--sample-seed", type=int, default=0)

    args = parser.parse_args()
    if args.command == "download":
        output = download_dataset(split=args.split, output_path=args.output, force=args.force)
        print(f"Wrote {output}")
        return

    records = select_records(
        load_records(args.data),
        limit=getattr(args, "limit", None),
        sample_size=getattr(args, "sample_size", None),
        sample_seed=getattr(args, "sample_seed", 0),
    )
    if args.command == "inspect":
        manifest_path = write_manifest(
            records=records,
            data_path=args.data,
            output_dir=args.output_dir,
            split_label=args.split_label,
        )
        print(json.dumps(inspect_records(records), indent=2))
        print(f"Wrote {manifest_path}")
        return
    if args.command == "make-prompts":
        output_jsonl = write_prompt_jsonl(
            records=records,
            output_jsonl=args.output_jsonl,
            max_sessions=args.max_sessions,
            history_format=args.history_format,
            reader_mode=args.reader_mode,
            include_evidence_markers=args.include_evidence_markers,
        )
        print(f"Wrote {output_jsonl}")
        return
    if args.command == "generate":
        stats = generate_hypotheses(
            records=records,
            output_jsonl=args.output_jsonl,
            model=args.model,
            max_sessions=args.max_sessions,
            history_format=args.history_format,
            reader_mode=args.reader_mode,
            resume=not args.no_resume,
            max_output_tokens=args.max_output_tokens,
            sleep_seconds=args.sleep_seconds,
        )
        print(json.dumps(asdict(stats), indent=2))
        return
    if args.command == "evaluate":
        stats = evaluate_hypotheses(
            records=records,
            hypotheses_jsonl=args.hypotheses,
            output_log=args.output_log,
            judge_model=args.judge_model,
        )
        print(json.dumps(asdict(stats), indent=2))
        return
    if args.command == "summarize":
        stats = summarize_eval_log(eval_log=args.eval_log, records=records)
        payload = asdict(stats)
        print(json.dumps(payload, indent=2))
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
