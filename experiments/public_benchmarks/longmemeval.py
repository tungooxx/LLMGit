"""LongMemEval public benchmark adapter.

This module does not change the frozen TruthGit synthetic benchmark. It adds a
small public-benchmark harness for inspecting official LongMemEval JSON files
and producing prompt JSONL records for a future answer-generation run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


OFFICIAL_DATASET_REPO = "xiaowu0162/longmemeval-cleaned"
OFFICIAL_CODE_REPO = "https://github.com/xiaowu0162/LongMemEval"
OFFICIAL_ARXIV = "https://arxiv.org/abs/2410.10813"


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
) -> Path:
    """Write prompts for a future LongMemEval answer-generation run."""

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            prompt_record = {
                "question_id": record.question_id,
                "question_type": record.question_type,
                "question": record.question,
                "question_date": record.question_date,
                "prompt": build_prompt(record, max_sessions=max_sessions),
            }
            handle.write(json.dumps(prompt_record, ensure_ascii=False) + "\n")
    return output_jsonl


def build_prompt(record: LongMemEvalRecord, *, max_sessions: int | None = None) -> str:
    """Format one LongMemEval instance as a deterministic answer prompt."""

    session_count = len(record.haystack_sessions)
    start = max(0, session_count - max_sessions) if max_sessions is not None else 0
    selected_sessions = record.haystack_sessions[start:]
    selected_dates = record.haystack_dates[start:]
    lines = [
        "Answer the LongMemEval question using only the timestamped chat history.",
        "If the history does not support an answer, say you do not know.",
        f"Question date: {record.question_date or 'unknown'}",
        f"Question type: {record.question_type}",
        "",
        "History:",
    ]
    for idx, session in enumerate(selected_sessions):
        session_id = start + idx
        date = selected_dates[idx] if idx < len(selected_dates) else "unknown"
        lines.append(f"[Session {session_id} | {date}]")
        lines.extend(_session_lines(session))
        lines.append("")
    lines.extend(["Question:", record.question, "Answer:"])
    return "\n".join(lines)


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


def _session_lines(session: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for turn in session:
        role = str(turn.get("role", "unknown"))
        content = str(turn.get("content", ""))
        evidence = " [evidence]" if turn.get("has_answer") else ""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--data", required=True, type=Path)
    inspect_parser.add_argument("--output-dir", default=Path("experiments/public_results/longmemeval"), type=Path)
    inspect_parser.add_argument("--split-label", default="longmemeval_s_cleaned")
    inspect_parser.add_argument("--limit", type=int, default=None)

    prompt_parser = subparsers.add_parser("make-prompts")
    prompt_parser.add_argument("--data", required=True, type=Path)
    prompt_parser.add_argument("--output-jsonl", default=Path("experiments/public_results/longmemeval/prompts.jsonl"), type=Path)
    prompt_parser.add_argument("--max-sessions", type=int, default=None)
    prompt_parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    records = load_records(args.data, limit=args.limit)
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
        )
        print(f"Wrote {output_jsonl}")
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
