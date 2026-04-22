"""Run the structural changing-world memory benchmark and export results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from experiments.baselines import SystemAnswer, default_systems
from experiments.benchmark import BenchmarkCase, BenchmarkQuestion, default_benchmark
from experiments.final_config import (
    BACKBONE,
    BENCHMARK_LOGIC_COMMIT_SHA,
    BENCHMARK_VERSION,
    PROMPT_TEMPLATE_PATH,
)
from experiments.metrics import aggregate_scores, score_questions, scores_to_dicts


def run_benchmark(
    cases: list[BenchmarkCase],
    *,
    include_ablations: bool = False,
    backbone: str = BACKBONE,
) -> dict[str, object]:
    """Run deterministic memory adapters on all benchmark cases."""

    all_questions: list[BenchmarkQuestion] = [
        question for case in cases for question in case.questions
    ]
    all_scores = []
    predictions = []

    for system in default_systems(include_ablations=include_ablations):
        system.reset()
        system_answers: list[SystemAnswer] = []
        try:
            for case in cases:
                for event in case.events:
                    system.ingest_event(event)
                for question in case.questions:
                    answer = system.answer(question)
                    system_answers.append(answer)
                    prediction = dict(answer.__dict__)
                    prediction["backbone"] = backbone
                    predictions.append(prediction)
            all_scores.extend(
                score_questions(system_name=system.name, questions=all_questions, answers=system_answers)
            )
        finally:
            close = getattr(system, "close", None)
            if close:
                close()

    return {
        "metadata": {
            "benchmark_version": BENCHMARK_VERSION,
            "evaluation_mode": "structural_memory_correctness",
            "backbone": backbone,
            "backbone_note": "Metadata label only; this structural runner does not call an LLM reader.",
            "prompt_template_path": PROMPT_TEMPLATE_PATH.as_posix(),
            "prompt_template_sha256": _sha256_file(PROMPT_TEMPLATE_PATH),
            "benchmark_logic_commit_sha": BENCHMARK_LOGIC_COMMIT_SHA,
            "include_ablations": include_ablations,
            "case_count": len(cases),
            "question_count": len(all_questions),
        },
        "benchmark": [case.to_json() for case in cases],
        "predictions": predictions,
        "question_scores": scores_to_dicts(all_scores),
        "metric_summary": aggregate_scores(all_scores),
    }


def export_results(results: dict[str, object], output_dir: Path) -> None:
    """Write detailed JSON and CSV result files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )
    _write_csv(output_dir / "metric_summary.csv", results["metric_summary"])
    _write_csv(output_dir / "question_scores.csv", results["question_scores"])
    _write_csv(output_dir / "predictions.csv", results["predictions"])


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
    parser.add_argument("--backbone", default=BACKBONE, help="Metadata label only for the structural table.")
    parser.add_argument("--include-ablations", action="store_true")
    args = parser.parse_args()
    results = run_benchmark(
        default_benchmark(),
        include_ablations=args.include_ablations,
        backbone=args.backbone,
    )
    export_results(results, Path(args.output_dir))
    print(f"Wrote benchmark outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
