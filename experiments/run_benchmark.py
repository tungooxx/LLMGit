"""Run the changing-world memory benchmark and export JSON/CSV results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.baselines import SystemAnswer, default_systems
from experiments.benchmark import BenchmarkCase, BenchmarkQuestion, default_benchmark
from experiments.metrics import aggregate_scores, score_questions, scores_to_dicts


def run_benchmark(cases: list[BenchmarkCase]) -> dict[str, object]:
    """Run all systems on all benchmark cases."""

    all_questions: list[BenchmarkQuestion] = [
        question for case in cases for question in case.questions
    ]
    all_scores = []
    predictions = []

    for system in default_systems():
        system.reset()
        system_answers: list[SystemAnswer] = []
        try:
            for case in cases:
                for event in case.events:
                    system.ingest_event(event)
                for question in case.questions:
                    answer = system.answer(question)
                    system_answers.append(answer)
                    predictions.append(answer.__dict__)
            all_scores.extend(
                score_questions(system_name=system.name, questions=all_questions, answers=system_answers)
            )
        finally:
            close = getattr(system, "close", None)
            if close:
                close()

    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/results")
    args = parser.parse_args()
    results = run_benchmark(default_benchmark())
    export_results(results, Path(args.output_dir))
    print(f"Wrote benchmark outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
