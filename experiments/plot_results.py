"""Plot benchmark metric summary CSV as grouped bars."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_results(summary_csv: Path, output_png: Path) -> None:
    """Create a grouped bar chart from `metric_summary.csv`."""

    rows: list[dict[str, str]] = []
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {summary_csv}")

    systems = sorted({row["system_name"] for row in rows})
    metrics = sorted({row["metric"] for row in rows})
    scores = {
        (row["system_name"], row["metric"]): float(row["score"])
        for row in rows
    }

    width = 0.8 / len(systems)
    x_positions = list(range(len(metrics)))
    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, system in enumerate(systems):
        offsets = [x + (idx - (len(systems) - 1) / 2) * width for x in x_positions]
        values = [scores.get((system, metric), 0.0) for metric in metrics]
        ax.bar(offsets, values, width=width, label=system)

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Changing-World Memory Benchmark")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", default="experiments/results/metric_summary.csv")
    parser.add_argument("--output-png", default="experiments/results/metric_summary.png")
    args = parser.parse_args()
    plot_results(Path(args.summary_csv), Path(args.output_png))
    print(f"Wrote plot to {args.output_png}")


if __name__ == "__main__":
    main()
