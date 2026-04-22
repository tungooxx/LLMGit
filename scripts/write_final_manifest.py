"""Write a manifest for the frozen final experiment outputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.final_config import (
    BACKBONE,
    BENCHMARK_LOGIC_COMMIT_SHA,
    BENCHMARK_VERSION,
    PROMPT_TEMPLATE_PATH,
    QUALITATIVE_FIGURE_PATH,
    RESULT_FILES,
)


def main() -> None:
    """Create `experiments/results/final_manifest.json`."""

    manifest_path = Path("experiments/results/final_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    exported_files = [
        file_record(path)
        for path in RESULT_FILES
        if path != manifest_path
    ]
    manifest: dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION,
        "backbone": BACKBONE,
        "benchmark_logic_commit_sha": BENCHMARK_LOGIC_COMMIT_SHA,
        "reproduced_from_commit_sha": git_sha(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_template": file_record(PROMPT_TEMPLATE_PATH),
        "qualitative_figure": file_record(QUALITATIVE_FIGURE_PATH),
        "exported_result_files": exported_files,
        "commands": [
            "python -m pytest -q",
            "python -m compileall app experiments tests",
            "python -m experiments.run_benchmark --output-dir experiments/results --backbone gpt-4o-mini --include-ablations",
            "python -m experiments.governance_benchmark --output-dir experiments/results",
            "python -m experiments.memory_ci_case_study --output experiments/results/memory_ci_case_study.json",
            "python -m experiments.plot_results --summary-csv experiments/results/metric_summary.csv --output-png experiments/results/metric_summary.png",
            "python scripts/write_final_manifest.py",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


def file_record(path: Path) -> dict[str, Any]:
    """Return path, size, and SHA-256 for a file."""

    return {
        "path": str(path).replace("\\", "/"),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def sha256_file(path: Path) -> str:
    """Hash a file with SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    """Return current git SHA, or unknown outside a git checkout."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":
    main()
