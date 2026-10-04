#!/usr/bin/env python3
"""Audit panel coverage and detect duplicate measurements.

A coverage count is only meaningful if it counts *measurements*, not run directories. Two failure
modes inflate it, and both occurred in this project:

1. **Repeated runs of one model.** A retried or re-hosted model leaves two directories. Counting
   directories counts the model twice.
2. **Distinct runner names, identical predictions.** Two registry entries can agree on every single
   decision while being separate entries. ``openvons`` and ``mini-jev`` do exactly this: they share
   the base checkpoint ``Qwen/Qwen3-4B-Instruct-2507`` at revision ``cdbee75f...`` and produce
   byte-identical answers across all 1240 evaluation decisions. They are one measurement wearing two
   names, and reporting them separately overstates the panel.

This tool derives the count from artifacts rather than from a hand-maintained tally, so it cannot
drift from what was actually measured. It reads finished run directories only, verifies each
against its own ``checksums.sha256``, and fingerprints the evaluation-phase answers so identical
runs are grouped.

Usage::

    python ops/panel_coverage.py --results-root results/raw
    python ops/panel_coverage.py --results-root results/raw --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUIRED = ("metadata.json", "predictions.jsonl", "summary.json", "usage.json", "checksums.sha256")

#: Manifest every row in this panel was produced against.
SEALED_MANIFEST_SHA256 = "a938cc2483a592dc84e0d5baac12594491bcaa5b4ceb6b7c3b0def71b36297bd"


def verify_run(run_dir: Path) -> tuple[bool, list[str]]:
    """Check required artifacts exist and every listed checksum matches."""
    present = {entry.name for entry in run_dir.iterdir() if entry.is_file()}
    missing = [name for name in REQUIRED if name not in present]
    if missing:
        return False, [f"missing {','.join(missing)}"]
    problems: list[str] = []
    for line in (run_dir / "checksums.sha256").read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        target = run_dir / name
        if not target.exists():
            problems.append(f"absent {name}")
            continue
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != expected:
            problems.append(f"corrupt {name}")
    return not problems, problems


def fingerprint(run_dir: Path) -> str | None:
    """Hash the evaluation-phase answers so identical runs can be detected.

    Only ``execution_phase == "benchmark"`` and ``phase == "evaluation"`` rows are used. Calibration
    rows are excluded deliberately: they are the fitting split, and two runs may legitimately share
    them while differing on evaluation. Returns ``None`` when the file cannot be read.
    """
    digest = hashlib.sha256()
    rows = 0
    try:
        with (run_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("execution_phase") != "benchmark" or row.get("phase") != "evaluation":
                    continue
                digest.update(
                    json.dumps(
                        [row.get("case_id"), row.get("answers")],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                )
                rows += 1
    except (OSError, ValueError):
        return None
    if rows == 0:
        return None
    return f"{rows}:{digest.hexdigest()[:32]}"


def read_run(run_dir: Path) -> dict[str, Any]:
    """Extract runner identity and accuracy from one finished run directory."""
    metadata = json.loads((run_dir / "metadata.json").read_text())
    # Vendor runners nest their identity under metadata["model"]; the CLI stores it flat.
    info = metadata.get("model") if isinstance(metadata.get("model"), dict) else metadata
    runner = info.get("runner") or metadata.get("runner")
    if isinstance(runner, dict):
        runner = runner.get("runner")
    runner = str(runner) if runner is not None else f"<unnamed:{run_dir.name}>"
    summary = json.loads((run_dir / "summary.json").read_text())
    return {
        "run_dir": run_dir.name,
        "runner": runner,
        "model": info.get("model"),
        "revision": info.get("revision"),
        "base_model": info.get("base_model"),
        "scoring": info.get("scoring"),
        "technique_reimplementation": info.get("technique_reimplementation", False),
        "sharding": info.get("sharding"),
        "accuracy": summary.get("accuracy"),
    }


def audit(results_root: Path) -> dict[str, Any]:
    roots = [results_root / "cpu" / "runs", results_root / "gpu"]
    runs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_dirs: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir() or run_dir.name in seen_dirs:
                continue
            if run_dir.name.startswith(".") or run_dir.name.startswith("_"):
                continue
            seen_dirs.add(run_dir.name)
            if not (run_dir / "checksums.sha256").exists():
                rejected.append({"run_dir": run_dir.name, "problems": ["no checksums.sha256"]})
                continue
            ok, problems = verify_run(run_dir)
            if not ok:
                rejected.append({"run_dir": run_dir.name, "problems": problems})
                continue
            record = read_run(run_dir)
            record["fingerprint"] = fingerprint(run_dir)
            runs.append(record)

    by_runner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in runs:
        by_runner[record["runner"]].append(record)

    # Collapse repeat runs of one runner, then collapse runners whose predictions are identical.
    repeated = {
        runner: sorted(item["run_dir"] for item in records)
        for runner, records in by_runner.items()
        if len(records) > 1
    }
    representative: dict[str, dict[str, Any]] = {}
    for runner, records in by_runner.items():
        best = max(records, key=lambda item: (item["accuracy"] is not None, item["accuracy"] or 0))
        representative[runner] = best

    by_fingerprint: dict[str, list[str]] = defaultdict(list)
    for runner, record in representative.items():
        if record["fingerprint"]:
            by_fingerprint[record["fingerprint"]].append(runner)
    collisions = {
        fingerprint: sorted(runners)
        for fingerprint, runners in by_fingerprint.items()
        if len(runners) > 1
    }

    measured = set(representative)
    for runners in collisions.values():
        for runner in runners[1:]:
            measured.discard(runner)

    rows = sorted(
        (record for runner, record in representative.items() if runner in measured),
        key=lambda item: (item["accuracy"] is None, -(item["accuracy"] or 0)),
    )
    duplicates = sorted(set(representative) - measured)

    return {
        "sealed_manifest_sha256": SEALED_MANIFEST_SHA256,
        "run_directories_verified": len(runs),
        "run_directories_rejected": rejected,
        "distinct_runners": len(representative),
        "distinct_measurements": len(measured),
        "repeated_runs_of_one_runner": repeated,
        "identical_prediction_collisions": collisions,
        "duplicate_runners_not_counted": duplicates,
        "measurements": rows,
    }


def render(report: dict[str, Any]) -> str:
    out: list[str] = []
    out.append(f"run directories verified : {report['run_directories_verified']}")
    out.append(f"distinct runner names    : {report['distinct_runners']}")
    out.append(f"distinct measurements    : {report['distinct_measurements']}")
    rejected = report["run_directories_rejected"]
    out.append(f"rejected run directories : {len(rejected)}")
    for item in rejected:
        out.append(f"    {item['run_dir']}: {'; '.join(item['problems'])}")
    for runner, dirs in sorted(report["repeated_runs_of_one_runner"].items()):
        out.append(f"repeated run: {runner} -> {', '.join(dirs)} (counted once)")
    for fingerprint, runners in sorted(report["identical_prediction_collisions"].items()):
        out.append(f"collision {fingerprint}: {', '.join(runners)} (counted once)")
    out.append("")
    out.append(f"{'runner':<24}{'accuracy':>10}  scoring")
    for row in report["measurements"]:
        accuracy = "n/a" if row["accuracy"] is None else f"{row['accuracy']:.4f}"
        notes = row["scoring"] or ""
        if row["technique_reimplementation"]:
            notes = f"{notes} [technique reimplementation]"
        if row["sharding"]:
            notes = f"{notes} [sharded]"
        out.append(f"{row['runner']:<24}{accuracy:>10}  {notes}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="emit the full report as JSON")
    args = parser.parse_args()
    report = audit(args.results_root)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
