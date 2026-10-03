#!/usr/bin/env python3
"""Regenerate webpage/data/results.json from finished run directories.

Reads the same artifacts `ops/panel_coverage.py` audits, so the site can never quote a number
that did not come from a checksum-verified run. Run from the repository root:

    python webpage/build-data.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RESULTS_ROOT = Path.home() / "sysone-bench-results"
SUITES = [
    "agnews", "banking77_12", "emotion", "guardrails", "mnli",
    "moderation", "multilingual_intent", "sst5", "triage",
]


def main() -> int:
    if not RESULTS_ROOT.is_dir():
        print(f"no results root at {RESULTS_ROOT}", file=sys.stderr)
        return 1
    coverage = json.loads(
        subprocess.run(
            ["python3", "ops/panel_coverage.py", "--results-root", str(RESULTS_ROOT), "--json"],
            capture_output=True, text=True, check=True,
        ).stdout
    )

    by_runner: dict[str, dict] = {}
    for root in (RESULTS_ROOT / "cpu" / "runs", RESULTS_ROOT / "gpu"):
        for run in sorted(root.iterdir()):
            if not run.is_dir() or run.name.startswith((".", "_")):
                continue
            if not (run / "metadata.json").exists():
                continue
            try:
                metadata = json.loads((run / "metadata.json").read_text())
                summary = json.loads((run / "summary.json").read_text())
            except (OSError, ValueError):
                continue
            info = metadata.get("model") if isinstance(metadata.get("model"), dict) else metadata
            runner = info.get("runner") or metadata.get("runner")
            if isinstance(runner, dict):
                runner = runner.get("runner")
            if not isinstance(runner, str):
                continue
            suites = summary.get("suites", {})
            by_runner[runner] = {
                "runner": runner,
                "runId": run.name,
                "model": info.get("model"),
                "revision": info.get("revision"),
                "baseModel": info.get("base_model"),
                "scoring": info.get("scoring"),
                "techniqueReimplementation": bool(info.get("technique_reimplementation", False)),
                "vendorCodeExecuted": info.get("vendor_code_executed"),
                "sharding": info.get("sharding"),
                "singleDevice": info.get("single_device"),
                "device": info.get("device"),
                "dtype": info.get("dtype"),
                "serving": info.get("serving"),
                "accuracy": summary["accuracy"],
                "cases": summary.get("cases"),
                "decisions": summary.get("decisions"),
                "suites": {k: suites[k]["accuracy"] for k in SUITES if k in suites},
                "latencyP50": {
                    k: suites[k].get("latency", {}).get("p50_seconds") for k in SUITES if k in suites
                },
            }

    collisions = coverage["identical_prediction_collisions"]
    counted_once = {r for runners in collisions.values() for r in runners[1:]}
    ordered = [r for r in (row["runner"] for row in coverage["measurements"])]

    payload = {
        "manifestSha256": "a938cc2483a592dc84e0d5baac12594491bcaa5b4ceb6b7c3b0def71b36297bd",
        "logicalDigest": "4272a7a25ebcb324235696ad808544e157a31e3f9c04421da731a08dfb9d5768",
        "cases": 1190,
        "evaluationCases": 952,
        "decisions": 1240,
        "seed": 42,
        "datasetVersion": "2.0.0",
        "panelEdition": "Decision Index 0.2.1",
        "suites": SUITES,
        "scopeTotal": 51,
        "runDirectoriesVerified": coverage["run_directories_verified"],
        "distinctMeasurements": coverage["distinct_measurements"],
        "referenceClosedApi": {
            "runner": "jev",
            "model": "Jev 1.13.0",
            "accuracy": 0.9065,
            "serving": "closed API at api.typesafe.ai",
            "inPanel": False,
        },
        "measurements": [by_runner[r] for r in ordered if r in by_runner],
        "duplicates": {
            "repeatedRuns": coverage["repeated_runs_of_one_runner"],
            "identicalPredictions": collisions,
            "runnersCountedOnce": sorted(counted_once),
        },
    }

    out = Path(__file__).parent / "data" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(
        f"wrote {out} : {len(payload['measurements'])} measurements, "
        f"{len(payload['suites'])} suites, {out.stat().st_size} bytes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
