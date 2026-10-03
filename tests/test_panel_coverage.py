"""Tests for the panel coverage auditor.

The auditor decides the panel's headline number, so its collapsing rules are pinned here. Two real
mistakes are guarded against: counting a retried run twice, and counting two runners that agree on
every decision as two measurements.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

import panel_coverage  # noqa: E402


def _write_run(
    root: Path,
    name: str,
    runner: str,
    accuracy: float,
    answers: dict[str, dict[str, object]],
    *,
    broken: bool = False,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    metadata = {
        "model": {
            "runner": runner,
            "model": f"org/{runner}",
            "revision": "0" * 40,
            "scoring": "test",
        }
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    rows = [
        {
            "execution_phase": "benchmark",
            "phase": "evaluation",
            "case_id": case_id,
            "answers": payload,
        }
        for case_id, payload in sorted(answers.items())
    ]
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"accuracy": accuracy}, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "usage.json").write_text("{}", encoding="utf-8")
    digest = hashlib.sha256((run_dir / "predictions.jsonl").read_bytes()).hexdigest()
    line = f"{digest}  predictions.jsonl\n"
    if broken:
        line = f"{'f' * 64}  predictions.jsonl\n"
    (run_dir / "checksums.sha256").write_text(line, encoding="utf-8")
    return run_dir


@pytest.fixture
def results_root(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    (root / "cpu" / "runs").mkdir(parents=True)
    (root / "gpu").mkdir(parents=True)
    return root


def test_counts_distinct_runners_when_no_collisions(results_root: Path) -> None:
    _write_run(results_root / "gpu", "a", "alpha", 0.9, {"c1": {"choice": "x"}})
    _write_run(results_root / "gpu", "b", "beta", 0.8, {"c1": {"choice": "y"}})
    report = panel_coverage.audit(results_root)
    assert report["run_directories_verified"] == 2
    assert report["distinct_measurements"] == 2
    assert report["identical_prediction_collisions"] == {}


def test_repeated_run_of_one_runner_counts_once(results_root: Path) -> None:
    _write_run(results_root / "gpu", "a1", "alpha", 0.9, {"c1": {"choice": "x"}})
    _write_run(results_root / "gpu", "a2", "alpha", 0.9, {"c1": {"choice": "x"}})
    report = panel_coverage.audit(results_root)
    assert report["run_directories_verified"] == 2
    assert report["distinct_runners"] == 1
    assert report["distinct_measurements"] == 1
    assert report["repeated_runs_of_one_runner"]["alpha"] == ["a1", "a2"]


def test_distinct_runners_with_identical_predictions_count_once(results_root: Path) -> None:
    """The `mini-jev` / `openvons` case: same base, same answers, one measurement."""
    shared = {"c1": {"choice": "x", "probabilities": {"x": 0.5, "y": 0.5}}, "c2": {"noul": 0.25}}
    _write_run(results_root / "gpu", "m", "mini-jev", 0.78, shared)
    _write_run(results_root / "gpu", "o", "openvons", 0.78, shared)
    report = panel_coverage.audit(results_root)
    assert report["distinct_runners"] == 2
    assert report["distinct_measurements"] == 1
    assert len(report["identical_prediction_collisions"]) == 1
    assert sorted(next(iter(report["identical_prediction_collisions"].values()))) == [
        "mini-jev",
        "openvons",
    ]


def test_calibration_rows_are_excluded_from_the_fingerprint(results_root: Path) -> None:
    """Two runs may share a calibration split and still differ on evaluation."""
    run_dir = results_root / "gpu" / "a"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"model": {"runner": "alpha"}}, sort_keys=True), encoding="utf-8"
    )
    lines = [
        json.dumps(
            {"execution_phase": "benchmark", "phase": "calibration", "case_id": "c0", "answers": {"q": 1}},
            sort_keys=True,
        ),
        json.dumps(
            {"execution_phase": "benchmark", "phase": "evaluation", "case_id": "c1", "answers": {"q": "x"}},
            sort_keys=True,
        ),
    ]
    (run_dir / "predictions.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text('{"accuracy": 0.5}', encoding="utf-8")
    (run_dir / "usage.json").write_text("{}", encoding="utf-8")
    digest = hashlib.sha256((run_dir / "predictions.jsonl").read_bytes()).hexdigest()
    (run_dir / "checksums.sha256").write_text(f"{digest}  predictions.jsonl\n", encoding="utf-8")
    fingerprint = panel_coverage.fingerprint(run_dir)
    assert fingerprint is not None
    assert fingerprint.startswith("1:"), "only the one evaluation row should be hashed"


def test_corrupt_run_is_rejected_not_counted(results_root: Path) -> None:
    _write_run(results_root / "gpu", "ok", "alpha", 0.9, {"c1": {"choice": "x"}})
    _write_run(results_root / "gpu", "bad", "beta", 0.8, {"c1": {"choice": "y"}}, broken=True)
    report = panel_coverage.audit(results_root)
    assert report["run_directories_verified"] == 1
    assert [item["run_dir"] for item in report["run_directories_rejected"]] == ["bad"]
    assert report["distinct_measurements"] == 1


def test_directory_without_checksums_is_rejected(results_root: Path) -> None:
    stray = results_root / "gpu" / "from-bundle"
    stray.mkdir()
    (stray / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    report = panel_coverage.audit(results_root)
    assert [item["run_dir"] for item in report["run_directories_rejected"]] == ["from-bundle"]
    assert report["distinct_measurements"] == 0


def test_measurements_are_ordered_by_accuracy(results_root: Path) -> None:
    _write_run(results_root / "gpu", "a", "alpha", 0.5, {"c1": {"choice": "x"}})
    _write_run(results_root / "gpu", "b", "beta", 0.9, {"c1": {"choice": "y"}})
    _write_run(results_root / "cpu" / "runs", "c", "gamma", 0.7, {"c1": {"choice": "z"}})
    report = panel_coverage.audit(results_root)
    assert [row["runner"] for row in report["measurements"]] == ["beta", "gamma", "alpha"]


def test_cpu_and_gpu_roots_are_both_scanned(results_root: Path) -> None:
    _write_run(results_root / "cpu" / "runs", "cpu-run", "alpha", 0.6, {"c1": {"choice": "x"}})
    _write_run(results_root / "gpu", "gpu-run", "beta", 0.7, {"c1": {"choice": "y"}})
    report = panel_coverage.audit(results_root)
    assert report["run_directories_verified"] == 2
    assert report["distinct_measurements"] == 2
