"""Tests for the orchestrator's aggregate of adapter-reported decision statuses."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from benchmark.orchestrator import _decision_statuses

CHOICE: dict[str, Any] = {"qid": "intent", "type": "choice", "criteria": {"a": "x", "b": "y"}}
NOUL: dict[str, Any] = {"qid": "urgent", "type": "noul"}


def _row(
    split: str,
    case_id: str,
    decisions: list[dict[str, Any]] | None,
    *,
    choice: str = "a",
    expected_choice: str = "a",
    noul: float = 0.9,
    expected_noul: int = 1,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "answers": {
            "intent": {"type": "choice", "choice": choice},
            "urgent": {"type": "noul", "noul": noul},
        },
        "case_id": case_id,
        "expected": {"intent": expected_choice, "urgent": expected_noul},
        "questions": [deepcopy(CHOICE), deepcopy(NOUL)],
        "split": split,
    }
    if decisions is not None:
        row["_raw_model"] = {"decisions": decisions}
    return row


def _decision(status: str, *, no_option_applied: bool = False) -> dict[str, Any]:
    return {"status": status, "no_option_applied": no_option_applied}


def test_counts_decisions_and_correct_answers_per_split_and_status() -> None:
    rows = [
        _row("calibration", "c0", [_decision("decided"), _decision("decided")]),
        _row("evaluation", "e0", [_decision("decided"), _decision("deferred")]),
        _row(
            "evaluation",
            "e1",
            [_decision("deferred", no_option_applied=True), _decision("deferred")],
            choice="b",
            noul=0.2,
        ),
    ]
    assert _decision_statuses(rows) == {
        "calibration": {
            "no_option_applied": 0,
            "statuses": {"decided": {"correct": 2, "decisions": 2}},
        },
        "evaluation": {
            "no_option_applied": 1,
            "statuses": {
                "decided": {"correct": 1, "decisions": 1},
                "deferred": {"correct": 1, "decisions": 3},
            },
        },
    }


def test_adapters_without_statuses_add_nothing() -> None:
    assert _decision_statuses([_row("evaluation", "e0", None)]) is None
    assert _decision_statuses([]) is None


def test_a_split_with_no_rows_reports_zero_counts() -> None:
    result = _decision_statuses([_row("evaluation", "e0", [_decision("decided")] * 2)])
    assert result is not None
    assert result["calibration"] == {"no_option_applied": 0, "statuses": {}}


def test_statuses_for_only_some_rows_are_rejected() -> None:
    rows = [
        _row("evaluation", "e0", [_decision("decided")] * 2),
        _row("evaluation", "e1", None),
    ]
    with pytest.raises(ValueError, match="only some prediction rows"):
        _decision_statuses(rows)


@pytest.mark.parametrize("decisions", [[], [_decision("decided")], [_decision("decided")] * 3])
def test_a_status_count_that_differs_from_the_question_count_names_the_case(
    decisions: list[dict[str, Any]],
) -> None:
    with pytest.raises(ValueError, match="e7: decision statuses do not match its questions"):
        _decision_statuses([_row("evaluation", "e7", decisions)])


def test_a_decision_without_a_status_names_the_case() -> None:
    with pytest.raises((ValueError, TypeError), match="e3"):
        _decision_statuses(
            [_row("evaluation", "e3", [{"no_option_applied": False}, _decision("x")])]
        )


def test_the_aggregate_conserves_every_decision() -> None:
    rows = [
        _row("evaluation", f"e{index}", [_decision("decided"), _decision("deferred")])
        for index in range(5)
    ]
    result = _decision_statuses(rows)
    assert result is not None
    statuses = result["evaluation"]["statuses"]
    assert sum(entry["decisions"] for entry in statuses.values()) == 5 * 2
