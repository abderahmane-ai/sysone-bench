"""This-that model 1.2 adapter.

The package is not on PyPI; it is installed from its GitHub repository at a pinned commit, and the
image layer records that commit so the code under test is immutable even though the package name is
not versioned on an index.

This family has the cleanest contract of the whole set. `decide(state, questions)` accepts a *list*
of `Question` and answers every one of them in a single forward pass, so one state costs one call
rather than one call per question. `Question` requires at least two options, which is satisfied by
choice questions, by score questions with an ordinal scale, and by expressing a noul as an explicit
yes/no pair.

`Decision` reports `options`, `index`, `probabilities`, and a `choice` property, with
`probabilities` aligned to `options` and summing to one. The adapter zips the declared options back
onto the returned distribution so the answer's probability labels always match the question's own
criteria, rather than trusting positional alignment implicitly.
"""

from __future__ import annotations

from runners.vendor.common import (
    VENDOR_ADAPTER_VERSION, VendorRunner, execution_provenance, resolve_device,
    resolve_dtype,)

import json
from collections.abc import Mapping
from typing import Any


ADAPTER_VERSION = VENDOR_ADAPTER_VERSION
CODE_COMMIT = "6af11c63c8bb260aa4bed2b7717f33f25decd16e"


def _options_for(question: Mapping[str, Any], qid: str) -> list[str]:
    question_type = question.get("type")
    criteria = question.get("criteria")
    if question_type == "choice":
        if not isinstance(criteria, Mapping) or not criteria:
            raise ValueError(f"{qid}: choice criteria must be a non-empty mapping")
        return [str(label) for label in criteria]
    if question_type == "score":
        if isinstance(criteria, Mapping):
            levels = [str(level) for level in criteria]
        elif isinstance(criteria, (list, tuple)):
            levels = [str(level) for level in criteria]
        else:
            raise ValueError(f"{qid}: score criteria must be an ordered sequence of levels")
        if len(levels) < 2:
            raise ValueError(f"{qid}: score needs at least two declared levels")
        return levels
    if question_type == "noul":
        # Question requires two or more options; a noul is expressed as an explicit pair.
        return ["no", "yes"]
    raise ValueError(f"{qid}: unsupported question type {question_type!r}")


class ThisThat12Runner(VendorRunner):
    name = "this-that-12"
    family = "this-that"
    default_model = "flock-io/this-that-model-1.2"
    default_revision = "c4d1c30b8d512d278726de439b8cc81fccc70c8f"
    license_name = "mit"

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        from huggingface_hub import snapshot_download

        from thisthat import TypedDecider

        # TypedDecider.from_pretrained forwards **kw to the backend, which expects a plain path, and
        # a "repo@revision" string is not a valid Hugging Face repo id. Resolving the pinned
        # revision to a local snapshot is the only way to keep the revision explicit.
        local = snapshot_download(
            self.default_model,
            revision=self.default_revision,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"],
        )
        decider = TypedDecider.from_pretrained(local, device=device)
        self._decider = decider
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "serving": "local",
            **execution_provenance(device, dtype),
            "adapter_version": ADAPTER_VERSION,
            "package": "thisthat",
            "package_commit": CODE_COMMIT,
        }

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        from thisthat import Question

        self._validate_phase(phase)
        copied_state, copied_questions = self._copy_inputs(state, questions)

        requested: list[Question] = []
        declared: list[list[str]] = []
        order: list[str] = []
        for qid, question in copied_questions.items():
            if not isinstance(question, Mapping):
                raise TypeError(f"{qid}: question must be a mapping")
            instructions = question.get("instructions")
            if not isinstance(instructions, str) or not instructions.strip():
                raise ValueError(f"{qid}: question instructions are required")
            options = _options_for(question, qid)
            requested.append(Question(instructions, options))
            declared.append(options)
            order.append(qid)

        # One state, every question, one forward pass.
        text = json.dumps(dict(copied_state), ensure_ascii=False, sort_keys=True)
        decisions = self._decider.decide(text, requested)
        if len(decisions) != len(order):
            raise ValueError(
                f"this-that returned {len(decisions)} decisions for {len(order)} questions"
            )

        answers: dict[str, Any] = {}
        for qid, question, options, decision in zip(
            order, copied_questions.values(), declared, decisions, strict=True
        ):
            probabilities = {
                label: float(value) for label, value in zip(options, decision.probabilities, strict=True)
            }
            question_type = question.get("type")
            if question_type == "noul":
                answers[qid] = {"type": "noul", "noul": probabilities["yes"]}
                continue
            if question_type == "score":
                index = int(decision.index)
                answers[qid] = {
                    "type": "score",
                    "score": float(index),
                    "confidence": max(probabilities.values()),
                }
                continue
            answers[qid] = {
                "type": "choice",
                "choice": str(options[int(decision.index)]),
                "probabilities": probabilities,
                "confidence": max(probabilities.values()),
            }
        return {
            "answers": self._validate_answer_ids(copied_questions, answers),
            "_usage": {"calls": 1},
        }
