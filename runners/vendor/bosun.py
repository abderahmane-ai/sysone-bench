"""Bosun adapters: 0.6B and 1.7B typed-decision checkpoints.

Bosun is the one family here that does not accept a questions mapping. `model.predict(...)` scores
exactly one typed question per call, taking the shared state, the instruction text, a
``decision_type``, a ``row_id``, and an explicit candidate list. The benchmark's contract is
state-plus-many-questions, so this adapter issues one call per question and reassembles the answers.

Candidate IDs are supplied by the caller and map back onto our choice labels unchanged, which keeps
the answer's probability labels identical to the question's declared criteria. Noul questions are
sent as an explicit yes/no pair, and score questions send their levels in declared order.

One call per question also means usage is accounted per question rather than per state, so the
``calls`` counter reflects the vendor's actual call count.
"""

from __future__ import annotations

from runners.vendor.common import (
    VENDOR_ADAPTER_VERSION, VendorRunner, execution_provenance, load_kwargs,
    place_on_device, project_vendor_answer, resolve_device, resolve_dtype,)

from collections.abc import Mapping
from typing import Any


ADAPTER_VERSION = VENDOR_ADAPTER_VERSION

def _option_text(label: str, description: object) -> str:
    if description is None or description == "":
        return label
    return f"{label}: {description}"


class _BosunRunner(VendorRunner):
    family = "bosun"

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            self.default_model,
            revision=self.default_revision,
            trust_remote_code=True,
            **load_kwargs(device, dtype),
        )
        place_on_device(model, device)
        model.eval()
        self._model = model
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "serving": "local",
            **execution_provenance(device, dtype),
            "adapter_version": ADAPTER_VERSION,
        }

    def _candidates(self, question: Mapping[str, Any], qid: str) -> list[dict[str, Any]]:
        question_type = question.get("type")
        instructions = question.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(f"{qid}: question instructions are required")
        if question_type == "choice":
            criteria = question.get("criteria")
            if not isinstance(criteria, Mapping) or not criteria:
                raise ValueError(f"{qid}: choice criteria must be a non-empty mapping")
            return [
                {"id": str(label), "label": _option_text(str(label), description)}
                for label, description in criteria.items()
            ]
        if question_type == "noul":
            return [
                {"id": "yes", "label": "Yes"},
                {"id": "no", "label": "No"},
            ]
        if question_type == "score":
            criteria = question.get("criteria")
            if isinstance(criteria, Mapping):
                ordered = [str(level) for level in criteria]
            elif isinstance(criteria, (list, tuple)):
                ordered = [str(level) for level in criteria]
            else:
                raise ValueError(f"{qid}: score criteria must be an ordered sequence of levels")
            if not ordered:
                raise ValueError(f"{qid}: score criteria resolved to no levels")
            # Bosun scores an ordered scale by index, so the candidate ID is the level position and
            # the label carries the rubric text.
            return [{"id": str(index), "label": level} for index, level in enumerate(ordered)]
        raise ValueError(f"{qid}: unsupported question type {question_type!r}")

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        self._validate_phase(phase)
        copied_state, copied_questions = self._copy_inputs(state, questions)
        answers: dict[str, Any] = {}
        calls = 0
        for qid, question in copied_questions.items():
            if not isinstance(question, Mapping):
                raise TypeError(f"{qid}: question must be a mapping")
            candidates = self._candidates(question, qid)
            raw = self._model.predict(
                state=dict(copied_state),
                instructions=str(question.get("instructions", "")),
                decision_type=str(question.get("type", "")),
                row_id=qid,
                candidates=candidates,
            )
            calls += 1
            # Verified against the real checkpoint: predict() returns `probabilities` as a POSITIONAL
            # list aligned to the candidates we passed, not a label-keyed mapping, and `content` is
            # the rendered prompt with `answer: null`. The labels are therefore reapplied here from
            # our own candidate order, which is what the projection needs to match question criteria.
            values = raw.get("probabilities") if isinstance(raw, Mapping) else None
            if not isinstance(values, (list, tuple)):
                raise TypeError(f"{qid}: bosun returned no positional probability list")
            if len(values) != len(candidates):
                raise ValueError(
                    f"{qid}: bosun returned {len(values)} probabilities for {len(candidates)} candidates"
                )
            labelled = {candidate["id"]: float(value) for candidate, value in zip(candidates, values, strict=True)}
            payload = dict(raw)
            payload["probabilities"] = labelled
            question_type = question.get("type")
            if question_type == "noul":
                # The declared pair is (yes, no); the contract wants P(true).
                payload["noul"] = labelled.get("yes")
            answers[qid] = project_vendor_answer(payload, question, qid)
        return {
            "answers": self._validate_answer_ids(copied_questions, answers),
            "_usage": {"calls": calls},
        }


class Bosun06BRunner(_BosunRunner):
    name = "bosun-06b"
    default_model = "Hanno-Labs/bosun-v3.1-0.6b"
    default_revision = "1d8b6f9611f9b64b514ce8b57cd86398fbc31a3b"
    license_name = "apache-2.0"


class Bosun17BRunner(_BosunRunner):
    name = "bosun-17b"
    default_model = "Hanno-Labs/bosun-v3.1-1.7b"
    default_revision = "1d8dc82a20e4a32ed60927a47272d6efff48eed2"
    license_name = "apache-2.0"
