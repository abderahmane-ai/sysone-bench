"""Lavoir adapter.

Lavoir is a Laya-derived decision head that adds an explicit "what information is still missing"
mechanism. It is installed from its GitHub repository at a pinned commit rather than from an index,
and it declares `laya>=0.3.11,<0.4`, which our frozen environment already satisfies.

The API is close to ours but not identical. `predict(state, question, slots)` takes ONE typed
question, not a questions mapping, and returns an object whose `probabilities` are a label-keyed
mapping. With no slots supplied it degrades to exactly Laya's input format and behaves as a plain
calibrated classifier, which is the mode this benchmark uses: our sealed manifest supplies the state
and the question, and expects a decision, not an interrogation.

The model can also emit a follow-up question of its own via `next_action`. That capability is
deliberately not exercised here. Our manifest asks a fixed set of questions per case and scores the
answers; letting the model interleave extra questions would change what is being measured, so the
adapter only ever calls `predict` with no slots and no thresholds.

Licence is CC BY-NC 4.0, so results must be labelled non-commercial.
"""

from __future__ import annotations

from runners.vendor.common import (
    VENDOR_ADAPTER_VERSION, VendorRunner, execution_provenance, project_vendor_answer,
    resolve_device, resolve_dtype,)

from collections.abc import Mapping
from typing import Any


ADAPTER_VERSION = VENDOR_ADAPTER_VERSION
PACKAGE_COMMIT = "af79b0a7a7f9864343223d3f7bd40a02747d0c1f"


def _vendor_state(state: Mapping[str, Any]) -> list[dict[str, str]]:
    """Render the state as a single-user turn, which is Lavoir's documented state form.

    Field names are preserved in the text because the sealed questions refer to them directly.
    """
    text = "\n".join(f"{key}: {value}" for key, value in state.items())
    return [{"role": "user", "text": text}]


def _question_for(question: Mapping[str, Any], qid: str) -> dict[str, Any]:
    question_type = question.get("type")
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(f"{qid}: question instructions are required")
    criteria = question.get("criteria")
    payload: dict[str, Any] = {"type": question_type, "instructions": instructions}
    if question_type == "choice":
        if not isinstance(criteria, Mapping) or not criteria:
            raise ValueError(f"{qid}: choice criteria must be a non-empty mapping")
        payload["criteria"] = {str(label): str(value) for label, value in criteria.items()}
    elif question_type == "score":
        if isinstance(criteria, Mapping):
            levels = [str(level) for level in criteria]
        elif isinstance(criteria, (list, tuple)):
            levels = [str(level) for level in criteria]
        else:
            raise ValueError(f"{qid}: score criteria must be an ordered sequence of levels")
        if len(levels) < 2:
            raise ValueError(f"{qid}: score needs at least two declared levels")
        payload["criteria"] = levels
    elif question_type == "noul":
        pass
    else:
        raise ValueError(f"{qid}: unsupported question type {question_type!r}")
    return payload


class LavoirRunner(VendorRunner):
    name = "lavoir"
    family = "lavoir"
    default_model = "moganai/lavoir"
    default_revision = "4c5eaeb99b2368f1416b9893c7aacc58235aa09f"
    license_name = "cc-by-nc-4.0"

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        from lavoir import Lavoir

        # Verified from the pinned source: Lavoir.from_pretrained(path_or_repo, device=None,
        # revision=None, **kw) forwards **kw to Lavoir.__init__(model, tokenizer, config, device,
        # batch_size). There is no `decision_mode` anywhere in the package, and passing one raises
        # "Lavoir.__init__() got an unexpected keyword argument". The measured mode is instead the
        # call shape: predict(state, question, slots) with slots=None answers the question asked,
        # while next_action() would interleave Lavoir's own questions and change what is measured.
        #
        # Device goes through the package's own `device` parameter, which it also uses to select its
        # autocast dtype, so no dtype is imposed from here and `place_on_device` is not needed:
        # Lavoir is a plain wrapper, not an nn.Module.
        model = Lavoir.from_pretrained(
            self.default_model,
            revision=self.default_revision,
            device=device,
        )
        self._model = model
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "serving": "local",
            **execution_provenance(device, "vendor-managed"),
            "dtype_requested": dtype,
            "adapter_version": ADAPTER_VERSION,
            "package": "lavoir",
            "package_commit": PACKAGE_COMMIT,
            "mode": "no_slots",
        }

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        self._validate_phase(phase)
        copied_state, copied_questions = self._copy_inputs(state, questions)
        rendered_state = _vendor_state(copied_state)
        answers: dict[str, Any] = {}
        calls = 0
        for qid, question in copied_questions.items():
            if not isinstance(question, Mapping):
                raise TypeError(f"{qid}: question must be a mapping")
            prediction = self._model.predict(
                rendered_state,
                _question_for(question, qid),
                None,
            )
            calls += 1
            probabilities = getattr(prediction, "probabilities", None)
            if probabilities is None and isinstance(prediction, Mapping):
                probabilities = prediction.get("probabilities")
            if probabilities is None:
                raise TypeError(f"{qid}: lavoir returned no probability mapping")
            label = getattr(prediction, "choice", None)
            if label is None and isinstance(prediction, Mapping):
                label = prediction.get("choice")
            payload: dict[str, Any] = {
                "probabilities": {str(k): float(v) for k, v in dict(probabilities).items()},
            }
            if label is not None:
                payload["choice"] = str(label)
            answers[qid] = project_vendor_answer(payload, question, qid)
        return {
            "answers": self._validate_answer_ids(copied_questions, answers),
            "_usage": {"calls": calls},
        }
