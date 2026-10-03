"""Julia-1 adapter.

Julia ships as a complete Python package *inside its own model repository* rather than on an index,
and `load_model` requires a local directory because it reads `julia_config.json` and `encoder/`
from disk. The adapter therefore snapshots the pinned repository and puts that snapshot on
`sys.path`, which imports `julia` from the checkpoint itself. No separate install step is needed.

That import executes vendor code, so it happens inside the locked-down container rather than on the
host.

Verified from the shipped `julia/typed.py` and `julia/inference.py`:

* `engine.predict(state=..., questions=...)` returns one dict per question with `type`,
  `probabilities` keyed by the declared option values, and `max_probability` rather than
  `confidence` for non-noul answers.
* A noul carries a bare `noul` set to `p[1]`, i.e. the second declared option, which is why the
  adapter sends `["no", "yes"]` for noul questions.
* A score is the probability-weighted expected level index.

The `julia` package also exposes a `bend`/`bend-dense` native backend. The torch backend is used
explicitly so the run stays on the verified CPU path; the native backend's numerics are unverified
here and would not be comparable to the rest of the panel.
"""

from __future__ import annotations

from runners.vendor.common import (
    VENDOR_ADAPTER_VERSION, VendorRunner, execution_provenance, project_vendor_answer,
    resolve_device, resolve_dtype,)

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ADAPTER_VERSION = VENDOR_ADAPTER_VERSION


def _vendor_question(question: Mapping[str, Any], qid: str) -> dict[str, Any]:
    """Translate one benchmark question into Julia's own typed-question shape.

    Read from the shipped `julia/typed.py`: a choice question takes `criteria` as a mapping of
    nonempty label IDs to descriptions, a score takes an ordered `criteria` list, and a noul takes
    `criteria` keyed by exactly `false` and `true` (or nothing at all). Julia builds the option list
    itself, so no `options` key is sent.
    """
    question_type = question.get("type")
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(f"{qid}: question instructions are required")
    payload: dict[str, Any] = {"type": question_type, "instructions": instructions}
    criteria = question.get("criteria")
    if question_type == "choice":
        if not isinstance(criteria, Mapping) or not criteria:
            raise ValueError(f"{qid}: choice criteria must be a non-empty mapping")
        payload["criteria"] = {str(label): "" if value is None else str(value)
                               for label, value in criteria.items()}
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
        # Julia's noul keys are exactly false/true and the descriptions are optional. They must be
        # sent as absent (None), not as empty strings: julia/typed.py treats a missing `criteria` as
        # labels=['false','true'], but renders whatever a supplied mapping holds, and
        # julia/data.py:validate_row then rejects the row with "options must contain 2-20 nonempty
        # rendered descriptions". Every noul in the sealed manifest omits `criteria`, so sending
        # blanks failed 480 questions on the first question of each triage-style case.
        payload["criteria"] = None
    else:
        raise ValueError(f"{qid}: unsupported question type {question_type!r}")
    return payload


def _restore_upstream_encoder_forward(engine: Any) -> bool:
    """Undo Julia's ModernBERT fast path so the encoder runs on stock transformers.

    julia/router/encoder.py:specialize_decision_encoder swaps ModernBertModel.forward for a leaner
    `_decision_forward` that calls `self._update_attention_mask(...)`. That private method no longer
    exists on ModernBertModel in transformers 5.15+, so every forward raises
    "AttributeError: 'ModernBertModel' object has no attribute '_update_attention_mask'" on the
    versions this benchmark runs. The specialization only skips outputs the benchmark never reads,
    so restoring the forward it saved computes the same hidden states on stock transformers. It is
    slower, not different.
    """
    encoder = getattr(engine, "encoder", None)
    if encoder is None:
        encoder = getattr(getattr(engine, "model", None), "encoder", None)
    original = getattr(encoder, "_julia_original_forward", None)
    if original is None:
        return False
    encoder.forward = original
    return True


class JuliaRunner(VendorRunner):
    name = "julia-1"
    family = "julia"
    default_model = "SupersonicLabs/Julia-1"
    default_revision = "a85b127321d580d65176c89ced8273f305745d85"
    license_name = "apache-2.0"

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                self.default_model,
                revision=self.default_revision,
                # The package lives in the repo, so the Python sources must be fetched too.
                allow_patterns=["*.json", "*.py", "*.safetensors", "*.model", "*.txt", "encoder/*"],
            )
        ).resolve()
        resolved = str(root)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        from julia import load_model

        # torch backend, cpu device: the native "bend" backends are not verified here.
        engine = load_model(resolved, device=device, backend="torch")
        _restore_upstream_encoder_forward(engine)
        self._engine = engine
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "serving": "local",
            **execution_provenance(device, dtype),
            "adapter_version": ADAPTER_VERSION,
            "package": "julia",
            "package_source": "in-repository",
            "backend": "torch",
        }

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        from runners.vendor.common import project_vendor_answer

        self._validate_phase(phase)
        copied_state, copied_questions = self._copy_inputs(state, questions)

        payload: dict[str, Any] = {}
        for qid, question in copied_questions.items():
            if not isinstance(question, Mapping):
                raise TypeError(f"{qid}: question must be a mapping")
            payload[qid] = _vendor_question(question, qid)

        text = "\n".join(f"{key}: {value}" for key, value in copied_state.items())
        # Verified from the pinned julia/typed.py: predict_typed returns dict(answers={qid: result}),
        # keyed by question id, not a positional list.
        response = self._engine.predict(state=text, questions=payload)
        results = response.get("answers") if isinstance(response, Mapping) else None
        if not isinstance(results, Mapping) or set(results) != set(copied_questions):
            raise ValueError(
                f"julia returned answers for "
                f"{sorted(results) if isinstance(results, Mapping) else '?'}, "
                f"expected {sorted(copied_questions)}"
            )

        answers: dict[str, Any] = {}
        for qid, question in copied_questions.items():
            result = results[qid]
            if not isinstance(result, Mapping):
                raise TypeError(f"{qid}: julia returned a non-mapping answer")
            answers[qid] = project_vendor_answer(result, question, qid)
        return {
            "answers": self._validate_answer_ids(copied_questions, answers),
            "_usage": {"calls": 1},
        }
