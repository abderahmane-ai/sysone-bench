"""JevK5 adapter.

JevK5 is a library, not a service: `JevK5.decide(state, question)` returns TypeSafe's answer shape
directly, so there is no subprocess and no HTTP hop. The constructor takes a device and a dtype, and
reads its calibration temperatures from `jevk5_config.json` next to the weights unless they are
passed explicitly.

Read from the pinned source and relied on here:

* `JevK5(source, device, dtype, graphs, temperature, method, knockout_temperature)`. Temperature
  defaults to the config's value (1.22 for this checkpoint) and `knockout_temperature` to 0.93. Both
  are left unset so the vendor's own calibration applies, and the resolved values are recorded.
* `JevK5.decide(state, question)` handles **one** typed question per call, so this adapter issues one
  call per question and records that in `questions_per_call`.
* `jevk5.prompt.answer` builds `{type, confidence, input_tokens}` plus `noul` for a noul,
  `{choice, probabilities}` for a choice and `{score, probabilities}` for a score. `confidence` is the
  largest probability, not a rescaled value.
* Questions with more than 16 options are read across several passes via `method` ("knockout" or
  "tree"). The sealed manifest's widest choice question carries 12 options, so every question here
  takes the single-pass path; the method is recorded regardless.
* `dtype` defaults to bfloat16, which a T4 does not implement natively, so fp16 is requested.

The repo is not on PyPI; it is installed from GitHub.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from runners.vendor.common import (
    VENDOR_ADAPTER_VERSION,
    VendorRunner,
    execution_provenance,
    project_vendor_answer,
    resolve_device,
    resolve_dtype,
)

ADAPTER_VERSION = VENDOR_ADAPTER_VERSION

PACKAGE_COMMIT = "main"
# The manifest's widest choice question has 12 options, under this bound.
SINGLE_PASS_OPTION_LIMIT = 16


class _JevK5Runner(VendorRunner):
    family = "jevk5"

    def _load(self) -> dict[str, Any]:
        import torch

        device, dtype = resolve_device(), resolve_dtype()
        if not device.startswith("cuda"):
            raise ValueError(
                f"{self.name} requires CUDA: JevK5's runtime captures CUDA graphs and its card "
                f"documents an H100 service path. Requested {device!r}."
            )
        from jevk5.runtime import JevK5

        # fp16 because the runtime defaults to bf16 and a T4 has no native bf16.
        torch_dtype = torch.float16 if dtype in ("float16", "fp16") else torch.bfloat16
        model = JevK5(
            source=self.default_model,
            device=device,
            dtype=torch_dtype,
            # Graphs are a latency optimisation; on a 14.6 GiB T4 the capture is not worth the
            # memory, and correctness does not depend on it.
            graphs=False,
        )
        self._model = model
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "serving": "local",
            **execution_provenance(device, "vendor-calibrated"),
            "adapter_version": ADAPTER_VERSION,
            "package": "jevk5",
            "package_commit": PACKAGE_COMMIT,
            "technique": "option-letter logits, one pass per question",
            "questions_per_call": 1,
            "temperature": getattr(model, "temperature", None),
            "knockout_temperature": getattr(model, "knockout_temperature", None),
            "method": getattr(model, "method", None),
            "single_pass_option_limit": SINGLE_PASS_OPTION_LIMIT,
            "cuda_graphs": False,
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

        answers: dict[str, Any] = {}
        calls = 0
        for qid, question in copied_questions.items():
            if not isinstance(question, Mapping):
                raise TypeError(f"{qid}: question must be a mapping")
            # decide() takes one typed question and already speaks the benchmark's
            # {type, instructions, criteria} shape.
            result = self._model.decide(copied_state, dict(question))
            calls += 1
            if not isinstance(result, Mapping):
                raise TypeError(f"{qid}: jevk5 returned a non-mapping answer")
            payload = {k: v for k, v in result.items() if k != "input_tokens"}
            answers[qid] = project_vendor_answer(payload, question, qid)
        return {
            "answers": self._validate_answer_ids(copied_questions, answers),
            "_usage": {"calls": calls},
        }


class JevK5Runner(_JevK5Runner):
    name = "jevk5"
    default_model = "alibiserikbay/JevK5"
    default_revision = "c4f7fdb3aeab5582336406e78d3bef11bf98833d"
    license_name = "apache-2.0"
