"""Decider-4B adapter.

`decider/` ships inside the pinned repository rather than on PyPI, so the adapter snapshots the
revision and puts that snapshot on ``sys.path``, exactly as the Julia adapter does.

The interesting part is ``Decider.system_one``: it already speaks TypeSafe's System One request
shape and returns ``{"model", "answers", "usage"}`` where each answer is built by
``decider.systemone.format_answer``. That means this family owns its own answer assembly, so the
adapter passes the benchmark's questions through unchanged and projects the vendor's answers rather
than re-deriving them.

Read from the pinned source, and relied on here:

* ``Decider(path, device=..., dtype=..., temperature=None)`` takes device and dtype directly.
* ``system_one(state, questions, independent=True)`` returns answers keyed by question id.
* ``format_answer`` spells the fields this adapter reads:
  ``noul`` -> ``{"type", "noul"}``;
  ``choice`` -> ``{"type", "choice", "choice", "confidence", "x_p_max", "certainty", "probabilities"}``;
  ``score`` -> ``{"type", "score", "score", "legend", "probabilities"}``.
* ``confidence`` for a choice is TypeSafe's rescaled ``(n * p_max - 1) / (n - 1)``, not the raw
  probability; ``x_p_max`` carries the raw one. The shared projection prefers ``x_p_max``, so the
  recorded confidence is the probability and the rescaled value is kept separately.

The per-type temperatures (choice 1.110, noul 1.560, score 1.287) live in ``decider_config.json``
and are loaded by the package itself, so no temperature is passed from here.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
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

# The package ships in-repo and imports sibling modules by top-level name, so the snapshot root (not
# the `decider/` directory itself) goes on the path.
_ALLOW_PATTERNS = ["*.json", "*.py", "*.safetensors", "*.model", "*.txt", "*.jinja", "decider/**"]


class _DeciderRunner(VendorRunner):
    family = "decider"

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                self.default_model,
                revision=self.default_revision,
                allow_patterns=_ALLOW_PATTERNS,
            )
        ).resolve()
        resolved = str(root)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        from decider.infer import Decider

        # Decider takes device and dtype itself. dtype=None means "use the checkpoint's own", which
        # is what this family documents (bf16 weights); an fp16 override would change numerics the
        # card calibrated against, so the resolved value is reported rather than imposed.
        #
        # use_graphs=False selects the eager path. The CUDA-graph path captures per (batch, length)
        # shape, and on a Tesla T4 torch.compile cannot use bf16 GEMM autotuning ("Tesla T4 does not
        # support bfloat16 compilation natively"), so every distinct prompt length triggers a
        # recompile. Observed on decider-4b: 54 decisions then zero progress with the process pinned at
        # ~91% CPU recompiling and the GPU at 0%. Eager is slower per call and does not recompile.
        model = Decider(resolved, device=device, dtype=None, use_graphs=False)
        self._model = model
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "serving": "local",
            **execution_provenance(device, "vendor-managed"),
            "dtype_requested": dtype,
            "adapter_version": ADAPTER_VERSION,
            "package": "decider (in-repository)",
            "package_source": self.default_model,
            "technique": "lettered-slot readout, one forward pass",
            "questions_per_call": "all",
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

        # system_one renders the state itself and takes the benchmark's question shape directly, so
        # the questions are forwarded untouched. independent=True keeps every answer independent of
        # the others, which is what this benchmark measures.
        response = self._model.system_one(copied_state, copied_questions, independent=True)
        results = response.get("answers") if isinstance(response, Mapping) else None
        if not isinstance(results, Mapping) or set(results) != set(copied_questions):
            raise ValueError(
                f"decider returned answers for "
                f"{sorted(results) if isinstance(results, Mapping) else '?'}, "
                f"expected {sorted(copied_questions)}"
            )

        answers: dict[str, Any] = {}
        for qid, question in copied_questions.items():
            result = results[qid]
            if not isinstance(result, Mapping):
                raise TypeError(f"{qid}: decider returned a non-mapping answer")
            answers[qid] = project_vendor_answer(result, question, qid)
        usage = response.get("usage") if isinstance(response, Mapping) else None
        return {
            "answers": self._validate_answer_ids(copied_questions, answers),
            "_usage": {
                "calls": 1,
                "input_tokens": (usage or {}).get("input_tokens") if isinstance(usage, Mapping) else None,
                "output_tokens": 0,
            },
        }


class Decider4BRunner(_DeciderRunner):
    name = "decider-4b"
    default_model = "Mapika/decider-4b"
    default_revision = "eb5fbdfc9448473ec25e399882912863afbdb70e"
    license_name = "apache-2.0"


class Decider2BRunner(_DeciderRunner):
    name = "decider-2b"
    default_model = "Mapika/decider-2b"
    default_revision = "533964dae8be954c5b5e19fa4948e48408094c1e"
    license_name = "apache-2.0"
