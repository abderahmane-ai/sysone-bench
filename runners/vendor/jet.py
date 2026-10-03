"""Jet adapter.

Jet ships its own runtime in the pinned repository (`jet.py`, `format.py`, `inference.py`,
`runtime.py`) rather than as an installable package, so the adapter snapshots the revision and puts
the snapshot root on ``sys.path``. The runtime's own module docstring states "Linux + NVIDIA CUDA,
Python 3.12", which is the only host this is verified on.

`Jet.decide(state, questions)` already takes the benchmark's question shape:

* ``Question.from_dict`` reads ``{type, instructions, criteria}``, exactly the manifest's shape, and
  validates it: a choice needs 2-255 criteria, a score an ordered list of levels, and a noul may
  omit ``criteria``.
* ``summarize`` builds the answers this adapter reads: a choice gets ``{type, choice,
  probabilities, confidence}`` where ``confidence`` is 1 - normalised entropy; a score gets
  ``{type, score, level, probabilities, confidence}``; a noul gets the ``true`` share.
* Per-type temperatures come from ``calibration.json`` in the snapshot and are applied by the
  runtime, so nothing is imposed from here.

The answer is keyed by question id, and ``decide`` returns ``{'model', 'answers'}``.
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

# The runtime imports its siblings as top-level modules (`from format import ...`), so the snapshot
# root goes on the path, not the file's own directory.
_ALLOW_PATTERNS = ["*.json", "*.py", "*.safetensors", "*.model", "*.txt", "*.jinja"]


class _JetRunner(VendorRunner):
    family = "jet"

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        if not device.startswith("cuda"):
            # runtime.load_model moves the model onto CUDA itself; there is no CPU path to select and
            # failing here is better than loading a 4B model and then failing on the first forward.
            raise ValueError(
                f"{self.name} requires a CUDA device: its shipped runtime loads with "
                f"device_map='cuda' and has no CPU branch. Requested {device!r}."
            )
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
        from jet import Jet

        model = Jet(resolved)
        self._model = model
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "serving": "local",
            **execution_provenance(device, "vendor-managed"),
            "dtype_requested": dtype,
            "adapter_version": ADAPTER_VERSION,
            "package": "jet (in-repository)",
            "package_source": self.default_model,
            "technique": "lettered-label logits, one forward pass per question",
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

        # Jet reads the benchmark's question shape directly and validates it, so the questions are
        # forwarded untouched.
        response = self._model.decide(copied_state, copied_questions)
        results = response.get("answers") if isinstance(response, Mapping) else None
        if not isinstance(results, Mapping) or set(results) != set(copied_questions):
            raise ValueError(
                f"jet returned answers for "
                f"{sorted(results) if isinstance(results, Mapping) else '?'}, "
                f"expected {sorted(copied_questions)}"
            )

        answers: dict[str, Any] = {}
        for qid, question in copied_questions.items():
            result = results[qid]
            if not isinstance(result, Mapping):
                raise TypeError(f"{qid}: jet returned a non-mapping answer")
            answers[qid] = project_vendor_answer(result, question, qid)
        # decide() loops per question internally, so calls track the question count.
        return {
            "answers": self._validate_answer_ids(copied_questions, answers),
            "_usage": {"calls": len(copied_questions)},
        }


class Jet4BRunner(_JetRunner):
    name = "jet"
    default_model = "michaljach/jet"
    default_revision = "fbc3d2daa679e0d4bd9f99c9912b6496d5a41f0a"
    license_name = "apache-2.0"