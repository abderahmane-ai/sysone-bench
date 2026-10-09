"""MoJev adapter.

MoJev is the only model in this panel with a native abstention capability, and its answer shape is
built by the vendor's own `build_answer`, which emits `choice` with a label-keyed `probabilities`
map, `noul` as a bare P(yes), and `score` as a weighted expected level plus a legend. All three are
already handled by the shared projection, so this adapter is mostly about loading.

Loading is the part that needs care. The checkpoint ships `modeling.py` behind `trust_remote_code`
and exposes a `PackedScorer` whose `forward()` consumes an already-packed batch. That packing is not
something this adapter reimplements: the `mojev` package's `Engine` owns the collator, the candidate
sort, and the answer assembly, and the adapter calls `Engine.answer(state, questions)` directly,
which is the same entry point its HTTP server uses. Every question in a state is scored in ONE
forward pass, so usage is counted per state rather than per question.

`Engine.answer` returns `(answers, usage)`; the answers are wired in the vendor's own shape and the
adapter projects them onto the benchmark contract, dropping `legend` and `source`.

Licence: the card states the code is MIT but that the Qwen base licence applies to the checkpoint,
and the repository is gated. `license_name` records both facts rather than choosing one.

Abstention: the adapter does not synthesise or suppress abstentions. If the vendor reports a deferred
or abstained decision, `build_answer` still returns a distribution over the declared options, and
that distribution is recorded as-is. No abstention is projected onto the argmax here; if a future
revision of the vendor API exposes an explicit abstention signal it is surfaced rather than erased.
"""

from __future__ import annotations

from runners.vendor.common import (
    VENDOR_ADAPTER_VERSION, VendorRunner, execution_provenance, resolve_device,
    resolve_dtype, state_to_text,)

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ADAPTER_VERSION = VENDOR_ADAPTER_VERSION
PACKAGE_COMMIT = "a74d58cd19ec573e83e8e27f9fecd837b8d830fb"


class MojevRunner(VendorRunner):
    name = "mojev"
    family = "mojev"
    default_model = "MoLeMo-Lab/mojev"
    default_revision = "0c8695b6252f4205907433d4e196a94f032e60c3"
    license_name = "mit (code); Qwen base licence applies to the checkpoint; gated repo"

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                self.default_model,
                revision=self.default_revision,
                # modeling.py/schema.py ship in the gated repo and provide the scorer.
                allow_patterns=["*.json", "*.txt", "*.py", "*.safetensors", "*.jinja"],
            )
        ).resolve()
        resolved = str(root)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        # The vendor's own engine: it owns candidate sorting, the packed collator, and answer
        # assembly, so nothing here re-implements the scorer's input format.
        from mojev.serve import Engine

        engine = Engine(checkpoint=resolved, device=device)
        self._engine = engine
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "serving": "local",
            **execution_provenance(device, dtype),
            "adapter_version": ADAPTER_VERSION,
            "package": "mojev",
            "package_commit": PACKAGE_COMMIT,
            "questions_per_call": "state",
        }

    def _call(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from runners.vendor.common import state_to_text

        payload = {
            qid: {
                key: value
                for key, value in question.items()
                if key not in {"qid", "max_score"}
            }
            for qid, question in questions.items()
        }
        # state_to_text keeps the field labels our sealed questions refer to by name.
        answers, usage = self._engine.answer(state_to_text(state), payload)
        return {"answers": answers, "usage": usage}
