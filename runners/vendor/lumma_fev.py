"""Lumma-Fev adapters: 0.1B and 0.6B decision checkpoints.

The card documents `model.decide(state=..., questions={...})` reachable through
`transformers.AutoModel` with `trust_remote_code=True`, and a separate `lumma-fev` package whose
`load()` picks a device. The Transformers path is used here so the adapter keeps the same loader
shape as every other vendor family and the revision stays pinned.

Answers arrive keyed by question ID with the native choice/noul/score shapes, so the shared
projection in `common` handles them unchanged.
"""

from __future__ import annotations

from runners.vendor.common import (
    VENDOR_ADAPTER_VERSION, VendorRunner, execution_provenance, load_kwargs,
    place_on_device, resolve_device, resolve_dtype,)

from collections.abc import Mapping
from typing import Any


ADAPTER_VERSION = VENDOR_ADAPTER_VERSION


class _LummaFevRunner(VendorRunner):
    family = "lumma-fev"

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
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

    def _call(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        payload = {
            qid: {
                key: value
                for key, value in question.items()
                if key not in {"qid", "max_score"}
            }
            for qid, question in questions.items()
        }
        return self._model.decide(state=dict(state), questions=payload)


class LummaFev01BRunner(_LummaFevRunner):
    name = "lumma-fev-01b"
    default_model = "FrontiersMind/Lumma-fev-0.1b"
    default_revision = "085f4705aa860a6404d6cc3ff17de8a2969ac0f4"
    license_name = "apache-2.0"


class LummaFev06BRunner(_LummaFevRunner):
    name = "lumma-fev-06b"
    default_model = "FrontiersMind/Lumma-fev-0.6b"
    default_revision = "59272e2c10506d99a464c999dcaa91b5108c3640"
    license_name = "apache-2.0"
