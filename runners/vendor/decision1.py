"""Decision 1.0 adapters: Kai, Lex, Eos, and Sol.

One repository family, one documented call. The model card exposes `system_one(state=..., questions=...)`
through `trust_remote_code=True` and states that nothing is generated: the same System One request
and response bodies as the hosted runtime.

The repository ships its own inference code, so the loader pins the repository revision and records
the resolved identity. Each checkpoint is a separate runner name so a run's metadata identifies the
exact model rather than a family.
"""

from __future__ import annotations

import os

from runners.vendor.common import (
    VENDOR_ADAPTER_VERSION, VendorRunner, execution_provenance, load_kwargs,
    place_on_device, resolve_device, resolve_dtype,)

from collections.abc import Mapping
from typing import Any


ADAPTER_VERSION = VENDOR_ADAPTER_VERSION


class _Decision1Runner(VendorRunner):
    family = "decision-1.0"

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        from transformers import AutoModel

        # This family's remote code owns its numerics and refuses an explicit dtype: it raises
        # "Decision 1.0 numerics are fixed (FP32 encoders; decoders BF16 on a GPU with an FP32
        # head, FP32 on CPU); pass None or 'auto'". It already selects GPU-appropriate precision on
        # its own, so a host-level dtype request has nothing to add and would only break the load.
        # `dtype` is therefore forced back to 'auto' and the resolved value is reported as "vendor".
        # A 9B panel model is ~16.8 GiB in bf16 against 14.56 GiB usable on one T4, so it only
        # loads with both cards visible. Sharding is an execution mode of this same model, not a
        # separate panel entry, so it is selected by the environment rather than a second registry
        # key that would look like an extra model. device_map reintroduces the double-allocation
        # behaviour load_kwargs documents, so low_cpu_mem_usage streams the weights instead.
        shard = os.environ.get("SYSONE_BENCH_SHARD") == "1"
        if shard:
            model = AutoModel.from_pretrained(
                self.default_model,
                revision=self.default_revision,
                trust_remote_code=True,
                dtype="auto",
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModel.from_pretrained(
                self.default_model,
                revision=self.default_revision,
                trust_remote_code=True,
                **load_kwargs(device, "auto"),
            )
            place_on_device(model, device)
        model.eval()
        self._model = model
        placement = getattr(model, "hf_device_map", None) if shard else None
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "serving": "local",
            **execution_provenance(device, "vendor-managed"),
            "dtype_requested": dtype,
            "adapter_version": ADAPTER_VERSION,
            **(
                {
                    "sharding": "device_map=auto across all visible GPUs",
                    "low_cpu_mem_usage": True,
                    "single_device": False,
                    "precision_confound": False,
                    "hf_device_map": {str(k): str(v) for k, v in (placement or {}).items()},
                }
                if shard
                else {}
            ),
        }

    def _call(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        # The vendor accepts our question objects directly; qid is our bookkeeping key and the
        # vendor keys its own answers by question name, so it is stripped to keep payloads clean.
        payload = {
            qid: {
                key: value
                for key, value in question.items()
                if key not in {"qid", "max_score"}
            }
            for qid, question in questions.items()
        }
        return self._model.system_one(state=dict(state), questions=payload)


class DecisionKaiRunner(_Decision1Runner):
    name = "decision-kai"
    default_model = "llm-semantic-router/Decision-1.0-Kai"
    default_revision = "69aef4060cafad7e8bf812e0014da8097b30bf69"
    license_name = "apache-2.0"


class DecisionLexRunner(_Decision1Runner):
    name = "decision-lex"
    default_model = "llm-semantic-router/decision-1.0-lex"
    default_revision = "1f9750a75ae9d32825e7f29ae44c1e1406676884"
    license_name = "apache-2.0"


class DecisionEosRunner(_Decision1Runner):
    name = "decision-eos"
    default_model = "llm-semantic-router/Decision-1.0-Eos-0.8B"
    default_revision = "bbdc2221d0ecb1a7929be9f5822bfc4394a4a1fb"
    license_name = "apache-2.0"


class DecisionSolRunner(_Decision1Runner):
    name = "decision-sol"
    default_model = "llm-semantic-router/Decision-1.0-Sol"
    default_revision = "fc210c8fcc7f1b86519aac34be810950b582e35e"
    license_name = "apache-2.0"


class DecisionNoxRunner(_Decision1Runner):
    name = "decision-nox"
    default_model = "llm-semantic-router/Decision-1.0-Nox"
    default_revision = "7f65e1db2c55a11ab8b558a8027c8883752f192d"
    license_name = "apache-2.0"


class DecisionLux9BRunner(_Decision1Runner):
    name = "decision-lux-9b"
    default_model = "llm-semantic-router/Decision-1.0-Lux-9B"
    default_revision = "2064c84daf599447f840719ef8e86db152a56851"
    license_name = "apache-2.0"
