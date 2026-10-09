"""Intern-Decision adapters: 0.8B and 2B single-token decision scorers.

This family is different in one important way. `inference.py` ships inside the checkpoint itself and
is imported from the downloaded snapshot rather than installed, so the adapter snapshots the
repository to a cache directory, adds it to `sys.path`, and imports `DecisionEngine` from there.
That import executes vendor code, which is why it happens inside the locked-down container rather
than on the host.

The response shape was verified by reading the shipped `inference.py` rather than trusting the
README: each answer carries `probabilities`, `confidence`, and `decision`, with `choice` on choice
questions, `noul` set to the `yes` probability, and `score` set to the probability-weighted expected
value plus a `legend`. The shared projection already handles the `no`/`yes` noul spelling and
drops `legend` and `source`, which the contract does not allow.

The request accepts our question objects nearly verbatim, so only our bookkeeping keys are removed.
"""

from __future__ import annotations

from runners.vendor.common import (
    VENDOR_ADAPTER_VERSION, VendorRunner, execution_provenance, resolve_device,
    resolve_dtype,)

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ADAPTER_VERSION = VENDOR_ADAPTER_VERSION


class _InternDecisionRunner(VendorRunner):
    family = "intern-decision"

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        from huggingface_hub import snapshot_download

        local = Path(
            snapshot_download(
                self.default_model,
                revision=self.default_revision,
                # inference.py is the entry point this adapter imports, and chat_template.jinja must
                # be present: the shipped engine calls apply_chat_template on the tokenizer, and the
                # pinned revision ships the template as a standalone file rather than inside
                # tokenizer_config.json, which transformers does not auto-load.
                allow_patterns=["*.json", "*.txt", "*.py", "*.safetensors", "*.model", "*.jinja"],
            )
        )
        resolved = str(local.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        self._attach_chat_template(resolved)
        from inference import DecisionEngine

        engine = DecisionEngine(checkpoint=resolved, device=device)
        self._engine = engine
        package_version = getattr(sys.modules.get("transformers"), "__version__", "unknown")
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "serving": "local",
            **execution_provenance(device, dtype),
            "adapter_version": ADAPTER_VERSION,
            "inference_module": "inference.py",
            "chat_template": "chat_template.jinja",
            "transformers_version": str(package_version),
        }

    @staticmethod
    def _attach_chat_template(root: str) -> None:
        """Point the tokenizer at the checkpoint's standalone chat template.

        The engine builds prompts with ``apply_chat_template(..., add_vision_id=True)``. Without a
        template the tokenizer raises before any inference happens, and the pinned revision keeps
        ``chat_template.jinja`` as its own file. Loading it here keeps the fix inside the adapter
        instead of editing the vendored inference module or the checkpoint.
        """
        from transformers import AutoTokenizer

        template = Path(root) / "chat_template.jinja"
        if not template.is_file():
            raise FileNotFoundError(f"checkpoint has no chat_template.jinja: {template}")
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        if getattr(tokenizer, "chat_template", None):
            return
        tokenizer.chat_template = template.read_text(encoding="utf-8")

    def _call(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        request = {
            "state": dict(state),
            "questions": {
                qid: {
                    key: value
                    for key, value in question.items()
                    if key not in {"qid", "max_score"}
                }
                for qid, question in questions.items()
            },
        }
        return self._engine.predict(request)


class InternDecision08BRunner(_InternDecisionRunner):
    name = "intern-decision-08b"
    default_model = "internlm/Intern-Decision-0.8B"
    default_revision = "85a0cc5a99d67ea8d56dfe98115689212867171d"
    license_name = "apache-2.0"


class InternDecision2BRunner(_InternDecisionRunner):
    name = "intern-decision-2b"
    default_model = "internlm/Intern-Decision-2B"
    default_revision = "8797836c65fc91a2435b1fb6850b5f0aabd75cc3"
    license_name = "apache-2.0"


class InternDecision4BRunner(_InternDecisionRunner):
    name = "intern-decision-4b"
    default_model = "internlm/Intern-Decision-4B"
    default_revision = "0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd"
    license_name = "apache-2.0"
