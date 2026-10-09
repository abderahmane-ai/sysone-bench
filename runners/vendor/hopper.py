"""Hopper (G) - PEFT LoRA on Qwen3.5-4B, option-letter readout with a per-kind calibration map.

The adapter repository ships three files that together define the whole contract:
``adapter_config.json`` (LoRA r=16, alpha=32 over Qwen3.5-4B), ``adapter_model.safetensors``, and
``hopper.json`` -- a ``per_kind`` temperature map fitted by the author on their own held-out
JevBench-style look-alike items, explicitly not on any JevBench item or file.

The README states the readout: "answers typed decision questions in one forward pass by reading the
probability of each option letter". So options are rendered as ``A``/``B``/``C`` labels, and the
decision is the argmax over the letter-token logits at the final position, tempered by the map for
that question kind. That is the vendor's own path, not an approximation of it.

The Decision Index row measures Hopper (G) 1.2, which the model card names as short revision
``d60a1d6``. That is resolved here to the full immutable
``d60a1d6ca3f3fa25e5daddea16b8ba2b431ebcee`` rather than the moving repository head
(``387b2b3995ff...``, Hopper (G) 1.3), so the recorded revision and the measured model agree.
"""

from __future__ import annotations

import json
import math
import urllib.request
from typing import Any, Mapping

from runners.vendor.common import (
    VendorRunner,
    execution_provenance,
    load_kwargs,
    place_on_device,
    resolve_device,
    resolve_dtype,
    state_to_text,
)

ADAPTER_VERSION = "1.0.0-option-letter"

#: Uppercase option labels, in order. A 9-question max keeps every prompt within one letter.
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class HopperGRunner(VendorRunner):
    """`HopitAI/hopper-g` - LoRA adapter over `Qwen/Qwen3.5-4B`."""

    name = "hopper-g"
    family = "peft-lora-option-letter"
    default_model = "HopitAI/hopper-g"
    default_revision = "d60a1d6ca3f3fa25e5daddea16b8ba2b431ebcee"
    base_model = "Qwen/Qwen3.5-4B"
    base_revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    license_name = "research-and-demo"

    def _temperatures(self) -> dict[str, float]:
        """Read the vendor's per-kind temperature map, failing closed if it is absent.

        The map is part of the model, not an optional refinement: without it the reported
        probabilities are the untempered logits and every calibration metric on this row would be
        wrong. So a missing or malformed map is a hard error rather than a silent default of 1.0.
        """
        url = (
            f"https://huggingface.co/{self.default_model}/resolve/"
            f"{self.default_revision}/hopper.json"
        )
        request = urllib.request.Request(url)
        token_path = "/root/.cache/huggingface/token"
        try:
            with open(token_path, encoding="utf-8") as handle:
                request.add_header("Authorization", f"Bearer {handle.read().strip()}")
        except OSError:
            pass
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
        temperatures = payload.get("temperatures")
        if not isinstance(temperatures, Mapping) or not temperatures:
            raise ValueError("hopper.json carries no per-kind temperature map")
        resolved: dict[str, float] = {}
        for kind, value in temperatures.items():
            number = float(value)
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"hopper.json temperature for {kind!r} is not usable")
            resolved[str(kind)] = number
        return resolved

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.base_model, revision=self.base_revision)
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model, revision=self.base_revision, **load_kwargs(device, dtype)
        )
        model = PeftModel.from_pretrained(
            base,
            self.default_model,
            revision=self.default_revision,
            torch_dtype=dtype,
        )
        place_on_device(model, device)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._temperatures_by_kind = self._temperatures()
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "base_model": self.base_model,
            "base_revision": self.base_revision,
            "serving": "local",
            **execution_provenance(device, dtype),
            "adapter_version": ADAPTER_VERSION,
            "scoring": "option_letter_logits",
            "calibration_map": "vendor_per_kind_temperatures",
            "calibration_temperatures": self._temperatures_by_kind,
        }

    def _options(self, question: Mapping[str, Any], qid: str) -> list[str]:
        kind = question.get("type")
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, Mapping) or not criteria:
                raise ValueError(f"{qid}: choice criteria must be a non-empty mapping")
            return [str(label) for label in criteria]
        if kind == "score":
            levels = (
                [str(level) for level in criteria]
                if isinstance(criteria, (list, tuple))
                else [str(level) for level in (criteria or [])]
            )
            if len(levels) < 2:
                raise ValueError(f"{qid}: score needs at least two levels")
            return levels
        if kind == "noul":
            return ["yes", "no"]
        raise ValueError(f"{qid}: unsupported question type {kind!r}")

    def _prompt(self, state: Mapping[str, Any], question: Mapping[str, Any], options: list[str]) -> str:
        if len(options) > len(LETTERS):
            raise ValueError(f"{len(options)} options exceeds the {len(LETTERS)} available letters")
        rendered = "\n".join(f"{LETTERS[index]}. {option}" for index, option in enumerate(options))
        return (
            f"{state_to_text(state)}\n\n"
            f"{str(question.get('instructions', '')).strip()}\n\n"
            f"{rendered}\n"
            "Answer with a single letter.\n"
            "Answer:"
        )

    def _raw_letter_logits(self, prompt: str, options: list[str]) -> dict[str, float]:
        """Untempered logit per option, read at the position that would generate its letter."""
        import torch

        tokenizer = self._tokenizer
        device = next(self._model.parameters()).device
        encoded = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = self._model(input_ids=encoded).logits[0, -1, :].float()
        picked: dict[str, float] = {}
        for index, option in enumerate(options):
            # The letter is read as it would be generated: a standalone token, no leading space.
            ids = tokenizer(LETTERS[index], add_special_tokens=False).input_ids
            if not ids:
                raise ValueError(f"tokenizer produced no id for option letter {LETTERS[index]!r}")
            picked[option] = float(logits[ids[-1]].item())
        return picked

    def _letter_distribution(self, prompt: str, options: list[str], temperature: float) -> dict[str, float]:
        scaled = {name: value / temperature for name, value in self._raw_letter_logits(prompt, options).items()}
        top = max(scaled.values())
        weights = {name: math.exp(value - top) for name, value in scaled.items()}
        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError("option letter logits collapsed to zero mass")
        return {name: weight / total for name, weight in weights.items()}

    def predict(
        self, state: Mapping[str, Any], questions: Mapping[str, Any], *, phase: str = "benchmark"
    ) -> dict[str, Any]:
        if phase not in ("warmup", "benchmark", "speed"):
            raise ValueError(f"invalid phase {phase!r}")
        answers: dict[str, Any] = {}
        for qid, question in questions.items():
            kind = str(question.get("type"))
            options = self._options(question, qid)
            if kind not in self._temperatures_by_kind:
                raise ValueError(f"hopper.json has no temperature for question kind {kind!r}")
            probs = self._letter_distribution(
                self._prompt(state, question, options), options, self._temperatures_by_kind[kind]
            )
            best = max(probs, key=lambda name: (probs[name], name))
            if kind == "noul":
                answers[qid] = {"type": "noul", "noul": float(probs["yes"])}
            elif kind == "score":
                answers[qid] = {
                    "type": "score",
                    "score": float(options.index(best)),
                    "confidence": float(probs[best]),
                }
            else:
                answers[qid] = {
                    "type": "choice",
                    "choice": best,
                    "probabilities": probs,
                    "confidence": float(probs[best]),
                }
        return {"answers": answers}
