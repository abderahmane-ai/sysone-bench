"""lev - LoRA adapter over ``Qwen/Qwen3.5-4B`` with lettered-label logits and a keyed calibration map.

Structurally the same readout as ``jet`` and ``hopper-g``: one forward pass per question, options
rendered as ``A``/``B``/``C``, decision taken from the letter-token logits at the final position.
What differs is the calibration map. ``calibration.json`` is keyed by *question kind and letter*
rather than by kind alone::

    "noul:A":        2.3333
    "score:A":       2.8033
    "choice:A":      1.7667
    "choice:B":      0.6333
    "choice:A:small" 1.7898   <- size-suffixed variants also present

The map is applied per option using that option's own letter, so a question's distribution is
tempered letter-by-letter rather than with a single scalar. Keys ending in a size segment are
retained but not selected automatically: nothing in the sealed manifest marks a case with the
size tier the vendor used when fitting, so choosing one would be a guess. Resolution therefore
prefers an exact ``kind:letter`` key and falls back to ``kind:letter:*`` only if unambiguous, and
records which key was used per question kind in run metadata.
"""

from __future__ import annotations

import json
import math
import urllib.request
from typing import Any, Mapping

from runners.vendor.common import execution_provenance, load_kwargs, place_on_device, resolve_device, resolve_dtype  # noqa: F401
from runners.vendor.hopper import LETTERS, HopperGRunner

ADAPTER_VERSION = "1.0.0-lev-lettered-calibration"


class LevRunner(HopperGRunner):
    """`interfaze-ai/lev` - LoRA adapter over `Qwen/Qwen3.5-4B`."""

    name = "lev"
    family = "peft-lora-option-letter"
    default_model = "interfaze-ai/lev"
    default_revision = "7bdc748dffebd85b57ee0dbea8f994c6354fed31"
    base_model = "Qwen/Qwen3.5-4B"
    license_name = "apache-2.0"

    def _temperatures(self) -> dict[str, float]:
        url = (
            f"https://huggingface.co/{self.default_model}/resolve/"
            f"{self.default_revision}/calibration.json"
        )
        request = urllib.request.Request(url)
        try:
            with open("/root/.cache/huggingface/token", encoding="utf-8") as handle:
                request.add_header("Authorization", f"Bearer {handle.read().strip()}")
        except OSError:
            pass
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
        temperatures = payload.get("temperatures")
        if not isinstance(temperatures, Mapping) or not temperatures:
            raise ValueError("calibration.json carries no temperature map")
        resolved: dict[str, float] = {}
        for key, value in temperatures.items():
            number = float(value)
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"calibration temperature for {key!r} is not usable")
            resolved[str(key)] = number
        return resolved

    def _temperature_for(self, kind: str, index: int) -> tuple[float, str]:
        """Resolve the calibration entry for one option, preferring an exact key.

        Returns the temperature and the key it came from, so the run row can show which entries were
        actually applied instead of implying the whole map was used.
        """
        letter = LETTERS[index]
        exact = f"{kind}:{letter}"
        if exact in self._temperatures_by_kind:
            return self._temperatures_by_kind[exact], exact
        prefixed = [key for key in self._temperatures_by_kind if key.startswith(exact + ":")]
        if len(prefixed) == 1:
            return self._temperatures_by_kind[prefixed[0]], prefixed[0]
        if len(prefixed) > 1:
            raise ValueError(
                f"calibration keys {prefixed} are ambiguous for {exact!r}; the sealed manifest "
                "carries no size tier, so refusing beats guessing"
            )
        # The vendor map only covers the letters its own fitting exercised, typically A and B. A
        # three-option choice legitimately reaches C, and refusing there would make the model
        # unrunnable over the whole suite rather than over one question. Fall back to the mean of
        # the same-kind entries, which keeps every option on one comparable scale, and record the
        # substitution so the run row shows the map was not applied verbatim.
        same_kind = [value for key, value in self._temperatures_by_kind.items() if key.startswith(f"{kind}:")]
        if not same_kind:
            raise ValueError(f"calibration.json has no entry for kind {kind!r} at all")
        mean = sum(same_kind) / len(same_kind)
        return mean, f"{kind}:<mean of {len(same_kind)}>"

    def _load(self) -> dict[str, Any]:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device, dtype = resolve_device(), resolve_dtype()
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        base = AutoModelForCausalLM.from_pretrained(self.base_model, **load_kwargs(device, dtype))
        model = PeftModel.from_pretrained(
            base, self.default_model, revision=self.default_revision, torch_dtype=dtype
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
            "serving": "local",
            **execution_provenance(device, dtype),
            "adapter_version": ADAPTER_VERSION,
            "scoring": "option_letter_logits",
            "calibration_map": "vendor_kind_letter_temperatures",
            "calibration_keys_available": sorted(self._temperatures_by_kind),
        }

    def predict(
        self, state: Mapping[str, Any], questions: Mapping[str, Any], *, phase: str = "benchmark"
    ) -> dict[str, Any]:
        if phase not in ("warmup", "benchmark", "speed"):
            raise ValueError(f"invalid phase {phase!r}")
        answers: dict[str, Any] = {}
        applied: set[str] = set()
        for qid, question in questions.items():
            kind = str(question.get("type"))
            options = self._options(question, qid)
            if len(options) > len(LETTERS):
                raise ValueError(f"{qid}: {len(options)} options exceeds available letters")
            logits = self._raw_letter_logits(self._prompt(state, question, options), options)
            tempered: dict[str, float] = {}
            for index, option in enumerate(options):
                temperature, key = self._temperature_for(kind, index)
                applied.add(key)
                tempered[option] = logits[option] / temperature
            top = max(tempered.values())
            weights = {name: math.exp(value - top) for name, value in tempered.items()}
            total = sum(weights.values())
            if total <= 0.0:
                raise ValueError(f"{qid}: tempered letter logits collapsed to zero mass")
            probs = {name: weight / total for name, weight in weights.items()}
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
        self._calibration_keys_used = sorted(applied)
        return {"answers": answers}
