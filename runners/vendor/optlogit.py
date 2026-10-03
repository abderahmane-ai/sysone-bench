"""Option-logprob scorers for vendor checkpoints that expose no serving API.

Two panel families land here because both are plain causal language models whose contribution is
an inference method rather than a tokenizer or chat contract we can call directly:

* ``metask-jev-4b`` is a full fine-tune of ``Qwen/Qwen3.5-4B`` described by its author as a
  "calibrated typed-decision model" returning "a probability for every option in a single forward
  pass", tagged ``candidate-logit``. The vendor ships no ``prompt_builder`` module, so the exact
  instruction wording it was fine-tuned on is not recoverable from the repository.
* ``LFM2.5-350M-RLCD`` and ``LFM2.5-2.6B-RLCD`` are byte-for-byte copies of stock LiquidAI
  weights plus a parallel constrained-decoding wrapper. Their READMEs state plainly that no
  training was performed and that they do not reproduce Jev's method. The wrapper's fast path
  branches attention *and convolution* state, which the vendored PCD engine in ``runners/pcd`` does
  not implement because it targets Qwen2.5.

Both are therefore scored the same way: build one prompt per question, then read the model's own
logprob of each candidate answer string and softmax across the declared options. That needs no
vendor prompt format and no cross-option state sharing, so LFM2's convolution layers never have to
be branched -- each option is an independent forward pass over the same cached prefix.

This is a faithful implementation of "score the declared options", not a reproduction of either
vendor's optimized readout. ``prompt_style`` and ``scoring`` in run metadata record that, so a
downstream reader never mistakes these rows for the vendor's intended inference path.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from runners.vendor.common import (
    VendorRunner,
    execution_provenance,
    load_kwargs,
    place_on_device,
    resolve_device,
    resolve_dtype,
    state_to_text,
)

ADAPTER_VERSION = "1.0.0-option-logprob"


class _OptionLogprobRunner(VendorRunner):
    """Score every declared option by the model's own conditional logprob."""

    family = "causal-option-logprob"
    max_new_tokens = 0

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.default_model, revision=self.default_revision)
        model = AutoModelForCausalLM.from_pretrained(
            self.default_model, revision=self.default_revision, **load_kwargs(device, dtype)
        )
        place_on_device(model, device)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "serving": "local",
            **execution_provenance(device, dtype),
            "adapter_version": ADAPTER_VERSION,
            "scoring": "per_option_conditional_logprob",
            "prompt_style": "vendor_prompt_not_published",
        }

    # -- prompt construction -------------------------------------------------
    def _question_text(self, state: Mapping[str, Any], question: Mapping[str, Any]) -> str:
        raise NotImplementedError

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

    def _render(self, options: list[str]) -> dict[str, str]:
        return {option: f" {option}" for option in options}

    # -- scoring -------------------------------------------------------------
    def _score_options(self, prompt: str, options: Mapping[str, str]) -> dict[str, float]:
        """Return a softmax over options of each option's mean conditional logprob.

        Mean rather than sum token logprob: option strings differ in token count, and a summed
        score would systematically favour the shortest option regardless of the model.
        """
        tokenizer = self._tokenizer
        device = next(self._model.parameters()).device
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]
        out: dict[str, float] = {}
        for option, continuation in options.items():
            cont_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False).input_ids
            cont_ids = cont_ids.to(device)
            if cont_ids.shape[1] == 0:
                out[option] = 0.0
                continue
            ids = torch.cat([prompt_ids, cont_ids], dim=1)
            with torch.no_grad():
                logits = self._model(input_ids=ids).logits
            # Predict each continuation token from the position before it.
            window = logits[0, prompt_len - 1 : prompt_len - 1 + cont_ids.shape[1], :]
            logprobs = torch.log_softmax(window.float(), dim=-1)
            targets = cont_ids[0]
            picked = logprobs[torch.arange(targets.shape[0], device=device), targets]
            out[option] = float(picked.mean().item())
        finite = [value for value in out.values() if math.isfinite(value)]
        if not finite:
            raise ValueError("no finite option logprob was produced")
        top = max(finite)
        weights = {name: math.exp(value - top) for name, value in out.items() if math.isfinite(value)}
        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError("option logprobs collapsed to zero mass")
        return {name: value / total for name, value in weights.items()}

    # -- contract ------------------------------------------------------------
    def predict(
        self, state: Mapping[str, Any], questions: Mapping[str, Any], *, phase: str = "benchmark"
    ) -> dict[str, Any]:
        if phase not in ("warmup", "benchmark", "speed"):
            raise ValueError(f"invalid phase {phase!r}")
        answers: dict[str, Any] = {}
        for qid, question in questions.items():
            options = self._options(question, qid)
            prompt = self._question_text(state, question)
            probs = self._score_options(prompt, self._render(options))
            probs = {name: probs.get(name, 0.0) for name in options}
            best = max(probs, key=lambda name: (probs[name], name))
            kind = question.get("type")
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


class MetaskRunner(_OptionLogprobRunner):
    """`wayfind/metask-jev-4b-policy-mix` - full Qwen3.5-4B fine-tune, candidate-logit readout."""

    name = "metask"
    default_model = "wayfind/metask-jev-4b-policy-mix"
    default_revision = "ea20fe85b28733b1522721dec119bec50947a869"
    license_name = "apache-2.0"

    def _question_text(self, state: Mapping[str, Any], question: Mapping[str, Any]) -> str:
        kind = question.get("type")
        criteria = question.get("criteria")
        if kind == "choice" and isinstance(criteria, Mapping):
            rendered = ", ".join(f"{label}={value}" for label, value in criteria.items())
        elif kind == "score":
            levels = criteria if isinstance(criteria, (list, tuple)) else list(criteria or [])
            rendered = ", ".join(f"{index}={level}" for index, level in enumerate(levels))
        else:
            rendered = "yes, no"
        instruction = str(question.get("instructions", "")).strip()
        return (
            f"{state_to_text(state)}\n\n"
            f"Question ({kind}): {instruction}\n"
            f"Options: {rendered}\n"
            "Answer:"
        )


class _LfmRunner(_OptionLogprobRunner):
    """Shared prompt for the stock-LiquidAI RLCD wrappers.

    Both LFM checkpoints ship unchanged base weights plus a parallel constrained-decoding wrapper,
    so neither has a fine-tuned instruction format to match and the same neutral prompt serves both.
    """

    def _question_text(self, state: Mapping[str, Any], question: Mapping[str, Any]) -> str:
        options = self._options(question, str(question.get("id", "q")))
        instruction = str(question.get("instructions", "")).strip()
        return (
            f"{state_to_text(state)}\n\n"
            f"{instruction}\n"
            f"Answer with one of: {', '.join(options)}.\n"
            "Answer:"
        )


class Lfm2600Runner(_LfmRunner):
    """`monotykamary/LFM2.5-2.6B-RLCD` - stock LiquidAI LFM2.5-2.6B weights under a PCD wrapper."""

    name = "lfm2600"
    default_model = "monotykamary/LFM2.5-2.6B-RLCD"
    default_revision = "31455458983bdbdc41b69ebbcedabd0d5de299c9"
    license_name = "lfm1.0"


class Lfm350Runner(_LfmRunner):
    """`notnotsamuel/LFM2.5-350M-RLCD` - stock LiquidAI LFM2.5-350M weights under a PCD wrapper."""

    name = "lfm350"
    default_model = "notnotsamuel/LFM2.5-350M-RLCD"
    default_revision = "deb589d803d141cabd158ef55f6617b128529f36"
    license_name = "lfm1.0"
