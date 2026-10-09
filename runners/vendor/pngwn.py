"""pngwn System One scorer - LoRA over ``Qwen/Qwen3.5-4B-Base`` with a scalar scoring head.

Unlike ``jet``, ``lev`` and ``hopper-g``, which read letter-token logits from a generative head,
this checkpoint is a *scorer*: the README states it "scores each option of a typed question
(yes/no, choice, score) in a single forward pass and softmaxes per question. No autoregressive
generation; the output space is exactly the option set supplied by the caller."

The adapter tensors confirm the shape. Its only non-layer key is::

    base_model.model.score.weight   BF16  [1, 2560]

a ``Linear(2560 -> 1)``, so the readout is: run the base model, take the final-token hidden state
of width 2560, project it through ``score`` to one scalar per option, then softmax across the
options the caller declared. One scalar per option means the vendor's fast path costs one forward
pass per option, which is what "single forward pass" refers to -- one pass per option, not one pass
per question.

The vendor fitted ``temperature 2.350`` on their validation split. It is applied here to the
option logits before the softmax. It is a single global value rather than a per-kind map like
``lev`` or ``hopper-g``, and it is applied as documented rather than refitted: refitting it here
would calibrate against the very split being measured.

Two vendor limits are enforced because exceeding them would silently change what is being scored:
the trained option cap of 16, and the training sequence length of 384.
"""

from __future__ import annotations

import math
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

ADAPTER_VERSION = "1.0.0-scalar-score-head"

#: Fitted by the vendor on their validation split; see README "temperature 2.350 (fitted on val)".
VENDOR_TEMPERATURE = 2.350
#: Trained option cap, from the README training summary.
OPTION_CAP = 16
#: Training sequence length, from the same summary.
MAX_SEQUENCE = 384


class PngwnRunner(VendorRunner):
    """`pngwn/system-one-qwen3.5-4b-scorer-v2b` - LoRA plus scalar scoring head over Qwen3.5-4B."""

    name = "pngwn"
    family = "peft-lora-scalar-scorer"
    default_model = "pngwn/system-one-qwen3.5-4b-scorer-v2b"
    default_revision = "3ec7785bf6aa1e8b0ade992c85206d46711d86c2"
    base_model = "Qwen/Qwen3.5-4B-Base"
    base_revision = "1001bb4d826a52d1f399e183466143f4da7b741b"
    license_name = "cc-by-nc-4.0"

    def _load(self) -> dict[str, Any]:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device, dtype = resolve_device(), resolve_dtype()
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, revision=self.base_revision)
        # The base must be loaded as a sequence classifier, not a causal LM. PEFT's
        # `modules_to_save: ["score", ...]` only creates a wrapper for modules that already exist
        # on the base, and a plain CausalLM has no `score`, so the adapter's
        # `base_model.model.score.weight` has nowhere to bind and is silently dropped. The
        # ForSequenceClassification head is itself named `score`, which is what the adapter expects.
        base = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, revision=self.base_revision, num_labels=1, **load_kwargs(device, dtype)
        )
        model = PeftModel.from_pretrained(
            base, self.default_model, revision=self.default_revision, torch_dtype=dtype
        )
        place_on_device(model, device)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        head = self._score_head(model)
        if head is None:
            raise ValueError(
                "adapter did not provide base_model.model.score; the scalar scoring head is the "
                "entire readout for this checkpoint, so its absence is not recoverable"
            )
        width = head.in_features
        if head.out_features != 1:
            raise ValueError(f"expected a scalar score head, found out_features={head.out_features}")
        if width != getattr(model.config, "hidden_size", width):
            raise ValueError(
                f"score head width {width} disagrees with model hidden_size "
                f"{getattr(model.config, 'hidden_size', None)}"
            )
        # The head ships in bf16 while the base may be loaded fp32; align them so the matmul does
        # not raise the dtype mismatch seen with GLiClass.
        head.to(device=device, dtype=next(model.parameters()).dtype)
        self._torch = torch
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "base_model": self.base_model,
            "base_revision": self.base_revision,
            "serving": "local",
            **execution_provenance(device, dtype),
            "adapter_version": ADAPTER_VERSION,
            "scoring": "per_option_scalar_head_softmax",
            "vendor_temperature": VENDOR_TEMPERATURE,
            "option_cap": OPTION_CAP,
            "max_sequence": MAX_SEQUENCE,
        }

    @staticmethod
    def _score_head(model: Any) -> Any:
        """Locate the scalar head on either a bare classifier or a PEFT wrapper."""
        inner = getattr(model, "base_model", None)
        inner = getattr(inner, "model", None) if inner is not None else None
        head = getattr(inner, "score", None)
        if head is None:
            head = getattr(model, "score", None)
        if head is None:
            # AutoModelForSequenceClassification wraps the real model one level deeper.
            deeper = getattr(inner, "model", None)
            head = getattr(deeper, "score", None)
        return head

    def _options(self, question: Mapping[str, Any], qid: str) -> list[str]:
        kind = question.get("type")
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, Mapping) or not criteria:
                raise ValueError(f"{qid}: choice criteria must be a non-empty mapping")
            options = [str(label) for label in criteria]
        elif kind == "score":
            options = (
                [str(level) for level in criteria]
                if isinstance(criteria, (list, tuple))
                else [str(level) for level in (criteria or [])]
            )
            if len(options) < 2:
                raise ValueError(f"{qid}: score needs at least two levels")
        elif kind == "noul":
            options = ["yes", "no"]
        else:
            raise ValueError(f"{qid}: unsupported question type {kind!r}")
        if len(options) > OPTION_CAP:
            raise ValueError(
                f"{qid}: {len(options)} options exceeds the vendor-trained cap of {OPTION_CAP}; "
                "padding past it would score outside the range the head was fitted on"
            )
        return options

    def _prompt(self, state: Mapping[str, Any], question: Mapping[str, Any], option: str) -> str:
        return (
            f"{state_to_text(state)}\n\n"
            f"{str(question.get('instructions', '')).strip()}\n\n"
            f"Answer: {option}"
        )

    def _score_one(self, prompt: str) -> float:
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        device = next(model.parameters()).device
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQUENCE,
        )
        ids = encoded.input_ids.to(device)
        mask = encoded.attention_mask.to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
        # num_labels=1, so the whole readout is this one value; the sequence classifier already
        # pools the last real token, so no hidden-state indexing is needed here.
        return float(logits.reshape(-1)[0].item())

    def predict(
        self, state: Mapping[str, Any], questions: Mapping[str, Any], *, phase: str = "benchmark"
    ) -> dict[str, Any]:
        if phase not in ("warmup", "benchmark", "speed"):
            raise ValueError(f"invalid phase {phase!r}")
        answers: dict[str, Any] = {}
        for qid, question in questions.items():
            kind = str(question.get("type"))
            options = self._options(question, qid)
            logits = {
                option: self._score_one(self._prompt(state, question, option)) / VENDOR_TEMPERATURE
                for option in options
            }
            top = max(logits.values())
            weights = {name: math.exp(value - top) for name, value in logits.items()}
            total = sum(weights.values())
            if total <= 0.0:
                raise ValueError(f"{qid}: option scores collapsed to zero mass")
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
        return {"answers": answers}
