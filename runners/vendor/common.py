"""Shared System One normalization for vendor decision-model adapters.

The vendor families disagree on nearly every detail a System One client would notice: whether a
noul answer is a bare ``P(true)`` scalar or a ``{"false": ..., "true": ...}`` distribution, whether
confidence is called ``confidence`` or ``max_probability`` or is absent, whether a score carries a
legend. This module absorbs those differences at the adapter edge so the orchestrator, the
validator, and the report see one shape.

Normalization is strict on purpose. A missing answer, an unknown question type, a non-positive
probability total, or a choice label that is not in the question's criteria raises rather than
being repaired, because a silently invented answer would corrupt the benchmark.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from runners.base import BaseRunner, renormalize_choice_probabilities

VENDOR_ADAPTER_VERSION = "1"

# Device and precision are process-level decisions, not per-model ones, so they are resolved once
# from the environment and then recorded verbatim in every run's provenance block. The defaults
# reproduce the original CPU/fp32 behaviour exactly, which is what every published run recorded.
_DEVICE_ENV = "SYSONE_BENCH_DEVICE"
_DTYPE_ENV = "SYSONE_BENCH_DTYPE"
_DTYPE_ALIASES = {
    "fp32": "float32",
    "float32": "float32",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}


def resolve_device() -> str:
    """Return the torch device string the adapters should load onto.

    Asking for ``cuda`` when torch cannot see a GPU is a hard error rather than a silent fallback to
    CPU: a run that silently changed host would otherwise be recorded as the wrong hardware.
    """
    requested = os.environ.get(_DEVICE_ENV, "cpu").strip().lower() or "cpu"
    if requested == "cpu":
        return "cpu"
    if not requested.startswith("cuda"):
        raise ValueError(f"{_DEVICE_ENV} must be 'cpu' or 'cuda...', got {requested!r}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency of every adapter
        raise ValueError(f"{_DEVICE_ENV}=cuda requires torch, which is not installed") from exc
    if not torch.cuda.is_available():
        raise ValueError(
            f"{_DEVICE_ENV}=cuda was requested but torch.cuda.is_available() is False; "
            "refusing to silently run on CPU and mislabel the host"
        )
    return requested


def resolve_dtype() -> str:
    """Return the torch dtype name the adapters should load weights in.

    ``auto`` is the honest default for the CPU runs already published: those checkpoints were loaded
    in whatever dtype their config declares, and overriding that would change the numbers.
    """
    requested = os.environ.get(_DTYPE_ENV, "auto").strip().lower() or "auto"
    if requested == "auto":
        return "auto"
    try:
        return _DTYPE_ALIASES[requested]
    except KeyError:
        raise ValueError(
            f"{_DTYPE_ENV} must be one of {sorted(_DTYPE_ALIASES)} or 'auto', got {requested!r}"
        ) from None


def load_kwargs(device: str, dtype: str) -> dict[str, Any]:
    """Build the shared ``from_pretrained`` kwargs, omitting no-op values.

    Only ``dtype`` is passed. Weights are always loaded without ``device_map`` and then moved by
    :func:`place_on_device`, because ``device_map`` makes transformers warm up the allocator by
    allocating a second full-size fp16 buffer before the first copy is freed. That OOMs on a 14.6 GiB
    T4 even for a 9.3 GB model. Loading then moving also keeps one code path for both hosts, so the
    only difference between a CPU and a GPU run is where the same weights end up.
    """
    kwargs: dict[str, Any] = {}
    if dtype != "auto":
        kwargs["dtype"] = dtype
    return kwargs


def place_on_device(model: Any, device: str) -> Any:
    """Move an already-constructed model onto ``device`` when it is not there already.

    Third-party loaders that do not accept a device argument are handled here, after construction,
    so that a CPU-only default never leaks into a GPU run.
    """
    if device == "cpu":
        return model
    place = getattr(model, "to", None)
    if place is None:
        raise ValueError(f"cannot move a {type(model).__name__} onto {device}: no .to() method")
    return place(device)


def execution_provenance(device: str, dtype: str) -> dict[str, str]:
    """The execution facts every adapter must copy verbatim into its provenance block."""
    return {"device": device, "dtype": dtype}

# Vendors spell top-label confidence four different ways across this model set. Probed in order.
_CONFIDENCE_KEYS = ("confidence", "max_probability", "certainty", "top_probability")

# Vendor noul answers arrive either as a bare P(true) or as a two-key distribution whose true side
# may be spelled "true" or as a yes/no candidate ID.
_NOUL_TRUE_KEYS = frozenset({"true", "yes", "1", "t", "y"})
_NOUL_FALSE_KEYS = frozenset({"false", "no", "0", "f", "n"})


def state_to_text(state: object) -> str:
    """Render a state mapping as labelled text.

    Vendors that take a string state need the field names preserved, because the sealed questions
    refer to them directly ("Does `message` suggest the customer may leave?"). Dropping the labels
    would make those questions unanswerable rather than merely awkward.
    """
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping):
        return "\n".join(f"{key}: {value}" for key, value in state.items())
    if isinstance(state, Sequence) and not isinstance(state, (str, bytes, bytearray)):
        return "\n".join(str(item) for item in state)
    return str(state)


def _criteria_labels(question: Mapping[str, Any], qid: str) -> tuple[str, ...]:
    criteria = question.get("criteria")
    if question.get("type") == "choice":
        if not isinstance(criteria, Mapping) or not criteria:
            raise ValueError(f"{qid}: choice criteria must be a non-empty mapping")
        labels = tuple(str(label) for label in criteria)
    else:
        if isinstance(criteria, Sequence) and not isinstance(criteria, (str, bytes, bytearray)):
            labels = tuple(str(level) for level in criteria)
        elif isinstance(criteria, Mapping):
            labels = tuple(str(level) for level in criteria)
        else:
            raise ValueError(f"{qid}: score criteria must be an ordered sequence of levels")
    if not labels:
        raise ValueError(f"{qid}: criteria resolved to no labels")
    return labels


def _probability(value: object, qid: str, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{qid}: {label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{qid}: {label} must be finite and nonnegative")
    return number


def _probability_map(raw: object, qid: str, labels: Sequence[str]) -> dict[str, float]:
    """Map a vendor probability payload onto the question's exact label set.

    Accepts a mapping keyed by label or by index, and a bare sequence aligned to the labels. Keys
    the question did not declare are dropped rather than added, because the validator rejects an
    answer whose probability labels differ from its choice labels.
    """
    values: dict[str, float] = {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            name = str(key)
            if name in labels:
                values[name] = _probability(value, qid, name)
        for index, label in enumerate(labels):
            if label in values:
                continue
            if str(index) in raw:
                values[label] = _probability(raw[str(index)], qid, label)
            elif index in raw:  # type: ignore[operator]
                values[label] = _probability(raw[index], qid, label)  # type: ignore[index]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        if len(raw) != len(labels):
            raise ValueError(
                f"{qid}: vendor returned {len(raw)} probabilities for {len(labels)} labels"
            )
        for label, value in zip(labels, raw, strict=True):
            values[label] = _probability(value, qid, label)
    else:
        raise TypeError(f"{qid}: vendor probabilities must be a mapping or sequence")
    missing = [label for label in labels if label not in values]
    if missing:
        raise ValueError(f"{qid}: vendor omitted probability for {missing}")
    return renormalize_choice_probabilities(values, qid)


def _confidence(raw: Mapping[str, Any], probabilities: Mapping[str, float]) -> float | None:
    for key in _CONFIDENCE_KEYS:
        if key in raw and raw[key] is not None:
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return float(value)
    if probabilities:
        return max(probabilities.values())
    return None


def noul_answer(raw: Mapping[str, Any], qid: str) -> dict[str, Any]:
    """Project a vendor noul answer onto ``{"type": "noul", "noul": P(true)}``.

    Vendors return this either as a bare ``P(true)`` scalar or as a two-key distribution, and some
    put it under ``probability`` or ``boolean``. The contract wants one probability of true.
    """
    candidate: object = None
    for key in ("noul", "probability", "boolean", "true", "p_true"):
        if key in raw and raw[key] is not None:
            candidate = raw[key]
            break
    if candidate is None and isinstance(raw.get("probabilities"), Mapping):
        # Some families report a noul only inside their probability map, using yes/no candidate
        # IDs, with no separate scalar field.
        candidate = raw["probabilities"]
    if candidate is None:
        raise ValueError(f"{qid}: vendor noul answer has no probability field")
    if isinstance(candidate, Mapping):
        # Two-key distributions spell the true side differently: "true"/"false", or the yes/no
        # candidate IDs some vendors require at the call site.
        true_key = next(
            (key for key in candidate if str(key).strip().casefold() in _NOUL_TRUE_KEYS),
            None,
        )
        if true_key is None:
            raise ValueError(f"{qid}: vendor noul distribution has no true key")
        probability = _probability(candidate[true_key], qid, "noul")
        false_key = next(
            (key for key in candidate if str(key).strip().casefold() in _NOUL_FALSE_KEYS),
            None,
        )
        if false_key is not None:
            # A declared distribution must agree with itself; disagreement means a vendor bug we
            # should not average away.
            other = _probability(candidate[false_key], qid, "noul")
            if abs(probability + other - 1.0) > 1e-3:
                raise ValueError(f"{qid}: vendor noul distribution does not sum to one")
    else:
        probability = _probability(candidate, qid, "noul")
    if probability > 1.0:
        raise ValueError(f"{qid}: noul probability above one")
    return {"type": "noul", "noul": probability}


def choice_answer(
    raw: Mapping[str, Any],
    question: Mapping[str, Any],
    qid: str,
) -> dict[str, Any]:
    """Project a vendor choice answer onto the exact contract field set."""
    labels = _criteria_labels(question, qid)
    probabilities = _probability_map(
        raw.get("probabilities", raw.get("probs", raw.get("scores"))), qid, labels
    )
    choice = raw.get("choice", raw.get("label", raw.get("answer")))
    if choice is None or (isinstance(choice, str) and choice not in labels):
        # Only fall back to the argmax when the vendor supplied no label at all. A supplied label
        # outside the declared criteria is an error, not something to overwrite.
        if choice is not None:
            raise ValueError(f"{qid}: vendor choice {choice!r} is not a declared criterion")
        choice = max(probabilities, key=lambda label: (probabilities[label], label))
    answer: dict[str, Any] = {
        "type": "choice",
        "choice": str(choice),
        "probabilities": probabilities,
    }
    # Confidence is taken from the renormalized map, not from the vendor's own field. The two must be
    # the same quantity for calibration to be meaningful, and renormalization is order-preserving,
    # so the top entry is the vendor's chosen label either way. Reading the vendor field instead
    # would leave confidence disagreeing with `probabilities` by the transport-rounding residual.
    answer["confidence"] = max(probabilities.values())
    return answer


def score_answer(
    raw: Mapping[str, Any],
    question: Mapping[str, Any],
    qid: str,
) -> dict[str, Any]:
    """Project a vendor score answer onto a level inside the question's declared range.

    Vendors report a score either as an expected level, as a per-level distribution, or as a named
    level from the legend. All three are accepted; the result is always the nearest legal integer
    level inside ``[0, max_score]``, which is the tolerance the benchmark scores against.
    """
    levels = _criteria_labels(question, qid)
    max_score = question.get("max_score")
    if isinstance(max_score, bool) or not isinstance(max_score, (int, float)):
        raise TypeError(f"{qid}: max_score must be a number")
    upper = float(max_score)
    if upper < 0:
        raise ValueError(f"{qid}: max_score cannot be negative")

    value = raw.get("score", raw.get("level"))
    if value is None:
        distribution = _probability_map(raw.get("probabilities"), qid, levels)
        value = math.fsum(index * probability for index, probability in enumerate(distribution.values()))
    elif isinstance(value, str):
        if value not in levels:
            raise ValueError(f"{qid}: vendor score {value!r} is not a declared level")
        value = float(levels.index(value))
    else:
        value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{qid}: score must be finite")
    # Half-up rounding keeps a 0.5 expected score on the higher level, matching the benchmark's
    # own tolerance rule rather than Python's banker's rounding.
    level = math.floor(value + 0.5)
    level = max(0, min(int(upper), level))
    answer: dict[str, Any] = {"type": "score", "score": level}
    probabilities = None
    if raw.get("probabilities") is not None:
        try:
            probabilities = _probability_map(raw["probabilities"], qid, levels)
        except (TypeError, ValueError):
            probabilities = None
    confidence = _confidence(raw, probabilities or {})
    if confidence is not None:
        answer["confidence"] = confidence
    return answer


def project_vendor_answer(
    raw: object,
    question: Mapping[str, Any],
    qid: str,
) -> dict[str, Any]:
    """Project one vendor answer onto the contract shape for its question type."""
    if not isinstance(raw, Mapping):
        raise TypeError(f"{qid}: vendor answer must be a mapping")
    question_type = question.get("type")
    if question_type == "choice":
        return choice_answer(raw, question, qid)
    if question_type == "noul":
        return noul_answer(raw, qid)
    if question_type == "score":
        return score_answer(raw, question, qid)
    raise ValueError(f"{qid}: unsupported question type {question_type!r}")


def vendor_answers_payload(raw: object, qid: str) -> Mapping[str, Any]:
    """Extract the answers mapping from a System One style vendor response."""
    if not isinstance(raw, Mapping):
        raise TypeError(f"{qid}: vendor response must be a mapping")
    if "answers" in raw:
        answers = raw["answers"]
    else:
        answers = raw
    if not isinstance(answers, Mapping):
        raise TypeError(f"{qid}: vendor answers must be a mapping")
    return answers


class VendorRunner(BaseRunner):
    """Base class for adapters whose vendor exposes a System One compatible call.

    Subclasses supply `_load()` and `_call()`. Everything else - input copying, phase validation,
    question translation, per-answer projection, and answer-ID validation - is shared so a family
    module only encodes what is genuinely vendor-specific.
    """

    name = "vendor"
    family = "vendor"
    default_model = ""
    default_revision = ""
    license_name = ""

    def __init__(self) -> None:
        self._metadata = self._load()

    def _load(self) -> dict[str, Any]:
        raise NotImplementedError

    def _call(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return the vendor's raw answers keyed by question ID."""
        raise NotImplementedError

    def _usage(self, raw: object) -> dict[str, int]:
        """Return per-call token usage when the vendor reports it."""
        if not isinstance(raw, Mapping):
            return {}
        usage = raw.get("usage")
        if not isinstance(usage, Mapping):
            return {}
        result: dict[str, int] = {}
        for key, value in usage.items():
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if value >= 0:
                result[str(key)] = value
        return result

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        self._validate_phase(phase)
        copied_state, copied_questions = self._copy_inputs(state, questions)
        raw = self._call(copied_state, copied_questions)
        answers = vendor_answers_payload(raw, "response")
        # Answer-ID equality is checked before projection: a vendor that silently omits or invents a
        # question must fail as an ID mismatch, not as a confusing per-answer type error.
        self._validate_answer_ids(copied_questions, dict(answers))
        projected = {
            qid: project_vendor_answer(answers[qid], question, qid)
            for qid, question in copied_questions.items()
        }
        return {
            "answers": self._validate_answer_ids(copied_questions, projected),
            "_usage": self._usage(raw),
        }

    def info(self) -> dict[str, Any]:
        snapshot = super().info()
        snapshot["family"] = self.family
        if self.license_name:
            snapshot["license"] = self.license_name
        return snapshot


def deep_copy_questions(questions: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached copy of a question mapping, for adapters that rewrite in place."""
    return deepcopy(dict(questions))
