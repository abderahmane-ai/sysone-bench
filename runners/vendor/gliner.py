"""GLiNER 2.5 adapters: the boundary extractors and the Decide specialist.

Four checkpoints share the `gliner2` package and `AutoExtractor` loader. Two shapes matter:

* `BoundaryExtractor` (small, base, multi) exposes `classify_text(text, schema)` for independent
  per-task classification. A schema is either ``{task: [labels]}`` or ``{task: {"labels": [...]}}``.
* `SpanExtractor` (GLiNER2.5-Decide) uses the same `classify_text` call and is documented as
  answering choice, score, and label-description tasks through it.

The documented return is a bare ``{"task": "label"}`` mapping with no probabilities. The benchmark
contract requires a full probability map summing to one, so `include_confidence` style detail is not
relied upon: this adapter requests the richest documented form and, when a checkpoint still returns
only a bare label, records a single-label distribution rather than fabricating spread. That is a
real limitation of these checkpoints for calibration and is surfaced in run metadata rather than
hidden.

Every call is one question, so `calls` is counted per question.
"""

from __future__ import annotations

from runners.vendor.common import (
    VENDOR_ADAPTER_VERSION, VendorRunner, execution_provenance, load_kwargs,
    place_on_device, resolve_device, resolve_dtype, state_to_text,)

import json
from collections.abc import Mapping
from typing import Any


ADAPTER_VERSION = VENDOR_ADAPTER_VERSION


class _GlinerRunner(VendorRunner):
    family = "gliner2.5"
    #: Set when the checkpoint's documented output carries no probability distribution.
    reports_probabilities = False

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        from gliner2 import AutoExtractor

        model = AutoExtractor.from_pretrained(
            self.default_model,
            revision=self.default_revision,
            **load_kwargs(device, dtype),
        )
        place_on_device(model, device)
        self._model = model
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "serving": "local",
            **execution_provenance(device, dtype),
            "adapter_version": ADAPTER_VERSION,
            "probability_source": "vendor" if self.reports_probabilities else "single_label",
        }

    def _schema_for(self, question: Mapping[str, Any], qid: str) -> dict[str, Any]:
        """Build a one-task GLiNER schema, folding the instruction into the task label text."""
        question_type = question.get("type")
        criteria = question.get("criteria")
        if question_type == "choice":
            if not isinstance(criteria, Mapping) or not criteria:
                raise ValueError(f"{qid}: choice criteria must be a non-empty mapping")
            labels = [str(label) for label in criteria]
        elif question_type == "score":
            if isinstance(criteria, Mapping):
                labels = [str(level) for level in criteria]
            elif isinstance(criteria, (list, tuple)):
                labels = [str(level) for level in criteria]
            else:
                raise ValueError(f"{qid}: score criteria must be an ordered sequence of levels")
            if len(labels) < 2:
                raise ValueError(f"{qid}: score needs at least two levels for a classification head")
        elif question_type == "noul":
            labels = ["yes", "no"]
        else:
            raise ValueError(f"{qid}: unsupported question type {question_type!r}")
        # GLiNER keys results by task name, so the question ID is reused as the task name and the
        # instruction is carried in the schema description slot the package accepts.
        return {qid: {"labels": labels, "description": str(question.get("instructions", ""))}}

    def _text_for(self, state: Mapping[str, Any]) -> str:
        from runners.vendor.common import state_to_text

        return state_to_text(state)

    def _one_label(self, label: str, labels: list[str], qid: str) -> dict[str, Any]:
        if label not in labels:
            raise ValueError(f"{qid}: vendor label {label!r} is not a declared option")
        return {
            "probabilities": {name: (1.0 if name == label else 0.0) for name in labels},
            "choice": label,
            "confidence": 1.0,
        }

    def _spread_from_confidence(
        self,
        result: Mapping[str, Any],
        qid: str,
        labels: list[str],
    ) -> dict[str, Any] | None:
        """Recover a distribution from a documented per-label confidence list, when present."""
        probabilities = result.get("probabilities")
        if isinstance(probabilities, Mapping):
            return {name: float(probabilities[name]) for name in labels if name in probabilities}
        entries = result.get("entities", result.get("labels"))
        if isinstance(entries, Mapping):
            entries = entries.get(qid)
        if isinstance(entries, list) and entries:
            total = 0.0
            values: dict[str, float] = {}
            for entry in entries:
                if isinstance(entry, Mapping) and "label" in entry:
                    values[str(entry["label"])] = float(entry.get("confidence", 0.0))
            for name in labels:
                total += values.get(name, 0.0)
            if total > 0.0 and len(values) > 1:
                return {name: values.get(name, 0.0) / total for name in labels}
        return None

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        self._validate_phase(phase)
        copied_state, copied_questions = self._copy_inputs(state, questions)
        text = self._text_for(copied_state)
        answers: dict[str, Any] = {}
        calls = 0
        for qid, question in copied_questions.items():
            if not isinstance(question, Mapping):
                raise TypeError(f"{qid}: question must be a mapping")
            schema = self._schema_for(question, qid)
            criteria = question.get("criteria")
            question_type = question.get("type")
            if question_type == "choice":
                labels = [str(label) for label in criteria]  # type: ignore[union-attr]
            elif question_type == "score":
                labels = (
                    [str(level) for level in criteria]
                    if isinstance(criteria, (list, tuple))
                    else [str(level) for level in criteria]  # type: ignore[union-attr]
                )
            else:
                labels = ["yes", "no"]
            raw = self._model.classify_text(text, schema)
            calls += 1
            if isinstance(raw, str):
                raw = json.loads(raw)
            if not isinstance(raw, Mapping):
                raise TypeError(f"{qid}: gliner2 returned a non-mapping result")
            entry = raw.get(qid)
            if isinstance(entry, Mapping):
                label = str(entry.get("label", entry.get("value", "")))
                probabilities = self._spread_from_confidence(entry, qid, labels)
            elif isinstance(entry, str):
                label = entry
                probabilities = None
            elif isinstance(entry, list) and entry:
                # A multi-label result has no single correct answer for our contract; take the
                # first declared label it reports and keep the reported confidence.
                label = str(entry[0])
                probabilities = None
            else:
                raise ValueError(f"{qid}: gliner2 returned no decision for {qid}")
            if probabilities:
                payload: dict[str, Any] = {
                    "probabilities": probabilities,
                    "choice": max(probabilities, key=lambda name: (probabilities[name], name)),
                    "confidence": max(probabilities.values()),
                }
            else:
                # Verified against the real checkpoints on 2026-10-02: gliner2's classify_text
                # returns a bare {"task": "label"} mapping with no probabilities or confidence for
                # these checkpoints. The contract requires a unit-sum map, so a single-label
                # distribution is emitted and `probability_source` in run metadata records that the
                # confidence is degenerate rather than vendor-reported. Any calibration metric over
                # a GLiNER column is therefore uninformative and must not be read as good ECE.
                payload = self._one_label(label, labels, qid)
            if question_type == "noul":
                answers[qid] = {"type": "noul", "noul": float(payload["probabilities"]["yes"])}
            elif question_type == "score":
                index = labels.index(str(payload["choice"]))
                answers[qid] = {
                    "type": "score",
                    "score": float(index),
                    "confidence": float(payload["confidence"]),
                }
            else:
                answers[qid] = {
                    "type": "choice",
                    "choice": str(payload["choice"]),
                    "probabilities": {
                        name: float(value) for name, value in payload["probabilities"].items()
                    },
                    "confidence": float(payload["confidence"]),
                }
        return {
            "answers": self._validate_answer_ids(copied_questions, answers),
            "_usage": {"calls": calls},
        }


class GlinerSmallRunner(_GlinerRunner):
    name = "gliner-small"
    default_model = "fastino/gliner2.5-small-v1"
    default_revision = "7132dc4561c3f94563c6147e75ffa8ef34c4964a"
    license_name = "apache-2.0"


class GlinerBaseRunner(_GlinerRunner):
    name = "gliner-base"
    default_model = "fastino/gliner2.5-base-v1"
    default_revision = "ca906247640776a07753514055be9726f9080ead"
    license_name = "apache-2.0"


class GlinerMultiRunner(_GlinerRunner):
    name = "gliner-multi"
    default_model = "fastino/gliner2.5-multi-v1"
    default_revision = "2ca71aafb3446d9014e1c55c7ff51c9bc7209c47"
    license_name = "apache-2.0"


class GlinerDecideRunner(_GlinerRunner):
    name = "gliner-decide"
    default_model = "fastino/GLiNER2.5-Decide"
    default_revision = "5a7adf72a23b4d311abae6ce050d7f0012bb3416"
    license_name = "apache-2.0"


class _GLiClassShim:
    """Present a ``gliner2.GLiClass`` checkpoint through the ``classify_text`` shape.

    ``verdict`` is a ``GLiClassModel``: a 25-slot non-autoregressive label composer, not a
    GLiNER2.5 span or boundary extractor. Its supported readout is ``predict(text, labels)``,
    which takes a flat label list rather than the task-keyed schema the shared runner builds.
    Wrapping it here lets every downstream answer-mapping rule in ``_GlinerRunner`` -- probability
    spreading, the noul and score projections, label validation -- apply unchanged, instead of
    duplicating that logic in a second runner.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def classify_text(self, text: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for task, spec in schema.items():
            if isinstance(spec, Mapping):
                labels = [str(label) for label in spec.get("labels", [])]
                description = str(spec.get("description", ""))
            else:
                labels = [str(label) for label in spec]
                description = ""
            if len(labels) < 2:
                raise ValueError(f"{task}: GLiClass needs at least two candidate labels")
            # GLiClass has no separate instruction channel; the task description is prepended to
            # the state text so the instruction still reaches the encoder.
            payload = f"{description}\n\n{text}" if description else text
            # return_hierarchical is the documented way to get scores for every declared label.
            # Without it the pipeline returns only the argmax, e.g. [[{'label': 'yes', 'score':
            # 0.76}]], which carries no distribution to convert. classification_type is pinned to
            # single-label so the scores compose to a proper distribution over the options rather
            # than independent per-label sigmoid masses.
            raw = self._model(payload, labels, return_hierarchical=True,
                              classification_type="single-label")
            scores = self._as_scores(raw, labels, task)
            best = max(scores, key=lambda name: (scores[name], name))
            results[task] = {
                "label": best,
                "confidence": float(scores[best]),
                "scores": {name: float(value) for name, value in scores.items()},
            }
        return results

    @staticmethod
    def _as_scores(raw: Any, labels: list[str], task: str) -> dict[str, float]:
        """Normalize every documented GLiClass return shape to ``{label: probability}``.

        ``predict`` has returned a list of ``{"label", "score"}`` mappings, a list of
        ``(label, score)`` pairs, and a bare sequence of scores across releases, so all three are
        accepted. Anything else fails loudly rather than being coerced into a distribution.
        """
        # return_hierarchical gives [{label: score}] directly. Unwrap a per-item batch and accept
        # a bare {label: score} dict too, since either can appear depending on input arity.
        if isinstance(raw, (list, tuple)) and len(raw) == 1 and isinstance(raw[0], (list, tuple)):
            raw = raw[0]
        if isinstance(raw, (list, tuple)) and len(raw) == 1 and isinstance(raw[0], Mapping) \
                and "label" not in raw[0] and all(isinstance(v, (int, float)) for v in raw[0].values()):
            raw = raw[0]
        if isinstance(raw, Mapping) and all(isinstance(v, (int, float)) for v in raw.values()):
            # return_hierarchical yields {label: score} directly. This must be an exclusive branch:
            # falling through would re-test a Mapping against the list shapes below and raise.
            scores = {str(label): float(value) for label, value in raw.items()}
        elif isinstance(raw, Mapping):
            raw = [raw]
            scores = {str(item["label"]): float(item.get("score", item.get("probability", 0.0))) for item in raw}
        elif isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], Mapping):
            scores = {str(item["label"]): float(item.get("score", item.get("probability", 0.0))) for item in raw}
        elif isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
            scores = {str(item[0]): float(item[1]) for item in raw}
        elif isinstance(raw, (list, tuple)) and len(raw) == len(labels):
            scores = {label: float(value) for label, value in zip(labels, raw)}
        else:
            raise TypeError(f"{task}: unrecognised GLiClass predict() return shape")
        missing = [label for label in labels if label not in scores]
        if missing:
            raise ValueError(f"{task}: GLiClass omitted declared labels {missing}")
        total = sum(max(0.0, value) for value in scores.values())
        if total <= 0.0:
            raise ValueError(f"{task}: GLiClass returned a non-positive score mass")
        return {label: max(0.0, scores[label]) / total for label in labels}


class VerdictRunner(_GlinerRunner):
    """`heman10x/rlcd-modernbert-151m` - OpenJev Verdict, 151M GLiClass over ModernBERT.

    Verified on 2026-10-03 against the real checkpoint. Two package behaviours forced decisions
    that are recorded here so the next reader does not rediscover them:

    - ``pipeline(text, labels)`` returns only the argmax, ``[[{'label': 'yes', 'score': 0.76}]]``,
      in both ``single-label`` and ``multi-label`` mode, even at ``threshold=0.0``. The
      distribution over declared labels is only available with ``return_hierarchical=True``, which
      yields ``[{'yes': 0.672, 'no': 0.313, 'unsure': 0.014}]``. ``_as_scores`` refuses the
      argmax-only shape rather than inventing the missing probability mass.
    - The pipeline builds its own fp16 label-embedding tensors, so a half-precision model raises
      ``mat1 and mat2 must have the same dtype, but got Float and Half``. Precision is pinned to
      fp32, which costs about 0.6 GiB at 151M parameters.
    """

    name = "verdict"
    family = "gliclass"
    default_model = "heman10x/rlcd-modernbert-151m"
    default_revision = "8af2496eb63c7fa66d7d234e1f62629380030eb4"
    license_name = "apache-2.0"

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        from transformers import AutoTokenizer

        from gliclass import GLiClassModel, ZeroShotClassificationPipeline

        # The GLiClass pipeline builds its own fp16 label-embedding tensors, so a half-precision
        # model raises "mat1 and mat2 must have the same dtype, but got Float and Half". At 151M
        # parameters fp32 costs about 0.6 GiB, so precision is pinned rather than taken from the
        # environment, and the deviation is recorded in run metadata below.
        dtype = "float32"
        kwargs = load_kwargs(device, dtype)
        model = GLiClassModel.from_pretrained(
            self.default_model, revision=self.default_revision, **kwargs
        )
        place_on_device(model, device)
        tokenizer = AutoTokenizer.from_pretrained(
            self.default_model, revision=self.default_revision
        )
        # max_classes=25 is the checkpoint's declared slot count, and single-label matches its
        # config.json problem_type, so the pipeline reports a distribution over declared labels
        # rather than a filtered multi-label shortlist.
        pipeline = ZeroShotClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            max_classes=25,
            max_length=1024,
            classification_type="single-label",
            device=str(device),
            progress_bar=False,
        )
        self._model = _GLiClassShim(pipeline)
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "serving": "local",
            **execution_provenance(device, dtype),
            "adapter_version": ADAPTER_VERSION,
            "probability_source": "vendor",
            "precision_deviation": (
                "fp32 pinned: the GLiClass pipeline builds fp16 label embeddings and raises a "
                "dtype mismatch against a half-precision model"
            ),
        }
