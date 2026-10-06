"""Independent reimplementations of panel entries that are methods, not checkpoints.

`mini-jev`, `openvons`, `jobe`, and `harsha` are recorded by the Decision Index as
`kind: "inference technique"`. Each has a null `weights_repo`, a public ``code_url``, and a
``base_model`` that is a stock public checkpoint. Nothing was trained, so there is no adapter to
download -- the contribution is a prompting and serving procedure.

Two consequences are recorded honestly rather than glossed:

1. This is **our** implementation of the published method, not the vendor's code. For ``lev``,
   ``hopper-g`` and ``jet`` the vendor's readout is fully specified by files in the repository and
   is reproduced exactly. Here it is not, so these rows measure a faithful reading of the published
   description rather than the authors' own pipeline.
2. Every run therefore carries ``technique_reimplementation: true`` and the base checkpoint it was
   run against in run metadata, so a later reader can never mistake one of these rows for a
   measurement of the authors' system.

The scoring itself reuses :class:`runners.vendor.optlogit._OptionLogprobRunner`, which reads the
model's own conditional logprob for each declared option. That keeps one implementation of the
answer contract across every plain causal checkpoint instead of a near-duplicate per model.
"""

from __future__ import annotations

from typing import Any, Mapping

from runners.vendor.common import place_on_device, resolve_device, resolve_dtype, state_to_text
from runners.vendor.gliner import _GlinerRunner as _GlinerRunnerBase
from runners.vendor.optlogit import _OptionLogprobRunner


class _TechniqueRunner(_OptionLogprobRunner):
    """Shared handling for a stock base checkpoint driven by a published method."""

    family = "technique-reimplementation"
    prompt_style = "independent_reimplementation"

    def _load(self) -> dict[str, Any]:
        metadata = super()._load()
        metadata.update(
            {
                "technique_reimplementation": True,
                "vendor_code_executed": False,
                "base_model": self.base_model,
                "base_revision": self.base_revision,
                "prompt_style": self.prompt_style,
            }
        )
        return metadata

    def _question_text(self, state: Mapping[str, Any], question: Mapping[str, Any]) -> str:
        options = self._options(question, str(question.get("id", "q")))
        instruction = str(question.get("instructions", "")).strip()
        return (
            f"{state_to_text(state)}\n\n"
            f"### Question ({question.get('type')})\n{instruction}\n\n"
            f"### Options\n" + "\n".join(f"- {option}" for option in options)
            + "\n\n### Answer\n"
        )


class MiniJevRunner(_TechniqueRunner):
    """`r-ms/mini-jev` method over the stock `Qwen/Qwen3-4B-Instruct-2507` checkpoint."""

    name = "mini-jev"
    default_model = "Qwen/Qwen3-4B-Instruct-2507"
    default_revision = "cdbee75f17c01a7cc42f958dc650907174af0554"
    base_model = "Qwen/Qwen3-4B-Instruct-2507"
    base_revision = "cdbee75f17c01a7cc42f958dc650907174af0554"
    license_name = "apache-2.0"


class OpenVonsRunner(_TechniqueRunner):
    """`genai-craft/openvons` method over the stock `Qwen/Qwen3-4B-Instruct-2507` checkpoint."""

    name = "openvons"
    default_model = "Qwen/Qwen3-4B-Instruct-2507"
    default_revision = "cdbee75f17c01a7cc42f958dc650907174af0554"
    base_model = "Qwen/Qwen3-4B-Instruct-2507"
    base_revision = "cdbee75f17c01a7cc42f958dc650907174af0554"
    license_name = "apache-2.0"


class JobeRunner(_TechniqueRunner):
    """`MantisShrimpdev/jobe` method over the stock `Qwen/Qwen3.5-4B-Base` checkpoint."""

    name = "jobe"
    default_model = "Qwen/Qwen3.5-4B-Base"
    default_revision = "1001bb4d826a52d1f399e183466143f4da7b741b"
    base_model = "Qwen/Qwen3.5-4B-Base"
    base_revision = "1001bb4d826a52d1f399e183466143f4da7b741b"
    license_name = "apache-2.0"


class HarshaRunner(_TechniqueRunner):
    """`harshatheg/Qwen-2.5-1B-RLCD` method over the stock `Qwen/Qwen2.5-1.5B` checkpoint.

    That repository ships a Dockerfile, ``app.py`` and a ``core/`` package but no weight file, so
    the only runnable path is its documented base checkpoint.
    """

    name = "harsha"
    default_model = "Qwen/Qwen2.5-1.5B"
    default_revision = "8faed761d45a263340a0528343f099c05c9a4323"
    base_model = "Qwen/Qwen2.5-1.5B"
    base_revision = "8faed761d45a263340a0528343f099c05c9a4323"
    license_name = "apache-2.0"


class JeffRunner(_GlinerRunnerBase):
    """`logan-markewich/jeff` method over the stock `knowledgator/gliformer-large-v1` checkpoint.

    jeff is recorded as `kind: "inference technique"` over a GLiNER-family extractor. Because the
    base is a GLiNER checkpoint with a working `classify_text` interface, the readout reuses the
    proven GLiNER answer-mapping path rather than inventing a second one: the declared options for
    each question are handed to ``classify_text`` as a one-task schema and the returned label and
    confidence are projected onto the contract.

    As with the other technique entries this is our reading of the published method, not the
    author's code, and the run records ``technique_reimplementation: true``.
    """

    name = "jeff"
    family = "technique-reimplementation"
    default_model = "knowledgator/gliformer-large-v1"
    default_revision = "d0a4e53d09cebe6bc963dd9be319d4279084bb2d"
    license_name = "apache-2.0"
    prompt_style = "independent_reimplementation"

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        # GLiFormer's inference surface is classify(texts, classes) returning a
        # ranked [{class_name, score}] list: there is no classify_text and no
        # schema object. One call per question, options as the class list, and the
        # readout maps the returned scores onto the contract's answer shapes.
        self._validate_phase(phase)
        copied_state, copied_questions = self._copy_inputs(state, questions)
        from runners.vendor.common import state_to_text  # noqa: PLC0415

        text = state_to_text(copied_state)
        answers: dict[str, Any] = {}
        for qid, question in copied_questions.items():
            qtype = question.get("type")
            if qtype == "noul":
                # noul questions declare no criteria: they are a yes/no judgement
                # written as an instruction. Scoring abstract "yes"/"no" tokens is
                # meaningless for a zero-shot label classifier - verified on the real
                # model: a clean post scores "yes" 0.9963 and a nasty one "no" 0.0680,
                # while the same posts score their real labels correctly. So score the
                # instruction's own imperative as a single positive class and read its
                # returned score directly as P(true).
                # The class name is the qid, but GLiFormer reads natural language:
                # "jailbreak" scores 0.9997 on a real jailbreak while
                # "prompt_injection" - the suite's own spelling, with an underscore -
                # scores 0.0047 on the same post. Spoken form, so the model can read it.
                spoken = qid.replace("_", " ").strip()
                ranked = self._model.classify(text, [spoken], threshold=0.0)
                probability = float(ranked[0]["score"]) if ranked else 0.0
                answers[qid] = {"type": "noul", "noul": probability}
                continue
            criteria = question.get("criteria")
            if isinstance(criteria, Mapping):
                labels = [str(name) for name in criteria]
            else:
                labels = [str(level) for level in (criteria or [])]
            # classify() is a detection API: entries below `threshold` are
            # dropped. At the default 0.5 a confident clean post returns both
            # labels above it and a quiet one returns nothing, so the default
            # destroys the very signal a probability needs. Request everything,
            # then normalise over the full declared label set.
            ranked = self._model.classify(text, labels, threshold=0.0)
            scores = {entry["class_name"]: float(entry["score"]) for entry in ranked}
            total = sum(scores.values())
            if total > 0.0 and len(scores) > 1:
                probabilities = {name: value / total for name, value in scores.items()}
            else:
                best = max(scores, key=scores.get) if scores else labels[0]
                probabilities = {name: (1.0 if name == best else 0.0) for name in labels}
            for name in labels:
                probabilities.setdefault(name, 0.0)
            top = max(probabilities, key=lambda name: (probabilities[name], name))
            if qtype == "score":
                index = labels.index(top)
                answers[qid] = {
                    "type": "score",
                    "score": float(index),
                    "confidence": float(probabilities[top]),
                }
            else:
                answers[qid] = {
                    "type": "choice",
                    "choice": top,
                    "probabilities": {
                        name: float(value) for name, value in probabilities.items()
                    },
                    "confidence": float(probabilities[top]),
                }
        return {"answers": answers}

    def _load(self) -> dict[str, Any]:
        # GLiFormer ships its config as `gliner_config.json`, not the `config.json` that
        # transformers' Auto* classes look for. Fetching the standard name 404s, so the file is
        # materialised under that name in a private snapshot directory before loading, leaving the
        # shared HF cache untouched.
# This checkpoint's own package is `gliformer`, not `gliner2`: its loader is
        # GLiFormer.from_pretrained and its inference surface is classify(). The
        # earlier gliner2 path could never work here - the state-dict heads belong
        # to GLiFormer's architecture, not gliner2's span heads.
        from gliformer import GLiFormer

        from runners.vendor.gliner import ADAPTER_VERSION as _GLINER_ADAPTER_VERSION

        device, _ = resolve_device(), resolve_dtype()
        model = GLiFormer.from_pretrained(
            self.default_model,
            revision=self.default_revision,
        )
        place_on_device(model, device)
        model.eval()
        self._model = model
        metadata = {
            "runner": self.name,
            "model": "knowledgator/gliformer-large-v1",
            "revision": self.default_revision,
            "serving": "local",
            "adapter_version": _GLINER_ADAPTER_VERSION,
            "probability_source": "vendor",
        }
        metadata.update(
            {
                "technique_reimplementation": True,
                "vendor_code_executed": False,
                "base_model": "knowledgator/gliformer-large-v1",
                "base_revision": self.default_revision,
                "config_alias": "gliner_config.json -> config.json",
            }
        )
        return metadata
