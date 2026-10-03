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

    def _load(self) -> dict[str, Any]:
        # GLiFormer ships its config as `gliner_config.json`, not the `config.json` that
        # transformers' Auto* classes look for. Fetching the standard name 404s, so the file is
        # materialised under that name in a private snapshot directory before loading, leaving the
        # shared HF cache untouched.
        from pathlib import Path

        from huggingface_hub import snapshot_download

        local = Path(
            snapshot_download(
                repo_id=self.default_model,
                revision=self.default_revision,
                allow_patterns=["gliner_config.json", "pytorch_model.bin", "tokenizer*"],
            )
        )
        if not (local / "config.json").exists():
            (local / "config.json").write_bytes((local / "gliner_config.json").read_bytes())
        self.default_model = str(local)
        # GLiFormer's AutoExtractor does not accept `dtype`; its loader takes `map_location`
        # instead, so the shared load_kwargs are bypassed rather than filtered.
        from gliner2 import AutoExtractor

        from runners.vendor.gliner import ADAPTER_VERSION as _GLINER_ADAPTER_VERSION

        device, _ = resolve_device(), resolve_dtype()
        model = AutoExtractor.from_pretrained(str(local))
        place_on_device(model, device)
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
