"""NeoHorse-Jev-4B adapter.

The package lives at ``package/src/neohorse_decision`` inside the pinned repository, so the adapter
snapshots the revision and puts ``package/src`` on ``sys.path``.

`DecisionEngine.predict` is documented as "One serialized GPU worker. Requests are never silently
truncated", and it takes a full System One request dict which it validates as
``SystemOneRequest(**request)``. That pydantic model declares ``questions: dict[str, Question]``, so
the vendor wants a mapping keyed by question id, not the list this adapter first assumed (which fails
with "questions Input should be a valid dictionary"). The adapter therefore forwards the benchmark's
own question mapping and adds only the ``model`` field, and reads the answers keyed by qid.

Read from the pinned source and relied on here:

* ``DecisionEngine(model_dir, device='cuda', ...)`` takes the device directly and defaults to CUDA.
* The engine enforces ``max_questions=16`` and a 1 MiB request cap. A manifest case carries at most a
  handful of questions, so this is not expected to bind; the limit is recorded rather than raised.
* ``neohorse_decision._inference.predict`` returns ``{answers: {id: ...}, input_tokens: n}`` where each
  answer is spelled ``{type, choice, probabilities}`` for a choice, ``{type, noul, noul,
  probabilities}`` for a noul, and ``{type, score, legend, probabilities}`` for a score.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from runners.vendor.common import (
    VENDOR_ADAPTER_VERSION,
    VendorRunner,
    execution_provenance,
    project_vendor_answer,
    resolve_device,
    resolve_dtype,
)

ADAPTER_VERSION = VENDOR_ADAPTER_VERSION

_ALLOW_PATTERNS = ["*.json", "*.py", "*.safetensors", "*.model", "*.txt", "*.jinja", "package/**"]


def _install_dist_metadata(package_dir: Path) -> str | None:
    """Make ``importlib.metadata.version("neohorse-decision")`` resolvable from a bare snapshot.

    The snapshot ships ``package/pyproject.toml`` and ``package/src/neohorse_decision`` but no
    installed distribution, so ``neohorse_decision/__init__.py`` fails at import time on
    ``_package_version('neohorse-decision')`` with "No package metadata was found".

    The snapshot directory is a Hugging Face cache of symlinks into read-only blobs, so the
    ``.dist-info`` is written to a private temporary directory that is added to ``sys.path`` instead
    of next to the pyproject. Nothing is imported from it and the environment is not mutated.
    """
    import tempfile
    import tomllib

    pyproject = package_dir / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        version = str(data.get("project", {}).get("version", "")).strip()
    except (OSError, tomllib.TOMLDecodeError):
        return None
    if not version:
        return None
    root = Path(tempfile.mkdtemp(prefix="neohorse-meta-"))
    dist_info = root / "neohorse_decision-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: neohorse-decision\nVersion: " + version + "\n",
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text("", encoding="utf-8")
    sys.path.append(str(root))
    return version


class _NeoHorseRunner(VendorRunner):
    family = "neohorse"

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                self.default_model,
                revision=self.default_revision,
                allow_patterns=_ALLOW_PATTERNS,
            )
        ).resolve()
        # The importable package lives one level down, at package/src/neohorse_decision.
        package_root = root / "package" / "src"
        for entry in (str(package_root), str(root)):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        _install_dist_metadata(root / "package")
        from neohorse_decision import DecisionEngine

        engine = DecisionEngine(str(root), device=device)
        self._engine = engine
        self._model_id = getattr(engine, "model_id", None)
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "serving": "local",
            **execution_provenance(device, "vendor-managed"),
            "dtype_requested": dtype,
            "adapter_version": ADAPTER_VERSION,
            "package": "neohorse-decision (in-repository)",
            "package_source": self.default_model,
            "max_questions": getattr(engine, "max_questions", None),
            "vendor_model_id": self._model_id,
        }

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        self._validate_phase(phase)
        copied_state, copied_questions = self._copy_inputs(state, questions)

        # SystemOneRequest wants questions as a mapping keyed by qid, which is exactly what the
        # benchmark provides, so only `model` is added.
        request: dict[str, Any] = {"state": copied_state, "questions": copied_questions}
        if self._model_id is not None:
            request["model"] = self._model_id
        qids = list(copied_questions)

        response = self._engine.predict(request)
        results = response.get("answers") if isinstance(response, Mapping) else None
        if not isinstance(results, Mapping) or set(results) != set(qids):
            raise ValueError(
                f"neohorse returned answers for "
                f"{sorted(results) if isinstance(results, Mapping) else '?'}, expected {sorted(qids)}"
            )

        answers: dict[str, Any] = {}
        for qid in qids:
            result = results[qid]
            if not isinstance(result, Mapping):
                raise TypeError(f"{qid}: neohorse returned a non-mapping answer")
            answers[qid] = project_vendor_answer(result, copied_questions[qid], qid)
        usage = response.get("usage") if isinstance(response, Mapping) else None
        input_tokens = response.get("input_tokens") if isinstance(response, Mapping) else None
        return {
            "answers": self._validate_answer_ids(copied_questions, answers),
            "_usage": {
                "calls": 1,
                "input_tokens": input_tokens if input_tokens is not None else (
                    (usage or {}).get("input_tokens") if isinstance(usage, Mapping) else None
                ),
                "output_tokens": 0,
            },
        }


class NeoHorse4BRunner(_NeoHorseRunner):
    name = "neohorse-4b"
    default_model = "TokenRhythm/NeoHorse-Jev-4B"
    default_revision = "434cb21d3a994a953d3ae5788405fcb2c4970554"
    license_name = "mit"