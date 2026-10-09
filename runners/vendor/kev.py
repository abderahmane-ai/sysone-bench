"""Kev adapter.

Kev is served, not imported: the card's own instructions are
`python -m kev.serve --run jaredpalmer/kev-0.8b --port 8008`, and the repo exposes a
TypeSafe-compatible `POST /v1/systemone`. The adapter therefore runs that server as a loopback
subprocess and speaks HTTP to it, which is also how the vendor's own quickstart consumes it. This
avoids reverse-engineering the in-process `Checkpoint` API, and it keeps the vendor's own loading,
prefix cache and CUDA-graph paths in charge of numerics.

Two facts read from the pinned source and honoured here:

* `kev/api.py:to_answers` spells the answers this adapter projects: a choice gets
  ``{type, choice, confidence, probabilities}``, a score ``{type, score, legend, probabilities,
  confidence}``, a noul ``{type, noul}``. As elsewhere in this benchmark, a choice's
  ``confidence`` is TypeSafe's rescaled value rather than the raw probability, so the projection
  prefers ``probabilities``.
* The card states the calibrated temperature is applied by default and that ``KEV_TEMPERATURE=1.0``
  returns raw probabilities. The default is therefore kept and recorded, and
  ``KEV_DTYPE=fp32`` is set because the card names fp32 as the exact path used for evaluation.

The `kev` name on PyPI is an unrelated key-value-store ORM, so this package is installed from its
GitHub tag rather than from the index.
"""

from __future__ import annotations

import atexit
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
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

# The card's exact evaluation path, and the run this card ships (Kev 1.0).
PACKAGE_COMMIT = "6b719c3c3f36"
DEFAULT_DTYPE = "fp32"
FALLBACK_DTYPE = "bf16"
STARTUP_TIMEOUT_SECONDS = 1800


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _KevRunner(VendorRunner):
    family = "kev"

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        if not device.startswith("cuda"):
            raise ValueError(
                f"{self.name} requires CUDA: kev.serve resolves its own device and the card "
                f"documents a CUDA/MLX serving path, not a CPU one. Requested {device!r}."
            )
        self._port = _free_port()
        self._base = f"http://127.0.0.1:{self._port}"
        environment = dict(os.environ)
        # The card names fp32 as the exact path its reported numbers use, and that is tried first.
        # fp32 needs roughly 4 bytes per parameter, so the 4B and 9B checkpoints cannot be held in
        # fp16-equivalent VRAM on a 14.56 GiB T4 (kev-4b OOMs at ~18.6 GB). Rather than hardcode a
        # size table, fp32 is attempted and bf16 is fallen back to, and whichever ran is recorded in
        # `serve_dtype` together with `precision_deviation` so the row is never mistaken for the
        # card's exact evaluation path.
        environment.setdefault("KEV_DTYPE", DEFAULT_DTYPE)
        command = [
            sys.executable,
            "-m",
            "kev.serve",
            "--run",
            self.default_model,
            "--port",
            str(self._port),
        ]
        self._process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
        )
        atexit.register(self._stop)
        if not self._await_ready():
            # Most likely fp32 VRAM on a size this card cannot hold; retry once in bf16.
            if environment["KEV_DTYPE"] == DEFAULT_DTYPE:
                self._stop()
                environment["KEV_DTYPE"] = FALLBACK_DTYPE
                self._process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=environment,
                )
                atexit.register(self._stop)
        if not self._await_ready():
            code = self._process.poll()
            self._stop()
            raise RuntimeError(
                f"kev.serve did not become ready (child exit={code}); "
                f"dtype={environment['KEV_DTYPE']} run={self.default_model}"
            )
        self._served = self._model_id()
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "serving": "loopback-subprocess",
            **execution_provenance(device, "vendor-managed"),
            "dtype_requested": dtype,
            "adapter_version": ADAPTER_VERSION,
            "package": "kev",
            "package_commit": PACKAGE_COMMIT,
            "serve_dtype": environment["KEV_DTYPE"],
            "card_evaluation_dtype": DEFAULT_DTYPE,
            "precision_deviation": (
                None if environment["KEV_DTYPE"] == DEFAULT_DTYPE
                else f"ran in {environment['KEV_DTYPE']}, not the card's {DEFAULT_DTYPE} "
                     f"evaluation path, which does not fit a 14.56 GiB T4 at this size"
            ),
            "temperature": "calibrated default (KEV_TEMPERATURE unset)",
            "served_model": self._served,
        }

    def _await_ready(self) -> bool:
        import requests

        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                return False
            try:
                if requests.get(f"{self._base}/v1/models", timeout=5).status_code == 200:
                    return True
            except requests.RequestException:
                time.sleep(3)
        return False

    def _model_id(self) -> str | None:
        """The run id the server reports, which is what /v1/systemone must be asked for."""
        import requests

        try:
            body = requests.get(f"{self._base}/v1/models", timeout=10).json()
        except Exception:  # noqa: BLE001 - the id is advisory metadata, not a correctness input
            return None
        data = body.get("data") if isinstance(body, Mapping) else None
        if isinstance(data, list) and data and isinstance(data[0], Mapping):
            return str(data[0].get("id"))
        return None

    def _stop(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            process.kill()

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        import requests

        self._validate_phase(phase)
        copied_state, copied_questions = self._copy_inputs(state, questions)

        payload: dict[str, Any] = {"state": copied_state}
        if self._served is not None:
            payload["model"] = self._served
        # kev.api.Choice/Noul/Score read the benchmark's own {type, instructions, criteria} shape.
        payload["questions"] = copied_questions

        response = requests.post(f"{self._base}/v1/systemone", json=payload, timeout=600)
        if response.status_code != 200:
            raise RuntimeError(
                f"kev /v1/systemone returned {response.status_code}: {response.text[:300]}"
            )
        body = response.json()
        results = body.get("answers") if isinstance(body, Mapping) else None
        if not isinstance(results, Mapping) or set(results) != set(copied_questions):
            raise ValueError(
                f"kev returned answers for "
                f"{sorted(results) if isinstance(results, Mapping) else '?'}, "
                f"expected {sorted(copied_questions)}"
            )

        answers: dict[str, Any] = {}
        for qid, question in copied_questions.items():
            result = results[qid]
            if not isinstance(result, Mapping):
                raise TypeError(f"{qid}: kev returned a non-mapping answer")
            answers[qid] = project_vendor_answer(result, question, qid)
        usage = body.get("usage") if isinstance(body, Mapping) else None
        return {
            "answers": self._validate_answer_ids(copied_questions, answers),
            "_usage": {
                "calls": 1,
                "input_tokens": (usage or {}).get("input_tokens") if isinstance(usage, Mapping) else None,
                "output_tokens": (usage or {}).get("output_tokens") if isinstance(usage, Mapping) else None,
            },
        }


class Kev08BRunner(_KevRunner):
    name = "kev-08b"
    default_model = "jaredpalmer/kev-0.8b"
    default_revision = "bf75a6a8848ea6960ff2ed108d9ed44c2941174f"
    license_name = "apache-2.0"


class Kev4BRunner(_KevRunner):
    name = "kev-4b"
    default_model = "jaredpalmer/kev-4b"
    default_revision = "6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
    license_name = "apache-2.0"


class Kev9BRunner(_KevRunner):
    name = "kev-9b"
    default_model = "jaredpalmer/kev-9b"
    default_revision = "db029f08b290afd9fee4aa4bbcd9ae48602d1eb0"
    license_name = "apache-2.0"
