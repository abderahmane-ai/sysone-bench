"""JPT adapters via the llm2jev decision service.

JPT ships no in-process inference class. Its documented architecture is a serving engine holding the
weights with `llm2jev` reading option probabilities off it and exposing `POST /v1/systemone`. The
CPU-feasible backend is llm2jev's own `hf` backend, which the card states "serializes requests; fine
on CPU at this size".

This adapter therefore starts llm2jev as a subprocess bound to loopback on a caller-chosen port,
performs one HTTP request per state through `requests`, and always tears the subprocess down. That
matters for the isolation contract: the worker never publishes a port, and the subprocess lives and
dies with this adapter.

Two documented details are load-bearing and are applied rather than ignored:
* JPT needs `--temperature 1.140`. The card is explicit that this constant is part of the model's
  calibration, so the default is used and recorded in metadata.
* noul criteria are spelled `{"true": ..., "false": ...}`, which the shared projection reads.
"""

from __future__ import annotations

from runners.vendor.common import (
    VENDOR_ADAPTER_VERSION, VendorRunner, _DEVICE_ENV, _DTYPE_ENV, execution_provenance,
    resolve_device, resolve_dtype,)

import atexit
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ADAPTER_VERSION = VENDOR_ADAPTER_VERSION
DEFAULT_TEMPERATURE = "1.140"  # JPT-0.8B; each size fits its own temperature once
STARTUP_TIMEOUT_SECONDS = 900


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _JptRunner(VendorRunner):
    # Each JPT size fits one temperature on a held-out split; the card states it, so it is
    # pinned here and recorded rather than searched for per benchmark.
    default_temperature = DEFAULT_TEMPERATURE

    family = "jpt"

    def _load(self) -> dict[str, Any]:
        device, dtype = resolve_device(), resolve_dtype()
        self._port = _free_port()
        self._base = f"http://127.0.0.1:{self._port}"
        environment = dict(os.environ)
        environment.setdefault("HF_HUB_OFFLINE", "1")
        environment.setdefault("TRANSFORMERS_OFFLINE", "1")
        # llm2jev owns device selection for its own `hf` backend, so the request is forwarded into
        # the child's environment and recorded below rather than asserted from here.
        environment[_DEVICE_ENV] = device
        environment[_DTYPE_ENV] = dtype
        # llm2jev forwards --model straight to AutoTokenizer.from_pretrained, and a "repo@revision"
        # string is not a valid repo id: transformers raises "Repo id must use alphanumeric chars".
        # Resolving the pinned revision to a local snapshot keeps the revision explicit, which is
        # what this adapter is required to do.
        from huggingface_hub import snapshot_download

        local = Path(
            snapshot_download(
                self.default_model,
                revision=self.default_revision,
                allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.jinja"],
            )
        ).resolve()
        command = [
            sys.executable,
            "-m",
            "llm2jev",
            "--model",
            str(local),
            "--backend",
            "hf",
            "--port",
            str(self._port),
            "--temperature",
            self.default_temperature,
        ]
        self._process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
        )
        atexit.register(self._stop)
        if not self._await_ready():
            # The child's streams are DEVNULL so a healthy run stays quiet, so on failure the exit
            # code is all the diagnostic there is. Report it rather than "did not become ready".
            returncode = self._process.poll()
            self._stop()
            raise RuntimeError(
                f"llm2jev did not become ready (child exit={returncode}); "
                f"model snapshot={local}"
            )
        return {
            "runner": self.name,
            "model": self.default_model,
            "revision": self.default_revision,
            "serving": "loopback-subprocess",
            **execution_provenance(device, dtype),
            "adapter_version": ADAPTER_VERSION,
            "temperature": self.default_temperature,
            "backend": "hf",
            "device_selection": "delegated to llm2jev hf backend",
        }

    def _await_ready(self) -> bool:
        import requests

        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                return False
            try:
                requests.get(self._base, timeout=2)
            except requests.RequestException:
                time.sleep(2.0)
                continue
            return True
        return False

    def _stop(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)

    def __del__(self) -> None:
        try:
            self._stop()
        except Exception:  # noqa: BLE001 - interpreter teardown must not raise
            pass

    def _call(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        import requests

        payload = {
            "state": state
            if isinstance(state, str)
            else "\n".join(f"{key}: {value}" for key, value in state.items()),
            "questions": {
                qid: {
                    key: value
                    for key, value in question.items()
                    if key not in {"qid", "max_score"}
                }
                for qid, question in questions.items()
            },
        }
        response = requests.post(f"{self._base}/v1/systemone", json=payload, timeout=600)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise TypeError("llm2jev returned a non-mapping body")
        return body


class Jpt08BRunner(_JptRunner):
    name = "jpt-08b"
    default_model = "kirp/jpt-0.8b"
    default_revision = "1431c0509bbc10772cd59964cc7af5835c8720d6"
    license_name = "cc-by-nc-4.0"


class Jpt4BRunner(_JptRunner):
    name = "jpt-4b"
    default_model = "kirp/jpt-4b"
    default_revision = "78312f855b8bebf83ae7e9e8f05b5a0f73f519a9"
    license_name = "cc-by-nc-4.0"
    default_temperature = "1.036"


class Jpt9BRunner(_JptRunner):
    name = "jpt-9b"
    default_model = "kirp/jpt-9b"
    default_revision = "b447cc7ee105c0a76a22f8fde8ecf05074dc8be0"
    license_name = "cc-by-nc-4.0"
    default_temperature = "1.087"
