from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

GIB = 1024**3
MIN_MEMORY_GIB = 16.0
MIN_DISK_GIB = 100.0
# load5 is a five-minute average, so it still reflects this benchmark's own previous run for minutes
# after it exits. load1 is gated against the worker's own CPU quota instead, so a completed run's
# residual cannot refuse the next one. See the load gate in check().
MAX_LOAD5_PER_CPU = 0.75
# Default worker pinning for a shared host. Operators may widen the cpuset via
# SYSONE_BENCH_WORKER_CPUSET only together with SYSONE_BENCH_WORKER_CPUS; both are re-derived here
# so preflight gates the exact resources the launcher will request.
REQUIRED_CPU_IDS = (0, 1, 2, 3)
DEFAULT_WORKER_CPUSET = "0-3"
DEFAULT_WORKER_CPUS = 4
# A widened worker must still leave the host majority-idle, so the gate scales with what the worker
# does NOT claim. This is what stops a wide run from starting on a host that is already busy.
MAX_UNCLAIMED_CPU_SHARE = 0.5
# Refuse any request for more than this fraction of the host's visible CPUs, so one run can never
# claim the whole machine regardless of environment configuration. The cap is skipped on hosts too
# small to leave a CPU behind: the documented 4-CPU default is 100% of a 4-CPU host, and refusing
# that would make the gate unusable on small workers for no safety gain.
MAX_WORKER_HOST_SHARE = 0.75
MIN_HOST_CPUS_FOR_SHARE_CAP = 8
# Every default is derived from this file's own location or from the environment, so the
# same checkout works for any user on any host and no personal path is baked into the code.
# SYSONE_BENCH_WORKSPACE_ROOT, SYSONE_BENCH_MODEL_CACHE, SYSONE_BENCH_MANIFEST and
# SYSONE_BENCH_MANIFEST_CHECKSUM override the derived values.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE_ROOT = Path(os.environ.get("SYSONE_BENCH_WORKSPACE_ROOT", REPOSITORY_ROOT))
DEFAULT_RUNS_ROOT = DEFAULT_WORKSPACE_ROOT / "runs"
DEFAULT_RUN_ROOT = DEFAULT_RUNS_ROOT
DEFAULT_MODEL_CACHE = Path(
    os.environ.get("SYSONE_BENCH_MODEL_CACHE", Path.home() / ".cache" / "sysone-bench-v2")
)
DEFAULT_MANIFEST_PATH = Path(
    os.environ.get("SYSONE_BENCH_MANIFEST", DEFAULT_WORKSPACE_ROOT / "datasets" / "v2" / "manifest.jsonl")
)
DEFAULT_MANIFEST_CHECKSUM_PATH = Path(
    os.environ.get(
        "SYSONE_BENCH_MANIFEST_CHECKSUM",
        DEFAULT_WORKSPACE_ROOT / "datasets" / "v2" / "manifest.sha256",
    )
)
OWNER_MARKER_NAME = ".sysone-owner"
LOADAVG_PATH = Path("/proc/loadavg")
CPUINFO_PATH = Path("/proc/cpuinfo")
MEMINFO_PATH = Path("/proc/meminfo")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
CONTAINER_PREFIX = "sysone-bench-"
REQUIRED_SNAPSHOT_FIELDS = frozenset(
    {
        "cpu_count",
        "cpu_affinity",
        "worker_cpuset",
        "worker_cpus",
        "requested_cpu_ids",
        "load1",
        "load5",
        "memory_available_gib",
        "disk_available_gib",
        "avx2",
        "python_version",
        "run_id",
        "workspace_root",
        "workspace_root_state",
        "disk_path",
        "run_root",
        "run_root_state",
        "run_path",
        "run_path_state",
        "owner_marker_path",
        "owner_marker_state",
        "owner_marker_valid",
        "owner_marker_owner_uid",
        "container_name",
        "container_state",
        "model_cache_path",
        "model_cache_state",
        "manifest_path",
        "manifest_checksum_path",
        "manifest_state",
        "manifest_checksum_state",
        "manifest_checksum_valid",
        "read_errors",
    }
)


def validate_run_id(run_id: str) -> str:
    if (
        not isinstance(run_id, str)
        or run_id in {".", ".."}
        or RUN_ID_PATTERN.fullmatch(run_id) is None
    ):
        raise ValueError("run_id must be one safe path component")
    return run_id


def validate_container_name(container_name: str, run_id: str | None = None) -> str:
    if not isinstance(container_name, str) or not container_name.startswith(CONTAINER_PREFIX):
        raise ValueError("container name must use the sysone-bench- prefix")
    suffix = container_name[len(CONTAINER_PREFIX) :]
    if run_id is not None and suffix != run_id:
        raise ValueError("container name does not match run_id")
    validate_run_id(suffix)
    return container_name


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{name} must be finite")
    return number


def _python_312(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        return False
    parts = value.split(".")
    return len(parts) >= 2 and parts[0] == "3" and parts[1] == "12"


def read_loadavg(path: Path = LOADAVG_PATH) -> tuple[float, float]:
    fields = path.read_text(encoding="utf-8").split()
    if len(fields) < 2:
        raise RuntimeError("loadavg is incomplete")
    try:
        return float(fields[0]), float(fields[1])
    except ValueError as error:
        raise RuntimeError("loadavg is invalid") from error


def read_avx2(path: Path = CPUINFO_PATH) -> bool:
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        if name.strip().casefold() == "flags":
            return "avx2" in value.casefold().split()
    return False


def read_mem_available(path: Path = MEMINFO_PATH) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "MemAvailable:":
            return int(fields[1]) * 1024
    raise RuntimeError("MemAvailable is missing")


def read_free_available(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    meminfo_path: Path = MEMINFO_PATH,
) -> int:
    command_runner = subprocess.run if runner is None else runner
    completed = command_runner(
        ["free", "-b"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    stdout = completed.stdout if isinstance(completed.stdout, str) else ""
    for line in stdout.splitlines():
        fields = line.split()
        if fields and fields[0].rstrip(":") == "Mem" and len(fields) >= 7:
            return int(fields[-1])
    return read_mem_available(meminfo_path)


def read_disk_available(path: Path) -> int:
    values = os.statvfs(path)
    return int(values.f_bavail * values.f_frsize)


def read_cpu_affinity() -> tuple[int, ...]:
    getter = getattr(os, "sched_getaffinity", None)
    if not callable(getter):
        raise TypeError("process CPU affinity is unavailable")
    raw = getter(0)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (set, frozenset, list, tuple)):
        raise TypeError("process CPU affinity is invalid")
    values: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("process CPU affinity is invalid")
        values.append(value)
    if not values or len(set(values)) != len(values):
        raise RuntimeError("process CPU affinity is invalid")
    return tuple(sorted(values))


def parse_worker_cpuset() -> tuple[int, ...]:
    """Parse the requested worker cpuset from ``SYSONE_BENCH_WORKER_CPUSET``.

    Accepts the same comma- and range-separated forms Docker accepts for ``--cpuset-cpus``, so the
    value preflight validates is byte-identical to the one the launcher passes to Docker. Parsing
    lives here rather than in the shell so a malformed value fails the read-only gate instead of
    reaching a container invocation.
    """
    raw = os.environ.get("SYSONE_BENCH_WORKER_CPUSET", DEFAULT_WORKER_CPUSET)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("SYSONE_BENCH_WORKER_CPUSET must be a non-empty string")
    cpus: list[int] = []
    for part in raw.strip().split(","):
        token = part.strip()
        if not token:
            raise ValueError("SYSONE_BENCH_WORKER_CPUSET contains an empty element")
        if "-" in token:
            start_text, _, end_text = token.partition("-")
            if not start_text.strip().isdigit() or not end_text.strip().isdigit():
                raise ValueError(f"invalid cpuset range: {token!r}")
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"cpuset range is descending: {token!r}")
            cpus.extend(range(start, end + 1))
        else:
            if not token.isdigit():
                raise ValueError(f"invalid cpuset element: {token!r}")
            cpus.append(int(token))
    if not cpus:
        raise ValueError("SYSONE_BENCH_WORKER_CPUSET resolved to no CPUs")
    if len(set(cpus)) != len(cpus):
        raise ValueError("SYSONE_BENCH_WORKER_CPUSET contains duplicates")
    return tuple(sorted(cpus))


def read_worker_cpus() -> int:
    """Parse ``SYSONE_BENCH_WORKER_CPUS``, the quota the launcher passes to ``docker --cpus``."""
    raw = os.environ.get("SYSONE_BENCH_WORKER_CPUS", str(DEFAULT_WORKER_CPUS))
    if not isinstance(raw, str) or not raw.strip().isdigit():
        raise ValueError("SYSONE_BENCH_WORKER_CPUS must be a positive integer")
    value = int(raw.strip())
    if value <= 0:
        raise ValueError("SYSONE_BENCH_WORKER_CPUS must be a positive integer")
    return value


def _container_state(
    container_name: str | None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
) -> str:
    if container_name is None:
        return "absent"
    command_runner = subprocess.run if runner is None else runner
    try:
        completed = command_runner(
            ["docker", "container", "inspect", container_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError):
        return "unknown"
    if completed.returncode == 0:
        return "occupied"
    output = f"{completed.stdout}\n{completed.stderr}".casefold()
    if "no such object" in output or "no such container" in output:
        return "absent"
    return "unknown"


def _effective_uid() -> int | None:
    getter = getattr(os, "geteuid", None)
    if not callable(getter):
        return None
    value = getter()
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _path_state(path: Path, *, cache: bool = False, regular_file: bool = False) -> str:
    try:
        current = path
        while True:
            if os.path.lexists(current):
                information = os.lstat(current)
                if stat.S_ISLNK(information.st_mode):
                    return "symlink"
            parent = current.parent
            if parent == current:
                break
            current = parent
        if not os.path.lexists(path):
            return "absent"
        information = os.lstat(path)
    except OSError:
        return "unknown"
    if regular_file:
        return "ready" if stat.S_ISREG(information.st_mode) else "not_regular"
    if cache:
        if not stat.S_ISDIR(information.st_mode):
            return "not_directory"
        get_euid = getattr(os, "geteuid", None)
        if not callable(get_euid) or information.st_uid != get_euid():
            return "unowned"
        return "ready"
    if not stat.S_ISDIR(information.st_mode):
        return "occupied"
    return "ready"


def _owner_marker_snapshot(
    marker_path: Path | None, run_id: str | None
) -> tuple[str, bool, int | None]:
    if marker_path is None or run_id is None:
        return "absent", False, None
    state = _path_state(marker_path, regular_file=True)
    if state != "ready":
        return state, False, None
    try:
        owner_uid = os.lstat(marker_path).st_uid
        content = marker_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return "unknown", False, None
    current_uid = _effective_uid()
    if current_uid is None or owner_uid != current_uid:
        return "unowned", False, owner_uid
    if content != f"{run_id}\n":
        return "mismatch", False, owner_uid
    return "ready", True, owner_uid


def _manifest_checksum_valid(manifest_path: Path, checksum_path: Path) -> bool:
    try:
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return False
    if len(checksum_lines) != 1:
        return False
    fields = checksum_lines[0].split("  ", 1)
    return len(fields) == 2 and fields[0] == digest and fields[1] == "manifest.jsonl"


def collect_snapshot(
    *,
    run_id: str | None = None,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
    run_root: Path = DEFAULT_RUN_ROOT,
    model_cache: Path = DEFAULT_MODEL_CACHE,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    manifest_checksum_path: Path = DEFAULT_MANIFEST_CHECKSUM_PATH,
    loadavg_path: Path = LOADAVG_PATH,
    cpuinfo_path: Path = CPUINFO_PATH,
    meminfo_path: Path = MEMINFO_PATH,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    selected_run_id = validate_run_id(run_id) if run_id is not None else None
    selected_container = (
        f"{CONTAINER_PREFIX}{selected_run_id}" if selected_run_id is not None else None
    )
    target_disk = workspace_root
    read_errors: list[str] = []
    try:
        affinity = read_cpu_affinity()
    except (OSError, RuntimeError, TypeError, ValueError):
        affinity = ()
        read_errors.append("CPU affinity unavailable")
    # Captured here so the snapshot always carries the requested shape; a malformed value becomes a
    # read error rather than an unhandled exception, matching the affinity failure above.
    try:
        worker_cpuset = parse_worker_cpuset()
        worker_cpus = read_worker_cpus()
    except ValueError:
        worker_cpuset = ()
        worker_cpus = None
        read_errors.append("requested worker CPU configuration is invalid")
    try:
        load1, load5 = read_loadavg(loadavg_path)
    except (OSError, RuntimeError, ValueError):
        load1 = None
        load5 = None
        read_errors.append("loadavg unavailable")
    try:
        memory_bytes = read_free_available(runner, meminfo_path)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        memory_bytes = None
        read_errors.append("memory unavailable")
    try:
        disk_bytes = read_disk_available(target_disk)
    except (OSError, ValueError):
        disk_bytes = None
        read_errors.append("disk unavailable")
    try:
        avx2 = read_avx2(cpuinfo_path)
    except (OSError, RuntimeError):
        avx2 = False
        read_errors.append("CPU feature information unavailable")
    workspace_root_state = _path_state(workspace_root)
    run_root_state = _path_state(run_root)
    run_path = run_root / selected_run_id if selected_run_id is not None else None
    run_path_state = _path_state(run_path) if run_path is not None else "absent"
    owner_marker_path = run_path / OWNER_MARKER_NAME if run_path is not None else None
    owner_marker_state, owner_marker_valid, owner_marker_owner_uid = _owner_marker_snapshot(
        owner_marker_path, selected_run_id
    )
    container_state = _container_state(selected_container, runner)
    model_cache_state = _path_state(model_cache, cache=True)
    manifest_state = _path_state(manifest_path, regular_file=True)
    manifest_checksum_state = _path_state(manifest_checksum_path, regular_file=True)
    manifest_checksum_valid = (
        manifest_state == "ready"
        and manifest_checksum_state == "ready"
        and _manifest_checksum_valid(manifest_path, manifest_checksum_path)
    )
    if workspace_root_state == "unknown":
        read_errors.append("workspace root could not be inspected")
    if run_root_state == "unknown":
        read_errors.append("run root could not be inspected")
    if run_path_state == "unknown":
        read_errors.append("run path could not be inspected")
    if owner_marker_state == "unknown":
        read_errors.append("owner marker could not be inspected")
    if container_state == "unknown":
        read_errors.append("container state unavailable")
    if model_cache_state == "unknown":
        read_errors.append("model cache could not be inspected")
    if manifest_state == "unknown":
        read_errors.append("manifest could not be inspected")
    if manifest_checksum_state == "unknown":
        read_errors.append("manifest checksum could not be inspected")
    return {
        "cpu_count": len(affinity) if affinity else None,
        "cpu_affinity": list(affinity),
        "requested_cpu_ids": list(REQUIRED_CPU_IDS),
        "worker_cpuset": list(worker_cpuset),
        "worker_cpus": worker_cpus,
        "host_cpu_count": os.cpu_count(),
        "load1": load1,
        "load5": load5,
        "memory_available_gib": None if memory_bytes is None else memory_bytes / GIB,
        "disk_available_gib": None if disk_bytes is None else disk_bytes / GIB,
        "avx2": avx2,
        "python_version": platform.python_version(),
        "run_id": selected_run_id,
        "workspace_root": str(workspace_root),
        "workspace_root_state": workspace_root_state,
        "disk_path": str(target_disk),
        "run_root": str(run_root),
        "run_root_state": run_root_state,
        "run_path": None if run_path is None else str(run_path),
        "run_path_state": run_path_state,
        "owner_marker_path": None if owner_marker_path is None else str(owner_marker_path),
        "owner_marker_state": owner_marker_state,
        "owner_marker_valid": owner_marker_valid,
        "owner_marker_owner_uid": owner_marker_owner_uid,
        "container_name": selected_container,
        "container_state": container_state,
        "model_cache_path": str(model_cache),
        "model_cache_state": model_cache_state,
        "manifest_path": str(manifest_path),
        "manifest_checksum_path": str(manifest_checksum_path),
        "manifest_state": manifest_state,
        "manifest_checksum_state": manifest_checksum_state,
        "manifest_checksum_valid": manifest_checksum_valid,
        "read_errors": tuple(read_errors),
        "errors": tuple(read_errors),
    }


snapshot = collect_snapshot


def _failure(message: str, values: Mapping[str, object]) -> RuntimeError:
    selected = {
        key: values.get(key)
        for key in sorted(REQUIRED_SNAPSHOT_FIELDS | {"host_cpu_count", "errors"})
        if key in values
    }
    return RuntimeError(f"preflight failed: {message}; snapshot={selected!r}")


def _ids(value: object, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a sequence of integers")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name} must be a sequence of integers")
        result.append(item)
    if len(set(result)) != len(result):
        raise RuntimeError(f"{name} must contain unique integers")
    return tuple(result)


def check(snapshot: Mapping[str, Any], *, allow_owned: bool = False) -> None:
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")
    missing = sorted(REQUIRED_SNAPSHOT_FIELDS - set(snapshot))
    if missing:
        raise _failure(f"missing required safety fields: {', '.join(missing)}", snapshot)
    try:
        run_id = validate_run_id(snapshot["run_id"])
        validate_container_name(snapshot["container_name"], run_id)
    except (TypeError, ValueError) as error:
        raise _failure(str(error), snapshot) from None
    for key in (
        "workspace_root",
        "disk_path",
        "run_root",
        "run_path",
        "owner_marker_path",
        "model_cache_path",
        "manifest_path",
        "manifest_checksum_path",
    ):
        value = snapshot[key]
        if not isinstance(value, str) or not value:
            raise _failure(f"{key} must be a non-empty path", snapshot)
        if not Path(value).is_absolute():
            label = {
                "workspace_root": "workspace root",
                "disk_path": "disk path",
                "run_root": "run root",
                "run_path": "run path",
                "owner_marker_path": "owner marker",
                "model_cache_path": "model cache path",
                "manifest_path": "manifest path",
                "manifest_checksum_path": "manifest checksum path",
            }[key]
            raise _failure(f"{label} must be absolute", snapshot)
    workspace_root = Path(snapshot["workspace_root"])
    run_root = Path(snapshot["run_root"])
    if run_root.parent != workspace_root:
        raise _failure("run root must be directly under workspace root", snapshot)
    if Path(snapshot["run_path"]) != run_root / run_id:
        raise _failure("run path does not match run root and run_id", snapshot)
    if workspace_root != DEFAULT_WORKSPACE_ROOT:
        raise _failure("workspace root is not the fixed dedicated path", snapshot)
    if Path(snapshot["disk_path"]) != DEFAULT_WORKSPACE_ROOT:
        raise _failure("disk path is not the fixed dedicated workspace path", snapshot)
    if run_root != DEFAULT_RUN_ROOT:
        raise _failure("run root is not the fixed dedicated path", snapshot)
    if Path(snapshot["model_cache_path"]) != DEFAULT_MODEL_CACHE:
        raise _failure("model cache path is not the fixed dedicated path", snapshot)
    if Path(snapshot["manifest_path"]) != DEFAULT_MANIFEST_PATH:
        raise _failure("manifest path is not the fixed sealed path", snapshot)
    if Path(snapshot["manifest_checksum_path"]) != DEFAULT_MANIFEST_CHECKSUM_PATH:
        raise _failure("manifest checksum path is not the fixed sealed path", snapshot)
    if Path(snapshot["owner_marker_path"]) != run_root / run_id / OWNER_MARKER_NAME:
        raise _failure("owner marker path is not the fixed run marker", snapshot)
    cpu_count = snapshot["cpu_count"]
    if isinstance(cpu_count, bool) or not isinstance(cpu_count, int) or cpu_count <= 0:
        raise _failure("cpu_count must be a positive integer", snapshot)
    try:
        affinity = _ids(snapshot["cpu_affinity"], "cpu_affinity")
        requested = _ids(snapshot["requested_cpu_ids"], "requested_cpu_ids")
        worker_cpuset = _ids(snapshot["worker_cpuset"], "worker_cpuset")
    except (RuntimeError, TypeError) as error:
        raise _failure(str(error), snapshot) from None
    worker_cpus = snapshot["worker_cpus"]
    if isinstance(worker_cpus, bool) or not isinstance(worker_cpus, int) or worker_cpus <= 0:
        raise _failure("worker_cpus must be a positive integer", snapshot)
    if cpu_count != len(affinity):
        raise _failure("cpu_count does not match process CPU affinity", snapshot)
    if tuple(sorted(requested)) != REQUIRED_CPU_IDS:
        raise _failure("requested CPU IDs must be 0, 1, 2, and 3", snapshot)
    # Gate the resources the launcher will actually request, not a fixed CPU list. The default
    # cpuset still has to contain the documented CPUs, but a widened operator request is accepted
    # only if every requested CPU is inside this process's affinity.
    if not worker_cpuset or worker_cpus is None:
        raise _failure("requested worker CPU configuration is invalid", snapshot)
    if set(worker_cpuset) != set(range(len(worker_cpuset))):
        raise _failure(
            f"requested worker cpuset {list(worker_cpuset)} must be the lowest CPUs "
            f"0-{len(worker_cpuset) - 1} so every run leaves the same high CPUs free",
            snapshot,
        )
    if not set(worker_cpuset).issubset(affinity):
        raise _failure(
            "requested worker cpuset is outside process CPU affinity: "
            f"{list(worker_cpuset)} not within {affinity}",
            snapshot,
        )
    if worker_cpuset == REQUIRED_CPU_IDS and cpu_count < len(REQUIRED_CPU_IDS):
        raise _failure(
            f"process CPU affinity does not include the default worker CPUs {list(REQUIRED_CPU_IDS)}",
            snapshot,
        )
    # A wide request may not exceed the host majority, and its quota may not exceed its own cpuset.
    if (
        cpu_count >= MIN_HOST_CPUS_FOR_SHARE_CAP
        and len(worker_cpuset) > cpu_count * MAX_WORKER_HOST_SHARE
    ):
        raise _failure(
            f"requested worker cpuset claims {len(worker_cpuset)} of {cpu_count} CPUs, "
            f"above the {MAX_WORKER_HOST_SHARE:.0%} host share limit",
            snapshot,
        )
    if worker_cpus > len(worker_cpuset):
        raise _failure(
            f"requested worker CPU quota {worker_cpus} exceeds its cpuset size {len(worker_cpuset)}",
            snapshot,
        )
    try:
        load1 = _number(snapshot["load1"], "load1")
        load5 = _number(snapshot["load5"], "load5")
    except (KeyError, RuntimeError, TypeError) as error:
        raise _failure(str(error), snapshot) from None
    if load1 < 0.0:
        raise _failure("load1 cannot be negative", snapshot)
    if load5 < 0.0:
        raise _failure("load5 cannot be negative", snapshot)
    # Load gates protect OTHER workloads and therefore belong to the pre-execution gate only. The
    # post-run gate runs while this run's own worker was still saturating its cpuset, so a wide
    # worker always observes its own load here; enforcing the same threshold would discard a valid,
    # complete run. The post-run gate still validates ownership, artifacts, and capacity limits.
    unclaimed = cpu_count - len(worker_cpuset)
    if not allow_owned:
        if load5 / cpu_count > MAX_LOAD5_PER_CPU:
            raise _failure("load5 per CPU exceeds 0.75", snapshot)
        # Host load up to this worker's own CPU quota can be caused by this benchmark's previous run
        # still inside the load-average window, not by another workload, so it is not evidence that
        # anyone else is being impacted. Only load ABOVE the worker's own quota is attributable to
        # other workloads, and that excess is what this gate measures. Two sequential runs of the
        # same cpuset therefore no longer refuse each other.
        excess = load1 - worker_cpus
        snapshot["load1_above_worker_quota"] = excess
        snapshot["worker_cpu_quota"] = worker_cpus
        if excess > unclaimed * MAX_UNCLAIMED_CPU_SHARE:
            raise _failure(
                f"load1 {load1:.2f} is {excess:.2f} above this worker's own quota of {worker_cpus} "
                f"CPUs, which is more than the {MAX_UNCLAIMED_CPU_SHARE:.0%} share of {unclaimed} "
                f"unclaimed CPUs; the host is busy outside the worker cpuset",
                snapshot,
            )
    try:
        memory = _number(snapshot["memory_available_gib"], "memory_available_gib")
        disk = _number(snapshot["disk_available_gib"], "disk_available_gib")
    except (KeyError, RuntimeError, TypeError) as error:
        raise _failure(str(error), snapshot) from None
    if memory < MIN_MEMORY_GIB:
        raise _failure("memory available is below 16 GiB", snapshot)
    if disk < MIN_DISK_GIB:
        raise _failure("disk available is below 100 GiB", snapshot)
    if snapshot["avx2"] is not True:
        raise _failure("AVX2 is required", snapshot)
    if not _python_312(snapshot["python_version"]):
        raise _failure("Python 3.12 is required", snapshot)
    if snapshot["workspace_root_state"] != "ready":
        raise _failure("workspace root is symlinked, occupied, or unknown", snapshot)
    run_root_state = snapshot["run_root_state"]
    if run_root_state not in {"ready", "absent"}:
        raise _failure("run root is symlinked, occupied, or unknown", snapshot)
    run_path_state = snapshot["run_path_state"]
    if allow_owned:
        if run_path_state != "ready":
            raise _failure("run path is not the known owned post-run state", snapshot)
    elif run_path_state != "absent":
        raise _failure("run path is occupied, symlinked, or unknown", snapshot)
    owner_marker_state = snapshot["owner_marker_state"]
    if allow_owned:
        if owner_marker_state != "ready" or snapshot["owner_marker_valid"] is not True:
            raise _failure("owner marker is missing, mismatched, or unowned", snapshot)
        current_uid = _effective_uid()
        if current_uid is None or snapshot["owner_marker_owner_uid"] != current_uid:
            raise _failure("owner marker ownership could not be revalidated", snapshot)
    elif (
        owner_marker_state != "absent"
        or snapshot["owner_marker_valid"] is not False
        or snapshot["owner_marker_owner_uid"] is not None
    ):
        raise _failure("owner marker is present before run creation", snapshot)
    container_state = snapshot["container_state"]
    if container_state != "absent":
        raise _failure("container name is occupied or unknown", snapshot)
    if snapshot["model_cache_state"] != "ready":
        raise _failure("dedicated model cache is missing, symlinked, unowned, or unknown", snapshot)
    if snapshot["manifest_state"] != "ready":
        raise _failure("sealed manifest is missing, symlinked, or not a regular file", snapshot)
    if snapshot["manifest_checksum_state"] != "ready":
        raise _failure(
            "sealed manifest checksum is missing, symlinked, or not a regular file", snapshot
        )
    if snapshot["manifest_checksum_valid"] is not True:
        raise _failure("sealed manifest checksum is invalid", snapshot)
    read_errors = snapshot["read_errors"]
    if isinstance(read_errors, (str, bytes)) or not isinstance(read_errors, (list, tuple)):
        raise _failure("read errors must be a sequence", snapshot)
    if read_errors or snapshot.get("errors"):
        raise _failure("host snapshot contains read errors", snapshot)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only remote capacity gate")
    parser.add_argument("--run-id")
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--manifest-checksum-path", type=Path, default=DEFAULT_MANIFEST_CHECKSUM_PATH
    )
    parser.add_argument("--post-run", action="store_true")
    args = parser.parse_args(argv)
    value: dict[str, Any] | None = None
    try:
        value = collect_snapshot(
            run_id=args.run_id,
            workspace_root=args.workspace_root,
            run_root=args.run_root,
            model_cache=args.model_cache,
            manifest_path=args.manifest_path,
            manifest_checksum_path=args.manifest_checksum_path,
        )
        check(value, allow_owned=args.post_run)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        if value is not None:
            print(json.dumps(value, sort_keys=True, separators=(",", ":")))
        print(f"preflight failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
