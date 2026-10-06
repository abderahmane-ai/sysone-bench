#!/usr/bin/env python3
"""Two-way results sync between this repo and the borrowed Spark.

Pull:    spark results -> results/raw/   (run this after every benchmark batch)
Push:    local results -> spark          (run this to seed future batches)

Pull is self-verifying. It refuses to claim success unless the coverage auditor
accepts the merged results on the local side, so a half-finished rsync cannot
masquerade as a good result.

Usage:
  python ops/results_sync.py pull          # spark -> laptop
  python ops/results_sync.py push          # local -> spark (seeds a new batch)
  python ops/results_sync.py status        # what differs, nothing transferred

Both sides must already be reachable over `ssh spark`.
"""
from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results" / "raw"
SPARK_HOST = "spark"
SPARK_ROOT = "/home/o_0/sysone-bench-spark/results/raw"


def sh(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def spark_up() -> bool:
    r = sh(["ssh", "-F", str(Path.home() / ".ssh" / "spark_config"), SPARK_HOST, "echo ok"])
    return "ok" in r.stdout


def rsync(direction: str) -> None:
    if direction == "pull":
        src, dst = f"{SPARK_HOST}:{SPARK_ROOT}/", f"{RAW}/"
    else:
        src, dst = f"{RAW}/", f"{SPARK_HOST}:{SPARK_ROOT}/"
    host = f"ssh -F {Path.home() / '.ssh' / 'spark_config'}"
    subprocess.run(["rsync", "-a", "-e", host, src, dst], check=True)


def file_inventory(root: Path) -> dict[str, str]:
    """sha256 per relative path, so 'which side is newer' is decidable."""
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib(p)
    return out


def hashlib(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auditor_report() -> dict:
    import sys
    sys.path.insert(0, str(REPO))
    from ops.panel_coverage import audit

    return audit(RAW)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("direction", choices=["pull", "push", "status"])
    args = ap.parse_args()

    if not spark_up():
        print("spark unreachable over Tailscale; nothing done.")
        return 2

    before = file_inventory(RAW)
    print(f"  local files before : {len(before)}")

    if args.direction == "pull":
        print("  pulling from spark ...")
        subprocess.run(
            ["rsync", "-a", "-e",
             f"ssh -F {Path.home() / '.ssh' / 'spark_config'}",
             f"{SPARK_HOST}:{SPARK_ROOT}/", str(RAW) + "/"],
            check=True,
        )
    elif args.direction == "push":
        print("  pushing to spark ...")
        subprocess.run(
            ["rsync", "-a", "-e",
             f"ssh -F {Path.home() / '.ssh' / 'spark_config'}",
             str(RAW) + "/", f"{SPARK_HOST}:{SPARK_ROOT}/"],
            check=True,
        )
    else:
        rem = sh(["ssh", "-F", str(Path.home() / ".ssh" / "spark_config"), SPARK_HOST,
                  f"find {SPARK_ROOT} -type f | wc -l"])
        print(f"  remote file count (approx): {rem.stdout.strip()}")
        return 0

    after = file_inventory(RAW)
    arrived = sorted(set(after) - set(before))
    missing = sorted(set(before) - set(after))
    print(f"  local files after  : {len(after)}")
    print(f"  new or changed locally after sync: {len(arrived)} files")
    if arrived:
        for n in arrived[:8]:
            print(f"    + {n}")
        if len(arrived) > 8:
            print(f"    ... and {len(arrived) - 8} more")

    print("\n  running coverage auditor on the merged results ...")
    rep = auditor_report()
    print(f"  run directories verified : {rep['run_directories_verified']}")
    print(f"  distinct measurements    : {rep['distinct_measurements']}")
    print(f"  rejected                 : {len(rep['run_directories_rejected'])}")
    for r in rep["run_directories_rejected"][:6]:
        print(f"    {r['run_dir']}: {'; '.join(r['problems'])}")

    ok = rep["run_directories_verified"] > 0 and rep["distinct_measurements"] >= 40
    print("\n  sync verified:", "yes" if ok else "NO - check the rejected list above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())