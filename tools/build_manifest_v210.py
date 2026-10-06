#!/usr/bin/env python3
"""Build the corrected manifest (dataset version 2.1.0).

Changes gold labels only. Answers, criteria, prompts, state, question ids and
order indices are untouched, so every stored run can be re-scored offline with
no inference and no GPU.

Corrected suites:
  agnews, banking77_12, mnli, sst5  -> adopt the public source label
  emotion                           -> the reviewed decision set, case by case

The old manifest is never modified. This writes a new file beside it.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import os
import pathlib
import sys
import urllib.request

# Defaults to the repo layout, but both the repo root and the source manifest can be
# overridden so this runs on a remote host where the script is copied in on its own.
REPO = pathlib.Path(os.environ.get("SYSONE_REPO", pathlib.Path(__file__).resolve().parent.parent))
V2 = pathlib.Path(os.environ.get("SYSONE_MANIFEST_DIR", REPO / "datasets" / "v2"))
# Allow an override so the script runs on a remote host where the review file
# was copied in beside it rather than at its local absolute path.
DECISIONS = pathlib.Path(
    os.environ.get("EMOTION_DECISIONS")
    or pathlib.Path(__file__).resolve().parent / "emotion_final.confirmed.json"
)

ADOPT_UPSTREAM = ("agnews", "banking77_12", "mnli", "sst5")
EMOTION = "emotion"
PROTOCOL = "source-adopted-and-reviewed-v1"


def upstream_labels(suite: str) -> dict[str, str]:
    """Public gold for a suite, keyed exactly as we key our own manifest."""
    if suite == "agnews":
        from datasets import load_dataset

        d = load_dataset("fancyzhx/ag_news", split="test")
        # Upstream ships Capitalised names with a slash; our manifest uses the
        # lowercase, de-slashed form. Compare and write in OUR vocabulary.
        names = ["world", "sports", "business", "scitech"]
        return {r["text"].strip(): names[int(r["label"])] for r in d}
    if suite == "emotion":
        from datasets import load_dataset

        d = load_dataset("dair-ai/emotion", split="test")
        names = d.features["label"].names
        return {r["text"].strip(): names[int(r["label"])] for r in d}
    if suite == "sst5":
        from datasets import load_dataset

        d = load_dataset("SetFit/sst5", split="test")
        # Our manifest stores the ordinal int directly, so keep the int.
        return {r["text"].strip(): int(r["label"]) for r in d}
    if suite == "mnli":
        from datasets import load_dataset

        out: dict[str, str] = {}
        for sp in ("validation_matched", "validation_mismatched", "test_matched", "test_mismatched"):
            d = load_dataset("nyu-mll/glue", "mnli", split=sp)
            names = d.features["label"].names
            for r in d:
                out[r["premise"].strip() + "\t" + r["hypothesis"].strip()] = names[int(r["label"])]
        return out
    if suite == "banking77_12":
        url = ("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
               "master/banking_data/test.csv")
        txt = urllib.request.urlopen(url, timeout=60).read().decode()
        return {r["text"].strip(): r["category"].lower() for r in csv.DictReader(io.StringIO(txt))}
    raise ValueError(suite)


def case_key(case: dict, suite: str) -> str | None:
    st = case["state"]
    if suite in ("agnews", "emotion", "sst5", "banking77_12"):
        return st["text"].strip()
    if suite == "mnli":
        return st["premise"].strip() + "\t" + st["hypothesis"].strip()
    return None


def qid_for(suite: str) -> str:
    return {"agnews": "topic", "emotion": "emotion", "sst5": "sentiment",
            "banking77_12": "intent", "mnli": "relation"}[suite]


def main() -> int:
    decisions = json.loads(DECISIONS.read_text())
    upstream = {s: upstream_labels(s) for s in (*ADOPT_UPSTREAM, EMOTION)}
    qid = {s: qid_for(s) for s in upstream}

    src = (V2 / "manifest.jsonl").read_text().splitlines()
    out_lines: list[str] = []
    stats = collections.Counter()
    per_suite = collections.defaultdict(lambda: collections.Counter())

    for line in src:
        case = json.loads(line)
        suite = case["suite_id"]

        if suite in ADOPT_UPSTREAM:
            k = case_key(case, suite)
            up = upstream[suite].get(k)
            if up is None:
                raise SystemExit(f"{case['case_id']}: no upstream label for {suite}")
            if suite == "banking77_12":
                cur = str(case["expected"][qid[suite]])
                if up != cur and cur.lower() == up.lower():
                    # cosmetic casing difference, no semantic change
                    up = cur
            q = qid[suite]
            if case["expected"][q] != up:
                stats["changed"] += 1
                per_suite[suite]["changed"] += 1
                case["expected"][q] = up
                case["label_provenance"] = PROTOCOL
            else:
                stats["already_correct"] += 1
                per_suite[suite]["already_correct"] += 1

        elif suite == EMOTION:
            cid = case["case_id"]
            if cid in decisions:
                want = decisions[cid]["final"]
                q = qid[suite]
                if case["expected"][q] != want:
                    stats["changed"] += 1
                    per_suite[suite]["changed"] += 1
                    case["expected"][q] = want
                case["label_provenance"] = PROTOCOL
            else:
                stats["untouched_review_agree"] += 1
                per_suite[suite]["untouched"] += 1

        case["dataset_version"] = "2.1.0"
        out_lines.append(json.dumps(case, sort_keys=True, separators=(",", ":")))

    out = V2 / "manifest.v2.1.0.jsonl"
    body = ("\n".join(out_lines) + "\n").encode()
    out.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    (V2 / "manifest.v2.1.0.sha256").write_text(f"{digest}  {out.name}\n")

    print(f"  wrote {out.name}: {len(out_lines)} cases")
    print(f"  sha256: {digest}")
    print(f"  changed gold        : {stats['changed']}")
    print(f"  already matched     : {stats['already_correct'] + stats['untouched_review_agree']}")
    print()
    print("  per suite:")
    for s in (*ADOPT_UPSTREAM, EMOTION):
        c = per_suite[s]
        print(f"    {s:14} changed {c['changed']:4}   already correct {c['already_correct'] + c['untouched']:4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())