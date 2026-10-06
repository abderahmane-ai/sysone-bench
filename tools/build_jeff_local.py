#!/usr/bin/env python3
"""Materialise a loadable jeff checkpoint from knowledgator/gliformer-large-v1.

The checkpoint stores the DeBERTa encoder under
token_rep_layer.bert_layer.model.*, while gliner2 2.0.0's span-head loader
expects it under encoder.*. The tensors are identical; only their names differ.

This writes a NEW checkpoint into a private directory - the upstream snapshot
and the shared HF cache are never modified. Reversible by construction: the
mapping is prefix-for-prefix, and every non-encoder key (rnn.*, heads.*) is
copied unchanged. The repair is recorded as loader_remap in the run's provenance
rather than left implicit.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

OUT_ROOT = pathlib.Path("/workspace/jeff-local")
SRC = ("~/.cache/huggingface/hub/models--knowledgator--gliformer-large-v1/"
       "snapshots/d0a4e53d09cebe6bc963dd9be319d4279084bb2d")
# Allow a mounted-cache override so the builder works in containers where the
# cache is bind-mounted at a different point.
SRC = pathlib.Path(os.environ.get("JEFF_SNAPSHOT", SRC)).expanduser()

PREFIX_FROM = "token_rep_layer.bert_layer.model."
PREFIX_TO = "encoder."


def main() -> int:
    import torch

    src = pathlib.Path(SRC).expanduser()
    sd = torch.load(src / "pytorch_model.bin", map_location="cpu", weights_only=True)
    remapped, moved = {}, 0
    for key, tensor in sd.items():
        if key.startswith(PREFIX_FROM):
            remapped[PREFIX_TO + key[len(PREFIX_FROM):]] = tensor
            moved += 1
        else:
            remapped[key] = tensor

    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(remapped, out_dir / "pytorch_model.bin")

    for name in ("gliner_config.json", "config.json", "tokenizer.json",
                 "tokenizer_config.json"):
        if (src / name).exists():
            (out_dir / name).write_bytes((src / name).read_bytes())
    enc = src / "encoder_config" / "config.json"
    (out_dir / "encoder_config").mkdir(exist_ok=True)
    if enc.exists():
        (out_dir / "encoder_config" / "config.json").write_bytes(enc.read_bytes())

    (out_dir / "loader_remap.json").write_text(json.dumps({
        "loader_remap": {
            "from": PREFIX_FROM,
            "to": PREFIX_TO,
            "tensors_moved": moved,
            "tensors_total": len(sd),
            "reason": ("gliner2 2.0.0 expects the encoder under encoder.*; the checkpoint "
                       "stores the same tensors under token_rep_layer.bert_layer.model.*"),
            "reversible": True,
        }
    }, indent=1))
    print(f"  remapped {moved}/{len(sd)} tensors")
    print(f"  wrote {out_dir/'pytorch_model.bin'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
