"""Import check for the consolidated vendor runtime.

Verifies that every SDK the registered adapters need is importable together, that torch is still the
CPU build, and that no CUDA-adjacent package slipped in. Run inside the image at build time so a
broken dependency fails the build rather than a benchmark run.
"""

from __future__ import annotations

import importlib.util
import sys


def main() -> int:
    import gliner2  # noqa: F401
    import lavoir  # noqa: F401
    import llm2jev  # noqa: F401
    import mojev  # noqa: F401
    import peft
    import thisthat  # noqa: F401
    import torch
    import torchvision
    import transformers
    from transformers import AutoProcessor, AutoTokenizer  # noqa: F401

    if torch.version.cuda is not None:
        raise SystemExit(f"expected a CPU torch build, found cuda={torch.version.cuda}")
    for name in ("nvidia", "triton"):
        if importlib.util.find_spec(name) is not None:
            raise SystemExit(f"CUDA guard broken: {name} is importable")

    print(f"torch {torch.__version__} | torchvision {torchvision.__version__}")
    print(f"transformers {transformers.__version__} | peft {peft.__version__}")
    print("lavoir, mojev, gliner2, llm2jev, thisthat, AutoProcessor: OK")
    print("CUDA guard held: no nvidia or triton modules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
