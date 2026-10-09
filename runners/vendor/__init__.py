"""Vendor decision-model adapters.

Every adapter here speaks the System One contract: a state plus a mapping of question ID to a typed
question (choice, noul, or score), returning an answer keyed by exactly those IDs. The families
differ in how they are loaded and which call they expose, not in the contract they answer, so the
shared normalization lives in `common` and each family module only supplies a loader and a call.

Adapters project rich vendor payloads onto the exact declared field set and drop everything else,
rather than loosening the contract. Nothing here reads the caller's inputs without deep-copying
them first.
"""

from runners.vendor.common import (
    VENDOR_ADAPTER_VERSION,
    VendorRunner,
    choice_answer,
    noul_answer,
    project_vendor_answer,
    score_answer,
    state_to_text,
)

__all__ = [
    "VENDOR_ADAPTER_VERSION",
    "VendorRunner",
    "choice_answer",
    "noul_answer",
    "project_vendor_answer",
    "score_answer",
    "state_to_text",
]
