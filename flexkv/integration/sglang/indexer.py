"""Resolve logical indexer layer IDs from the pool metadata."""

from typing import Optional, Sequence


def active_index_layers(mask: Optional[Sequence[bool]], count: int) -> tuple[int, ...]:
    if mask is None:
        return tuple(range(count))
    if len(mask) != count or any(type(skip) is not bool for skip in mask):
        raise ValueError("Expected one skip_topk boolean per indexer layer")
    return tuple(i for i, skip in enumerate(mask) if not skip)
