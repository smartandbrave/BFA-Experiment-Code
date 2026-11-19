"""Read and write blocked parameter indices using a stable JSON format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

BlockedIndices = dict[str, set[int]]


def normalize_blocked_indices(
    values: Mapping[str, Iterable[int]] | None,
) -> BlockedIndices:
    if not values:
        return {}
    normalized: BlockedIndices = {}
    for module_name, indices in values.items():
        normalized[str(module_name)] = {int(index) for index in indices}
    return normalized


def load_blocked_indices(path: str | Path | None) -> BlockedIndices:
    """Load blocked indices. A missing path means an empty initial set."""
    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Blocked-index file does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Blocked-index JSON must be an object keyed by module name.")
    return normalize_blocked_indices(payload)


def save_blocked_indices(path: str | Path, values: Mapping[str, Iterable[int]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        module_name: sorted({int(index) for index in indices})
        for module_name, indices in sorted(values.items())
    }
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def add_indices(
    destination: BlockedIndices,
    module_name: str,
    indices: Iterable[int],
) -> int:
    """Add indices and return the number of newly added positions."""
    target = destination.setdefault(module_name, set())
    size_before = len(target)
    target.update(int(index) for index in indices)
    return len(target) - size_before


def count_indices(values: Mapping[str, Iterable[int]]) -> int:
    return sum(len(set(indices)) for indices in values.values())
