"""Lightweight registry pattern for pluggable components."""
from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name -> factory mapping for swappable components.

    Used by encoders, clusterers, feature extractors, and pathology rule sets
    so that experiments can be assembled from YAML config files without
    touching call sites.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        # Decorator that records a factory under `name`.
        def deco(obj: T) -> T:
            if name in self._items:
                raise ValueError(
                    f"Duplicate registration in {self._kind} registry: {name}"
                )
            self._items[name] = obj
            return obj

        return deco

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(
                f"{self._kind!r} has no entry {name!r}. "
                f"Available: {sorted(self._items)}"
            )
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)


# Global registries — modules call `register` at import time.
ENCODERS: Registry = Registry("encoders")
CLUSTERERS: Registry = Registry("clusterers")
FEATURE_EXTRACTORS: Registry = Registry("feature_extractors")
PATHOLOGY_RULE_SETS: Registry = Registry("pathology_rule_sets")
