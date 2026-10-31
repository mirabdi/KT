"""A tiny, generic plug-in registry.

Both models and dataset adapters are discovered through registries so the
pipeline never needs ``if name == "akt": ...`` branches. Registering a new
component is a one-line decorator::

    @MODEL_REGISTRY.register("my_model")
    class MyModel(BaseKTModel): ...
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterator, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: Dict[str, T] = {}

    def register(self, name: str | None = None) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            key = (name or getattr(obj, "name", None) or obj.__name__).lower()  # type: ignore[attr-defined]
            if key in self._items and self._items[key] is not obj:
                raise KeyError(f"{self.kind} '{key}' is already registered")
            self._items[key] = obj
            # Expose the registry key on the class for introspection.
            try:
                setattr(obj, "registry_name", key)
            except (AttributeError, TypeError):
                pass
            return obj

        return deco

    def get(self, name: str) -> T:
        key = name.lower()
        if key not in self._items:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}"
            )
        return self._items[key]

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def names(self) -> list[str]:
        return sorted(self._items)
