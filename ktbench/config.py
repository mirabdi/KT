"""Typed experiment configuration.

Configuration is split into independent sections that map 1:1 to the
framework layers, so changing one concern never requires touching another:

* :class:`DataConfig`    -> ``ktbench.data``
* :class:`ModelConfig`   -> ``ktbench.models`` (``params`` validated by the model's own ``Config``)
* :class:`TrainConfig`   -> ``ktbench.training.trainer``
* :class:`MemoryConfig`  -> ``ktbench.training.memory``

Everything round-trips through plain dicts / YAML and supports dotted CLI
overrides such as ``train.lr=3e-4 model.params.d_model=128``.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    adapter: str = "synthetic"
    """Registered dataset adapter name (see ``ktbench.data.adapters``)."""
    adapter_params: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments forwarded to the adapter constructor (paths, column maps...)."""
    split: List[float] = field(default_factory=lambda: [0.7, 0.1, 0.2])
    """Student-level train/valid/test fractions (ignored if the adapter supplies a split)."""
    max_seq_len: int = 200
    """Sequence length cap. Longer histories are windowed (like the AKT reference code)."""
    window_stride: Optional[int] = None
    """Stride for windowing; ``None`` = non-overlapping windows of ``max_seq_len``."""
    min_seq_len: int = 2
    """Windows / students with fewer interactions are dropped."""
    num_workers: int = 0
    """DataLoader workers. 0 is usually fastest on macOS for these small tensors."""
    length_bucketing: bool = True
    """Group similar-length sequences into the same batch to minimise padding."""


@dataclass
class ModelConfig:
    name: str = "akt"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 24
    """*Logical* batch size (sequences per optimiser step). Physically split into
    micro-batches by the memory manager when necessary (gradient accumulation)."""
    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 0.0
    """Max global grad-norm; ``<= 0`` disables clipping."""
    loss_reduction: str = "sum"
    """``sum`` (paper default: backprop summed token loss) or ``mean`` (divide by batch tokens)."""
    patience: int = 10
    """Early-stopping patience in epochs, monitored on validation AUC."""
    monitor: str = "auc"
    log_every: int = 0
    """Log training progress every N optimiser steps (0 = only per epoch)."""
    compile: bool = False
    """Wrap the model with ``torch.compile`` (experimental on MPS)."""
    save_checkpoints: bool = True


@dataclass
class MemoryConfig:
    memory_cap_gb: float = 11.0
    """Hard ceiling for accelerator allocations. On a 16 GB M3 the OS needs ~5 GB."""
    enforce_cap: bool = True
    """On MPS, call ``torch.mps.set_per_process_memory_fraction`` so we OOM (and
    recover) instead of pushing macOS into swap."""
    safety_factor: float = 0.75
    """Fraction of the cap the auto-sizer is allowed to plan for."""
    auto_budget: bool = True
    """Probe the model on the device to derive the micro-batch cost budget."""
    cost_budget: Optional[float] = None
    """Manual micro-batch budget in model cost units (B * L^p). Overrides probing."""
    max_micro_batch: int = 256
    oom_retry: bool = True
    """On OOM, halve the budget, clear cache and retry the step."""
    max_oom_retries: int = 6
    empty_cache_each_epoch: bool = True
    amp: bool = False
    """Mixed precision via ``torch.autocast`` (bf16 on MPS/CUDA)."""
    amp_dtype: str = "bfloat16"


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 42
    device: str = "auto"
    output_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        return _from_dict(cls, d)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Optional[List[str]] = None) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        raw = apply_overrides(raw, overrides or [])
        return cls.from_dict(raw)

    def save_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def copy(self, **changes: Any) -> "ExperimentConfig":
        new = copy.deepcopy(self)
        for k, v in changes.items():
            setattr(new, k, v)
        return new


# ---------------------------------------------------------------------- helpers
def _from_dict(cls: Any, d: Dict[str, Any]) -> Any:
    if not dataclasses.is_dataclass(cls):
        return d
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    hints = _resolved_types(cls)
    for name, value in d.items():
        sub = hints.get(name)
        if dataclasses.is_dataclass(sub) and isinstance(value, dict):
            kwargs[name] = _from_dict(sub, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _resolved_types(cls: Any) -> Dict[str, Any]:
    import typing

    try:
        return typing.get_type_hints(cls)
    except Exception:  # pragma: no cover - defensive
        return {}


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(raw: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` style overrides (values parsed as YAML)."""
    raw = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        key, value = item.split("=", 1)
        node = raw
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_scalar(value)
    return raw


def build_dataclass(cls: Any, params: Dict[str, Any], owner: str) -> Any:
    """Instantiate a (model-specific) Config dataclass with a helpful error."""
    try:
        return _from_dict(cls, params or {})
    except (TypeError, ValueError) as e:
        valid = [f.name for f in dataclasses.fields(cls)]
        raise ValueError(f"Invalid params for {owner}: {e}. Valid keys: {valid}") from e
