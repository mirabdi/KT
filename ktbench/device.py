"""Device selection and backend-agnostic memory introspection.

Priority for ``device="auto"``: Apple Silicon ``mps`` -> ``cuda`` -> ``cpu``.
All memory helpers degrade gracefully to no-ops on backends that do not
expose the corresponding statistics.
"""

from __future__ import annotations

import gc
import logging
import random
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

GB = 1024**3


def resolve_device(spec: str = "auto") -> torch.device:
    spec = (spec or "auto").lower()
    if spec == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    dev = torch.device(spec)
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available on this machine")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine")
    return dev


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds mps / cuda generators


def apply_memory_cap(device: torch.device, cap_gb: float) -> Optional[float]:
    """Install a hard allocator limit so the process OOMs (recoverably) instead
    of driving macOS into swap. Returns the applied fraction, if any."""
    if device.type == "mps":
        try:
            recommended = torch.mps.recommended_max_memory()
        except AttributeError:  # older torch: assume ~2/3 of system RAM
            recommended = None
        if not recommended:
            return None
        fraction = min(max(cap_gb * GB / recommended, 0.05), 2.0)
        torch.mps.set_per_process_memory_fraction(fraction)
        log.info(
            "MPS memory cap: %.2f GB (fraction %.3f of recommended %.2f GB)",
            cap_gb, fraction, recommended / GB,
        )
        return fraction
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        fraction = min(cap_gb * GB / total, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        return fraction
    return None


def empty_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def allocated_bytes(device: torch.device) -> int:
    """Bytes currently held by live tensors on ``device`` (0 if unknown)."""
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    return 0


def driver_bytes(device: torch.device) -> int:
    """Total bytes reserved from the driver (live + cached)."""
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.memory_reserved(device))
    return 0


def is_oom_error(err: BaseException) -> bool:
    msg = str(err).lower()
    return isinstance(err, torch.OutOfMemoryError) or "out of memory" in msg


def autocast_context(device: torch.device, enabled: bool, dtype: str = "bfloat16"):
    if not enabled or device.type == "cpu":
        import contextlib

        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=getattr(torch, dtype))
