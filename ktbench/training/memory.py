"""Memory management for training inside a fixed unified-memory budget.

Strategies (all composable, all model-agnostic):

1. **Hard cap** - ``torch.mps.set_per_process_memory_fraction`` turns "macOS
   starts swapping / freezes" into a catchable ``OutOfMemoryError``.
2. **Sequence length capping** - done upstream by windowing (``data.max_seq_len``).
3. **Cost-model-based dynamic micro-batching** - every model declares
   ``memory_complexity = p`` and activation memory is modelled as
   ``bytes ~ a * B * L**p``. The coefficient ``a`` is *measured* by probing the
   real model on the real device; this yields a budget in cost units. Each
   logical batch is then split into the largest micro-batches that fit,
   *depending on the actual lengths in that batch* (short batches -> fewer
   splits, long batches -> more).
4. **Gradient accumulation** - micro-batch gradients are accumulated and
   normalised by the logical batch's token count, so the optimisation is
   independent of how the batch was split.
5. **OOM recovery** - if a step still OOMs, free the cache, halve the budget
   and retry the same logical batch (deterministic, no data skipped).
"""

from __future__ import annotations

import logging
import math
from typing import Callable, List, Optional

import torch
from torch import nn

from ktbench.config import MemoryConfig
from ktbench.data.dataset import Batch
from ktbench.device import GB, allocated_bytes, apply_memory_cap, empty_cache, is_oom_error, synchronize

log = logging.getLogger(__name__)

BACKWARD_FACTOR = 2.0
"""Peak-during-backward ~= this many times the activations saved by forward."""


class MemoryManager:
    def __init__(self, cfg: MemoryConfig, device: torch.device, complexity: int = 1) -> None:
        self.cfg = cfg
        self.device = device
        self.p = max(int(complexity), 1)
        self.budget: float = math.inf if cfg.cost_budget is None else float(cfg.cost_budget)
        self.oom_events = 0
        self.peak_allocated = 0
        self.bytes_per_unit: Optional[float] = None

    # ----------------------------------------------------------------- setup
    def install_cap(self) -> None:
        if self.cfg.enforce_cap:
            apply_memory_cap(self.device, self.cfg.memory_cap_gb)

    def cost(self, batch_size: int, seq_len: int) -> float:
        return float(batch_size) * float(seq_len) ** self.p

    def calibrate(
        self,
        model: nn.Module,
        make_batch: Callable[[int, int], Batch],
        loss_fn: Callable[[nn.Module, Batch], torch.Tensor],
        seq_len: int,
        optimizer_state_factor: float = 2.0,
    ) -> float:
        """Measure activation bytes per cost unit and derive the micro-batch budget."""
        if self.cfg.cost_budget is not None:
            self.budget = float(self.cfg.cost_budget)
            log.info("Memory: using manual cost budget %.3g", self.budget)
            return self.budget
        if not self.cfg.auto_budget or self.device.type == "cpu":
            # CPU RAM is not separately accountable here; rely on OOM retry.
            self.budget = math.inf
            return self.budget

        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        # params + grads + optimizer moments (Adam: 2x)
        fixed = param_bytes * (2.0 + optimizer_state_factor)
        was_training = model.training
        model.train()
        sizes = (2, 8)
        measured = []
        try:
            for b in sizes:
                model.zero_grad(set_to_none=True)
                empty_cache(self.device)
                synchronize(self.device)
                before = allocated_bytes(self.device)
                batch = make_batch(b, seq_len).to(self.device)
                loss = loss_fn(model, batch)
                synchronize(self.device)
                after_fwd = allocated_bytes(self.device)
                loss.backward()
                synchronize(self.device)
                measured.append(max(after_fwd - before, 1))
                del loss, batch
        except RuntimeError as e:
            if not is_oom_error(e):
                raise
            log.warning("Memory probe hit OOM; falling back to a conservative budget")
            measured = []
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)
            empty_cache(self.device)

        cap = self.cfg.memory_cap_gb * GB * self.cfg.safety_factor
        if len(measured) == 2 and measured[1] > measured[0]:
            per_unit = (measured[1] - measured[0]) / (self.cost(sizes[1], seq_len) - self.cost(sizes[0], seq_len))
        elif measured:
            per_unit = measured[-1] / self.cost(sizes[-1], seq_len)
        else:
            per_unit = None

        if per_unit is None:
            self.budget = self.cost(1, seq_len)
        else:
            self.bytes_per_unit = per_unit
            available = max(cap - fixed, 0.0)
            self.budget = available / (per_unit * BACKWARD_FACTOR)
        min_budget = self.cost(1, seq_len)
        if self.budget < min_budget:
            log.warning(
                "Memory: a single sequence of length %d may not fit the %.1f GB cap; "
                "consider lowering data.max_seq_len. Proceeding with 1-sequence micro-batches.",
                seq_len, self.cfg.memory_cap_gb,
            )
            self.budget = min_budget
        log.info(
            "Memory: fixed=%.1f MB, act=%.3g B/unit, budget=%.3g units (~%d seqs of len %d per micro-batch)",
            fixed / 2**20, per_unit or float("nan"), self.budget,
            self.max_rows(seq_len), seq_len,
        )
        return self.budget

    # ------------------------------------------------------------- planning
    def max_rows(self, seq_len: int) -> int:
        if math.isinf(self.budget):
            return self.cfg.max_micro_batch
        return int(max(1, min(self.cfg.max_micro_batch, self.budget // self.cost(1, max(seq_len, 1)))))

    def plan(self, batch: Batch) -> List[Batch]:
        """Split a logical batch into memory-safe micro-batches.

        Rows are sorted by length (desc) so each micro-batch is padded only to
        its own longest row; greedy packing under ``rows * L_max**p <= budget``.
        """
        n_rows = self.max_rows(batch.seq_len)
        if n_rows >= batch.batch_size:
            return [batch]
        batch = batch.sort_by_length()
        lengths = batch.mask.sum(1).tolist()
        chunks: List[Batch] = []
        start = 0
        while start < batch.batch_size:
            cap_rows = self.max_rows(int(lengths[start]))
            end = min(start + cap_rows, batch.batch_size)
            chunks.append(batch.slice(start, end))
            start = end
        return chunks

    # ------------------------------------------------------------- recovery
    def on_oom(self, batch: Batch) -> None:
        self.oom_events += 1
        current = min(self.budget, self.cost(batch.batch_size, batch.seq_len))
        self.budget = max(current / 2.0, self.cost(1, 1))
        empty_cache(self.device)
        log.warning("OOM caught -> budget halved to %.3g units (event #%d)", self.budget, self.oom_events)

    def track(self) -> None:
        self.peak_allocated = max(self.peak_allocated, allocated_bytes(self.device))

    def end_epoch(self) -> None:
        if self.cfg.empty_cache_each_epoch:
            empty_cache(self.device)
