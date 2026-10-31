"""Model-agnostic training / evaluation engine.

The trainer only relies on the :class:`BaseKTModel` contract (``forward(batch)
-> ModelOutput``) and the :class:`Batch` structure, so every registered model
is trained and evaluated by byte-identical code - a prerequisite for fair
head-to-head comparisons.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn

from ktbench.config import MemoryConfig, TrainConfig
from ktbench.data.dataset import Batch
from ktbench.device import GB, autocast_context, is_oom_error
from ktbench.models.base import BaseKTModel
from ktbench.training.losses import kt_loss_sum
from ktbench.training.memory import MemoryManager
from ktbench.training.metrics import MetricAccumulator

log = logging.getLogger(__name__)

LOWER_IS_BETTER = {"rmse", "mae", "bce", "loss"}


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    name = cfg.optimizer.lower()
    params = model.parameter_groups(cfg.lr) if hasattr(model, "parameter_groups") else model.parameters()
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer '{cfg.optimizer}'")


class Trainer:
    def __init__(
        self,
        model: BaseKTModel,
        train_cfg: TrainConfig,
        memory_cfg: MemoryConfig,
        device: torch.device,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = train_cfg
        self.device = device
        self.model = model.to(device)
        self.forward_model: nn.Module = torch.compile(self.model) if train_cfg.compile else self.model
        self.optimizer = build_optimizer(self.model, train_cfg)
        self.memory = MemoryManager(memory_cfg, device, complexity=type(model).memory_complexity)
        self.mem_cfg = memory_cfg
        self.output_dir = Path(output_dir) if output_dir else None
        self.history: List[Dict[str, Any]] = []
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch = -1
        self.global_step = 0

    # ------------------------------------------------------------ internals
    def _forward_loss(self, batch: Batch):
        with autocast_context(self.device, self.mem_cfg.amp, self.mem_cfg.amp_dtype):
            out = self.forward_model(batch)
        total, bce = kt_loss_sum(out, batch)
        return out, total, bce

    def calibrate_memory(self, make_batch, seq_len: int) -> None:
        self.memory.install_cap()

        def loss_fn(model: nn.Module, batch: Batch) -> torch.Tensor:
            _, total, _ = self._forward_loss(batch)
            return total / max(batch.num_tokens, 1)

        self.memory.calibrate(self.model, make_batch, loss_fn, seq_len)

    def _train_step(self, batch: Batch, acc: MetricAccumulator) -> float:
        """One optimiser step on a logical batch, with micro-batching + OOM retry."""
        n_tokens = max(batch.num_tokens, 1)
        loss_scale = float(n_tokens) if self.cfg.loss_reduction == "mean" else 1.0
        for attempt in range(self.mem_cfg.max_oom_retries + 1):
            pending = []
            try:
                self.optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                for mb in self.memory.plan(batch):
                    mb = mb.to(self.device)
                    out, total, bce = self._forward_loss(mb)
                    self.memory.track()  # activations are live here -> near the step's peak
                    (total / loss_scale).backward()
                    pending.append((out.logits.detach(), mb.responses, mb.mask, bce.detach(), total.detach()))
                    del out, total, bce
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
                for logits, tgt, msk, bce_t, tot_t in pending:
                    acc.update(logits, tgt, msk, float(bce_t))
                    step_loss += float(tot_t)
                return step_loss / n_tokens
            except RuntimeError as e:
                if not (is_oom_error(e) and self.mem_cfg.oom_retry) or attempt == self.mem_cfg.max_oom_retries:
                    raise
                pending.clear()
                self.optimizer.zero_grad(set_to_none=True)
                self.memory.on_oom(batch)
        raise RuntimeError("unreachable")

    def train_epoch(self, loader: Iterable[Batch]) -> Dict[str, float]:
        self.model.train()
        acc = MetricAccumulator()
        for i, batch in enumerate(loader):
            loss = self._train_step(batch, acc)
            self.global_step += 1
            if self.cfg.log_every and self.global_step % self.cfg.log_every == 0:
                log.info("step %d | loss %.4f", self.global_step, loss)
        return acc.compute()

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Batch]) -> Dict[str, float]:
        self.model.eval()
        acc = MetricAccumulator()
        for batch in loader:
            for attempt in range(self.mem_cfg.max_oom_retries + 1):
                try:
                    results = []
                    for mb in self.memory.plan(batch):
                        mb = mb.to(self.device)
                        out, _, bce = self._forward_loss(mb)
                        results.append((out.logits, mb.responses, mb.mask, float(bce)))
                    for r in results:
                        acc.update(*r)
                    break
                except RuntimeError as e:
                    if not is_oom_error(e) or attempt == self.mem_cfg.max_oom_retries:
                        raise
                    self.memory.on_oom(batch)
        return acc.compute()

    # ----------------------------------------------------------------- fit
    def _is_better(self, value: float, best: float) -> bool:
        if math.isnan(value):
            return False
        return value < best if self.cfg.monitor in LOWER_IS_BETTER else value > best

    def fit(self, train_loader, valid_loader) -> List[Dict[str, Any]]:
        best = math.inf if self.cfg.monitor in LOWER_IS_BETTER else -math.inf
        wait = 0
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.perf_counter()
            train_m = self.train_epoch(train_loader)
            if hasattr(self.model, "calibrate_rasch"):
                self.model.calibrate_rasch(train_loader, self.device)
            valid_m = self.evaluate(valid_loader)
            self.memory.end_epoch()
            rec = {"epoch": epoch, "time_s": round(time.perf_counter() - t0, 2),
                   "peak_alloc_gb": round(self.memory.peak_allocated / GB, 3),
                   **{f"train_{k}": v for k, v in train_m.items()},
                   **{f"valid_{k}": v for k, v in valid_m.items()}}
            self.history.append(rec)
            log.info(
                "epoch %3d | train loss %.4f auc %.4f | valid auc %.4f rmse %.4f acc %.4f | %.1fs | peak %.2f GB",
                epoch, train_m.get("loss", float("nan")), train_m.get("auc", float("nan")),
                valid_m.get("auc", float("nan")), valid_m.get("rmse", float("nan")),
                valid_m.get("acc", float("nan")), rec["time_s"], rec["peak_alloc_gb"],
            )
            score = valid_m.get(self.cfg.monitor, float("nan"))
            if self._is_better(score, best):
                best, wait, self.best_epoch = score, 0, epoch
                self.best_state = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}
                if self.output_dir and self.cfg.save_checkpoints:
                    self.save_checkpoint(self.output_dir / "best.pt", epoch)
            else:
                wait += 1
                if wait >= self.cfg.patience:
                    log.info("Early stopping at epoch %d (best epoch %d, %s=%.4f)",
                             epoch, self.best_epoch, self.cfg.monitor, best)
                    break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if self.output_dir:
            with open(self.output_dir / "history.json", "w") as f:
                json.dump(self.history, f, indent=2)
        return self.history

    # ---------------------------------------------------------- checkpoints
    def save_checkpoint(self, path: Path, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model": type(self.model).registry_name,
            "hparams": self.model.hparams(),
            "meta": self.model.meta.to_dict(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt
